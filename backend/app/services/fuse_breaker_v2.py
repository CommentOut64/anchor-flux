"""
FuseBreakerV2 - 熔断决策器 V2 增强版

负责在SenseVoice转录过程中监控质量，当检测到低置信度+BGM标签时触发升级分离。

关键配置：
- max_fuse_retry=1：默认只允许一次熔断升级（NONE → HTDEMUCS）
- auto_upgrade=False：默认不启用第二次自动升级到 MDX_EXTRA
- 第二次熔断升级作为可选配置暴露，需显式启用
"""

import logging
from typing import Optional, Dict

from app.services.audio.chunk_engine import AudioChunk
from app.services.demucs_service import DemucsService
from app.models.circuit_breaker_models import (
    FuseAction,
    FuseDecision,
    SeparationLevel
)


class FuseBreakerV2:
    """
    熔断决策器 V2 - 增强版

    职责：
    - 监控SenseVoice转录质量
    - 根据置信度和事件标签决定是否触发熔断
    - 执行升级分离并回溯重做
    """

    def __init__(
        self,
        max_retry: int = 1,
        confidence_threshold: float = 0.5,
        auto_upgrade: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化熔断决策器

        Args:
            max_retry: 最大重试次数，默认1次（只升级到 HTDEMUCS）
            confidence_threshold: 置信度阈值，默认0.5
            auto_upgrade: 是否启用第二次自动升级到 MDX_EXTRA，默认False
            logger: 日志记录器，如果为None则创建新的
        """
        self.max_retry = max_retry
        self.confidence_threshold = confidence_threshold
        self.auto_upgrade = auto_upgrade
        self.logger = logger or logging.getLogger(__name__)

        # 事件标签权重
        self.event_weights: Dict[str, float] = {
            'BGM': 1.0,      # 最高优先级
            'Music': 0.9,
            'Noise': 0.8,
            'Applause': 0.6  # 较低优先级
        }

    def should_fuse(
        self,
        chunk: AudioChunk,
        sv_result: dict
    ) -> FuseDecision:
        """
        熔断决策

        根据SenseVoice转录结果判断是否需要触发熔断升级

        Args:
            chunk: 待判断的AudioChunk
            sv_result: SenseVoice转录结果

        Returns:
            FuseDecision: 熔断决策结果
        """
        confidence = sv_result.get('confidence', 1.0)
        event_tag = sv_result.get('event_tag')

        # 检查是否达到重试上限
        if chunk.fuse_retry_count >= self.max_retry:
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason="已达最大重试次数"
            )

        # 检查置信度
        if confidence >= self.confidence_threshold:
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason=f"置信度足够 ({confidence:.2f})"
            )

        # 检查事件标签
        if not event_tag or event_tag not in self.event_weights:
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason="无背景干扰标签"
            )

        # 计算加权置信度
        weight = self.event_weights[event_tag]
        weighted_confidence = confidence * (1 + weight)

        if weighted_confidence >= self.confidence_threshold:
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason=f"加权置信度足够 ({weighted_confidence:.2f})"
            )

        # 触发熔断升级
        next_level = self._get_next_level(chunk)
        if next_level is None:
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason="已达最高分离级别"
            )

        return FuseDecision(
            action=FuseAction.UPGRADE_SEPARATION,
            target_level=next_level,
            reason=f"置信度低 ({confidence:.2f}) + {event_tag}标签"
        )

    def _get_next_level(self, chunk: AudioChunk) -> Optional[SeparationLevel]:
        """
        获取下一个分离级别

        升级路径：
        - 第一次重试：NONE → HTDEMUCS
        - 第二次重试（auto_upgrade=True）：HTDEMUCS → MDX_EXTRA（跳级）
        - 第二次重试（auto_upgrade=False）：HTDEMUCS → MDX_EXTRA（顺序）

        Args:
            chunk: AudioChunk

        Returns:
            下一个分离级别，如果无法升级则返回None
        """
        current = chunk.separation_level

        # 第一次重试：NONE → HTDEMUCS
        if current == SeparationLevel.NONE:
            return SeparationLevel.HTDEMUCS

        # 第二次重试：自动升级到最高级别
        if self.auto_upgrade and chunk.fuse_retry_count == 1:
            return SeparationLevel.MDX_EXTRA

        # 否则按顺序升级
        return current.next_level()

    async def execute_upgrade(
        self,
        chunk: AudioChunk,
        target_level: SeparationLevel,
        demucs_service: DemucsService
    ) -> AudioChunk:
        """
        执行升级分离

        使用原始音频和更强的模型重新分离

        Args:
            chunk: 待升级的AudioChunk
            target_level: 目标分离级别
            demucs_service: Demucs服务实例

        Returns:
            更新后的AudioChunk

        Raises:
            ValueError: 如果原始音频不存在
        """
        self.logger.info(
            f"Chunk {chunk.index}: 升级分离 "
            f"{chunk.separation_level.value} → {target_level.value}"
        )

        # 使用原始音频
        if chunk.original_audio is None:
            raise ValueError(
                f"Chunk {chunk.index}: 原始音频不存在，无法回溯"
            )

        # 执行分离
        model = target_level.value
        self.logger.debug(
            f"Chunk {chunk.index}: 使用模型 {model} 重新分离"
        )

        # 调用DemucsService的chunk级别分离方法
        separated_audio = demucs_service.separate_chunk(
            audio=chunk.original_audio,
            model=model,
            sr=chunk.sample_rate
        )

        self.logger.info(
            f"Chunk {chunk.index}: 分离完成，模型={model}"
        )

        # 更新chunk
        chunk.audio = separated_audio
        chunk.separation_level = target_level
        chunk.separation_model = model
        chunk.is_separated = True
        chunk.fuse_retry_count += 1

        self.logger.info(
            f"Chunk {chunk.index}: 升级分离完成，重试次数: {chunk.fuse_retry_count}"
        )

        return chunk

    def get_statistics(self) -> dict:
        """
        获取熔断统计信息

        Returns:
            统计信息字典
        """
        return {
            "max_retry": self.max_retry,
            "confidence_threshold": self.confidence_threshold,
            "auto_upgrade": self.auto_upgrade,
            "event_weights": self.event_weights
        }
