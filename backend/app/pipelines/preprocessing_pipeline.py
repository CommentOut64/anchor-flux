"""
PreprocessingPipeline - 预处理流水线（新架构）

负责统一管理所有预处理步骤：
1. 音频提取 + VAD切分
2. 频谱分诊（可选）
3. 人声分离（可选，支持全局/按需模式）

与旧的 AudioProcessingPipeline 的区别：
- 旧版：整轨分离 + VAD切分
- 新版：VAD切分 + 频谱分诊 + 按需分离（Stage模式）

V3.7 更新：
- 集成 CancellationToken 支持暂停/取消
- 支持断点续传检查点保存
"""

import logging
from typing import List, Optional, TYPE_CHECKING
from pathlib import Path

from app.services.audio.chunk_engine import ChunkEngine, AudioChunk
from app.services.audio.vad_service import VADConfig
from app.pipelines.stages.spectral_triage_stage import SpectralTriageStage
from app.pipelines.stages.separation_stage import SeparationStage
from app.models.job_models import PreprocessingConfig, JobState

# V3.7: 导入取消令牌
if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken


class PreprocessingPipeline:
    """
    预处理流水线 - 统一管理所有预处理步骤

    采用 Stage 模式，支持灵活的预处理流程配置。

    V3.7: 支持 CancellationToken 实现暂停/取消/断点续传
    """

    def __init__(
        self,
        config: PreprocessingConfig,
        chunk_engine: Optional[ChunkEngine] = None,
        logger: Optional[logging.Logger] = None,
        cancellation_token: Optional["CancellationToken"] = None  # V3.7: 新增
    ):
        """
        初始化预处理流水线

        Args:
            config: 预处理配置
            chunk_engine: 音频切分引擎（可选）
            logger: 日志记录器（可选）
            cancellation_token: 取消令牌（可选，V3.7）
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.cancellation_token = cancellation_token  # V3.7

        # 初始化 ChunkEngine（用于音频提取和VAD切分）
        self.chunk_engine = chunk_engine or ChunkEngine(logger=self.logger)

        # 初始化频谱分诊阶段（如果启用）
        if config.enable_spectral_triage:
            self.spectral_triage_stage = SpectralTriageStage(
                threshold=config.spectrum_threshold,
                logger=self.logger,
                cancellation_token=cancellation_token  # V3.7: 传递令牌
            )
            self.logger.info(
                f"频谱分诊已启用: threshold={config.spectrum_threshold}"
            )
        else:
            self.spectral_triage_stage = None
            self.logger.info("频谱分诊已禁用")

        # 初始化人声分离阶段（如果启用）
        enable_demucs = config.demucs_strategy != "off"
        if enable_demucs:
            self.separation_stage = SeparationStage(
                mode=config.separation_mode,
                logger=self.logger,
                cancellation_token=cancellation_token  # V3.7: 传递令牌
            )
            self.logger.info(
                f"人声分离已启用: mode={config.separation_mode}, "
                f"model={config.demucs_model}"
            )
        else:
            self.separation_stage = None
            self.logger.info("人声分离已禁用")

    async def process(
        self,
        video_path: str,
        job_state: Optional[JobState] = None,
        job_dir: Optional[Path] = None  # V3.7: 用于保存检查点
    ) -> List[AudioChunk]:
        """
        执行完整的预处理流程

        流程：
        1. 音频提取 + VAD切分 -> List[AudioChunk]
        2. 频谱分诊（可选）-> 标记 needs_separation
        3. 人声分离（可选）-> 分离标记的chunk

        Args:
            video_path: 视频/音频文件路径
            job_state: 任务状态（可选，用于进度回调）
            job_dir: 任务目录（可选，V3.7 用于保存检查点）

        Returns:
            List[AudioChunk]: 预处理完成的 Chunk 列表
        """
        self.logger.info(f"开始预处理流程: {video_path}")
        token = self.cancellation_token  # V3.7: 简化引用

        # V3.7.2: Stage 1 拆分为两个原子区域
        # Stage 1a: 音频提取（FFmpeg，耗时 1-5 秒）
        self.logger.info("Stage 1a: 音频提取")

        if token:
            token.enter_atomic_region("ffmpeg_extract")

        try:
            # 音频提取（包含 FFmpeg 转码和降采样）
            chunks = await self._extract_and_vad(video_path, job_state)
            self.logger.info(f"音频提取和 VAD 切分完成: {len(chunks)} 个chunk")
        finally:
            if token:
                has_pending = token.exit_atomic_region()
                if has_pending:
                    self.logger.info("[V3.7.2] FFmpeg/VAD 完成后检测到待处理请求")

        # V3.7.2: Stage 1 检查点（音频提取 + VAD 完成）
        if token and job_dir:
            checkpoint_data = {
                "audio_extracted": True,
                "vad_completed": True,
                "total_chunks": len(chunks)
            }
            token.check_and_save(checkpoint_data, job_dir)

        # Stage 2: 频谱分诊（如果启用，逐chunk可中断）
        if self.spectral_triage_stage:
            self.logger.info("Stage 2: 频谱分诊")
            chunks = await self.spectral_triage_stage.process(chunks, job_dir=job_dir)

            # 统计分诊结果
            stats = self.spectral_triage_stage.get_statistics(chunks)
            self.logger.info(
                f"频谱分诊完成: {stats['need_separation']}/{stats['total_chunks']} "
                f"个chunk需要分离 (比例: {stats['separation_ratio']:.2%})"
            )

            # V3.7: 频谱分诊完成后检查点
            if token and job_dir:
                checkpoint_data = {
                    "spectral_triage": {
                        "completed": True,
                        "need_separation_count": stats['need_separation']
                    }
                }
                token.check_and_save(checkpoint_data, job_dir)
        else:
            self.logger.info("Stage 2: 频谱分诊已跳过")

        # Stage 3: 人声分离（如果启用）
        if self.separation_stage:
            self.logger.info("Stage 3: 人声分离")

            # 根据分离模式传递不同的参数
            if self.config.separation_mode == "global":
                # 全局分离模式：需要传递原始音频路径
                chunks = await self.separation_stage.process(
                    chunks=chunks,
                    audio_path=video_path,
                    job_dir=job_dir  # V3.7: 传递job_dir
                )
            else:
                # 按需分离模式：只分离标记的chunk
                chunks = await self.separation_stage.process(
                    chunks=chunks,
                    job_dir=job_dir  # V3.7: 传递job_dir
                )

            # 统计分离结果
            stats = self.separation_stage.get_statistics(chunks)
            self.logger.info(
                f"人声分离完成: {stats['separated']}/{stats['total_chunks']} "
                f"个chunk已分离 (比例: {stats['separation_ratio']:.2%})"
            )

            # V3.7: 人声分离完成后检查点
            if token and job_dir:
                checkpoint_data = {
                    "separation": {
                        "completed": True,
                        "mode": self.config.separation_mode,
                        "separated_count": stats['separated']
                    }
                }
                token.check_and_save(checkpoint_data, job_dir)
        else:
            self.logger.info("Stage 3: 人声分离已跳过")

        self.logger.info(f"预处理流程完成: {len(chunks)} 个chunk准备就绪")

        return chunks

    async def _extract_and_vad(
        self,
        video_path: str,
        job_state: Optional[JobState] = None
    ) -> List[AudioChunk]:
        """
        音频提取 + VAD切分

        使用 ChunkEngine 完成音频提取和VAD切分，但不执行Demucs分离。

        Args:
            video_path: 视频/音频文件路径
            job_state: 任务状态（可选）

        Returns:
            List[AudioChunk]: VAD切分后的 Chunk 列表
        """
        # 构建 VAD 配置（使用默认配置）
        vad_config = VADConfig()

        # 定义进度回调
        def progress_callback(progress: float, message: str):
            if job_state:
                job_state.phase_percent = progress * 100
                job_state.message = message
                self.logger.debug(f"进度: {progress:.1%} - {message}")

        # 使用 ChunkEngine 处理音频（不启用Demucs）
        chunks, full_audio, sr = self.chunk_engine.process_audio(
            audio_path=video_path,
            enable_demucs=False,  # 关键：不在这里执行Demucs
            vad_config=vad_config,
            progress_callback=progress_callback
        )

        self.logger.info(
            f"VAD切分完成: {len(chunks)} 个chunk, "
            f"采样率: {sr}Hz, "
            f"总时长: {len(full_audio) / sr:.2f}s"
        )

        return chunks

    def get_statistics(self, chunks: List[AudioChunk]) -> dict:
        """
        获取预处理统计信息

        Args:
            chunks: 已处理的 AudioChunk 列表

        Returns:
            dict: 统计信息
        """
        total = len(chunks)

        # 频谱分诊统计
        need_separation = sum(1 for c in chunks if c.needs_separation)

        # 人声分离统计
        separated = sum(1 for c in chunks if c.is_separated)

        # 分离级别统计
        from app.models.circuit_breaker_models import SeparationLevel
        htdemucs_count = sum(
            1 for c in chunks
            if c.is_separated and c.separation_level == SeparationLevel.HTDEMUCS
        )
        mdx_extra_count = sum(
            1 for c in chunks
            if c.is_separated and c.separation_level == SeparationLevel.MDX_EXTRA
        )

        # 熔断回溯统计
        fuse_retry_count = sum(c.fuse_retry_count for c in chunks)
        max_retry = max((c.fuse_retry_count for c in chunks), default=0)

        return {
            "total_chunks": total,
            "need_separation": need_separation,
            "separated": separated,
            "not_separated": total - separated,
            "htdemucs_count": htdemucs_count,
            "mdx_extra_count": mdx_extra_count,
            "separation_ratio": separated / total if total > 0 else 0.0,
            "fuse_retry_total": fuse_retry_count,
            "fuse_retry_max": max_retry,
        }


# 便捷函数
def get_preprocessing_pipeline(
    config: PreprocessingConfig,
    chunk_engine: Optional[ChunkEngine] = None,
    logger: Optional[logging.Logger] = None,
    cancellation_token: Optional["CancellationToken"] = None  # V3.7: 新增
) -> PreprocessingPipeline:
    """
    获取预处理流水线实例

    Args:
        config: 预处理配置
        chunk_engine: 音频切分引擎（可选）
        logger: 日志记录器（可选）
        cancellation_token: 取消令牌（可选，V3.7）

    Returns:
        PreprocessingPipeline 实例
    """
    return PreprocessingPipeline(
        config=config,
        chunk_engine=chunk_engine,
        logger=logger,
        cancellation_token=cancellation_token  # V3.7
    )
