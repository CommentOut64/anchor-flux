"""
WhisperExecutor - Whisper 推理执行器

Phase 3 实现 - 2025-12-10

封装 WhisperService，提供统一的执行器接口。
"""

import logging
from typing import Optional, Dict, Any, Union
import numpy as np

from app.services.whisper_service import WhisperService


class WhisperExecutor:
    """
    Whisper 推理执行器
    
    封装 WhisperService，提供统一的执行器接口。
    """
    
    def __init__(
        self,
        service: Optional[WhisperService] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 Whisper 执行器
        
        Args:
            service: Whisper 服务实例
            logger: 日志记录器
        """
        self.service = service or WhisperService()
        self.logger = logger or logging.getLogger(__name__)
    
    async def execute(
        self,
        audio: Union[str, np.ndarray],
        start_time: float,
        end_time: float,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3
    ) -> Dict[str, Any]:
        """
        执行 Whisper 推理

        Args:
            audio: 音频数组（已切片的 Chunk 音频，不需要再次切片）
            start_time: Chunk 起始时间（秒）- 仅用于日志，不用于切片
            end_time: Chunk 结束时间（秒）- 仅用于日志，不用于切片
            language: 语言代码（zh/en/auto）
            initial_prompt: 初始提示词（用于引导识别）
            repetition_penalty: 重复惩罚系数，>1 抑制重复（默认 1.2，推荐 1.1-1.3）
            no_repeat_ngram_size: 禁止重复的 N-gram 大小（默认 3，0=禁用）

        Returns:
            Dict: 推理结果
                - text: 识别文本
                - confidence: 置信度估算
                - language: 检测到的语言

        Note:
            传入的 audio 应该是已经切片好的 Chunk 音频，
            不会再进行二次切片。
        """
        # 自动加载模型（如果未加载）
        if not self.is_loaded():
            self.logger.info('Whisper 模型未加载，正在加载...')
            self.service.load_model()
            self.logger.info('Whisper 模型加载完成')

        duration = end_time - start_time
        self.logger.debug(
            f'执行 Whisper 推理: start={start_time:.2f}s, end={end_time:.2f}s, '
            f'duration={duration:.2f}s, '
            f'prompt={initial_prompt[:50] if initial_prompt else None}'
        )

        # 自适应 beam_size：短 chunk 用 greedy（快），长 chunk 用 beam（质量）
        if duration < 10.0:
            beam_size = 1  # <10s：greedy 解码，速度提升 3倍
            self.logger.debug(f'短 chunk ({duration:.1f}s)，使用 beam_size=1 快速模式')
        elif duration < 15.0:
            beam_size = 2  # 10-15s：小 beam，平衡
            self.logger.debug(f'中等 chunk ({duration:.1f}s)，使用 beam_size=2 平衡模式')
        else:
            beam_size = 5  # >15s：大 beam，保证质量
            self.logger.debug(f'长 chunk ({duration:.1f}s)，使用 beam_size=5 高质量模式')

        # 直接调用 transcribe，不进行二次切片
        # 传入的 audio 已经是切片后的 Chunk 音频
        # 禁用 condition_on_previous_text 避免基于前文截断音频末尾内容
        result = self.service.transcribe(
            audio=audio,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=False,  # 使用伪对齐，不需要词级时间戳
            beam_size=beam_size,  # 自适应 beam_size
            vad_filter=False,  # 已经是 VAD 切片，不需要再次 VAD
            condition_on_previous_text=False,  # 禁用前文条件化，保留 prompt 用于词汇引导
            repetition_penalty=repetition_penalty,  # 重复惩罚
            no_repeat_ngram_size=no_repeat_ngram_size  # N-gram 重复抑制
        )
        
        # 估算置信度
        confidence = self.service.estimate_confidence(result)

        # 提取文本
        text = result.get('text', '').strip()

        self.logger.debug(f'Whisper 推理完成: text={text}, confidence={confidence:.2f}')
        
        return {
            'text': text,
            'confidence': confidence,
            'language': result.get('language', language),
            'raw_result': result
        }
    
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 是否已加载
        """
        return self.service.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息
        """
        return {
            'model_name': self.service.config.model_name if self.service.config else 'unknown',
            'model_type': 'Faster-Whisper',
            'is_loaded': self.is_loaded(),
            'supported_languages': ['zh', 'en', 'ja', 'ko', 'auto']
        }
