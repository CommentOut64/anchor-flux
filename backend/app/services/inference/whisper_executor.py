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
        initial_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行 Whisper 推理
        
        Args:
            audio: 音频文件路径或音频数组
            start_time: 起始时间（秒）
            end_time: 结束时间（秒）
            language: 语言代码（zh/en/auto）
            initial_prompt: 初始提示词（用于引导识别）
        
        Returns:
            Dict: 推理结果
                - text: 识别文本
                - confidence: 置信度估算
                - language: 检测到的语言
        """
        self.logger.debug(
            f'执行 Whisper 推理: start={start_time:.2f}s, end={end_time:.2f}s, '
            f'prompt={initial_prompt[:50] if initial_prompt else None}'
        )
        
        # 调用底层服务
        result = self.service.transcribe_segment(
            audio=audio,
            start_time=start_time,
            end_time=end_time,
            language=language,
            initial_prompt=initial_prompt
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
