"""
流水线模块

包含:
- base_pipeline: 流水线基类
- dual_alignment_pipeline: 双流对齐流水线（核心）
- audio_processing_pipeline: 音频前处理流水线
"""

from app.pipelines.audio_processing_pipeline import (
    AudioProcessingPipeline,
    AudioProcessingConfig,
    AudioProcessingResult,
    SeparationStrategy,
    get_audio_processing_pipeline
)

from app.pipelines.dual_alignment_pipeline import (
    DualAlignmentPipeline,
    DualAlignmentConfig,
    ChunkProcessingResult,
    AlignmentLevel,
    get_dual_alignment_pipeline
)

__all__ = [
    # 音频前处理流水线
    'AudioProcessingPipeline',
    'AudioProcessingConfig',
    'AudioProcessingResult',
    'SeparationStrategy',
    'get_audio_processing_pipeline',

    # 双流对齐流水线
    'DualAlignmentPipeline',
    'DualAlignmentConfig',
    'ChunkProcessingResult',
    'AlignmentLevel',
    'get_dual_alignment_pipeline',
]
