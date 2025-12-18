"""
SpectralTriageStage - 频谱分诊阶段

负责为每个AudioChunk进行频谱分析，判断是否需要人声分离，并推荐合适的模型。
"""

import logging
from typing import List, Optional

from app.services.audio.chunk_engine import AudioChunk
from app.services.audio_spectrum_classifier import AudioSpectrumClassifier, get_spectrum_classifier


class SpectralTriageStage:
    """
    频谱分诊阶段

    职责：
    - 为每个AudioChunk进行频谱分析
    - 判断是否需要人声分离
    - 推荐合适的分离模型（htdemucs/mdx_extra）
    """

    def __init__(
        self,
        classifier: Optional[AudioSpectrumClassifier] = None,
        threshold: float = 0.35,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化频谱分诊阶段

        Args:
            classifier: 频谱分类器实例，如果为None则使用全局单例
            threshold: 分诊阈值，默认0.35
            logger: 日志记录器，如果为None则创建新的
        """
        self.classifier = classifier or get_spectrum_classifier()
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)

    async def process(self, chunks: List[AudioChunk]) -> List[AudioChunk]:
        """
        批量分诊所有chunk

        Args:
            chunks: 待分诊的AudioChunk列表

        Returns:
            带有分诊结果标记的AudioChunk列表
        """
        if not chunks:
            self.logger.warning("收到空的chunk列表，跳过频谱分诊")
            return chunks

        self.logger.info(f"开始频谱分诊，共 {len(chunks)} 个chunk")

        # 提取音频数据和采样率
        audio_list = [chunk.audio for chunk in chunks]
        sample_rate = chunks[0].sample_rate if chunks else 16000

        # 批量分诊
        diagnoses = self.classifier.diagnose_chunks(
            chunks=audio_list,
            sr=sample_rate,
            threshold=self.threshold
        )

        # 为每个chunk添加分诊结果
        for chunk, diagnosis in zip(chunks, diagnoses):
            chunk.needs_separation = diagnosis.need_separation
            chunk.recommended_model = diagnosis.recommended_model
            chunk.spectrum_diagnosis = diagnosis

        # 统计
        need_sep_count = sum(1 for c in chunks if c.needs_separation)
        self.logger.info(
            f"频谱分诊完成: {need_sep_count}/{len(chunks)} 个chunk需要分离"
        )

        # 详细统计
        if need_sep_count > 0:
            htdemucs_count = sum(
                1 for c in chunks
                if c.needs_separation and c.recommended_model == 'htdemucs'
            )
            mdx_extra_count = sum(
                1 for c in chunks
                if c.needs_separation and c.recommended_model == 'mdx_extra'
            )
            self.logger.info(
                f"推荐模型分布: htdemucs={htdemucs_count}, mdx_extra={mdx_extra_count}"
            )

        return chunks

    def get_statistics(self, chunks: List[AudioChunk]) -> dict:
        """
        获取分诊统计信息

        Args:
            chunks: 已分诊的AudioChunk列表

        Returns:
            统计信息字典
        """
        total = len(chunks)
        need_sep = sum(1 for c in chunks if c.needs_separation)
        htdemucs = sum(
            1 for c in chunks
            if c.needs_separation and c.recommended_model == 'htdemucs'
        )
        mdx_extra = sum(
            1 for c in chunks
            if c.needs_separation and c.recommended_model == 'mdx_extra'
        )

        return {
            "total_chunks": total,
            "need_separation": need_sep,
            "no_separation": total - need_sep,
            "recommended_htdemucs": htdemucs,
            "recommended_mdx_extra": mdx_extra,
            "separation_ratio": need_sep / total if total > 0 else 0.0
        }
