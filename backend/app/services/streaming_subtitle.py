"""
流式字幕管理系统

核心职责：
1. 管理字幕句子列表（支持原地更新）
2. 协调 SSE 事件推送（统一 Tag）
3. 支持多阶段增量更新（SV → Whisper → LLM）
"""
from typing import Dict, List, Optional
from app.models.sensevoice_models import SentenceSegment, TextSource
from app.services.sse_service import get_sse_manager
import logging

logger = logging.getLogger(__name__)


def push_subtitle_event(sse_manager, job_id: str, event_type: str, data: dict):
    """
    推送字幕事件（统一封装）

    Args:
        sse_manager: SSE 管理器
        job_id: 任务 ID
        event_type: 事件类型
        data: 事件数据
    """
    sse_manager.broadcast_sync(
        f"job:{job_id}",
        f"subtitle.{event_type}",
        data
    )


class StreamingSubtitleManager:
    """流式字幕管理器"""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.sentences: Dict[int, SentenceSegment] = {}  # key = sentence_index
        self.sentence_count = 0
        self.sse_manager = get_sse_manager()

    def add_sentence(self, sentence: SentenceSegment) -> int:
        """
        添加新句子（SenseVoice 阶段）

        Args:
            sentence: 句子段落

        Returns:
            int: 句子索引
        """
        index = self.sentence_count
        self.sentences[index] = sentence
        self.sentence_count += 1

        # 推送 SSE 事件
        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "sv_sentence",
            {
                "index": index,
                "sentence": sentence.to_dict(),
                "source": "sensevoice"
            }
        )

        logger.debug(f"添加句子 {index}: {sentence.text[:30]}...")
        return index

    def update_sentence(
        self,
        index: int,
        new_text: str,
        source: TextSource,
        confidence: float = None,
        perplexity: float = None
    ):
        """
        更新已有句子（Whisper 补刀或 LLM 校对）

        Args:
            index: 句子索引
            new_text: 新文本
            source: 文本来源
            confidence: 新置信度（可选）
            perplexity: LLM 困惑度（可选）
        """
        if index not in self.sentences:
            logger.warning(f"句子 {index} 不存在，无法更新")
            return

        sentence = self.sentences[index]

        # 应用伪对齐
        from .pseudo_alignment import PseudoAlignment
        PseudoAlignment.apply_to_sentence(sentence, new_text, source)

        # 更新置信度和困惑度
        if confidence is not None:
            sentence.confidence = confidence
        if perplexity is not None:
            sentence.perplexity = perplexity
            sentence.warning_type = sentence.compute_warning_type()

        # 推送 SSE 事件
        event_type = {
            TextSource.WHISPER_PATCH: "whisper_patch",
            TextSource.LLM_CORRECTION: "llm_proof",
            TextSource.LLM_TRANSLATION: "llm_trans",
        }.get(source, "batch_update")

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            event_type,
            {
                "index": index,
                "sentence": sentence.to_dict(),
                "source": source.value,
                "is_update": True
            }
        )

        logger.debug(f"更新句子 {index} ({source.value}): {new_text[:30]}...")

    def set_translation(self, index: int, translation: str, confidence: float = None):
        """
        设置翻译结果

        Args:
            index: 句子索引
            translation: 翻译文本
            confidence: 翻译置信度
        """
        if index not in self.sentences:
            return

        sentence = self.sentences[index]
        sentence.translation = translation
        if confidence is not None:
            sentence.translation_confidence = confidence

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "llm_trans",
            {
                "index": index,
                "translation": translation,
                "confidence": confidence
            }
        )

    def get_all_sentences(self) -> List[SentenceSegment]:
        """获取所有句子（按时间排序）"""
        sentences = list(self.sentences.values())
        sentences.sort(key=lambda s: s.start)
        return sentences

    def get_context_window(self, index: int, window_size: int = 3) -> str:
        """
        获取上下文窗口（用于 LLM 校对）

        Args:
            index: 当前句子索引
            window_size: 上下文窗口大小

        Returns:
            str: 上下文文本
        """
        context_indices = range(max(0, index - window_size), index)
        context_texts = [
            self.sentences[i].text
            for i in context_indices
            if i in self.sentences
        ]
        return " ".join(context_texts)


# ========== 单例工厂 ==========

_subtitle_managers: Dict[str, StreamingSubtitleManager] = {}


def get_streaming_subtitle_manager(job_id: str) -> StreamingSubtitleManager:
    """获取或创建流式字幕管理器"""
    global _subtitle_managers
    if job_id not in _subtitle_managers:
        _subtitle_managers[job_id] = StreamingSubtitleManager(job_id)
    return _subtitle_managers[job_id]


def remove_streaming_subtitle_manager(job_id: str):
    """移除流式字幕管理器"""
    global _subtitle_managers
    if job_id in _subtitle_managers:
        del _subtitle_managers[job_id]
