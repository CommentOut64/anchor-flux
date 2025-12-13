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

        # Phase 4: Chunk 级别的句子索引映射
        # chunk_sentences[chunk_index] = [sentence_index_1, sentence_index_2, ...]
        self.chunk_sentences: Dict[int, List[int]] = {}

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

        # 推送 SSE 事件（使用清洗后的文本）
        sentence_dict = sentence.to_dict() if hasattr(sentence, 'to_dict') else {
            "index": index,
            "text": sentence.text_clean or sentence.text,  # 优先使用清洗后的文本
            "start": sentence.start,
            "end": sentence.end,
            "confidence": sentence.confidence,
            "source": sentence.source.value if hasattr(sentence.source, 'value') else str(sentence.source),
            "words": [w.to_dict() if hasattr(w, 'to_dict') else w for w in getattr(sentence, 'words', [])]  # 确保包含 words
        }

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "sv_sentence",
            {
                "index": index,
                "sentence": sentence_dict,
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

    def mark_for_deletion(self, index: int, reason: str = "garbage"):
        """
        标记句子为待删除（Whisper 仲裁后确认为垃圾）

        Args:
            index: 句子索引
            reason: 删除原因
        """
        if index not in self.sentences:
            return

        sentence = self.sentences[index]
        sentence.marked_for_deletion = True
        sentence.deletion_reason = reason

        # 推送 SSE 事件通知前端
        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "sentence_deleted",
            {
                "index": index,
                "reason": reason
            }
        )
        logger.info(f"标记删除句子 {index}: {reason}")

    def remove_marked_sentences(self) -> int:
        """
        物理删除被标记为垃圾的句子

        Returns:
            int: 删除的句子数量
        """
        marked_indices = [
            idx for idx, s in self.sentences.items()
            if getattr(s, 'marked_for_deletion', False)
        ]

        for idx in marked_indices:
            del self.sentences[idx]

        if marked_indices:
            logger.info(f"已删除 {len(marked_indices)} 个垃圾句子: {marked_indices}")

        return len(marked_indices)

    def get_all_sentences(self) -> List[SentenceSegment]:
        """获取所有句子（按时间排序，排除已标记删除的）"""
        sentences = [
            s for s in self.sentences.values()
            if not getattr(s, 'marked_for_deletion', False)
        ]
        sentences.sort(key=lambda s: s.start)
        return sentences

    def get_context_window(self, index: int, window_size: int = 3) -> str:
        """
        获取上下文窗口（用于 LLM 校对和 Whisper 补刀）

        Args:
            index: 当前句子索引
            window_size: 上下文窗口大小

        Returns:
            str: 上下文文本（清洗后的文本，避免传递 SenseVoice 原始 token）
        """
        context_indices = range(max(0, index - window_size), index)
        context_texts = [
            self.sentences[i].text_clean or self.sentences[i].text  # 优先使用清洗后的文本，避免传递下划线等原始 token
            for i in context_indices
            if i in self.sentences
        ]
        return " ".join(context_texts)

    # ========== Phase 4: 双流对齐专用方法 ==========

    def add_draft_sentences(
        self,
        chunk_index: int,
        sentences: List[SentenceSegment]
    ) -> List[int]:
        """
        添加草稿句子（快流推送）

        Phase 4 双流对齐专用方法。
        推送多个草稿句子，并记录 Chunk 级别的索引映射。

        Args:
            chunk_index: Chunk 索引
            sentences: 句子列表

        Returns:
            List[int]: 句子索引列表
        """
        sentence_indices = []

        for sentence in sentences:
            index = self.sentence_count
            self.sentences[index] = sentence
            self.sentence_count += 1
            sentence_indices.append(index)

            # 推送 SSE 事件（草稿）
            sentence_dict = sentence.to_dict() if hasattr(sentence, 'to_dict') else {
                "index": index,
                "text": sentence.text_clean or sentence.text,
                "start": sentence.start,
                "end": sentence.end,
                "confidence": sentence.confidence,
                "source": sentence.source.value if hasattr(sentence.source, 'value') else str(sentence.source),
                "is_draft": True,
                "words": [w.to_dict() if hasattr(w, 'to_dict') else w for w in getattr(sentence, 'words', [])]  # 确保包含 words
            }

            push_subtitle_event(
                self.sse_manager,
                self.job_id,
                "draft",  # 新事件类型
                {
                    "index": index,
                    "chunk_index": chunk_index,
                    "sentence": sentence_dict
                }
            )

        # 记录 Chunk 级别的索引映射
        self.chunk_sentences[chunk_index] = sentence_indices

        logger.info(
            f"添加草稿句子: Chunk {chunk_index}, "
            f"{len(sentences)} 个句子, 索引 {sentence_indices}"
        )

        return sentence_indices

    def replace_chunk(
        self,
        chunk_index: int,
        sentences: List[SentenceSegment]
    ) -> List[int]:
        """
        替换 Chunk 的所有句子（慢流推送）

        Phase 4 双流对齐专用方法。
        用定稿句子替换整个 Chunk 的草稿句子。

        流程：
        1. 删除旧的草稿句子
        2. 添加新的定稿句子
        3. 更新 Chunk 索引映射
        4. 推送 replace_chunk 事件

        Args:
            chunk_index: Chunk 索引
            sentences: 定稿句子列表

        Returns:
            List[int]: 新的句子索引列表
        """
        # 删除旧的草稿句子
        old_indices = self.chunk_sentences.get(chunk_index, [])
        for old_index in old_indices:
            if old_index in self.sentences:
                del self.sentences[old_index]

        # 添加新的定稿句子
        new_indices = []
        for sentence in sentences:
            index = self.sentence_count
            self.sentences[index] = sentence
            self.sentence_count += 1
            new_indices.append(index)

        # 更新 Chunk 索引映射
        self.chunk_sentences[chunk_index] = new_indices

        # 推送 SSE 事件（批量替换）
        sentences_data = [
            sentence.to_dict() if hasattr(sentence, 'to_dict') else {
                "index": new_indices[i],
                "text": sentence.text_clean or sentence.text,
                "start": sentence.start,
                "end": sentence.end,
                "confidence": sentence.confidence,
                "source": sentence.source.value if hasattr(sentence.source, 'value') else str(sentence.source),
                "is_draft": False,
                "is_finalized": True,
                "words": [w.to_dict() if hasattr(w, 'to_dict') else w for w in getattr(sentence, 'words', [])]  # 确保包含 words
            }
            for i, sentence in enumerate(sentences)
        ]

        push_subtitle_event(
            self.sse_manager,
            self.job_id,
            "replace_chunk",  # 新事件类型
            {
                "chunk_index": chunk_index,
                "old_indices": old_indices,
                "new_indices": new_indices,
                "sentences": sentences_data
            }
        )

        logger.info(
            f"替换 Chunk {chunk_index}: "
            f"删除 {len(old_indices)} 个草稿, "
            f"添加 {len(new_indices)} 个定稿"
        )

        return new_indices


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
