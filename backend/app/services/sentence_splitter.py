"""
分句算法

核心原则：
- 标点符号（。？！）必切
- 长停顿（>0.4s）必切
- 强制长度（5秒或30字）强制切
- VAD 负责物理切分，分句算法负责语义切分
"""
import re
import logging
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..models.sensevoice_models import WordTimestamp, SentenceSegment

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """分句配置"""
    # 标点切分
    sentence_end_punctuation: str = "。？！.?!"  # 句末标点
    clause_punctuation: str = "，；：,;:"        # 分句标点（可选切分点）

    # 停顿切分
    pause_threshold: float = 0.4                 # 停顿阈值（秒）
    long_pause_threshold: float = 0.8            # 长停顿阈值（强制切分）

    # 长度限制
    max_duration: float = 5.0                    # 最大时长（秒）
    max_chars: int = 30                          # 最大字符数
    min_chars: int = 2                           # 最小字符数（避免过短）

    # 特殊处理
    merge_short_sentences: bool = True           # 合并过短句子
    short_sentence_threshold: int = 3            # 短句阈值


class SentenceSplitter:
    """分句器"""

    def __init__(self, config: SplitConfig = None):
        self.config = config or SplitConfig()

    def split(
        self,
        words: List['WordTimestamp'],
        text: str = None
    ) -> List['SentenceSegment']:
        """
        将字级时间戳切分为句子

        Args:
            words: 字级时间戳列表
            text: 原始文本（可选，用于验证）

        Returns:
            句子列表
        """
        from ..models.sensevoice_models import SentenceSegment, WordTimestamp

        if not words:
            return []

        sentences = []
        current_words: List[WordTimestamp] = []
        current_start = words[0].start

        for i, word in enumerate(words):
            current_words.append(word)

            # 检查是否需要切分
            should_split = False
            split_reason = ""

            # 1. 句末标点切分
            if word.word in self.config.sentence_end_punctuation:
                should_split = True
                split_reason = "punctuation"

            # 2. 停顿切分（检查与下一个词的间隔）
            elif i < len(words) - 1:
                next_word = words[i + 1]
                pause = next_word.start - word.end

                if pause >= self.config.long_pause_threshold:
                    should_split = True
                    split_reason = "long_pause"
                elif pause >= self.config.pause_threshold:
                    # 中等停顿 + 分句标点
                    if word.word in self.config.clause_punctuation:
                        should_split = True
                        split_reason = "pause_with_clause"

            # 3. 强制长度切分
            current_duration = word.end - current_start
            current_text = "".join(w.word for w in current_words)

            if current_duration >= self.config.max_duration:
                should_split = True
                split_reason = "max_duration"
            elif len(current_text) >= self.config.max_chars:
                should_split = True
                split_reason = "max_chars"

            # 4. 最后一个词强制切分
            if i == len(words) - 1:
                should_split = True
                split_reason = "end_of_input"

            # 执行切分
            if should_split and current_words:
                sentence = self._create_sentence(current_words)
                if sentence:
                    sentences.append(sentence)
                    logger.debug(f"分句: '{sentence.text}' ({split_reason})")

                # 重置
                current_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        # 合并过短句子
        if self.config.merge_short_sentences:
            sentences = self._merge_short_sentences(sentences)

        return sentences

    def _create_sentence(self, words: List['WordTimestamp']) -> Optional['SentenceSegment']:
        """创建句子对象"""
        from ..models.sensevoice_models import SentenceSegment

        if not words:
            return None

        text = "".join(w.word for w in words)

        # 过滤过短句子
        if len(text.strip()) < self.config.min_chars:
            return None

        # 计算平均置信度
        avg_confidence = sum(w.confidence for w in words) / len(words)

        return SentenceSegment(
            text=text,
            start=words[0].start,
            end=words[-1].end,
            words=words.copy(),
            confidence=avg_confidence
        )

    def _merge_short_sentences(
        self,
        sentences: List['SentenceSegment']
    ) -> List['SentenceSegment']:
        """合并过短句子"""
        from ..models.sensevoice_models import SentenceSegment

        if len(sentences) <= 1:
            return sentences

        merged = []
        i = 0

        while i < len(sentences):
            current = sentences[i]

            # 检查是否为短句
            if len(current.text) <= self.config.short_sentence_threshold:
                # 尝试与下一句合并
                if i + 1 < len(sentences):
                    next_sent = sentences[i + 1]
                    merged_text = current.text + next_sent.text
                    merged_words = current.words + next_sent.words

                    # 检查合并后是否超限
                    merged_duration = next_sent.end - current.start
                    if (merged_duration <= self.config.max_duration and
                        len(merged_text) <= self.config.max_chars):

                        avg_confidence = (
                            (current.confidence * len(current.words) +
                             next_sent.confidence * len(next_sent.words)) /
                            len(merged_words)
                        )

                        merged_sentence = SentenceSegment(
                            text=merged_text,
                            start=current.start,
                            end=next_sent.end,
                            words=merged_words,
                            confidence=avg_confidence
                        )
                        merged.append(merged_sentence)
                        i += 2
                        continue

            merged.append(current)
            i += 1

        return merged

    def split_by_punctuation_only(self, text: str) -> List[str]:
        """
        仅按标点切分文本（不考虑时间戳）

        Args:
            text: 输入文本

        Returns:
            切分后的句子列表
        """
        if not text:
            return []

        # 使用正则按句末标点切分
        pattern = f"([{re.escape(self.config.sentence_end_punctuation)}])"
        parts = re.split(pattern, text)

        sentences = []
        current = ""

        for part in parts:
            current += part
            if part in self.config.sentence_end_punctuation:
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        # 处理最后一部分
        if current.strip():
            sentences.append(current.strip())

        return sentences


# 单例访问
_splitter_instance = None


def get_sentence_splitter(config: SplitConfig = None) -> SentenceSplitter:
    """获取分句器单例"""
    global _splitter_instance
    if _splitter_instance is None:
        _splitter_instance = SentenceSplitter(config)
    return _splitter_instance
