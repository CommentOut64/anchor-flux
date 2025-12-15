"""
分句算法 (Layer 1 优化版)

核心原则：
- 标点符号（。？！）必切
- 长停顿（>0.4s）必切
- 强制长度（5秒或30字）强制切
- VAD 负责物理切分，分句算法负责语义切分

Layer 1 优化:
- 动态停顿阈值（根据语速自适应）
- 边界修剪（消除静音段）
- 智能长句拆分（优先在标点和停顿处）
- 语言策略抽象（支持多语言扩展）
"""
import re
import logging
import statistics
from abc import ABC, abstractmethod
from typing import List, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..models.sensevoice_models import WordTimestamp, SentenceSegment

logger = logging.getLogger(__name__)


# ============================================================================
# 语言策略抽象 (解决硬编码语言规则问题)
# ============================================================================

class LanguageStrategy(ABC):
    """语言策略抽象基类，用于处理不同语言的分句规则"""

    @abstractmethod
    def get_sentence_end_chars(self) -> Set[str]:
        """获取句子结束标点"""
        pass

    @abstractmethod
    def get_clause_break_chars(self) -> Set[str]:
        """获取从句断句标点"""
        pass

    @abstractmethod
    def get_continuation_words(self) -> List[str]:
        """获取续接词列表"""
        pass

    @abstractmethod
    def is_continuation(self, text: str) -> bool:
        """判断文本是否以续接词开头"""
        pass

    def is_semantically_complete(self, text: str) -> bool:
        """
        检查文本是否语义完整，避免在不完整位置切分
        默认实现：总是返回 True（允许切分）
        子类可重写此方法以提供语言特定的语义完整性检查
        """
        return True

    def is_incomplete_ending(self, word: str) -> bool:
        """
        检查单个词是否为不完整结尾词

        Args:
            word: 单个词（token）

        Returns:
            True 表示该词是不完整结尾词，不应在此处切分
        """
        return False


class ChineseStrategy(LanguageStrategy):
    """中文语言策略"""

    def get_sentence_end_chars(self) -> Set[str]:
        return {'。', '？', '！', '.', '?', '!'}

    def get_clause_break_chars(self) -> Set[str]:
        return {'，', '、', '；', '：', ',', ';', ':'}

    def get_continuation_words(self) -> List[str]:
        return ['但', '可', '然', '所', '因', '如', '虽', '而', '不过', '或者',
                '但是', '可是', '然后', '所以', '因为', '如果', '虽然', '而且']

    def is_continuation(self, text: str) -> bool:
        return any(text.strip().startswith(word) for word in self.get_continuation_words())


class EnglishStrategy(LanguageStrategy):
    """英文语言策略"""

    # 不完整结尾词集合 - 这些词结尾的句子语义不完整，不应在此处切分
    # 分类整理，便于维护和扩展
    INCOMPLETE_ENDINGS = {
        # ============ 1. 代词 ============
        # 人称代词（主格）- 需要后接谓语动词
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        # 人称代词（宾格）- 某些情况下也不宜结尾（如 "give me" 后需接宾语）
        'me', 'him', 'us', 'them',
        # 反身代词 - 通常作为宾语，但某些结构需要后续
        # 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
        # 关系代词/疑问代词 - 引导从句，需要后续内容
        'who', 'whom', 'whose', 'which', 'what', 'whoever', 'whatever', 'whichever',
        # 不定代词 - 部分情况下需要后续
        'some', 'any', 'no', 'every', 'each', 'either', 'neither', 'both', 'all', 'none',
        'someone', 'anyone', 'everyone', 'something', 'anything', 'everything',

        # ============ 2. 限定词 ============
        # 冠词 - 必须接名词
        'the', 'a', 'an',
        # 指示词 - 作为限定词时需要接名词
        'this', 'that', 'these', 'those', 'such',
        # 所有格 - 必须接名词
        'my', 'your', 'his', 'her', 'its', 'our', 'their',
        # 数量词/量词 - 通常需要接名词
        'many', 'much', 'few', 'little', 'several', 'most', 'more', 'less', 'fewer',
        'enough', 'plenty', 'lots', 'another', 'other',

        # ============ 3. 介词 ============
        # 常用介词 - 必须接宾语
        'to', 'of', 'in', 'on', 'at', 'for', 'with', 'about', 'from', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among',
        'under', 'over', 'behind', 'beside', 'beyond', 'within', 'without',
        'against', 'along', 'around', 'across', 'towards', 'toward', 'upon',
        'like', 'unlike', 'except', 'besides', 'including', 'excluding',
        'near', 'inside', 'outside', 'beneath', 'underneath', 'alongside',
        # 复合介词（常见部分）
        'onto', 'throughout', 'despite', 'regarding', 'concerning',

        # ============ 4. 连词 ============
        # 并列连词
        'and', 'or', 'but', 'nor', 'yet', 'so',
        # 从属连词 - 引导从句，需要后续
        'because', 'if', 'when', 'while', 'although', 'though', 'unless', 'until',
        'since', 'whether', 'whereas', 'wherever', 'whenever', 'however',
        'as', 'than', 'once', 'provided', 'supposing', 'considering',
        # 关联连词组件
        'either', 'neither', 'both', 'not',

        # ============ 5. 动词（必须后接内容的形式）============
        # be动词 - 必须需要补语（不能独立结尾）
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
        # 助动词 - 必须后接动词（不能独立结尾）
        'have', 'has', 'had',  # 注: having 可独立使用
        'will', 'would', 'shall', 'should',
        'can', 'could', 'may', 'might', 'must',
        'do', 'does', 'did',  # 注: doing 可独立使用
        # 注意: 及物动词（如 want, get, make 等）虽然通常需要宾语，
        # 但它们可以在句末独立出现（如 "That's what I wanted."），
        # 因此不应加入此集合

        # ============ 6. 副词（修饰后续内容）============
        # 程度副词 - 通常修饰形容词或动词
        'very', 'really', 'quite', 'rather', 'pretty', 'fairly', 'extremely',
        'absolutely', 'completely', 'totally', 'entirely', 'perfectly',
        'almost', 'nearly', 'barely', 'hardly', 'scarcely',
        # 频率/时间副词（某些位置）
        'just', 'only', 'even', 'still', 'already', 'also', 'always', 'never',
        'ever', 'often', 'usually', 'sometimes', 'rarely', 'seldom',
        # 方式/情态副词
        'maybe', 'perhaps', 'probably', 'possibly', 'certainly', 'definitely',
        'obviously', 'clearly', 'apparently', 'actually', 'basically', 'essentially',
        # 否定副词
        "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
        "shouldn't", "mustn't", "isn't", "aren't", "wasn't", "weren't", "haven't",
        "hasn't", "hadn't",

        # ============ 7. 特殊结构词 ============
        # There be 结构
        'there',
        # It 形式主语结构
        # 'it' 已在代词中
        # 比较结构
        'more', 'less', 'most', 'least', 'better', 'worse', 'best', 'worst',
        # 不定式标记
        # 'to' 已在介词中
    }

    def get_sentence_end_chars(self) -> Set[str]:
        return {'.', '?', '!'}

    def get_clause_break_chars(self) -> Set[str]:
        return {',', ';', ':'}

    def get_continuation_words(self) -> List[str]:
        return ['but', 'and', 'or', 'so', 'because', 'however', 'therefore',
                'although', 'though', 'yet', 'still', 'moreover', 'furthermore']

    def is_continuation(self, text: str) -> bool:
        first_word = text.strip().split()[0].lower() if text.strip() else ''
        return first_word in self.get_continuation_words()

    def is_incomplete_ending(self, word: str) -> bool:
        """
        检查单个词是否为不完整结尾词

        Args:
            word: 单个词（token），可能包含标点

        Returns:
            True 表示该词是不完整结尾词，不应在此处切分
        """
        # 清理词：去除标点和空格标记
        clean_word = word.strip().lower().rstrip('.,;:!?').lstrip('▁')
        return clean_word in self.INCOMPLETE_ENDINGS

    def is_semantically_complete(self, text: str) -> bool:
        """
        检查文本是否语义完整，避免在不完整位置切分

        Args:
            text: 待检查的文本（可能包含 ▁ 作为空格标记，或无空格）

        Returns:
            True 表示语义完整可以切分，False 表示语义不完整不应切分
        """
        import re

        if not text or not text.strip():
            return True

        # 处理可能的 ▁ 空格标记（SentencePiece tokenizer 格式）
        normalized_text = text.replace('▁', ' ').strip()

        # 尝试用空格分词
        words = normalized_text.split()

        # 如果只有一个"词"（可能是无空格拼接的文本），用正则提取最后一个英文单词
        if len(words) == 1:
            # 匹配最后一个英文单词（字母序列）
            match = re.search(r"[a-zA-Z']+$", normalized_text.rstrip('.,;:!?'))
            if match:
                last_word = match.group().lower().rstrip("'")
            else:
                return True  # 无法提取单词，默认允许切分
        else:
            last_word = words[-1].lower().rstrip('.,;:!?')

        # 检查是否为不完整结尾词
        is_incomplete = last_word in self.INCOMPLETE_ENDINGS

        return not is_incomplete


class JapaneseStrategy(LanguageStrategy):
    """日文语言策略"""

    def get_sentence_end_chars(self) -> Set[str]:
        return {'。', '？', '！', '.', '?', '!'}

    def get_clause_break_chars(self) -> Set[str]:
        return {'、', '，', ','}

    def get_continuation_words(self) -> List[str]:
        return ['でも', 'しかし', 'だから', 'そして', 'それで', 'ただ', 'けど']

    def is_continuation(self, text: str) -> bool:
        return any(text.strip().startswith(word) for word in self.get_continuation_words())


def get_language_strategy(language: str) -> LanguageStrategy:
    """根据语言代码获取对应的语言策略"""
    strategies = {
        'zh': ChineseStrategy(),
        'en': EnglishStrategy(),
        'ja': JapaneseStrategy(),
        'auto': ChineseStrategy(),  # 默认使用中文策略
    }
    return strategies.get(language, ChineseStrategy())


@dataclass
class SplitConfig:
    """分句配置 (Layer 1 优化版)"""
    # ============================================================================
    # 现有配置 (保持兼容性)
    # ============================================================================
    # 标点切分
    sentence_end_punctuation: str = "。？！.?!"  # 句末标点
    clause_punctuation: str = "，；：,;:"        # 分句标点（可选切分点）

    # 停顿切分
    pause_threshold: float = 0.7                 # 停顿阈值（秒），从0.5提高到0.7
    long_pause_threshold: float = 1.5            # 长停顿阈值（强制切分），从1.0提高到1.5

    # 长度限制
    max_duration: float = 5.0                    # 最大时长（秒），软上限，会检查语义完整性
    max_duration_hard_ratio: float = 1.5         # 硬上限比例，超过 max_duration * 此比例 无论如何都切分
    max_chars: int = 0                           # 最大字符数，0表示不启用字符数限制
    min_chars: int = 2                           # 最小字符数（避免过短）

    # 特殊处理
    merge_short_sentences: bool = True           # 合并过短句子
    short_sentence_threshold: int = 3            # 短句阈值

    # ============================================================================
    # Layer 1 新增配置
    # ============================================================================
    # 语言设置 (解决硬编码问题)
    language: str = "auto"                       # 语言: auto, zh, en, ja, ko 等

    # 边界修剪
    trim_leading_silence: bool = True            # 修剪句首静音
    trim_trailing_silence: bool = True           # 修剪句尾静音
    max_boundary_gap: float = 0.3                # 最大允许的边界间隙(秒)，放宽至0.3避免过度修剪

    # 动态停顿
    use_dynamic_pause: bool = True               # 启用动态停顿阈值
    speech_rate_window: int = 10                 # 语速计算窗口(字数)
    pause_multiplier: float = 2.5                # 停顿阈值 = 百分位字间隔 * multiplier (提高至2.5)
    min_pause_threshold: float = 0.5             # 最小停顿阈值(秒)，从0.3提高到0.5，避免自然语流中短停顿触发切分

    # 智能分句
    prefer_punctuation_break: bool = True        # 优先在标点处断句
    soft_break_threshold: float = 0.8            # 软断点阈值(秒)

    # max_duration 延迟切分到标点符号
    delay_split_to_punctuation: bool = True      # 达到 max_duration 后延迟切分，等待遇到标点符号
    delay_split_ignore_clause_punct: bool = True # 忽略从句标点(逗号、顿号等)，只等待句末标点(.?!)
    delay_split_max_wait: float = 2.0            # 延迟等待的最大额外时长(秒)，超过则强制切分

    # 语义分组 (预留给 Layer 2)
    enable_semantic_grouping: bool = True        # 启用语义分组
    max_group_duration: float = 10.0             # 单个语义组最大时长(秒)

    def get_strategy(self) -> LanguageStrategy:
        """获取当前语言对应的策略"""
        return get_language_strategy(self.language)


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

        # 调试日志：确认当前语言配置
        logger.debug(f"分句器语言配置: language={self.config.language}, strategy={type(self.config.get_strategy()).__name__}")

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
            # Layer 1 优化: 使用动态停顿阈值
            elif i < len(words) - 1:
                next_word = words[i + 1]
                pause = next_word.start - word.end

                # 计算动态停顿阈值（根据局部语速）
                dynamic_threshold = self._calculate_dynamic_pause_threshold(words, i)

                if pause >= self.config.long_pause_threshold:
                    should_split = True
                    split_reason = "long_pause"
                elif pause >= dynamic_threshold:
                    # 动态停顿 + 分句标点
                    if word.word in self.config.clause_punctuation:
                        # 方案C: 直接检查最后一个词是否为不完整结尾词
                        strategy = self.config.get_strategy()
                        last_word = current_words[-1].word if current_words else ''
                        if not strategy.is_incomplete_ending(last_word):
                            should_split = True
                            split_reason = "dynamic_pause_with_clause"
                        else:
                            # 语义不完整，跳过切分
                            logger.debug(f"语义不完整(词='{last_word}')，跳过切分")

            # 3. 强制长度切分
            current_duration = word.end - current_start

            # 计算硬上限时长
            hard_limit_duration = self.config.max_duration * self.config.max_duration_hard_ratio
            # 计算延迟等待的最大时长
            delay_max_duration = self.config.max_duration + self.config.delay_split_max_wait

            if current_duration >= hard_limit_duration:
                # 超过硬上限，无论如何都要切分
                should_split = True
                split_reason = "max_duration_hard_limit"
            elif current_duration >= self.config.max_duration:
                # 超过软上限，检查是否应该延迟切分
                strategy = self.config.get_strategy()
                last_word = current_words[-1].word if current_words else ''
                is_incomplete = strategy.is_incomplete_ending(last_word)

                # 检查当前词是否带标点
                last_word_stripped = last_word.rstrip()
                has_sentence_end_punct = any(last_word_stripped.endswith(p) for p in '.?!')
                has_clause_punct = any(last_word_stripped.endswith(p) for p in ',;:')

                logger.debug(f"max_duration 检查: duration={current_duration:.2f}s, last_word='{last_word}', "
                           f"is_incomplete={is_incomplete}, has_end_punct={has_sentence_end_punct}, has_clause_punct={has_clause_punct}")

                # 决定是否延迟切分
                should_delay = False

                # 检查1: 语义不完整 -> 延迟
                if is_incomplete:
                    should_delay = True
                    logger.debug(f"max_duration 语义不完整(词='{last_word}')，延迟切分...")

                # 检查2: 启用了延迟到标点功能，且当前词没有结束标点 -> 延迟
                elif self.config.delay_split_to_punctuation:
                    if self.config.delay_split_ignore_clause_punct:
                        # 只等待句末标点，忽略从句标点
                        if not has_sentence_end_punct:
                            # 检查是否超过延迟等待上限
                            if current_duration < delay_max_duration:
                                should_delay = True
                                logger.debug(f"max_duration 等待句末标点(当前词='{last_word}')，延迟切分...")
                            else:
                                logger.debug(f"max_duration 延迟等待超时({current_duration:.2f}s >= {delay_max_duration:.2f}s)，强制切分")
                    else:
                        # 等待任意标点（包括逗号等）
                        if not has_sentence_end_punct and not has_clause_punct:
                            if current_duration < delay_max_duration:
                                should_delay = True
                                logger.info(f"max_duration 等待任意标点(当前词='{last_word}')，延迟切分...")
                            else:
                                logger.info(f"max_duration 延迟等待超时({current_duration:.2f}s >= {delay_max_duration:.2f}s)，强制切分")

                if not should_delay:
                    should_split = True
                    split_reason = "max_duration"
            elif self.config.max_chars > 0:
                # max_chars > 0 时才启用字符数限制
                current_text = "".join(w.word for w in current_words)
                if len(current_text) >= self.config.max_chars:
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
                    logger.debug(f"分句: '{sentence.text[-50:]}' (原因: {split_reason}, 时长: {current_duration:.2f}s)")

                # 重置
                current_words = []
                if i < len(words) - 1:
                    current_start = words[i + 1].start

        # 合并过短句子
        if self.config.merge_short_sentences:
            sentences = self._merge_short_sentences(sentences)

        return sentences

    # ============================================================================
    # 辅助方法
    # ============================================================================

    def _is_chinese_char(self, char: str) -> bool:
        """判断字符是否为中文字符"""
        if not char:
            return False
        code = ord(char)
        # 常用汉字范围: CJK Unified Ideographs
        return (0x4E00 <= code <= 0x9FFF or   # 基本汉字
                0x3400 <= code <= 0x4DBF or   # CJK扩展A
                0x20000 <= code <= 0x2A6DF or # CJK扩展B
                0xF900 <= code <= 0xFAFF)     # CJK兼容汉字

    def _is_punctuation(self, char: str) -> bool:
        """判断字符是否为标点符号"""
        if not char:
            return False
        # 包含中英文常见标点
        punctuation = set(',.!?;:\'\"()[]{}，。！？；：""''（）【】《》、')
        return char in punctuation

    # ============================================================================
    # Layer 1 优化算法
    # ============================================================================

    def _trim_sentence_boundaries(self, sentence: 'SentenceSegment') -> 'SentenceSegment':
        """
        修剪句子边界，消除静音段

        核心思想：句子的 start/end 时间戳应该紧贴实际语音，而非简单取首尾词的时间戳

        Args:
            sentence: 待修剪的句子对象

        Returns:
            修剪后的句子对象
        """
        if not sentence.words:
            return sentence

        first_word = sentence.words[0]
        last_word = sentence.words[-1]

        # 修剪句首: 如果句子start比第一个词start早太多，说明包含了静音
        if self.config.trim_leading_silence:
            gap = first_word.start - sentence.start
            if gap > self.config.max_boundary_gap:
                sentence.start = first_word.start
                logger.debug(f"修剪句首静音: {gap:.3f}s -> 0s")

        # 修剪句尾: 如果句子end比最后一个词end晚太多，说明包含了静音
        if self.config.trim_trailing_silence:
            gap = sentence.end - last_word.end
            if gap > self.config.max_boundary_gap:
                sentence.end = last_word.end
                logger.debug(f"修剪句尾静音: {gap:.3f}s -> 0s")

        return sentence

    def _calculate_dynamic_pause_threshold(
        self,
        words: List['WordTimestamp'],
        current_index: int
    ) -> float:
        """
        根据局部语速计算动态停顿阈值

        优化策略:
        - 使用 75th 百分位数，抗噪性强且避免对快速语音过度敏感
        - 混合动态权重(50%)和静态基准(50%)，保持稳定性
        - 使用可配置的最小阈值，避免极端情况

        Args:
            words: 字级时间戳列表
            current_index: 当前词索引

        Returns:
            动态停顿阈值（秒）
        """
        if not self.config.use_dynamic_pause:
            return self.config.pause_threshold

        # 计算窗口范围
        window_size = self.config.speech_rate_window
        start_idx = max(0, current_index - window_size // 2)
        end_idx = min(len(words), current_index + window_size // 2)

        # 计算窗口内的字间隔
        intervals = []
        for i in range(start_idx, end_idx - 1):
            interval = words[i + 1].start - words[i].end
            # 只统计正常间隔，排除长停顿
            if 0 < interval < self.config.long_pause_threshold:
                intervals.append(interval)

        if not intervals:
            return self.config.pause_threshold

        # 优化: 使用 75th 百分位数，避免快速语音场景下阈值过低
        intervals_sorted = sorted(intervals)
        percentile_75_idx = int(len(intervals_sorted) * 0.75)
        percentile_75 = intervals_sorted[min(percentile_75_idx, len(intervals_sorted) - 1)]

        # 优化: 混合动态权重(50%)和静态基准(50%)，提高稳定性
        # 动态权重 0.5 + 静态基准 0.5
        weighted_threshold = (
            percentile_75 * self.config.pause_multiplier * 0.5 +
            self.config.pause_threshold * 0.5
        )

        # 优化: 使用可配置的最小阈值，确保快速语音场景下不会过度分句
        min_threshold = getattr(self.config, 'min_pause_threshold', 0.3)
        final_threshold = max(min(weighted_threshold, 2.0), min_threshold)

        logger.debug(
            f"动态停顿阈值: 75th百分位={percentile_75:.3f}s, "
            f"加权阈值={weighted_threshold:.3f}s, "
            f"最终阈值={final_threshold:.3f}s"
        )

        return final_threshold

    def _smart_split_long_sentence(
        self,
        words: List['WordTimestamp'],
        text: str
    ) -> List[List['WordTimestamp']]:
        """
        智能拆分过长的句子

        拆分优先级:
        1. 在句中标点处拆分（逗号、分号等）
        2. 在较长停顿处拆分
        3. 在接近中点的位置拆分

        性能优化:
        - 使用 running_length 计数器，避免循环内重复计算 (O(N) 而非 O(N^2))

        Args:
            words: 字级时间戳列表
            text: 文本内容

        Returns:
            拆分后的词列表（列表的列表）
        """
        if len(text) <= self.config.max_chars:
            return [words]

        # 寻找最佳拆分点
        best_split_idx = None
        best_score = -1
        target_len = self.config.max_chars * 0.7  # 目标长度为最大长度的70%

        # 性能优化: 维护一个 running_length 计数器，避免重复计算
        running_length = 0

        for i, word in enumerate(words[:-1]):
            running_length += len(word.word)
            score = 0

            # 1. 标点加分 (最高优先级)
            if self.config.prefer_punctuation_break:
                if word.word[-1] in ',;:':
                    score += 100
                elif word.word[-1] in '，、；：':
                    score += 100

            # 2. 停顿加分
            pause = words[i + 1].start - word.end
            if pause > 0.3:
                score += 50 * min(pause, 1.0)

            # 3. 接近目标长度加分
            length_diff = abs(running_length - target_len)
            score += max(0, 50 - length_diff)

            if score > best_score:
                best_score = score
                best_split_idx = i + 1

        if best_split_idx is None:
            best_split_idx = len(words) // 2

        # 递归拆分
        left_words = words[:best_split_idx]
        right_words = words[best_split_idx:]
        left_text = ''.join(w.word for w in left_words)
        right_text = ''.join(w.word for w in right_words)

        result = self._smart_split_long_sentence(left_words, left_text)
        result.extend(self._smart_split_long_sentence(right_words, right_text))

        logger.debug(
            f"智能拆分: 原长度={len(text)}, "
            f"拆分点={best_split_idx}, "
            f"得分={best_score:.1f}, "
            f"左={len(left_text)}, 右={len(right_text)}"
        )

        return result

    def _create_sentence(self, words: List['WordTimestamp']) -> Optional['SentenceSegment']:
        """创建句子对象 (集成边界修剪)"""
        from ..models.sensevoice_models import SentenceSegment
        from ..services.text_normalizer import get_text_normalizer

        if not words:
            return None

        # 智能构建文本：根据词的特性决定是否添加空格
        # 英文单词之间需要空格，中文字符之间不需要
        text_parts = []
        for i, w in enumerate(words):
            word = w.word
            text_parts.append(word)
            
            # 判断是否需要在此词后添加空格
            if i < len(words) - 1:
                next_word = words[i + 1].word
                # 如果当前词或下一词是中文字符，不加空格
                # 如果当前词以标点结尾且下一词不是标点，不加空格（标点已含空格语义）
                # 否则（英文单词之间）加空格
                if not self._is_chinese_char(word[-1] if word else '') and \
                   not self._is_chinese_char(next_word[0] if next_word else '') and \
                   not self._is_punctuation(next_word[0] if next_word else ''):
                    text_parts.append(' ')
        
        text_raw = "".join(text_parts)

        # 清洗文本（去除标签和连字符）
        normalizer = get_text_normalizer()
        text_clean = normalizer.clean(text_raw)

        # 过滤过短句子（基于清洗后的文本）
        if len(text_clean.strip()) < self.config.min_chars:
            return None

        # 计算平均置信度
        avg_confidence = sum(w.confidence for w in words) / len(words)

        sentence = SentenceSegment(
            text=text_raw,  # 保留原始文本用于调试
            text_clean=text_clean,  # 清洗后的文本用于展示
            start=words[0].start,
            end=words[-1].end,
            words=words.copy(),
            confidence=avg_confidence
        )

        # Layer 1 优化: 修剪边界静音
        sentence = self._trim_sentence_boundaries(sentence)

        return sentence

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
