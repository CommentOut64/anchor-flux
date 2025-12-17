"""
SenseVoice 文本后处理器
清洗特殊标签、重复字符，统一标点符号
"""
import re
import logging

logger = logging.getLogger(__name__)


class TextNormalizer:
    """SenseVoice 文本后处理器"""

    # SenseVoice 特殊标签模式（如 <|zh|>, <|HAPPY|>, <|BGM|> 等）
    SPECIAL_TAGS = re.compile(r'<\|.*?\|>')

    # SentencePiece 分词符号（用于标记空格）
    SENTENCEPIECE_SYMBOL = '▁'

    # 异常重复字符（3个及以上相同字符）
    REPEATED_CHARS = re.compile(r'(.)\1{2,}')

    # 多余空白字符
    EXTRA_SPACES = re.compile(r'\s+')

    # 错误的数字格式（如 10,00 应该是 10,000）
    # 匹配：数字 + 逗号 + 2位数字 + 空格或结尾
    MALFORMED_NUMBER = re.compile(r'(\d+),(\d{2})(?=\s|$)')

    # 标点符号映射（全角 -> 半角）
    PUNCTUATION_MAP = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '；': ';', '：': ':', '"': '"', '"': '"',
        ''': "'", ''': "'", '（': '(', '）': ')',
        '【': '[', '】': ']', '《': '<', '》': '>',
    }

    # Whisper 幻觉检测模式
    # 重复子串检测: 长度>=4 且重复>=3次的子串
    REPEATED_PATTERN = re.compile(r'(.{4,})\1{2,}')

    # 特定幻觉句式（开头匹配）
    HALLUCINATION_PATTERNS = [
        re.compile(r'^Questions?\s+\d+', re.IGNORECASE),           # "Questions 19..."
        re.compile(r'^Subtitles?\s+by', re.IGNORECASE),            # "Subtitles by..."
        re.compile(r'^Copyright\s+', re.IGNORECASE),               # "Copyright 2024..."
        re.compile(r'^Thanks?\s+for\s+watching', re.IGNORECASE),   # "Thanks for watching"
        re.compile(r'^Please\s+subscribe', re.IGNORECASE),         # "Please subscribe"
    ]

    @staticmethod
    def clean(text: str) -> str:
        """
        清洗文本（移除特殊标签和异常重复）

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # 1. 移除 SenseVoice 特殊标签
        text = TextNormalizer.SPECIAL_TAGS.sub('', text)

        # 2. 替换 SentencePiece 分词符号为空格
        text = text.replace(TextNormalizer.SENTENCEPIECE_SYMBOL, ' ')

        # 3. 处理异常重复字符（保留最多2个）
        text = TextNormalizer.REPEATED_CHARS.sub(r'\1\1', text)

        # 4. 修复错误的数字格式（10,00 -> 10,000）
        text = TextNormalizer._fix_malformed_numbers(text)

        # 5. 规范化空白字符
        text = TextNormalizer.EXTRA_SPACES.sub(' ', text)

        return text.strip()

    @staticmethod
    def _fix_malformed_numbers(text: str) -> str:
        """
        修复错误的数字格式

        常见错误：
        - "10,00" -> "10,000"（千位分隔符后只有2位数字）
        - "15,00" -> "15,000"

        Args:
            text: 输入文本

        Returns:
            修复后的文本
        """
        if not text:
            return ""

        # 修复模式：数字 + 逗号 + 2位数字 -> 数字 + 逗号 + 3位数字（补0）
        # 例如：10,00 -> 10,000
        fixed_text = TextNormalizer.MALFORMED_NUMBER.sub(r'\1,\g<2>0', text)

        return fixed_text

    @classmethod
    def is_whisper_hallucination(cls, text: str) -> bool:
        """
        检测 Whisper 输出是否为幻觉文本

        Args:
            text: Whisper 输出的文本

        Returns:
            bool: True 表示检测到幻觉
        """
        if not text:
            return False

        text = text.strip()

        # 检测1: 重复子串模式
        if cls.REPEATED_PATTERN.search(text):
            return True

        # 检测2: 特定幻觉句式
        for pattern in cls.HALLUCINATION_PATTERNS:
            if pattern.match(text):
                return True

        # 检测3: 纯下划线/符号（清洗后为空或大幅缩减）
        cleaned = cls.clean(text)
        if not cleaned:
            # 清洗后完全为空
            return True
        
        # 如果清洗后损失严重(>60%)且原文或清洗后很短，可能是垃圾字符
        loss_ratio = 1.0 - (len(cleaned) / len(text))
        if loss_ratio > 0.6 and (len(text) <= 10 or len(cleaned) <= 3):
            return True

        return False

    @classmethod
    def clean_whisper_output(cls, text: str) -> str:
        """
        清洗 Whisper 输出（比 SenseVoice 清洗更激进）

        Args:
            text: Whisper 原始输出

        Returns:
            str: 清洗后的文本，如果是幻觉则返回空字符串
        """
        if not text:
            return ""

        # 先用基础清洗
        cleaned = cls.clean(text)

        # 再检测是否为幻觉
        if cls.is_whisper_hallucination(text):
            return ""

        return cleaned

    @staticmethod
    def normalize_punctuation(text: str, to_fullwidth: bool = True) -> str:
        """
        统一标点符号（全角/半角）

        Args:
            text: 输入文本
            to_fullwidth: True=转全角, False=转半角

        Returns:
            标点统一后的文本
        """
        if not text:
            return ""

        if to_fullwidth:
            # 半角转全角
            mapping = {v: k for k, v in TextNormalizer.PUNCTUATION_MAP.items()}
        else:
            # 全角转半角
            mapping = TextNormalizer.PUNCTUATION_MAP

        for old, new in mapping.items():
            text = text.replace(old, new)

        return text

    @staticmethod
    def extract_tags(text: str) -> dict:
        """
        提取 SenseVoice 特殊标签信息

        Args:
            text: 原始文本

        Returns:
            提取的标签信息字典
        """
        tags = {
            "language": None,
            "emotion": None,
            "event": None,
            "raw_tags": []
        }

        if not text:
            return tags

        # 查找所有标签
        found_tags = TextNormalizer.SPECIAL_TAGS.findall(text)
        tags["raw_tags"] = found_tags

        for tag in found_tags:
            # 移除 <| 和 |>
            tag_content = tag[2:-2].lower()

            # 语言标签
            if tag_content in ['zh', 'en', 'ja', 'ko', 'yue', 'auto']:
                tags["language"] = tag_content
            # 情感标签
            elif tag_content in ['happy', 'sad', 'angry', 'neutral', 'fearful', 'disgusted', 'surprised']:
                tags["emotion"] = tag_content
            # 事件标签
            elif tag_content in ['bgm', 'noise', 'speech', 'applause', 'laughter']:
                tags["event"] = tag_content

        return tags

    @staticmethod
    def process(text: str, to_fullwidth: bool = None, extract_info: bool = False, language: str = None) -> dict:
        """
        完整处理流程

        Args:
            text: 原始文本
            to_fullwidth: 是否转换为全角标点（None=自动根据语言判断）
            extract_info: 是否提取标签信息
            language: 语言代码（zh/en/ja等），用于自适应标点归一化

        Returns:
            处理结果字典，包含 text_clean 和可选的 tags
        """
        result = {
            "text_original": text,
            "text_clean": "",
            "tags": None
        }

        if not text:
            return result

        # 提取标签信息（在清洗前）
        if extract_info:
            result["tags"] = TextNormalizer.extract_tags(text)

        # 清洗文本
        clean_text = TextNormalizer.clean(text)

        # 自适应标点归一化：根据语言决定使用全角还是半角标点
        if to_fullwidth is None:
            # 从标签中提取语言（如果没有提取标签，先提取）
            if language is None and result["tags"] is not None:
                language = result["tags"].get("language")

            # 自动判断：中文/日文使用全角，英文使用半角
            if language in ['zh', 'ja', 'ko', 'yue']:
                to_fullwidth = True
            elif language in ['en']:
                to_fullwidth = False
            else:
                # 未知语言，根据文本特征判断
                to_fullwidth = TextNormalizer._detect_language_by_text(clean_text) in ['zh', 'ja']

        # 统一标点
        clean_text = TextNormalizer.normalize_punctuation(clean_text, to_fullwidth)

        result["text_clean"] = clean_text

        return result

    @staticmethod
    def _detect_language_by_text(text: str) -> str:
        """
        根据文本特征简单判断语言（仅用于标点归一化）

        Args:
            text: 文本

        Returns:
            str: 语言代码 zh/en/ja
        """
        if not text:
            return "en"

        # 统计中文字符比例
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.strip())

        if total_chars == 0:
            return "en"

        chinese_ratio = chinese_chars / total_chars

        # 中文字符超过30%，认为是中文
        if chinese_ratio > 0.3:
            return "zh"
        else:
            return "en"


# 单例实例
_normalizer_instance = None


def get_text_normalizer() -> TextNormalizer:
    """获取文本清洗器单例"""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = TextNormalizer()
    return _normalizer_instance
