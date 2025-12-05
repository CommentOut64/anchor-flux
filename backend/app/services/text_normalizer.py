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

    # 异常重复字符（3个及以上相同字符）
    REPEATED_CHARS = re.compile(r'(.)\1{2,}')

    # 多余空白字符
    EXTRA_SPACES = re.compile(r'\s+')

    # 标点符号映射（全角 -> 半角）
    PUNCTUATION_MAP = {
        '，': ',', '。': '.', '！': '!', '？': '?',
        '；': ';', '：': ':', '"': '"', '"': '"',
        ''': "'", ''': "'", '（': '(', '）': ')',
        '【': '[', '】': ']', '《': '<', '》': '>',
    }

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

        # 2. 处理异常重复字符（保留最多2个）
        text = TextNormalizer.REPEATED_CHARS.sub(r'\1\1', text)

        # 3. 规范化空白字符
        text = TextNormalizer.EXTRA_SPACES.sub(' ', text)

        return text.strip()

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
    def process(text: str, to_fullwidth: bool = True, extract_info: bool = False) -> dict:
        """
        完整处理流程

        Args:
            text: 原始文本
            to_fullwidth: 是否转换为全角标点
            extract_info: 是否提取标签信息

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

        # 统一标点
        clean_text = TextNormalizer.normalize_punctuation(clean_text, to_fullwidth)

        result["text_clean"] = clean_text

        return result


# 单例实例
_normalizer_instance = None


def get_text_normalizer() -> TextNormalizer:
    """获取文本清洗器单例"""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = TextNormalizer()
    return _normalizer_instance
