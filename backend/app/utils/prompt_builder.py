"""
Prompt Builder - 安全的 Whisper Prompt 构建器

Phase 4 实现 - 2025-12-10

提供轻量级的关键词提取，避免 initial_prompt 干扰 Whisper 识别。
"""

import re
from typing import List, Set, Optional


class PromptBuilder:
    """
    安全的 Whisper Prompt 构建器

    核心策略：
    1. 从前文中提取专有名词（大写开头的词）
    2. 过滤停用词
    3. 构建 "Glossary: word1, word2, ..." 格式的 prompt
    4. 避免使用完整句子，防止 Whisper 误判
    """

    # 停用词表（句首常见的大写词）
    STOPWORDS = {
        "The", "And", "But", "This", "That", "There", "Here",
        "When", "What", "Who", "Where", "Why", "How",
        "Then", "Now", "So", "Or", "If", "As", "At",
        "It", "Its", "I", "You", "He", "She", "We", "They"
    }

    def __init__(self, max_keywords: int = 20, min_word_length: int = 2):
        """
        初始化 Prompt 构建器

        Args:
            max_keywords: 最多提取的关键词数量
            min_word_length: 最小词长度
        """
        self.max_keywords = max_keywords
        self.min_word_length = min_word_length

    def extract_keywords(self, text: str) -> Set[str]:
        """
        从文本中提取关键词（专有名词）

        策略：
        1. 提取所有大写开头的词
        2. 过滤停用词
        3. 过滤短词（长度 <= min_word_length）
        4. 去重

        Args:
            text: 输入文本

        Returns:
            Set[str]: 关键词集合
        """
        if not text:
            return set()

        # 分词（按空格和标点）
        words = re.findall(r"\b[A-Za-z0-9']+\b", text)

        keywords = set()

        for word in words:
            # 条件：
            # 1. 首字母大写
            # 2. 不在停用词表
            # 3. 长度 > min_word_length
            if (word[0].isupper() and
                word not in self.STOPWORDS and
                len(word) > self.min_word_length):
                keywords.add(word)

        return keywords

    def build_prompt(
        self,
        previous_text: Optional[str] = None,
        user_glossary: Optional[List[str]] = None
    ) -> str:
        """
        构建安全的 Whisper Prompt

        格式："Glossary: word1, word2, word3."

        这种格式明确告诉 Whisper："这些是提示词"，而不是"这是上一句话"

        Args:
            previous_text: 上一个 Chunk 的定稿文本
            user_glossary: 用户自定义词表

        Returns:
            str: Prompt 字符串
        """
        keywords = set()

        # 1. 从前文中提取关键词
        if previous_text:
            keywords.update(self.extract_keywords(previous_text))

        # 2. 添加用户词表
        if user_glossary:
            keywords.update(user_glossary)

        # 3. 限制关键词数量
        if len(keywords) > self.max_keywords:
            # 按字母顺序排序后取前 N 个
            keywords = set(sorted(keywords)[:self.max_keywords])

        # 4. 构建 Prompt
        if not keywords:
            return ""

        prompt = "Glossary: " + ", ".join(sorted(keywords)) + "."
        return prompt


# 全局单例
_prompt_builder = None


def get_prompt_builder() -> PromptBuilder:
    """
    获取全局 Prompt 构建器单例

    Returns:
        PromptBuilder: Prompt 构建器实例
    """
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder
