"""
KeywordExtractor - 关键词提取服务

Phase 3 实现 - 2025-12-10

从 SenseVoice 草稿中提取关键词（人名、生僻词等），用于构建 Whisper Prompt。
不要把整句草稿放进 Whisper Prompt，只提取关键词。
"""

import logging
import re
from typing import List, Set, Optional
from dataclasses import dataclass

from app.models.sensevoice_models import WordTimestamp, SentenceSegment


@dataclass
class KeywordExtractionConfig:
    """关键词提取配置"""
    # 最小词长度
    min_word_length: int = 2
    
    # 最大关键词数量
    max_keywords: int = 10
    
    # 置信度阈值（只提取高置信度的词）
    confidence_threshold: float = 0.8
    
    # 是否提取数字
    extract_numbers: bool = True
    
    # 是否提取英文词
    extract_english: bool = True


class KeywordExtractor:
    """
    关键词提取服务
    
    从 SenseVoice 草稿中提取关键词，用于构建 Whisper Prompt。
    """
    
    def __init__(
        self,
        config: Optional[KeywordExtractionConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化关键词提取服务
        
        Args:
            config: 关键词提取配置
            logger: 日志记录器
        """
        self.config = config or KeywordExtractionConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # 常见停用词（中文）
        self.stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那'
        }
    
    def extract_from_tokens(
        self,
        tokens: List[WordTimestamp],
        max_keywords: Optional[int] = None
    ) -> List[str]:
        """
        从 SenseVoice 词级时间戳中提取关键词
        
        Args:
            tokens: SenseVoice 词级时间戳列表
            max_keywords: 最大关键词数量（覆盖配置）
        
        Returns:
            List[str]: 关键词列表
        """
        max_keywords = max_keywords or self.config.max_keywords
        
        keywords = []
        
        for token in tokens:
            # 跳过低置信度的词
            if token.confidence < self.config.confidence_threshold:
                continue
            
            word = token.word.strip()
            
            # 跳过空词
            if not word:
                continue
            
            # 跳过短词
            if len(word) < self.config.min_word_length:
                continue
            
            # 跳过停用词
            if word in self.stopwords:
                continue
            
            # 判断是否为关键词
            if self._is_keyword(word):
                keywords.append(word)
        
        # 去重并限制数量
        keywords = list(dict.fromkeys(keywords))[:max_keywords]
        
        self.logger.debug(f'提取关键词: {keywords}')
        
        return keywords
    
    def extract_from_sentence(
        self,
        sentence: SentenceSegment,
        max_keywords: Optional[int] = None
    ) -> List[str]:
        """
        从 SenseVoice 句子中提取关键词
        
        Args:
            sentence: SenseVoice 句子段
            max_keywords: 最大关键词数量
        
        Returns:
            List[str]: 关键词列表
        """
        if sentence.words:
            return self.extract_from_tokens(sentence.words, max_keywords)
        else:
            # 如果没有词级时间戳，从文本中提取
            return self.extract_from_text(sentence.text, max_keywords)
    
    def extract_from_text(
        self,
        text: str,
        max_keywords: Optional[int] = None
    ) -> List[str]:
        """
        从文本中提取关键词
        
        简单的基于规则的提取。
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
        
        Returns:
            List[str]: 关键词列表
        """
        max_keywords = max_keywords or self.config.max_keywords
        
        # 简单分词
        words = text.strip().split()
        
        keywords = []
        
        for word in words:
            word = word.strip()
            
            if not word:
                continue
            
            if len(word) < self.config.min_word_length:
                continue
            
            if word in self.stopwords:
                continue
            
            if self._is_keyword(word):
                keywords.append(word)
        
        # 去重并限制数量
        keywords = list(dict.fromkeys(keywords))[:max_keywords]
        
        return keywords
    
    def _is_keyword(self, word: str) -> bool:
        """
        判断是否为关键词
        
        关键词特征：
        1. 包含大写字母（可能是人名、地名、专有名词）
        2. 包含数字（如果配置允许）
        3. 纯英文词（如果配置允许）
        4. 中文长词（>= 3个字符）
        
        Args:
            word: 词
        
        Returns:
            bool: 是否为关键词
        """
        # 包含大写字母（专有名词）
        if any(c.isupper() for c in word):
            return True
        
        # 包含数字
        if self.config.extract_numbers and any(c.isdigit() for c in word):
            return True
        
        # 纯英文词
        if self.config.extract_english and word.isascii() and word.isalpha():
            return True
        
        # 中文长词
        if self._is_chinese(word) and len(word) >= 3:
            return True
        
        return False
    
    def _is_chinese(self, text: str) -> bool:
        """
        判断是否为中文
        
        Args:
            text: 文本
        
        Returns:
            bool: 是否为中文
        """
        # 简单判断：是否包含中文字符
        return bool(re.search(r'[一-鿿]', text))
    
    def build_whisper_prompt(
        self,
        keywords: List[str],
        previous_text: Optional[str] = None,
        user_glossary: Optional[List[str]] = None
    ) -> str:
        """
        构建 Whisper Prompt
        
        Prompt 格式：
        - 上一句 Whisper 文本（如果有）
        - 用户词表（如果有）
        - SenseVoice 关键词
        
        Args:
            keywords: 关键词列表
            previous_text: 上一句 Whisper 文本
            user_glossary: 用户词表
        
        Returns:
            str: Whisper Prompt
        """
        prompt_parts = []
        
        # 上一句文本
        if previous_text:
            prompt_parts.append(previous_text.strip())
        
        # 用户词表
        if user_glossary:
            prompt_parts.extend(user_glossary)
        
        # SenseVoice 关键词
        prompt_parts.extend(keywords)
        
        # 拼接
        prompt = ' '.join(prompt_parts)
        
        # 限制长度（Whisper Prompt 不宜过长）
        max_prompt_length = 200
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length]
        
        return prompt
