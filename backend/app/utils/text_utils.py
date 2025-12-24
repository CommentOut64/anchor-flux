"""
文本处理工具函数

提供智能文本拼接等功能
"""
from typing import List, Union, Dict, Any


def smart_join_words(words: List[Union[Dict[str, Any], Any]], word_key: str = "word") -> str:
    """
    智能拼接 words，处理标点符号空格问题

    规则：
    - 标点符号前不加空格
    - 英文单词之间加空格
    - 中文字符之间不加空格

    Args:
        words: 词列表，可以是字典列表或对象列表
        word_key: 如果是字典，指定 word 字段的 key；如果是对象，指定属性名

    Returns:
        拼接后的文本

    Examples:
        >>> words = [{"word": "It"}, {"word": "'"}, {"word": "s"}]
        >>> smart_join_words(words)
        "It's"

        >>> words = [{"word": "Hello"}, {"word": ","}, {"word": "world"}]
        >>> smart_join_words(words)
        "Hello, world"
    """
    if not words:
        return ""

    result = []
    punctuation = set(",.!?;:'\"()[]{}，。！？；：""''（）【】《》、")

    for i, word_obj in enumerate(words):
        # 获取 word 字符串
        if isinstance(word_obj, dict):
            word = word_obj.get(word_key, "")
        else:
            word = getattr(word_obj, word_key, "")

        if not word:
            continue

        if i == 0:
            result.append(word)
        else:
            # 获取前一个 word
            prev_obj = words[i-1]
            if isinstance(prev_obj, dict):
                prev_word = prev_obj.get(word_key, "")
            else:
                prev_word = getattr(prev_obj, word_key, "")

            # 判断是否需要添加空格
            # 标点符号前不加空格
            if word in punctuation:
                result.append(word)
            # 前一个是标点符号，根据情况决定
            elif prev_word in punctuation:
                # 如果前一个是引号、括号等，可能需要空格
                if prev_word in "\"'([{""''（【《":
                    result.append(word)
                else:
                    result.append(" " + word)
            # 中文字符之间不加空格
            elif any('\u4e00' <= c <= '\u9fff' for c in word) or \
                 any('\u4e00' <= c <= '\u9fff' for c in prev_word):
                result.append(word)
            # 英文单词之间加空格
            else:
                result.append(" " + word)

    return "".join(result)
