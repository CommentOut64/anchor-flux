"""
置信度与困惑度阈值配置

基于实际测试和业界经验设定阈值：
- 转录置信度：基于 Whisper avg_logprob 和 no_speech_prob
- 校对困惑度：基于 LLM 输出的 perplexity
"""
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """置信度等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"


@dataclass
class ThresholdConfig:
    """阈值配置"""

    # ========== 转录置信度阈值 ==========
    # SenseVoice 置信度阈值
    sv_confidence_high: float = 0.85      # 高置信度
    sv_confidence_medium: float = 0.6     # 中等置信度
    sv_confidence_low: float = 0.4        # 低置信度（触发补刀）

    # Whisper 阈值
    whisper_logprob_good: float = -0.5    # 好的 logprob
    whisper_logprob_ok: float = -0.8      # 可接受的 logprob
    whisper_no_speech_medium: float = 0.6 # 中等无语音概率

    # ========== 校对困惑度阈值 ==========
    llm_perplexity_good: float = 35.0     # 好的困惑度
    llm_perplexity_acceptable: float = 50.0  # 可接受的困惑度
    llm_perplexity_poor: float = 80.0     # 差的困惑度

    # ========== 触发阈值 ==========
    # Whisper 补刀触发条件
    whisper_patch_trigger_confidence: float = 0.6  # 低于此值触发补刀

    # 短片段强制补刀条件 (应对 CTC 对快速语音的限制)
    short_segment_duration: float = 1.0  # 短片段时长阈值(秒)
    short_segment_chars: int = 3         # 短片段字符数阈值
    single_char_force_patch: bool = True # 单字符结果强制补刀

    # LLM 校对触发条件
    llm_proof_trigger_confidence: float = 0.7      # 低于此值触发校对
    llm_proof_trigger_modified: bool = True        # 被修改过的句子是否触发校对

    # ========== 警告高亮阈值 ==========
    # 字级警告
    word_warning_confidence: float = 0.5   # 字级警告阈值
    word_critical_confidence: float = 0.3  # 字级严重警告阈值

    # 句级警告
    sentence_warning_confidence: float = 0.6   # 句级置信度警告阈值
    sentence_warning_perplexity: float = 50.0  # 句级困惑度警告阈值


# 默认阈值配置
DEFAULT_THRESHOLDS = ThresholdConfig()


def get_confidence_level(confidence: float, config: ThresholdConfig = None) -> ConfidenceLevel:
    """
    根据置信度获取等级

    Args:
        confidence: 置信度值 (0-1)
        config: 阈值配置

    Returns:
        置信度等级
    """
    if config is None:
        config = DEFAULT_THRESHOLDS

    if confidence >= config.sv_confidence_high:
        return ConfidenceLevel.HIGH
    elif confidence >= config.sv_confidence_medium:
        return ConfidenceLevel.MEDIUM
    elif confidence >= config.sv_confidence_low:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.CRITICAL


def needs_whisper_patch(
    confidence: float,
    duration: float = None,
    text_length: int = None,
    config: ThresholdConfig = None
) -> bool:
    """
    判断是否需要 Whisper 补刀

    触发条件（满足任一即触发）：
    1. 置信度低于阈值
    2. 短片段 + 少字符（CTC 解码限制）
    3. 单字符结果（可能是 CTC 漏字）

    Args:
        confidence: 置信度值
        duration: 片段时长(秒)，用于短片段检测
        text_length: 文本字符数（清洗后），用于短片段检测
        config: 阈值配置

    Returns:
        是否需要补刀
    """
    if config is None:
        config = DEFAULT_THRESHOLDS

    # 条件 1: 低置信度触发
    if confidence < config.whisper_patch_trigger_confidence:
        return True

    # 条件 2: 短片段 + 少字符 (应对 CTC 对快速语音的限制)
    # 时长 < 1s 且字符数 < 3 时，CTC 可能遗漏快速语音
    if duration is not None and text_length is not None:
        if (duration < config.short_segment_duration and
            text_length < config.short_segment_chars):
            return True

    # 条件 3: 单字符结果强制补刀 (极可能是 CTC 漏字)
    if config.single_char_force_patch and text_length is not None:
        if text_length == 1:
            return True

    return False


def needs_llm_proof(
    confidence: float,
    is_modified: bool = False,
    mode: str = 'sparse',
    config: ThresholdConfig = None
) -> bool:
    """
    判断是否需要 LLM 校对

    Args:
        confidence: 置信度值
        is_modified: 是否被修改过
        mode: 校对模式 ('sparse' 或 'full')
        config: 阈值配置

    Returns:
        是否需要校对
    """
    if config is None:
        config = DEFAULT_THRESHOLDS

    # 全量模式：所有句子都校对
    if mode == 'full':
        return True

    # 稀疏模式：只校对低置信度或被修改的句子
    if mode == 'sparse':
        if confidence < config.llm_proof_trigger_confidence:
            return True
        if is_modified and config.llm_proof_trigger_modified:
            return True

    return False


def get_warning_level(
    confidence: float,
    perplexity: float = None,
    config: ThresholdConfig = None
) -> str:
    """
    获取警告级别

    Args:
        confidence: 置信度值
        perplexity: 困惑度值
        config: 阈值配置

    Returns:
        警告级别: 'none', 'low_confidence', 'high_perplexity', 'both'
    """
    if config is None:
        config = DEFAULT_THRESHOLDS

    has_low_confidence = confidence < config.sentence_warning_confidence
    has_high_perplexity = perplexity is not None and perplexity > config.sentence_warning_perplexity

    if has_low_confidence and has_high_perplexity:
        return 'both'
    elif has_low_confidence:
        return 'low_confidence'
    elif has_high_perplexity:
        return 'high_perplexity'
    return 'none'
