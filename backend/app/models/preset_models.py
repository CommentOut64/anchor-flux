"""
预设配置数据模型

定义 SenseVoice 预设方案和硬件自适应配置
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class PresetTier(Enum):
    """预设等级"""
    LITE = "lite"           # 轻量级，低显存
    STANDARD = "standard"   # 标准，平衡
    QUALITY = "quality"     # 高质量，高显存
    CUSTOM = "custom"       # 自定义


class EnhancementMode(Enum):
    """增强模式"""
    OFF = "off"                    # 仅 SenseVoice
    SMART_PATCH = "smart_patch"   # 低置信度自动 Whisper 补刀
    DEEP_LISTEN = "deep_listen"   # Whisper 全文转录


class ProofreadMode(Enum):
    """校对模式"""
    OFF = "off"           # 不校对
    SPARSE = "sparse"     # 仅校对低置信度和疑问片段
    FULL = "full"         # 滑动窗口全量校对润色


class TranslateMode(Enum):
    """翻译模式"""
    OFF = "off"           # 不翻译
    FULL = "full"         # 全量翻译
    PARTIAL = "partial"   # 部分翻译（仅低置信度）


@dataclass
class WhisperPatchConfig:
    """Whisper 补刀配置"""
    enabled: bool = True
    model: str = "medium"                    # Whisper 模型
    confidence_threshold: float = 0.5        # 触发补刀的置信度阈值
    short_segment_threshold: float = 0.3     # 短片段阈值（秒）
    single_char_force_patch: bool = True     # 单字符强制补刀
    use_prompt: bool = True                  # 是否使用上下文 Prompt
    max_prompt_tokens: int = 224             # 最大 Prompt token 数

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "confidence_threshold": self.confidence_threshold,
            "short_segment_threshold": self.short_segment_threshold,
            "single_char_force_patch": self.single_char_force_patch,
            "use_prompt": self.use_prompt,
            "max_prompt_tokens": self.max_prompt_tokens
        }


@dataclass
class LLMConfig:
    """LLM 配置"""
    enabled: bool = False
    provider: str = "openai"                 # openai, claude, local
    model: str = "gpt-4o-mini"               # 模型名称
    api_key: Optional[str] = None            # API Key（运行时注入）
    base_url: Optional[str] = None           # 自定义 API 地址

    # 校对配置
    proofread_mode: ProofreadMode = field(default=ProofreadMode.OFF)
    proofread_window_size: int = 5           # 滑动窗口大小（句数）
    proofread_overlap: int = 2               # 窗口重叠句数

    # 翻译配置
    translate_mode: TranslateMode = field(default=TranslateMode.OFF)
    target_language: str = "en"              # 目标语言

    # 性能配置
    max_concurrent: int = 3                  # 最大并发请求数
    timeout: float = 30.0                    # 请求超时（秒）
    retry_count: int = 2                     # 重试次数

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "proofread_mode": self.proofread_mode.value,
            "proofread_window_size": self.proofread_window_size,
            "proofread_overlap": self.proofread_overlap,
            "translate_mode": self.translate_mode.value,
            "target_language": self.target_language,
            "max_concurrent": self.max_concurrent,
            "timeout": self.timeout,
            "retry_count": self.retry_count
        }


@dataclass
class PresetConfig:
    """预设配置"""
    id: str                                  # 预设 ID
    name: str                                # 预设名称
    description: str                         # 预设描述
    tier: PresetTier = field(default=PresetTier.STANDARD)

    # 显存要求
    min_vram_mb: int = 2000                  # 最低显存要求（MB）
    recommended_vram_mb: int = 4000          # 推荐显存（MB）

    # 增强模式
    enhancement_mode: EnhancementMode = field(default=EnhancementMode.OFF)

    # Whisper 补刀配置
    whisper_patch: WhisperPatchConfig = field(default_factory=WhisperPatchConfig)

    # LLM 配置
    llm: LLMConfig = field(default_factory=LLMConfig)

    # 置信度阈值
    confidence_threshold: float = 0.6        # 低置信度阈值
    warning_threshold: float = 0.4           # 警告阈值

    # 性能配置
    batch_size: int = 16                     # 批处理大小
    enable_vad_optimization: bool = True     # 启用 VAD 优化

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tier": self.tier.value,
            "min_vram_mb": self.min_vram_mb,
            "recommended_vram_mb": self.recommended_vram_mb,
            "enhancement_mode": self.enhancement_mode.value,
            "whisper_patch": self.whisper_patch.to_dict(),
            "llm": self.llm.to_dict(),
            "confidence_threshold": self.confidence_threshold,
            "warning_threshold": self.warning_threshold,
            "batch_size": self.batch_size,
            "enable_vad_optimization": self.enable_vad_optimization
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PresetConfig":
        """从字典创建预设配置"""
        whisper_patch_data = data.get("whisper_patch", {})
        whisper_patch = WhisperPatchConfig(
            enabled=whisper_patch_data.get("enabled", True),
            model=whisper_patch_data.get("model", "medium"),
            confidence_threshold=whisper_patch_data.get("confidence_threshold", 0.5),
            short_segment_threshold=whisper_patch_data.get("short_segment_threshold", 0.3),
            single_char_force_patch=whisper_patch_data.get("single_char_force_patch", True),
            use_prompt=whisper_patch_data.get("use_prompt", True),
            max_prompt_tokens=whisper_patch_data.get("max_prompt_tokens", 224)
        )

        llm_data = data.get("llm", {})
        llm = LLMConfig(
            enabled=llm_data.get("enabled", False),
            provider=llm_data.get("provider", "openai"),
            model=llm_data.get("model", "gpt-4o-mini"),
            base_url=llm_data.get("base_url"),
            proofread_mode=ProofreadMode(llm_data.get("proofread_mode", "off")),
            proofread_window_size=llm_data.get("proofread_window_size", 5),
            proofread_overlap=llm_data.get("proofread_overlap", 2),
            translate_mode=TranslateMode(llm_data.get("translate_mode", "off")),
            target_language=llm_data.get("target_language", "en"),
            max_concurrent=llm_data.get("max_concurrent", 3),
            timeout=llm_data.get("timeout", 30.0),
            retry_count=llm_data.get("retry_count", 2)
        )

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            tier=PresetTier(data.get("tier", "standard")),
            min_vram_mb=data.get("min_vram_mb", 2000),
            recommended_vram_mb=data.get("recommended_vram_mb", 4000),
            enhancement_mode=EnhancementMode(data.get("enhancement_mode", "off")),
            whisper_patch=whisper_patch,
            llm=llm,
            confidence_threshold=data.get("confidence_threshold", 0.6),
            warning_threshold=data.get("warning_threshold", 0.4),
            batch_size=data.get("batch_size", 16),
            enable_vad_optimization=data.get("enable_vad_optimization", True)
        )


# 预定义预设方案
PRESET_DEFINITIONS: Dict[str, PresetConfig] = {
    # 预设 0: 纯 SenseVoice（最快）
    "preset_0_pure_sv": PresetConfig(
        id="preset_0_pure_sv",
        name="Pure SenseVoice",
        description="仅使用 SenseVoice，最快速度，适合清晰人声",
        tier=PresetTier.LITE,
        min_vram_mb=1500,
        recommended_vram_mb=2000,
        enhancement_mode=EnhancementMode.OFF,
        whisper_patch=WhisperPatchConfig(enabled=False),
        llm=LLMConfig(enabled=False),
        confidence_threshold=0.6,
        batch_size=32
    ),

    # 预设 1: 智能补刀（推荐）
    "preset_1_smart_patch": PresetConfig(
        id="preset_1_smart_patch",
        name="Smart Patch",
        description="SenseVoice + 低置信度 Whisper 补刀，平衡速度和质量",
        tier=PresetTier.STANDARD,
        min_vram_mb=3000,
        recommended_vram_mb=4000,
        enhancement_mode=EnhancementMode.SMART_PATCH,
        whisper_patch=WhisperPatchConfig(
            enabled=True,
            model="medium",
            confidence_threshold=0.5
        ),
        llm=LLMConfig(enabled=False),
        confidence_threshold=0.6,
        batch_size=16
    ),

    # 预设 2: 深度聆听
    "preset_2_deep_listen": PresetConfig(
        id="preset_2_deep_listen",
        name="Deep Listen",
        description="SenseVoice 时间戳 + Whisper 全文转录，最高准确度",
        tier=PresetTier.QUALITY,
        min_vram_mb=6000,
        recommended_vram_mb=8000,
        enhancement_mode=EnhancementMode.DEEP_LISTEN,
        whisper_patch=WhisperPatchConfig(
            enabled=True,
            model="large-v3",
            confidence_threshold=0.0  # 全量补刀
        ),
        llm=LLMConfig(enabled=False),
        confidence_threshold=0.6,
        batch_size=8
    ),

    # 预设 3: 智能补刀 + LLM 校对
    "preset_3_with_proofread": PresetConfig(
        id="preset_3_with_proofread",
        name="Smart Patch + Proofread",
        description="智能补刀 + LLM 稀疏校对，适合专业场景",
        tier=PresetTier.QUALITY,
        min_vram_mb=4000,
        recommended_vram_mb=6000,
        enhancement_mode=EnhancementMode.SMART_PATCH,
        whisper_patch=WhisperPatchConfig(
            enabled=True,
            model="medium",
            confidence_threshold=0.5
        ),
        llm=LLMConfig(
            enabled=True,
            proofread_mode=ProofreadMode.SPARSE
        ),
        confidence_threshold=0.6,
        batch_size=16
    ),

    # 预设 4: 深度聆听 + 全量校对
    "preset_4_full_quality": PresetConfig(
        id="preset_4_full_quality",
        name="Full Quality",
        description="深度聆听 + LLM 全量校对，最高质量",
        tier=PresetTier.QUALITY,
        min_vram_mb=8000,
        recommended_vram_mb=12000,
        enhancement_mode=EnhancementMode.DEEP_LISTEN,
        whisper_patch=WhisperPatchConfig(
            enabled=True,
            model="large-v3",
            confidence_threshold=0.0
        ),
        llm=LLMConfig(
            enabled=True,
            proofread_mode=ProofreadMode.FULL,
            proofread_window_size=5
        ),
        confidence_threshold=0.6,
        batch_size=8
    ),

    # 预设 5: 带翻译
    "preset_5_with_translation": PresetConfig(
        id="preset_5_with_translation",
        name="With Translation",
        description="智能补刀 + LLM 翻译，适合多语言场景",
        tier=PresetTier.QUALITY,
        min_vram_mb=4000,
        recommended_vram_mb=6000,
        enhancement_mode=EnhancementMode.SMART_PATCH,
        whisper_patch=WhisperPatchConfig(
            enabled=True,
            model="medium",
            confidence_threshold=0.5
        ),
        llm=LLMConfig(
            enabled=True,
            translate_mode=TranslateMode.FULL,
            target_language="en"
        ),
        confidence_threshold=0.6,
        batch_size=16
    )
}


def get_preset(preset_id: str) -> Optional[PresetConfig]:
    """获取预设配置"""
    return PRESET_DEFINITIONS.get(preset_id)


def get_all_presets() -> List[PresetConfig]:
    """获取所有预设配置"""
    return list(PRESET_DEFINITIONS.values())


def get_recommended_preset(vram_mb: int) -> PresetConfig:
    """根据显存推荐预设"""
    if vram_mb >= 8000:
        return PRESET_DEFINITIONS["preset_2_deep_listen"]
    elif vram_mb >= 4000:
        return PRESET_DEFINITIONS["preset_1_smart_patch"]
    else:
        return PRESET_DEFINITIONS["preset_0_pure_sv"]
