# Demucs 分级人声分离系统设计文档

## 版本信息

| 版本 | 日期 | 作者 | 说明 |
|------|------|------|------|
| v1.0 | 2025-12-03 | - | 初始版本，基于现有代码架构设计 |

---

## 1. 系统概述

### 1.1 设计目标

构建一个**分级人声分离系统**，根据背景音乐（BGM）强度智能选择不同质量/速度的 Demucs 模型，并通过动态熔断机制在质量不达标时自动升级模型，同时提供用户可配置的选项。

### 1.2 核心理念

```
┌─────────────────────────────────────────────────────────────────┐
│                        分级策略矩阵                              │
├─────────────┬─────────────────┬─────────────────────────────────┤
│  BGM 级别   │   初始模型       │   熔断升级路径                   │
├─────────────┼─────────────────┼─────────────────────────────────┤
│  NONE       │   不启用分离     │   -                             │
│  LIGHT      │   htdemucs_ft   │   → mdx_extra                   │
│  HEAVY      │   mdx_extra_q   │   → mdx_extra                   │
└─────────────┴─────────────────┴─────────────────────────────────┘
```

### 1.3 设计原则

1. **渐进式质量提升**：先用轻量模型，不够再升级
2. **用户可控**：所有策略参数可配置
3. **向后兼容**：不破坏现有 API 和配置
4. **可观测性**：SSE 实时推送状态变化

---

## 2. 现有代码分析

### 2.1 当前架构

```
backend/app/
├── models/
│   └── job_models.py          # DemucsSettings, JobSettings, JobState
├── services/
│   ├── demucs_service.py      # DemucsService, DemucsConfig, BGMLevel
│   └── transcription_service.py  # TranscriptionService, CircuitBreakerState
```

### 2.2 现有类结构

#### DemucsSettings（job_models.py）- 用户配置
```python
@dataclass
class DemucsSettings:
    enabled: bool = True
    mode: str = "auto"                      # auto/always/never/on_demand
    retry_threshold_logprob: float = -0.8
    retry_threshold_no_speech: float = 0.6
    circuit_breaker_enabled: bool = True
    consecutive_threshold: int = 3
    ratio_threshold: float = 0.2
```

#### DemucsConfig（demucs_service.py）- 服务配置
```python
@dataclass
class DemucsConfig:
    model_name: str = "mdx_extra"
    device: str = "cuda"
    shifts: int = 2
    overlap: float = 0.5
    segment_length: int = 10
    segment_buffer_sec: float = 2.0
    bgm_sample_duration: float = 10.0
    bgm_light_threshold: float = 0.02
    bgm_heavy_threshold: float = 0.15
    available_models: List[str] = [...]
```

#### CircuitBreakerState（transcription_service.py）- 熔断状态
```python
@dataclass
class CircuitBreakerState:
    consecutive_retries: int = 0
    total_retries: int = 0
    total_segments: int = 0
    processed_segments: int = 0
```

### 2.3 现有不足

1. **单一模型**：DemucsConfig 只有一个 `model_name`，无法实现分级
2. **熔断后无升级**：熔断触发后只是切换到全局分离，不会更换模型
3. **用户配置有限**：DemucsSettings 缺少模型选择和分级策略配置

---

## 3. 改造方案

### 3.1 配置结构改造

#### 3.1.1 新增 ModelTierConfig（分级模型配置）

```python
# demucs_service.py 新增

@dataclass
class ModelTierConfig:
    """分级模型配置"""
    # 弱BGM场景使用的模型（速度优先）
    weak_model: str = "htdemucs_ft"
    
    # 强BGM场景使用的模型（质量优先）
    strong_model: str = "mdx_extra_q"
    
    # 兜底模型（熔断升级后使用）
    fallback_model: str = "mdx_extra"
    
    # 模型质量参数（按模型分别配置）
    model_quality: Dict[str, Dict] = field(default_factory=lambda: {
        "htdemucs": {"shifts": 1, "overlap": 0.25},      # 最快
        "htdemucs_ft": {"shifts": 1, "overlap": 0.25},   # 快速+人声优化
        "mdx_extra_q": {"shifts": 2, "overlap": 0.5},    # 中等
        "mdx_extra": {"shifts": 2, "overlap": 0.5},      # 最高质量
    })
```

#### 3.1.2 扩展 DemucsSettings（用户配置）

```python
# job_models.py 修改

@dataclass
class DemucsSettings:
    """Demucs人声分离配置（用户可配置）"""
    # === 基础开关 ===
    enabled: bool = True
    mode: str = "auto"  # auto/always/never/on_demand
    
    # === 分级模型配置 ===
    weak_model: str = "htdemucs_ft"       # 弱BGM使用的模型
    strong_model: str = "mdx_extra_q"     # 强BGM使用的模型
    fallback_model: str = "mdx_extra"     # 兜底模型
    auto_escalation: bool = True          # 是否允许自动升级模型
    max_escalations: int = 1              # 最大升级次数
    
    # === BGM检测阈值 ===
    bgm_light_threshold: float = 0.02     # 轻微BGM阈值
    bgm_heavy_threshold: float = 0.15     # 强BGM阈值
    
    # === 质量评估阈值 ===
    retry_threshold_logprob: float = -0.8
    retry_threshold_no_speech: float = 0.6
    
    # === 熔断配置 ===
    circuit_breaker_enabled: bool = True
    consecutive_threshold: int = 3
    ratio_threshold: float = 0.2
    
    # === 质量预设（简化配置）===
    quality_preset: str = "balanced"  # fast/balanced/quality
```

#### 3.1.3 质量预设映射

```python
QUALITY_PRESETS = {
    "fast": {
        "weak_model": "htdemucs",
        "strong_model": "htdemucs_ft",
        "fallback_model": "mdx_extra_q",
        "shifts": 1,
        "overlap": 0.25,
    },
    "balanced": {
        "weak_model": "htdemucs_ft",
        "strong_model": "mdx_extra_q",
        "fallback_model": "mdx_extra",
        "shifts": 2,
        "overlap": 0.5,
    },
    "quality": {
        "weak_model": "mdx_extra_q",
        "strong_model": "mdx_extra",
        "fallback_model": "mdx_extra",
        "shifts": 3,
        "overlap": 0.5,
    },
}
```

### 3.2 服务层改造

#### 3.2.1 新增 SeparationStrategy（策略类）

```python
# demucs_service.py 新增

@dataclass
class SeparationStrategy:
    """分离策略决策结果"""
    should_separate: bool           # 是否需要分离
    initial_model: str              # 初始使用的模型
    fallback_model: Optional[str]   # 升级后的模型（如果允许升级）
    reason: str                     # 决策原因
    bgm_level: BGMLevel             # BGM级别
    allow_escalation: bool          # 是否允许升级
    
    def to_dict(self) -> dict:
        return {
            "should_separate": self.should_separate,
            "initial_model": self.initial_model,
            "fallback_model": self.fallback_model,
            "reason": self.reason,
            "bgm_level": self.bgm_level.value,
            "allow_escalation": self.allow_escalation,
        }
```

#### 3.2.2 新增 SeparationStrategyResolver（策略解析器）

```python
# demucs_service.py 新增

class SeparationStrategyResolver:
    """
    分离策略解析器
    
    根据 BGM 检测结果和用户配置，决定使用哪个模型以及升级路径
    """
    
    def __init__(self, settings: DemucsSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
    
    def resolve(self, bgm_level: BGMLevel) -> SeparationStrategy:
        """
        根据 BGM 级别解析分离策略
        
        策略矩阵：
        ┌───────────┬─────────────────┬─────────────────┬──────────────┐
        │ BGM Level │ mode=auto       │ mode=always     │ mode=never   │
        ├───────────┼─────────────────┼─────────────────┼──────────────┤
        │ NONE      │ 不分离          │ weak_model      │ 不分离        │
        │ LIGHT     │ weak_model      │ weak_model      │ 不分离        │
        │ HEAVY     │ strong_model    │ strong_model    │ 不分离        │
        └───────────┴─────────────────┴─────────────────┴──────────────┘
        """
        mode = self.settings.mode
        
        # mode=never: 始终不分离
        if mode == "never":
            return SeparationStrategy(
                should_separate=False,
                initial_model=None,
                fallback_model=None,
                reason="用户禁用人声分离",
                bgm_level=bgm_level,
                allow_escalation=False,
            )
        
        # mode=always: 始终分离（根据BGM级别选模型）
        if mode == "always":
            if bgm_level == BGMLevel.HEAVY:
                model = self.settings.strong_model
            else:
                model = self.settings.weak_model
            
            return SeparationStrategy(
                should_separate=True,
                initial_model=model,
                fallback_model=self.settings.fallback_model if self.settings.auto_escalation else None,
                reason=f"始终分离模式，BGM={bgm_level.value}",
                bgm_level=bgm_level,
                allow_escalation=self.settings.auto_escalation,
            )
        
        # mode=auto: 根据BGM级别决定
        if bgm_level == BGMLevel.NONE:
            return SeparationStrategy(
                should_separate=False,
                initial_model=None,
                fallback_model=None,
                reason="未检测到背景音乐",
                bgm_level=bgm_level,
                allow_escalation=False,
            )
        
        elif bgm_level == BGMLevel.LIGHT:
            return SeparationStrategy(
                should_separate=True,
                initial_model=self.settings.weak_model,
                fallback_model=self.settings.fallback_model if self.settings.auto_escalation else None,
                reason=f"检测到轻微BGM，使用 {self.settings.weak_model}",
                bgm_level=bgm_level,
                allow_escalation=self.settings.auto_escalation,
            )
        
        else:  # HEAVY
            return SeparationStrategy(
                should_separate=True,
                initial_model=self.settings.strong_model,
                fallback_model=self.settings.fallback_model if self.settings.auto_escalation else None,
                reason=f"检测到强BGM，使用 {self.settings.strong_model}",
                bgm_level=bgm_level,
                allow_escalation=self.settings.auto_escalation,
            )
    
    def get_escalation_model(self, current_model: str) -> Optional[str]:
        """
        获取升级后的模型
        
        升级路径：
        - htdemucs → htdemucs_ft → mdx_extra_q → mdx_extra
        - 如果当前已是最高级模型，返回 None
        """
        if not self.settings.auto_escalation:
            return None
        
        # 已经是 fallback 模型，无法再升级
        if current_model == self.settings.fallback_model:
            return None
        
        return self.settings.fallback_model
```

#### 3.2.3 扩展 CircuitBreakerState（熔断升级）

```python
# transcription_service.py 修改

@dataclass
class CircuitBreakerState:
    """熔断器状态（支持模型升级）"""
    consecutive_retries: int = 0
    total_retries: int = 0
    total_segments: int = 0
    processed_segments: int = 0
    
    # === 新增：升级跟踪 ===
    escalation_count: int = 0           # 已升级次数
    current_model: Optional[str] = None # 当前使用的模型
    escalation_history: List[str] = field(default_factory=list)  # 升级历史
    
    def record_escalation(self, new_model: str):
        """记录一次模型升级"""
        if self.current_model:
            self.escalation_history.append(f"{self.current_model} → {new_model}")
        self.current_model = new_model
        self.escalation_count += 1
        # 升级后重置连续重试计数
        self.consecutive_retries = 0
    
    def should_escalate(self, settings: DemucsSettings) -> bool:
        """
        判断是否应该升级模型（而非熔断）
        
        升级条件：
        1. 允许自动升级
        2. 未达到最大升级次数
        3. 满足熔断条件
        """
        if not settings.auto_escalation:
            return False
        
        if self.escalation_count >= settings.max_escalations:
            return False
        
        return self.should_break(settings)
    
    def should_break(self, settings: DemucsSettings) -> bool:
        """判断是否触发熔断（原有逻辑）"""
        if not settings.circuit_breaker_enabled:
            return False
        
        if self.consecutive_retries >= settings.consecutive_threshold:
            return True
        
        if self.processed_segments >= 5:
            retry_ratio = self.total_retries / self.processed_segments
            if retry_ratio >= settings.ratio_threshold:
                return True
        
        return False
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "consecutive_retries": self.consecutive_retries,
            "total_retries": self.total_retries,
            "processed_segments": self.processed_segments,
            "retry_ratio": self.total_retries / max(1, self.processed_segments),
            "escalation_count": self.escalation_count,
            "current_model": self.current_model,
            "escalation_history": self.escalation_history,
        }
```

### 3.3 DemucsService 改造

```python
# demucs_service.py 修改

class DemucsService:
    """Demucs人声分离服务（支持分级模型）"""
    
    _instance = None
    _model = None
    _model_name_loaded = None
    _model_lock = None
    _model_quality_cache: Dict[str, Dict] = {}  # 模型质量参数缓存
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = DemucsConfig()
        self._cache_dir = Path("models/demucs")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 分级模型配置
        self.tier_config = ModelTierConfig()
    
    def set_model_for_tier(self, bgm_level: BGMLevel, settings: DemucsSettings):
        """
        根据 BGM 级别设置对应的模型
        
        Args:
            bgm_level: 检测到的 BGM 级别
            settings: 用户配置
        """
        resolver = SeparationStrategyResolver(settings)
        strategy = resolver.resolve(bgm_level)
        
        if strategy.should_separate and strategy.initial_model:
            self.set_model(strategy.initial_model)
            self._apply_quality_settings(strategy.initial_model)
        
        return strategy
    
    def escalate_model(self, current_model: str, settings: DemucsSettings) -> Optional[str]:
        """
        升级到更好的模型
        
        Args:
            current_model: 当前模型名称
            settings: 用户配置
            
        Returns:
            新模型名称，如果无法升级则返回 None
        """
        resolver = SeparationStrategyResolver(settings)
        new_model = resolver.get_escalation_model(current_model)
        
        if new_model:
            self.logger.info(f"模型升级: {current_model} → {new_model}")
            self.set_model(new_model)
            self._apply_quality_settings(new_model)
        
        return new_model
    
    def _apply_quality_settings(self, model_name: str):
        """
        应用模型对应的质量参数
        """
        quality = self.tier_config.model_quality.get(model_name, {})
        if quality:
            self.config.shifts = quality.get("shifts", self.config.shifts)
            self.config.overlap = quality.get("overlap", self.config.overlap)
            self.logger.debug(f"应用质量参数: shifts={self.config.shifts}, overlap={self.config.overlap}")
    
    def separate_vocals_with_strategy(
        self,
        audio_path: str,
        strategy: SeparationStrategy,
        progress_callback: Optional[callable] = None
    ) -> Tuple[str, str]:
        """
        使用策略执行人声分离
        
        Returns:
            Tuple[output_path, used_model]: 输出路径和实际使用的模型
        """
        if not strategy.should_separate:
            return None, None
        
        self.set_model(strategy.initial_model)
        self._apply_quality_settings(strategy.initial_model)
        
        output_path = self.separate_vocals(audio_path, progress_callback=progress_callback)
        return output_path, strategy.initial_model
```

---

## 4. 运行流程

### 4.1 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            转录任务启动                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. 提取音频                                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. BGM 检测（分位数采样：15%, 50%, 85%）                                     │
│     ├── NONE (ratio < 0.02)                                                 │
│     ├── LIGHT (0.02 ≤ ratio < 0.15)                                         │
│     └── HEAVY (ratio ≥ 0.15)                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. 策略决策（SeparationStrategyResolver）                                   │
│     ├── NONE  → 不分离，直接转录                                             │
│     ├── LIGHT → 使用 weak_model (htdemucs_ft)                               │
│     └── HEAVY → 使用 strong_model (mdx_extra_q)                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────────────┐
        │ 不分离            │           │ 全局人声分离               │
        │ (BGM=NONE)        │           │ (使用决策的模型)           │
        └───────────────────┘           └───────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. 分段转录                                                                 │
│     for each segment:                                                       │
│       ├── 转录 → 评估质量                                                    │
│       ├── 质量OK → record_success()                                         │
│       └── 质量差 → record_retry()                                           │
│                    │                                                        │
│                    ├── should_escalate()? ──Yes──► 模型升级                  │
│                    │                              │                         │
│                    │                              ▼                         │
│                    │                    ┌─────────────────────┐             │
│                    │                    │ 升级到 fallback_model│             │
│                    │                    │ (mdx_extra)         │             │
│                    │                    └─────────────────────┘             │
│                    │                              │                         │
│                    │                              ▼                         │
│                    │                    ┌─────────────────────┐             │
│                    │                    │ 重新全局分离         │             │
│                    │                    │ 继续剩余段落         │             │
│                    │                    └─────────────────────┘             │
│                    │                                                        │
│                    └── should_break() && 已达max_escalations ──► 熔断异常    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. 批次对齐 + 时间微调 (start_delay=25ms, end_padding=25ms)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. 生成 SRT 文件                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 熔断升级决策流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     质量检测失败                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ record_retry()  │
                    └─────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │ auto_escalation = True?       │
              └───────────────────────────────┘
                     │              │
                    Yes             No
                     │              │
                     ▼              ▼
         ┌───────────────────┐  ┌─────────────────────┐
         │ escalation_count  │  │ should_break()?     │
         │ < max_escalations?│  │                     │
         └───────────────────┘  └─────────────────────┘
              │          │              │         │
             Yes         No            Yes        No
              │          │              │         │
              ▼          ▼              ▼         ▼
    ┌─────────────┐ ┌─────────────┐ ┌───────┐ ┌──────┐
    │ should_     │ │ should_     │ │ 熔断  │ │ 继续 │
    │ escalate()? │ │ break()?    │ │ 异常  │ │      │
    └─────────────┘ └─────────────┘ └───────┘ └──────┘
         │                │
        Yes              Yes
         │                │
         ▼                ▼
    ┌─────────────┐  ┌─────────────┐
    │ 模型升级    │  │ 熔断异常    │
    │ + 重新分离  │  │             │
    └─────────────┘  └─────────────┘
```

### 4.3 熔断异常处理策略

熔断异常（`BreakToGlobalSeparation`）触发后，系统有多种处理策略可选：

#### 4.3.1 处理策略选项

| 策略 | 说明 | 适用场景 | 用户配置 |
|------|------|----------|----------|
| **继续处理** | 标记问题段落，继续转录剩余内容 | 大部分正常，少数有问题 | `on_break: "continue"` |
| **降级完成** | 使用原始音频完成剩余转录 | 分离后反而更差 | `on_break: "fallback_original"` |
| **任务失败** | 终止任务，返回错误 | 严格质量要求 | `on_break: "fail"` |
| **人工介入** | 暂停任务，等待用户决定 | 需要人工判断 | `on_break: "pause"` |

#### 4.3.2 推荐策略：继续处理 + 标记问题段落

```python
class CircuitBreakAction(Enum):
    """熔断后的处理动作"""
    CONTINUE = "continue"           # 继续处理，标记问题段落
    FALLBACK_ORIGINAL = "fallback"  # 降级使用原始音频
    FAIL = "fail"                   # 任务失败
    PAUSE = "pause"                 # 暂停等待人工介入
```

#### 4.3.3 详细处理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          熔断异常触发                                        │
│  (已达 max_escalations 且仍满足熔断条件)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 检查 on_break 配置            │
                    └───────────────────────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬───────────────┐
        │               │           │           │               │
        ▼               ▼           ▼           ▼               │
   ┌─────────┐    ┌─────────┐  ┌─────────┐  ┌─────────┐        │
   │CONTINUE │    │FALLBACK │  │  FAIL   │  │  PAUSE  │        │
   └─────────┘    └─────────┘  └─────────┘  └─────────┘        │
        │               │           │           │               │
        ▼               ▼           ▼           ▼               │
┌─────────────┐ ┌─────────────┐ ┌───────┐ ┌─────────────┐      │
│ 1.记录问题  │ │ 1.卸载分离  │ │ 抛出  │ │ 1.保存状态  │      │
│   段落索引  │ │   后的音频  │ │ 异常  │ │ 2.推送SSE   │      │
│ 2.继续转录  │ │ 2.切回原始  │ │ 终止  │ │   等待事件  │      │
│   剩余段落  │ │   音频      │ │ 任务  │ │ 3.等待用户  │      │
│ 3.最终汇总  │ │ 3.继续转录  │ │       │ │   操作      │      │
│   问题报告  │ │ 4.标记为    │ │       │ │             │      │
│             │ │   降级处理  │ │       │ │             │      │
└─────────────┘ └─────────────┘ └───────┘ └─────────────┘      │
        │               │                       │               │
        └───────────────┴───────────────────────┘               │
                                    │                           │
                                    ▼                           │
                    ┌───────────────────────────────┐           │
                    │ 推送 circuit_breaker_handled  │◄──────────┘
                    │ SSE 事件                      │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ 继续后续流程（对齐、生成SRT） │
                    └───────────────────────────────┘
```

#### 4.3.4 代码实现示例

```python
# transcription_service.py

class CircuitBreakHandler:
    """熔断异常处理器"""
    
    def __init__(self, job: JobState, settings: DemucsSettings):
        self.job = job
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.problem_segments: List[int] = []  # 记录问题段落索引
    
    def handle(
        self,
        breaker_state: CircuitBreakerState,
        current_segment_idx: int,
        remaining_segments: List[Dict],
        audio_source: Any,
        original_audio: Any
    ) -> Tuple[str, Any]:
        """
        处理熔断异常
        
        Args:
            breaker_state: 熔断器状态
            current_segment_idx: 当前段落索引
            remaining_segments: 剩余待处理段落
            audio_source: 当前使用的音频（可能是分离后的）
            original_audio: 原始音频
            
        Returns:
            Tuple[action, audio_to_use]: 处理动作和后续使用的音频
        """
        action = self.settings.on_break  # 用户配置的处理策略
        
        # 记录问题段落
        self.problem_segments.append(current_segment_idx)
        
        # 推送 SSE 事件
        self._push_circuit_break_event(breaker_state, action)
        
        if action == CircuitBreakAction.FAIL:
            raise TranscriptionError(
                f"熔断触发，任务终止。问题段落: {self.problem_segments}",
                error_code="CIRCUIT_BREAK_FAIL"
            )
        
        elif action == CircuitBreakAction.PAUSE:
            # 保存断点，等待人工介入
            self._save_checkpoint(breaker_state, current_segment_idx)
            raise PauseForUserIntervention(
                f"熔断触发，等待人工介入。问题段落: {self.problem_segments}"
            )
        
        elif action == CircuitBreakAction.FALLBACK_ORIGINAL:
            self.logger.warning(
                f"熔断触发，降级使用原始音频继续处理。"
                f"问题段落: {self.problem_segments}"
            )
            return action, original_audio
        
        else:  # CONTINUE (默认)
            self.logger.warning(
                f"熔断触发，标记问题段落并继续处理。"
                f"问题段落: {self.problem_segments}"
            )
            return action, audio_source
    
    def get_problem_report(self) -> Dict:
        """获取问题报告"""
        return {
            "total_problem_segments": len(self.problem_segments),
            "problem_indices": self.problem_segments,
            "suggestion": self._get_suggestion()
        }
    
    def _get_suggestion(self) -> str:
        """根据问题段落数量给出建议"""
        count = len(self.problem_segments)
        if count == 0:
            return "所有段落处理正常"
        elif count <= 3:
            return "少量段落可能需要手动调整时间轴"
        elif count <= 10:
            return "建议检查这些段落的字幕准确性"
        else:
            return "大量段落有问题，建议使用更高质量的模型重新处理"
    
    def _push_circuit_break_event(
        self, 
        state: CircuitBreakerState, 
        action: CircuitBreakAction
    ):
        """推送熔断处理事件"""
        from services.sse_service import push_sse_event
        
        push_sse_event(
            self.job.job_id,
            "circuit_breaker_handled",
            {
                "action": action.value,
                "problem_segments": self.problem_segments,
                "stats": state.get_stats(),
                "suggestion": self._get_suggestion()
            }
        )
```

#### 4.3.5 集成到转录主流程

```python
# transcription_service.py - _transcribe_segments 方法修改

async def _transcribe_segments(self, job, segments, audio_source, ...):
    """分段转录（集成熔断处理）"""
    
    breaker_state = CircuitBreakerState(current_model=strategy.initial_model)
    break_handler = CircuitBreakHandler(job, job.settings.demucs)
    
    for idx, segment in enumerate(segments):
        try:
            # 转录单个段落
            result = self._transcribe_single_segment(segment, audio_source)
            
            # 评估质量
            if self._is_quality_acceptable(result):
                breaker_state.record_success()
                results.append(result)
            else:
                breaker_state.record_retry()
                
                # 检查是否需要升级或熔断
                if breaker_state.should_escalate(job.settings.demucs):
                    # 模型升级
                    new_model = self._escalate_model(breaker_state, job.settings.demucs)
                    if new_model:
                        # 重新分离并继续
                        audio_source = self._re_separate_with_model(new_model)
                        breaker_state.record_escalation(new_model)
                    else:
                        # 无法升级，触发熔断处理
                        action, audio_source = break_handler.handle(
                            breaker_state, idx, segments[idx:], 
                            audio_source, original_audio
                        )
                
                elif breaker_state.should_break(job.settings.demucs):
                    # 已达升级上限，触发熔断处理
                    action, audio_source = break_handler.handle(
                        breaker_state, idx, segments[idx:],
                        audio_source, original_audio
                    )
        
        except PauseForUserIntervention:
            # 暂停任务，保存状态
            job.status = "paused"
            job.paused = True
            return partial_results
    
    # 转录完成，附加问题报告
    return results, break_handler.get_problem_report()
```

#### 4.3.6 用户配置扩展

```python
# job_models.py - DemucsSettings 扩展

@dataclass
class DemucsSettings:
    # ... 现有字段 ...
    
    # === 熔断处理配置 ===
    on_break: str = "continue"  # continue/fallback/fail/pause
    
    # 问题段落标记方式
    mark_problem_segments: bool = True      # 是否在结果中标记问题段落
    problem_segment_suffix: str = "[?]"     # 问题段落的标记后缀
```

#### 4.3.7 前端展示问题段落

转录完成后，问题段落会在编辑器中特殊标记：

```
┌─────────────────────────────────────────────────────────────────┐
│  字幕编辑器                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1  00:00:01,000 --> 00:00:03,500                              │
│     这是正常的字幕内容                                          │
│                                                                 │
│  2  00:00:04,000 --> 00:00:06,500  ⚠️ [需要检查]               │
│     这是可能有问题的字幕 [?]                                    │
│     └─ 提示: 此段落在转录时触发了质量警告                       │
│                                                                 │
│  3  00:00:07,000 --> 00:00:09,500                              │
│     这是正常的字幕内容                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.3.8 SSE 事件：circuit_breaker_handled

```json
{
  "event": "circuit_breaker_handled",
  "data": {
    "action": "continue",
    "problem_segments": [12, 13, 14],
    "stats": {
      "consecutive_retries": 3,
      "total_retries": 8,
      "processed_segments": 25,
      "retry_ratio": 0.32,
      "escalation_count": 1,
      "escalation_history": ["htdemucs_ft → mdx_extra"]
    },
    "suggestion": "少量段落可能需要手动调整时间轴"
  }
}
```

---

## 5. API 接口设计

### 5.1 启动任务接口（扩展）

**POST /api/start**

```json
{
  "job_id": "xxx",
  "settings": {
    "model": "medium",
    "compute_type": "float16",
    "device": "cuda",
    "batch_size": 16,
    "demucs": {
      "enabled": true,
      "mode": "auto",
      
      "weak_model": "htdemucs_ft",
      "strong_model": "mdx_extra_q",
      "fallback_model": "mdx_extra",
      "auto_escalation": true,
      "max_escalations": 1,
      
      "bgm_light_threshold": 0.02,
      "bgm_heavy_threshold": 0.15,
      
      "retry_threshold_logprob": -0.8,
      "retry_threshold_no_speech": 0.6,
      "circuit_breaker_enabled": true,
      "consecutive_threshold": 3,
      "ratio_threshold": 0.2,
      
      "quality_preset": "balanced"
    }
  }
}
```

### 5.2 SSE 事件扩展

#### 5.2.1 bgm_detected（现有，保持不变）

```json
{
  "event": "bgm_detected",
  "data": {
    "level": "light",
    "ratios": [0.03, 0.05, 0.04],
    "max_ratio": 0.05,
    "thresholds": {
      "light": 0.02,
      "heavy": 0.15
    }
  }
}
```

#### 5.2.2 separation_strategy（新增）

```json
{
  "event": "separation_strategy",
  "data": {
    "should_separate": true,
    "initial_model": "htdemucs_ft",
    "fallback_model": "mdx_extra",
    "reason": "检测到轻微BGM，使用 htdemucs_ft",
    "bgm_level": "light",
    "allow_escalation": true
  }
}
```

#### 5.2.3 model_escalated（新增）

```json
{
  "event": "model_escalated",
  "data": {
    "from_model": "htdemucs_ft",
    "to_model": "mdx_extra",
    "reason": "连续3个segment重试，触发模型升级",
    "escalation_count": 1,
    "max_escalations": 1,
    "stats": {
      "consecutive_retries": 3,
      "total_retries": 5,
      "processed_segments": 12,
      "retry_ratio": 0.417
    }
  }
}
```

#### 5.2.4 circuit_breaker_triggered（现有，扩展）

```json
{
  "event": "circuit_breaker_triggered",
  "data": {
    "triggered": true,
    "reason": "已达最大升级次数，触发熔断",
    "action": "停止处理",
    "stats": {
      "consecutive_retries": 3,
      "total_retries": 8,
      "processed_segments": 20,
      "retry_ratio": 0.4,
      "escalation_count": 1,
      "escalation_history": ["htdemucs_ft → mdx_extra"]
    }
  }
}
```

---

## 6. 配置文件设计

### 6.1 默认配置（config/demucs_tiers.json）

```json
{
  "version": "1.0",
  "description": "Demucs分级模型配置",
  
  "presets": {
    "fast": {
      "description": "速度优先，适合低配机器",
      "weak_model": "htdemucs",
      "strong_model": "htdemucs_ft",
      "fallback_model": "mdx_extra_q",
      "quality": {
        "shifts": 1,
        "overlap": 0.25
      }
    },
    "balanced": {
      "description": "平衡模式（默认推荐）",
      "weak_model": "htdemucs_ft",
      "strong_model": "mdx_extra_q",
      "fallback_model": "mdx_extra",
      "quality": {
        "shifts": 2,
        "overlap": 0.5
      }
    },
    "quality": {
      "description": "质量优先，处理时间较长",
      "weak_model": "mdx_extra_q",
      "strong_model": "mdx_extra",
      "fallback_model": "mdx_extra",
      "quality": {
        "shifts": 3,
        "overlap": 0.5
      }
    }
  },
  
  "models": {
    "htdemucs": {
      "description": "Hybrid Transformer，快速模式",
      "size_mb": 80,
      "quality_score": 3,
      "speed_score": 5,
      "recommended_for": ["fast", "low_vram"]
    },
    "htdemucs_ft": {
      "description": "Fine-tuned版本，人声优化",
      "size_mb": 80,
      "quality_score": 4,
      "speed_score": 4,
      "recommended_for": ["balanced", "vocals"]
    },
    "mdx_extra_q": {
      "description": "MDX-Net量化版",
      "size_mb": 25,
      "quality_score": 4,
      "speed_score": 4,
      "recommended_for": ["balanced", "low_vram"]
    },
    "mdx_extra": {
      "description": "MDX-Net完整版，最高质量",
      "size_mb": 600,
      "quality_score": 5,
      "speed_score": 2,
      "recommended_for": ["quality", "heavy_bgm"]
    }
  },
  
  "bgm_thresholds": {
    "light": 0.02,
    "heavy": 0.15
  },
  
  "circuit_breaker": {
    "consecutive_threshold": 3,
    "ratio_threshold": 0.2,
    "max_escalations": 1
  }
}
```

### 6.2 用户配置覆盖（user_config.json）

```json
{
  "demucs": {
    "quality_preset": "balanced",
    "auto_escalation": true,
    "bgm_light_threshold": 0.02,
    "bgm_heavy_threshold": 0.15
  }
}
```

---

## 7. 前端配置界面设计

### 7.1 简化模式（推荐）

```
┌─────────────────────────────────────────────────────────────────┐
│  人声分离设置                                              [✓] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  工作模式：  ○ 自动检测(推荐)  ○ 始终分离  ○ 禁用             │
│                                                                 │
│  质量预设：  ○ 快速  ● 平衡(推荐)  ○ 高质量                   │
│                                                                 │
│  [✓] 质量不足时自动升级模型                                    │
│                                                                 │
│  [展开高级选项 ▼]                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 高级模式

```
┌─────────────────────────────────────────────────────────────────┐
│  人声分离高级设置                                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  分级模型配置：                                                 │
│    弱BGM模型:    [htdemucs_ft    ▼]                            │
│    强BGM模型:    [mdx_extra_q    ▼]                            │
│    兜底模型:     [mdx_extra      ▼]                            │
│                                                                 │
│  BGM检测阈值：                                                  │
│    轻微BGM:     [====●==========] 0.02                         │
│    强BGM:       [=======●=======] 0.15                         │
│                                                                 │
│  熔断配置：                                                     │
│    [✓] 启用熔断机制                                            │
│    连续重试阈值:  [3]                                          │
│    重试比例阈值:  [====●==========] 20%                        │
│    最大升级次数:  [1]                                          │
│                                                                 │
│  质量参数：                                                     │
│    增强次数(shifts): [2]   (1=快速, 2=平衡, 5=最高)            │
│    重叠率(overlap):  [0.5] (0.25=快速, 0.5=高质量)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 进度显示扩展

```
┌─────────────────────────────────────────────────────────────────┐
│  任务进度                                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [■■■■■■■■■■■■■■■■■■□□] 75%                                    │
│                                                                 │
│  当前阶段: 转录中 (45/60 段)                                   │
│                                                                 │
│  人声分离状态:                                                  │
│    ├─ BGM检测: LIGHT (max_ratio=0.05)                          │
│    ├─ 当前模型: htdemucs_ft                                    │
│    └─ 升级历史: (无)                                           │
│                                                                 │
│  [!] 注意: 第23-25段质量较低，可能需要手动调整                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 代码改动清单

### 8.1 需要修改的文件

| 文件 | 改动类型 | 改动说明 |
|------|----------|----------|
| `models/job_models.py` | 扩展 | DemucsSettings 新增分级配置字段 |
| `services/demucs_service.py` | 扩展 | 新增 ModelTierConfig, SeparationStrategy, SeparationStrategyResolver |
| `services/transcription_service.py` | 修改 | CircuitBreakerState 支持升级，集成策略解析器 |
| `api/routes.py` | 扩展 | 解析新的配置字段 |

### 8.2 需要新增的文件

| 文件 | 说明 |
|------|------|
| `config/demucs_tiers.json` | 分级模型默认配置 |
| `services/separation_strategy.py` | （可选）策略解析器独立模块 |

### 8.3 改动优先级

1. **Phase 1（核心功能）**
   - DemucsSettings 扩展
   - SeparationStrategyResolver 实现
   - CircuitBreakerState 升级支持

2. **Phase 2（集成测试）**
   - TranscriptionService 集成
   - SSE 事件推送

3. **Phase 3（用户体验）**
   - 配置文件支持
   - 前端界面

---

## 9. 测试用例

### 9.1 策略解析测试

```python
def test_strategy_resolver():
    settings = DemucsSettings(
        mode="auto",
        weak_model="htdemucs_ft",
        strong_model="mdx_extra_q",
        fallback_model="mdx_extra",
        auto_escalation=True,
    )
    resolver = SeparationStrategyResolver(settings)
    
    # NONE → 不分离
    strategy = resolver.resolve(BGMLevel.NONE)
    assert strategy.should_separate == False
    
    # LIGHT → weak_model
    strategy = resolver.resolve(BGMLevel.LIGHT)
    assert strategy.should_separate == True
    assert strategy.initial_model == "htdemucs_ft"
    assert strategy.fallback_model == "mdx_extra"
    
    # HEAVY → strong_model
    strategy = resolver.resolve(BGMLevel.HEAVY)
    assert strategy.should_separate == True
    assert strategy.initial_model == "mdx_extra_q"
```

### 9.2 熔断升级测试

```python
def test_circuit_breaker_escalation():
    settings = DemucsSettings(
        auto_escalation=True,
        max_escalations=1,
        consecutive_threshold=3,
    )
    state = CircuitBreakerState(current_model="htdemucs_ft")
    
    # 连续3次重试
    for _ in range(3):
        state.record_retry()
    
    # 应该升级而非熔断
    assert state.should_escalate(settings) == True
    
    # 升级后
    state.record_escalation("mdx_extra")
    assert state.escalation_count == 1
    assert state.consecutive_retries == 0  # 重置
    
    # 再次触发，已达上限，应该熔断
    for _ in range(3):
        state.record_retry()
    
    assert state.should_escalate(settings) == False
    assert state.should_break(settings) == True
```

### 9.3 端到端测试场景

| 场景 | 输入 | 期望行为 |
|------|------|----------|
| 纯人声 | 播客录音 | BGM=NONE，不分离 |
| 轻BGM | 轻音乐访谈 | BGM=LIGHT，用htdemucs_ft |
| 强BGM | 音乐MV | BGM=HEAVY，用mdx_extra_q |
| 质量差 | 嘈杂环境 | 触发升级到mdx_extra |
| 升级后仍差 | 极端噪音 | 达到max_escalations后熔断 |

---

## 10. 性能预估

### 10.1 模型处理时间（10分钟音频，RTX 3080）

| 模型 | shifts=1 | shifts=2 | shifts=5 |
|------|----------|----------|----------|
| htdemucs | ~30s | ~50s | ~120s |
| htdemucs_ft | ~30s | ~50s | ~120s |
| mdx_extra_q | ~40s | ~70s | ~150s |
| mdx_extra | ~90s | ~150s | ~350s |

### 10.2 典型场景耗时估算

| 场景 | 分离耗时 | 总耗时增加 |
|------|----------|------------|
| BGM=NONE | 0s | +30s（仅检测） |
| BGM=LIGHT, 无升级 | ~50s | +80s |
| BGM=LIGHT, 升级1次 | ~50s + ~150s | +230s |
| BGM=HEAVY | ~70s | +100s |

---

## 11. 后续优化方向

1. **并行分离**：在转录同时异步执行分离
2. **增量分离**：检测到质量问题时只分离有问题的段落
3. **模型预热**：启动时预加载常用模型
4. **质量学习**：根据历史数据自动调整阈值
5. **GPU显存优化**：支持多模型切换时的显存管理

---

## 附录 A：模型对比

| 模型 | 大小 | 质量 | 速度 | 适用场景 |
|------|------|------|------|----------|
| htdemucs | 80MB | ★★★☆☆ | ★★★★★ | 快速预览 |
| htdemucs_ft | 80MB | ★★★★☆ | ★★★★☆ | 弱BGM场景 |
| mdx_extra_q | 25MB | ★★★★☆ | ★★★★☆ | 平衡选择 |
| mdx_extra | 600MB | ★★★★★ | ★★☆☆☆ | 强BGM场景 |

## 附录 B：错误码

| 错误码 | 说明 | 处理建议 |
|--------|------|----------|
| DEMUCS_MODEL_LOAD_FAILED | 模型加载失败 | 检查模型文件是否完整 |
| DEMUCS_OOM | 显存不足 | 降级使用量化模型 |
| DEMUCS_ESCALATION_LIMIT | 达到升级上限 | 检查音频质量 |
| DEMUCS_CIRCUIT_BREAK | 熔断触发 | 手动检查问题段落 |
