# Demucs 分级人声分离系统 - 分段重构计划

## 版本信息

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2025-12-03 | 初始版本，基于设计文档与现有代码分析 |

---

## 1. 现状分析

### 1.1 当前代码结构

```
backend/app/
├── models/
│   └── job_models.py          # DemucsSettings（基础配置）
├── services/
│   ├── demucs_service.py      # DemucsService, DemucsConfig, BGMLevel
│   └── transcription_service.py  # CircuitBreakerState（基础熔断）
```

### 1.2 现有实现 vs 目标设计对比

| 功能点 | 当前状态 | 目标状态 | 差距 |
|--------|----------|----------|------|
| BGM 检测 | 已实现（分位数采样） | 保持 | 无 |
| BGM 级别分类 | NONE/LIGHT/HEAVY | 保持 | 无 |
| 模型选择 | 单一模型（mdx_extra） | 分级模型（weak/strong/fallback） | **需重构** |
| 熔断机制 | 基础熔断（切换全局分离） | 熔断+模型升级 | **需扩展** |
| 用户配置 | 7个字段 | 15+字段（含预设） | **需扩展** |
| 质量预设 | 无 | fast/balanced/quality | **需新增** |
| SSE 事件 | bgm_detected, circuit_breaker | +separation_strategy, model_escalated | **需扩展** |
| 前端配置 | 无 | 简化/高级模式 | **需预留接口** |

### 1.3 现有代码关键类

#### DemucsSettings（job_models.py:14-22）
```python
@dataclass
class DemucsSettings:
    enabled: bool = True
    mode: str = "auto"                          # auto/always/never/on_demand
    retry_threshold_logprob: float = -0.8
    retry_threshold_no_speech: float = 0.6
    circuit_breaker_enabled: bool = True
    consecutive_threshold: int = 3
    ratio_threshold: float = 0.2
```

#### DemucsConfig（demucs_service.py:36-60）
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

#### CircuitBreakerState（transcription_service.py:72-131）
```python
@dataclass
class CircuitBreakerState:
    consecutive_retries: int = 0
    total_retries: int = 0
    total_segments: int = 0
    processed_segments: int = 0
    # 方法: record_retry(), record_success(), should_break()
```

---

## 2. 重构阶段划分

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           重构阶段总览                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: 数据模型扩展                                                       │
│  ├─ 扩展 DemucsSettings（用户配置）                                          │
│  ├─ 新增 ModelTierConfig（分级模型配置）                                      │
│  └─ 新增 SeparationStrategy（策略决策结果）                                   │
│                                                                             │
│  Phase 2: 策略解析器                                                         │
│  ├─ 新增 SeparationStrategyResolver                                         │
│  ├─ 新增 QUALITY_PRESETS 质量预设                                            │
│  └─ 集成到 DemucsService                                                     │
│                                                                             │
│  Phase 3: 熔断升级机制                                                       │
│  ├─ 扩展 CircuitBreakerState（升级跟踪）                                      │
│  ├─ 新增 CircuitBreakHandler（熔断处理器）                                    │
│  └─ 集成到 TranscriptionService                                              │
│                                                                             │
│  Phase 4: SSE 事件扩展                                                       │
│  ├─ 新增 separation_strategy 事件                                            │
│  ├─ 新增 model_escalated 事件                                                │
│  └─ 扩展 circuit_breaker_triggered 事件                                      │
│                                                                             │
│  Phase 5: 配置文件与接口                                                     │
│  ├─ 新增 config/demucs_tiers.json                                            │
│  ├─ 扩展 API 接口                                                            │
│  └─ 预留前端配置接口                                                          │
│                                                                             │
│  Phase 6: 文档更新                                                           │
│  └─ 更新 llmdoc 文档系统                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: 数据模型扩展

### 3.1 目标
扩展数据模型以支持分级策略，同时保持向后兼容。

### 3.2 改动文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `models/job_models.py` | 扩展 | DemucsSettings 新增字段 |
| `services/demucs_service.py` | 新增 | ModelTierConfig, SeparationStrategy |

### 3.3 DemucsSettings 扩展

**修改位置**: `models/job_models.py:14-22`

```python
@dataclass
class DemucsSettings:
    """Demucs人声分离配置（用户可配置）"""
    # === 基础开关（保持现有） ===
    enabled: bool = True
    mode: str = "auto"  # auto/always/never/on_demand

    # === 新增：分级模型配置 ===
    weak_model: str = "htdemucs_ft"       # 弱BGM使用的模型
    strong_model: str = "mdx_extra_q"     # 强BGM使用的模型
    fallback_model: str = "mdx_extra"     # 兜底模型（升级后使用）
    auto_escalation: bool = True          # 是否允许自动升级模型
    max_escalations: int = 1              # 最大升级次数

    # === 新增：BGM检测阈值（从DemucsConfig移入，用户可配置） ===
    bgm_light_threshold: float = 0.02     # 轻微BGM阈值
    bgm_heavy_threshold: float = 0.15     # 强BGM阈值

    # === 质量评估阈值（保持现有） ===
    retry_threshold_logprob: float = -0.8
    retry_threshold_no_speech: float = 0.6

    # === 熔断配置（保持现有） ===
    circuit_breaker_enabled: bool = True
    consecutive_threshold: int = 3
    ratio_threshold: float = 0.2

    # === 新增：熔断处理策略 ===
    on_break: str = "continue"  # continue/fallback/fail/pause
    mark_problem_segments: bool = True
    problem_segment_suffix: str = "[?]"

    # === 新增：质量预设（简化配置入口） ===
    quality_preset: str = "balanced"  # fast/balanced/quality
```

**向后兼容说明**:
- 所有新增字段都有默认值
- 现有 `from_meta_dict` 方法需要更新以支持新字段（缺失时使用默认值）
- API 接口无需强制传递新字段

### 3.4 ModelTierConfig 新增

**新增位置**: `services/demucs_service.py`（在 DemucsConfig 之后）

```python
@dataclass
class ModelTierConfig:
    """
    分级模型配置
    定义不同场景下使用的模型及其质量参数
    """
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

### 3.5 SeparationStrategy 新增

**新增位置**: `services/demucs_service.py`（在 ModelTierConfig 之后）

```python
@dataclass
class SeparationStrategy:
    """
    分离策略决策结果
    由 SeparationStrategyResolver 生成，描述本次任务应采用的分离策略
    """
    should_separate: bool           # 是否需要分离
    initial_model: Optional[str]    # 初始使用的模型
    fallback_model: Optional[str]   # 升级后的模型（如果允许升级）
    reason: str                     # 决策原因（用于日志和SSE）
    bgm_level: BGMLevel             # 检测到的BGM级别
    allow_escalation: bool          # 是否允许升级

    def to_dict(self) -> dict:
        """转换为字典（用于SSE推送）"""
        return {
            "should_separate": self.should_separate,
            "initial_model": self.initial_model,
            "fallback_model": self.fallback_model,
            "reason": self.reason,
            "bgm_level": self.bgm_level.value,
            "allow_escalation": self.allow_escalation,
        }
```

### 3.6 质量预设常量

**新增位置**: `services/demucs_service.py`（在类定义之前）

```python
# 质量预设映射
QUALITY_PRESETS = {
    "fast": {
        "weak_model": "htdemucs",
        "strong_model": "htdemucs_ft",
        "fallback_model": "mdx_extra_q",
        "description": "速度优先，适合低配机器",
    },
    "balanced": {
        "weak_model": "htdemucs_ft",
        "strong_model": "mdx_extra_q",
        "fallback_model": "mdx_extra",
        "description": "平衡模式（默认推荐）",
    },
    "quality": {
        "weak_model": "mdx_extra_q",
        "strong_model": "mdx_extra",
        "fallback_model": "mdx_extra",
        "description": "质量优先，处理时间较长",
    },
}
```

---

## 4. Phase 2: 策略解析器

### 4.1 目标
实现策略解析器，根据 BGM 检测结果和用户配置智能选择模型。

### 4.2 改动文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `services/demucs_service.py` | 新增 | SeparationStrategyResolver 类 |
| `services/demucs_service.py` | 修改 | DemucsService 集成策略解析 |

### 4.3 SeparationStrategyResolver 新增

**新增位置**: `services/demucs_service.py`

```python
class SeparationStrategyResolver:
    """
    分离策略解析器

    根据 BGM 检测结果和用户配置，决定使用哪个模型以及升级路径

    策略矩阵：
    ┌───────────┬─────────────────┬─────────────────┬──────────────┐
    │ BGM Level │ mode=auto       │ mode=always     │ mode=never   │
    ├───────────┼─────────────────┼─────────────────┼──────────────┤
    │ NONE      │ 不分离          │ weak_model      │ 不分离        │
    │ LIGHT     │ weak_model      │ weak_model      │ 不分离        │
    │ HEAVY     │ strong_model    │ strong_model    │ 不分离        │
    └───────────┴─────────────────┴─────────────────┴──────────────┘
    """

    def __init__(self, settings: DemucsSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        # 如果使用质量预设，覆盖模型配置
        self._apply_preset()

    def _apply_preset(self):
        """应用质量预设"""
        preset = QUALITY_PRESETS.get(self.settings.quality_preset)
        if preset:
            # 仅当用户未自定义时才应用预设
            if self.settings.weak_model == "htdemucs_ft":  # 默认值检查
                self.settings.weak_model = preset["weak_model"]
                self.settings.strong_model = preset["strong_model"]
                self.settings.fallback_model = preset["fallback_model"]

    def resolve(self, bgm_level: BGMLevel) -> SeparationStrategy:
        """
        根据 BGM 级别解析分离策略

        Args:
            bgm_level: 检测到的 BGM 级别

        Returns:
            SeparationStrategy: 分离策略决策结果
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
            model = self.settings.strong_model if bgm_level == BGMLevel.HEAVY else self.settings.weak_model
            return SeparationStrategy(
                should_separate=True,
                initial_model=model,
                fallback_model=self.settings.fallback_model if self.settings.auto_escalation else None,
                reason=f"始终分离模式，BGM={bgm_level.value}，使用 {model}",
                bgm_level=bgm_level,
                allow_escalation=self.settings.auto_escalation,
            )

        # mode=auto（默认）: 根据BGM级别决定
        if bgm_level == BGMLevel.NONE:
            return SeparationStrategy(
                should_separate=False,
                initial_model=None,
                fallback_model=None,
                reason="未检测到背景音乐，跳过人声分离",
                bgm_level=bgm_level,
                allow_escalation=False,
            )

        elif bgm_level == BGMLevel.LIGHT:
            return SeparationStrategy(
                should_separate=True,
                initial_model=self.settings.weak_model,
                fallback_model=self.settings.fallback_model if self.settings.auto_escalation else None,
                reason=f"检测到轻微BGM，使用轻量模型 {self.settings.weak_model}",
                bgm_level=bgm_level,
                allow_escalation=self.settings.auto_escalation,
            )

        else:  # HEAVY
            return SeparationStrategy(
                should_separate=True,
                initial_model=self.settings.strong_model,
                fallback_model=self.settings.fallback_model if self.settings.auto_escalation else None,
                reason=f"检测到强BGM，使用高质量模型 {self.settings.strong_model}",
                bgm_level=bgm_level,
                allow_escalation=self.settings.auto_escalation,
            )

    def get_escalation_model(self, current_model: str) -> Optional[str]:
        """
        获取升级后的模型

        Args:
            current_model: 当前模型名称

        Returns:
            新模型名称，如果无法升级则返回 None
        """
        if not self.settings.auto_escalation:
            return None

        # 已经是 fallback 模型，无法再升级
        if current_model == self.settings.fallback_model:
            return None

        return self.settings.fallback_model
```

### 4.4 DemucsService 扩展

**修改位置**: `services/demucs_service.py` DemucsService 类

```python
class DemucsService:
    """Demucs人声分离服务（支持分级模型）"""

    # ... 保持现有属性和方法 ...

    def __init__(self):
        # ... 保持现有初始化 ...

        # 新增：分级模型配置
        self.tier_config = ModelTierConfig()

    # 新增方法
    def resolve_strategy(
        self,
        bgm_level: BGMLevel,
        settings: DemucsSettings
    ) -> SeparationStrategy:
        """
        根据 BGM 级别解析分离策略

        Args:
            bgm_level: 检测到的 BGM 级别
            settings: 用户配置

        Returns:
            SeparationStrategy: 分离策略
        """
        resolver = SeparationStrategyResolver(settings)
        return resolver.resolve(bgm_level)

    def set_model_for_strategy(self, strategy: SeparationStrategy):
        """
        根据策略设置模型

        Args:
            strategy: 分离策略
        """
        if strategy.should_separate and strategy.initial_model:
            self.set_model(strategy.initial_model)
            self._apply_quality_settings(strategy.initial_model)

    def escalate_model(
        self,
        current_model: str,
        settings: DemucsSettings
    ) -> Optional[str]:
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
            self.logger.info(f"模型升级: {current_model} -> {new_model}")
            self.set_model(new_model)
            self._apply_quality_settings(new_model)

        return new_model

    def _apply_quality_settings(self, model_name: str):
        """
        应用模型对应的质量参数

        Args:
            model_name: 模型名称
        """
        quality = self.tier_config.model_quality.get(model_name, {})
        if quality:
            self.config.shifts = quality.get("shifts", self.config.shifts)
            self.config.overlap = quality.get("overlap", self.config.overlap)
            self.logger.debug(
                f"应用质量参数: model={model_name}, "
                f"shifts={self.config.shifts}, overlap={self.config.overlap}"
            )
```

---

## 5. Phase 3: 熔断升级机制

### 5.1 目标
扩展熔断机制，支持在触发熔断时先升级模型，而非直接切换全局分离。

### 5.2 改动文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `services/transcription_service.py` | 扩展 | CircuitBreakerState 支持升级跟踪 |
| `services/transcription_service.py` | 新增 | CircuitBreakHandler 熔断处理器 |
| `services/transcription_service.py` | 修改 | 转录主流程集成 |

### 5.3 CircuitBreakerState 扩展

**修改位置**: `services/transcription_service.py:72-131`

```python
@dataclass
class CircuitBreakerState:
    """
    熔断器状态（支持模型升级）

    扩展功能：
    - 跟踪模型升级历史
    - 支持升级优先于熔断的决策逻辑
    """
    consecutive_retries: int = 0
    total_retries: int = 0
    total_segments: int = 0
    processed_segments: int = 0

    # === 新增：升级跟踪 ===
    escalation_count: int = 0                           # 已升级次数
    current_model: Optional[str] = None                 # 当前使用的模型
    escalation_history: List[str] = field(default_factory=list)  # 升级历史

    def record_retry(self):
        """记录一次重试"""
        self.consecutive_retries += 1
        self.total_retries += 1

    def record_success(self):
        """记录一次成功（重置连续计数）"""
        self.consecutive_retries = 0
        self.processed_segments += 1

    def record_escalation(self, new_model: str):
        """
        记录一次模型升级

        Args:
            new_model: 升级后的模型名称
        """
        if self.current_model:
            self.escalation_history.append(f"{self.current_model} -> {new_model}")
        self.current_model = new_model
        self.escalation_count += 1
        # 升级后重置连续重试计数，给新模型机会
        self.consecutive_retries = 0

    def should_escalate(self, demucs_settings) -> bool:
        """
        判断是否应该升级模型（优先于熔断）

        升级条件：
        1. 允许自动升级 (auto_escalation=True)
        2. 未达到最大升级次数
        3. 满足熔断条件（连续重试或比例过高）

        Args:
            demucs_settings: Demucs配置对象

        Returns:
            bool: True表示应该升级模型
        """
        if not demucs_settings.auto_escalation:
            return False

        if self.escalation_count >= demucs_settings.max_escalations:
            return False

        # 满足熔断条件时，优先升级
        return self._check_break_condition(demucs_settings)

    def should_break(self, demucs_settings) -> bool:
        """
        判断是否应该触发熔断

        注意：只有在无法升级时才触发熔断

        Args:
            demucs_settings: Demucs配置对象

        Returns:
            bool: True表示应该触发熔断
        """
        if not demucs_settings.circuit_breaker_enabled:
            return False

        # 如果还能升级，不触发熔断
        if self.should_escalate(demucs_settings):
            return False

        return self._check_break_condition(demucs_settings)

    def _check_break_condition(self, demucs_settings) -> bool:
        """检查是否满足熔断/升级条件"""
        # 条件1：连续重试次数
        if self.consecutive_retries >= demucs_settings.consecutive_threshold:
            return True

        # 条件2：总重试比例（至少处理5个segment后才检查）
        if self.processed_segments >= 5:
            retry_ratio = self.total_retries / self.processed_segments
            if retry_ratio >= demucs_settings.ratio_threshold:
                return True

        return False

    def get_stats(self) -> Dict:
        """获取统计信息（扩展）"""
        return {
            "consecutive_retries": self.consecutive_retries,
            "total_retries": self.total_retries,
            "total_segments": self.total_segments,
            "processed_segments": self.processed_segments,
            "retry_ratio": self.total_retries / max(1, self.processed_segments),
            # 新增
            "escalation_count": self.escalation_count,
            "current_model": self.current_model,
            "escalation_history": self.escalation_history,
        }
```

### 5.4 CircuitBreakHandler 新增

**新增位置**: `services/transcription_service.py`（在 CircuitBreakerState 之后）

```python
class CircuitBreakAction(Enum):
    """熔断后的处理动作"""
    CONTINUE = "continue"           # 继续处理，标记问题段落
    FALLBACK_ORIGINAL = "fallback"  # 降级使用原始音频
    FAIL = "fail"                   # 任务失败
    PAUSE = "pause"                 # 暂停等待人工介入


class CircuitBreakHandler:
    """
    熔断异常处理器

    负责在熔断触发时执行用户配置的处理策略
    """

    def __init__(self, job: JobState, settings: DemucsSettings):
        self.job = job
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.problem_segments: List[int] = []  # 记录问题段落索引

    def handle(
        self,
        breaker_state: CircuitBreakerState,
        current_segment_idx: int,
        sse_manager = None
    ) -> CircuitBreakAction:
        """
        处理熔断异常

        Args:
            breaker_state: 熔断器状态
            current_segment_idx: 当前段落索引
            sse_manager: SSE管理器（用于推送事件）

        Returns:
            CircuitBreakAction: 处理动作
        """
        action_str = self.settings.on_break
        action = CircuitBreakAction(action_str) if action_str in [e.value for e in CircuitBreakAction] else CircuitBreakAction.CONTINUE

        # 记录问题段落
        self.problem_segments.append(current_segment_idx)

        # 推送 SSE 事件
        if sse_manager:
            self._push_circuit_break_event(breaker_state, action, sse_manager)

        if action == CircuitBreakAction.FAIL:
            raise BreakToGlobalSeparation(
                f"熔断触发，任务终止。问题段落: {self.problem_segments}"
            )

        elif action == CircuitBreakAction.PAUSE:
            self.job.paused = True
            self.job.status = "paused"
            self.job.message = f"熔断触发，等待人工介入。问题段落: {self.problem_segments}"
            raise BreakToGlobalSeparation(self.job.message)

        else:  # CONTINUE 或 FALLBACK_ORIGINAL
            self.logger.warning(
                f"熔断触发，采用 {action.value} 策略继续处理。"
                f"问题段落: {self.problem_segments}"
            )

        return action

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
        action: CircuitBreakAction,
        sse_manager
    ):
        """推送熔断处理事件"""
        sse_manager.push_event(
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

### 5.5 转录主流程集成要点

**修改位置**: `services/transcription_service.py` `_transcribe_segments` 或相关方法

关键改动点：
1. 初始化 `CircuitBreakerState` 时设置 `current_model`
2. 在 `should_break` 检查前先检查 `should_escalate`
3. 升级时调用 `demucs_service.escalate_model()` 并重新分离
4. 推送 `model_escalated` SSE 事件

```python
# 伪代码示意
async def _transcribe_segments(self, job, segments, audio_source, ...):
    # 初始化熔断状态
    breaker_state = CircuitBreakerState(current_model=strategy.initial_model)
    break_handler = CircuitBreakHandler(job, job.settings.demucs)

    for idx, segment in enumerate(segments):
        result = self._transcribe_single_segment(segment, audio_source)

        if self._is_quality_acceptable(result):
            breaker_state.record_success()
        else:
            breaker_state.record_retry()

            # 优先检查升级
            if breaker_state.should_escalate(job.settings.demucs):
                new_model = demucs_service.escalate_model(
                    breaker_state.current_model,
                    job.settings.demucs
                )
                if new_model:
                    breaker_state.record_escalation(new_model)
                    # 推送升级事件
                    self._push_sse_model_escalated(job, breaker_state)
                    # 重新分离
                    audio_source = self._re_separate_with_model(new_model)

            # 无法升级，检查熔断
            elif breaker_state.should_break(job.settings.demucs):
                action = break_handler.handle(breaker_state, idx, self.sse_manager)
                # 根据 action 决定后续处理...
```

---

## 6. Phase 4: SSE 事件扩展

### 6.1 目标
扩展 SSE 事件以支持前端实时展示分级策略和模型升级状态。

### 6.2 新增事件

#### 6.2.1 separation_strategy（策略决策）

**触发时机**: BGM 检测完成后，开始分离前

```json
{
  "event": "separation_strategy",
  "data": {
    "should_separate": true,
    "initial_model": "htdemucs_ft",
    "fallback_model": "mdx_extra",
    "reason": "检测到轻微BGM，使用轻量模型 htdemucs_ft",
    "bgm_level": "light",
    "allow_escalation": true
  }
}
```

#### 6.2.2 model_escalated（模型升级）

**触发时机**: 熔断条件满足但选择升级模型时

```json
{
  "event": "model_escalated",
  "data": {
    "from_model": "htdemucs_ft",
    "to_model": "mdx_extra",
    "reason": "连续3个segment质量不达标，升级模型",
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

#### 6.2.3 circuit_breaker_handled（熔断处理）

**触发时机**: 熔断触发且无法再升级时

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
      "escalation_history": ["htdemucs_ft -> mdx_extra"]
    },
    "suggestion": "少量段落可能需要手动调整时间轴"
  }
}
```

### 6.3 改动文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `services/transcription_service.py` | 新增 | `_push_sse_separation_strategy` 方法 |
| `services/transcription_service.py` | 新增 | `_push_sse_model_escalated` 方法 |
| `services/transcription_service.py` | 修改 | `_push_sse_circuit_breaker_triggered` 扩展字段 |

---

## 7. Phase 5: 配置文件与接口

### 7.1 目标
新增配置文件支持，扩展 API 接口，预留前端配置入口。

### 7.2 配置文件 config/demucs_tiers.json

**新增位置**: `backend/config/demucs_tiers.json`

```json
{
  "version": "1.0",
  "description": "Demucs分级模型配置",

  "presets": {
    "fast": {
      "description": "速度优先，适合低配机器",
      "weak_model": "htdemucs",
      "strong_model": "htdemucs_ft",
      "fallback_model": "mdx_extra_q"
    },
    "balanced": {
      "description": "平衡模式（默认推荐）",
      "weak_model": "htdemucs_ft",
      "strong_model": "mdx_extra_q",
      "fallback_model": "mdx_extra"
    },
    "quality": {
      "description": "质量优先，处理时间较长",
      "weak_model": "mdx_extra_q",
      "strong_model": "mdx_extra",
      "fallback_model": "mdx_extra"
    }
  },

  "models": {
    "htdemucs": {
      "description": "Hybrid Transformer，快速模式",
      "size_mb": 80,
      "quality_score": 3,
      "speed_score": 5
    },
    "htdemucs_ft": {
      "description": "Fine-tuned版本，人声优化",
      "size_mb": 80,
      "quality_score": 4,
      "speed_score": 4
    },
    "mdx_extra_q": {
      "description": "MDX-Net量化版",
      "size_mb": 25,
      "quality_score": 4,
      "speed_score": 4
    },
    "mdx_extra": {
      "description": "MDX-Net完整版，最高质量",
      "size_mb": 600,
      "quality_score": 5,
      "speed_score": 2
    }
  },

  "defaults": {
    "bgm_light_threshold": 0.02,
    "bgm_heavy_threshold": 0.15,
    "consecutive_threshold": 3,
    "ratio_threshold": 0.2,
    "max_escalations": 1
  }
}
```

### 7.3 API 接口扩展

#### 7.3.1 启动任务接口 POST /api/start

**请求体扩展**:

```json
{
  "job_id": "xxx",
  "settings": {
    "model": "medium",
    "demucs": {
      "enabled": true,
      "mode": "auto",
      "quality_preset": "balanced",

      "weak_model": "htdemucs_ft",
      "strong_model": "mdx_extra_q",
      "fallback_model": "mdx_extra",
      "auto_escalation": true,
      "max_escalations": 1,

      "bgm_light_threshold": 0.02,
      "bgm_heavy_threshold": 0.15,

      "on_break": "continue"
    }
  }
}
```

#### 7.3.2 新增配置查询接口 GET /api/demucs/config

**响应**:

```json
{
  "presets": { ... },
  "models": { ... },
  "defaults": { ... }
}
```

### 7.4 前端配置接口预留

#### 7.4.1 设置菜单集成点

前端设置菜单需要预留 Demucs 配置区块，建议接口：

```typescript
// types/demucs.ts

export interface DemucsPreset {
  id: string;           // "fast" | "balanced" | "quality"
  description: string;
  weak_model: string;
  strong_model: string;
  fallback_model: string;
}

export interface DemucsModel {
  id: string;
  description: string;
  size_mb: number;
  quality_score: number;  // 1-5
  speed_score: number;    // 1-5
}

export interface DemucsUserSettings {
  // 基础配置
  enabled: boolean;
  mode: "auto" | "always" | "never" | "on_demand";
  quality_preset: "fast" | "balanced" | "quality";

  // 高级配置（可折叠）
  weak_model?: string;
  strong_model?: string;
  fallback_model?: string;
  auto_escalation?: boolean;
  max_escalations?: number;
  bgm_light_threshold?: number;
  bgm_heavy_threshold?: number;
  on_break?: "continue" | "fallback" | "fail" | "pause";
}

// API
export async function getDemucsConfig(): Promise<{
  presets: Record<string, DemucsPreset>;
  models: Record<string, DemucsModel>;
  defaults: DemucsUserSettings;
}>;

export async function saveDemucsSettings(settings: DemucsUserSettings): Promise<void>;
```

#### 7.4.2 UI 组件建议结构

```
设置菜单
├─ 转录设置
│   └─ Whisper模型选择 ...
├─ 人声分离设置  <-- 新增区块
│   ├─ [开关] 启用人声分离
│   ├─ 工作模式: [自动检测] [始终分离] [禁用]
│   ├─ 质量预设: [快速] [平衡(推荐)] [高质量]
│   ├─ [复选框] 质量不足时自动升级模型
│   └─ [展开高级选项]
│       ├─ 弱BGM模型: [下拉选择]
│       ├─ 强BGM模型: [下拉选择]
│       ├─ 兜底模型: [下拉选择]
│       ├─ BGM检测阈值: [滑块]
│       └─ 熔断策略: [继续] [降级] [失败] [暂停]
└─ 其他设置 ...
```

---

## 8. Phase 6: 文档更新

### 8.1 目标
更新项目文档系统，确保开发者和用户理解新功能。

### 8.2 需更新的文档

| 文档 | 更新内容 |
|------|----------|
| `llmdoc/architecture/demucs.md` | 新增分级策略架构说明 |
| `llmdoc/reference/api.md` | 新增 Demucs 配置 API |
| `llmdoc/guides/demucs-config.md` | 新增用户配置指南 |

---

## 9. 重构实施顺序

```
Week 1: Phase 1 + Phase 2
├─ Day 1-2: DemucsSettings 扩展 + 单元测试
├─ Day 3-4: ModelTierConfig + SeparationStrategy
└─ Day 5-7: SeparationStrategyResolver + 集成测试

Week 2: Phase 3
├─ Day 1-2: CircuitBreakerState 扩展
├─ Day 3-4: CircuitBreakHandler 实现
└─ Day 5-7: TranscriptionService 集成

Week 3: Phase 4 + Phase 5
├─ Day 1-2: SSE 事件扩展
├─ Day 3-4: 配置文件 + API 扩展
└─ Day 5-7: 前端接口预留 + 端到端测试

Week 4: Phase 6 + 发布
├─ Day 1-3: 文档更新
├─ Day 4-5: 回归测试
└─ Day 6-7: 发布
```

---

## 10. 测试用例清单

### 10.1 单元测试

| 测试类 | 测试点 |
|--------|--------|
| TestDemucsSettings | 默认值、质量预设覆盖、向后兼容 |
| TestSeparationStrategy | to_dict 序列化 |
| TestSeparationStrategyResolver | 各模式+各BGM级别组合（9种） |
| TestCircuitBreakerState | 升级记录、升级优先熔断、统计信息 |
| TestCircuitBreakHandler | 各处理策略、问题报告生成 |

### 10.2 集成测试

| 场景 | 输入 | 期望 |
|------|------|------|
| 纯人声 | 播客录音 | BGM=NONE，不分离 |
| 轻BGM | 轻音乐访谈 | BGM=LIGHT，用 htdemucs_ft |
| 强BGM | 音乐MV | BGM=HEAVY，用 mdx_extra_q |
| 质量差触发升级 | 嘈杂环境 | 升级到 mdx_extra |
| 升级后仍差 | 极端噪音 | 达到 max_escalations 后熔断 |
| mode=always | 任意 | 始终分离 |
| mode=never | 有BGM | 不分离 |

### 10.3 端到端测试

| 测试 | 验证点 |
|------|--------|
| SSE 事件流 | separation_strategy → 进度 → model_escalated(可选) → 完成 |
| API 配置 | 传递自定义模型配置，验证生效 |
| 向后兼容 | 不传新字段，验证使用默认值 |

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 模型切换导致显存不足 | 任务失败 | 切换前检查显存，不足时自动降级 |
| 升级后重新分离耗时长 | 用户体验 | SSE 推送升级进度，让用户知晓 |
| 向后兼容性问题 | 旧配置失效 | 所有新字段设默认值，from_meta_dict 容错 |
| 前端未适配 | 功能不可见 | 后端 API 先上线，前端渐进式集成 |

---

## 附录: 文件改动汇总

| 文件 | Phase | 改动类型 | 改动说明 |
|------|-------|----------|----------|
| `models/job_models.py` | 1 | 扩展 | DemucsSettings 新增 8 个字段 |
| `services/demucs_service.py` | 1,2 | 新增+扩展 | ModelTierConfig, SeparationStrategy, SeparationStrategyResolver, DemucsService 新方法 |
| `services/transcription_service.py` | 3,4 | 扩展+新增 | CircuitBreakerState 扩展, CircuitBreakHandler, SSE 方法 |
| `config/demucs_tiers.json` | 5 | 新增 | 配置文件 |
| `api/routes.py` | 5 | 扩展 | 新增 /api/demucs/config |
