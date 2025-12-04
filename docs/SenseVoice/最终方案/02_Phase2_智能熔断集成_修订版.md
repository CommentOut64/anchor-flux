# Phase 2: 智能熔断集成（修订版）

> 目标：实现频谱指纹检测器、熔断决策器和动态权重计算
>
> 工期：2-3天

---

## ⚠️ 重要修订

- ✅ **保留**：所有新文件创建（无重复）
- ✅ **修改**：动态权重计算方法集成到现有 `config.py`
- ✅ **无冲突**：本阶段文件均为新建，无需复用现有服务

---

## 一、任务清单

| 任务 | 文件 | 优先级 | 状态 |
|------|------|--------|------|
| 频谱指纹检测器 | `audio_circuit_breaker.py` | P0 | 新建 |
| 熔断决策器 | `fuse_breaker.py` | P0 | 新建 |
| 动态权重计算 | `config.py` (修改) | P1 | 修改 |
| 熔断数据模型 | `circuit_breaker_models.py` | P0 | 新建 |

---

## 二、核心概念

### 2.1 熔断机制设计原则

1. **熔断升级优先于 ASR 补刀**
   - 在噪音环境下，优先升级分离模型去除噪音
   - 已升级后才允许 Whisper 补刀

2. **频谱预判 + Demucs 验证（可选）**
   - 默认信任频谱检测（极速模式）
   - 可选 Demucs 验证（精确模式）

3. **决策状态记录**
   - 记录每个片段的分离级别
   - 避免重复升级

---

## 三、文件1：熔断数据模型

**路径**: `backend/app/models/circuit_breaker_models.py`

```python
"""
熔断机制数据模型
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BGMLevel(Enum):
    """BGM 级别"""
    NONE = "none"           # 无 BGM
    LIGHT = "light"         # 轻度 BGM
    HEAVY = "heavy"         # 重度 BGM


class SeparationLevel(Enum):
    """人声分离级别"""
    NONE = "none"           # 未分离
    HTDEMUCS = "htdemucs"   # 轻量分离
    MDX_EXTRA = "mdx_extra" # 高质量分离


class FuseAction(Enum):
    """熔断动作"""
    ACCEPT = "accept"                       # 接受结果
    UPGRADE_SEPARATION = "upgrade_separation"  # 升级分离模型
    WHISPER_RETRY = "whisper_retry"        # Whisper 补刀


@dataclass
class CircuitBreakerDecision:
    """频谱检测决策结果"""
    segment_index: int
    should_separate: bool           # 是否需要分离
    confidence: float               # 置信度评分
    reason: str                     # 决策原因
    spectral_score: float           # 频谱音乐性评分


@dataclass
class FuseState:
    """熔断状态"""
    segment_id: int
    separation_level: SeparationLevel
    bgm_level: BGMLevel
    retry_count: int = 0


@dataclass
class FuseDecision:
    """熔断决策结果"""
    action: FuseAction
    reason: str
    next_separation_level: Optional[SeparationLevel] = None
```

---

## 四、文件2：频谱指纹检测器

**路径**: `backend/app/services/audio_circuit_breaker.py`

（代码与原 Phase 2 相同，无需修改）

---

## 五、文件3：熔断决策器

**路径**: `backend/app/services/fuse_breaker.py`

（代码与原 Phase 2 相同，无需修改）

---

## 六、文件4：动态权重计算（⚠️ 重要修订）

**修改文件**: `backend/app/core/config.py`

在 `ProjectConfig` 类中添加以下方法：

```python
def calculate_dynamic_weights(
    self,
    engine: str,
    total_segments: int,
    segments_to_separate: int,
    segments_to_retry: int
) -> Dict[str, int]:
    """
    根据引擎和实际场景动态计算权重

    Args:
        engine: 'whisperx' | 'sensevoice'
        total_segments: 总片段数
        segments_to_separate: 需要分离的片段数
        segments_to_retry: 需要补刀的片段数

    Returns:
        动态权重字典
    """
    if engine == 'whisperx':
        # WhisperX 使用现有固定权重
        return self.PHASE_WEIGHTS.copy()

    # SenseVoice 动态权重
    base_weights = self.PHASE_WEIGHTS.copy()

    if total_segments > 0:
        sep_ratio = segments_to_separate / total_segments
        retry_ratio = segments_to_retry / total_segments

        # 调整权重（SenseVoice 特有）
        base_weights['demucs_global'] = int(15 * sep_ratio)  # 最多15%
        base_weights['transcribe'] = 70  # SenseVoice 转录+分句
        base_weights['align'] = 0        # SenseVoice 无需对齐

        # 新增补刀阶段权重
        if 'retry' not in base_weights:
            base_weights['retry'] = 0
        base_weights['retry'] = int(15 * retry_ratio)  # Whisper 补刀

        # 平衡到 100%
        used = sum(base_weights.values())
        if used < 100:
            base_weights['transcribe'] += (100 - used)

    return base_weights
```

**注意**：此方法需要导入 `Dict` 类型：

```python
from typing import Optional, Dict  # 在文件顶部添加 Dict
```

---

## 七、快速测试（修订版）

创建测试脚本验证熔断逻辑：

**路径**: `backend/test_phase2_revised.py`

```python
"""
Phase 2 快速测试脚本（修订版）
"""
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.audio_circuit_breaker import AudioCircuitBreaker
from app.services.fuse_breaker import FuseBreaker
from app.models.circuit_breaker_models import SeparationLevel
from app.core.config import config


def test_circuit_breaker():
    """测试频谱检测"""
    print("\n=== 测试频谱检测器 ===")

    breaker = AudioCircuitBreaker()

    # 生成测试音频（模拟纯人声）
    sr = 16000
    duration = 5
    clean_audio = np.random.randn(sr * duration) * 0.1

    decision = breaker.analyze_segment(clean_audio, segment_index=0)

    print(f"片段 0:")
    print(f"  需要分离: {decision.should_separate}")
    print(f"  置信度: {decision.confidence:.2f}")
    print(f"  原因: {decision.reason}")
    print(f"  频谱评分: {decision.spectral_score:.2f}")


def test_fuse_breaker():
    """测试熔断决策器"""
    print("\n=== 测试熔断决策器 ===")

    fuse = FuseBreaker()

    # 场景1: 高置信度，直接接受
    print("\n场景1: 高置信度")
    decision = fuse.decide_action(
        segment_id=1,
        confidence=0.85,
        event_tag=None,
        current_separation_level=SeparationLevel.NONE
    )
    print(f"  动作: {decision.action.value}")
    print(f"  原因: {decision.reason}")

    # 场景2: 低置信度 + 噪音事件，升级分离
    print("\n场景2: 低置信度 + 噪音事件")
    decision = fuse.decide_action(
        segment_id=2,
        confidence=0.45,
        event_tag='BGM',
        current_separation_level=SeparationLevel.NONE
    )
    print(f"  动作: {decision.action.value}")
    print(f"  原因: {decision.reason}")
    print(f"  下一级别: {decision.next_separation_level.value if decision.next_separation_level else 'N/A'}")

    # 场景3: 低置信度 + 已升级，Whisper补刀
    print("\n场景3: 低置信度 + 已升级")
    decision = fuse.decide_action(
        segment_id=3,
        confidence=0.55,
        event_tag=None,
        current_separation_level=SeparationLevel.MDX_EXTRA
    )
    print(f"  动作: {decision.action.value}")
    print(f"  原因: {decision.reason}")


def test_dynamic_weights():
    """测试动态权重"""
    print("\n=== 测试动态权重计算 ===")

    # 场景1: WhisperX（固定权重）
    print("\n场景1: WhisperX 引擎")
    weights = config.calculate_dynamic_weights(
        engine='whisperx',
        total_segments=100,
        segments_to_separate=0,
        segments_to_retry=0
    )
    print(f"  转录: {weights['transcribe']}%")
    print(f"  对齐: {weights['align']}%")
    print(f"  分离: {weights['demucs_global']}%")

    # 场景2: SenseVoice 纯净语音（无分离，无补刀）
    print("\n场景2: SenseVoice 纯净语音")
    weights = config.calculate_dynamic_weights(
        engine='sensevoice',
        total_segments=100,
        segments_to_separate=0,
        segments_to_retry=0
    )
    print(f"  转录: {weights['transcribe']}%")
    print(f"  对齐: {weights['align']}%")
    print(f"  分离: {weights['demucs_global']}%")
    print(f"  补刀: {weights.get('retry', 0)}%")

    # 场景3: SenseVoice 重度BGM（50%分离，10%补刀）
    print("\n场景3: SenseVoice 重度BGM")
    weights = config.calculate_dynamic_weights(
        engine='sensevoice',
        total_segments=100,
        segments_to_separate=50,
        segments_to_retry=10
    )
    print(f"  转录: {weights['transcribe']}%")
    print(f"  对齐: {weights['align']}%")
    print(f"  分离: {weights['demucs_global']}%")
    print(f"  补刀: {weights.get('retry', 0)}%")


if __name__ == "__main__":
    test_circuit_breaker()
    test_fuse_breaker()
    test_dynamic_weights()

    print("\n=== Phase 2 测试完成 ===")
```

---

## 八、验收标准

- [ ] 频谱检测器可正确识别纯人声和BGM片段
- [ ] 熔断决策器遵循"升级优先于补刀"原则
- [ ] 动态权重计算符合预期（不同场景权重不同）
- [ ] WhisperX 引擎保持固定权重
- [ ] SenseVoice 引擎使用动态权重

---

## 九、注意事项

1. **频谱检测参数调优**：
   - `music_score_threshold` 默认 0.35，可根据实际效果调整
   - 如果误判率高，适当提高阈值

2. **熔断决策器**：
   - `max_retry_count` 默认 1，避免无限重试
   - 超过重试次数后会接受低质量结果

3. **动态权重**：
   - 权重分配确保总和为 100%
   - 转录权重最低 40%，避免进度条失真
   - **重要**：方法需要在现有 `ProjectConfig` 类中添加，不创建新类

---

## 十、下一步

完成 Phase 2（修订版）后，进入 [Phase 3: 转录服务重构（修订版）](./03_Phase3_转录服务重构_修订版.md)
