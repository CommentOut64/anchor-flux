# Phase 2: 智能熔断集成

> 目标：实现频谱指纹检测器、熔断决策器和动态权重计算
>
> 工期：2-3天

---

## 一、任务清单

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 频谱指纹检测器 | `audio_circuit_breaker.py` | P0 |
| 熔断决策器 | `fuse_breaker.py` | P0 |
| 动态权重计算 | `config.py` (修改) | P1 |
| 熔断数据模型 | `circuit_breaker_models.py` | P0 |

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

```python
"""
音频熔断器 - 轻量级 BGM 预判
基于频谱特征（ZCR 方差 + 谱质心标准差）
"""
import logging
import numpy as np
from collections import deque
from typing import Optional

from ..models.circuit_breaker_models import CircuitBreakerDecision

logger = logging.getLogger(__name__)


class AudioCircuitBreaker:
    """
    音频熔断器

    功能:
    1. 逐片段频谱分析（纯 CPU）
    2. 滑动窗口历史惯性机制
    3. 快速判断是否需要人声分离
    """

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.history = deque(maxlen=self.config['history_size'])

        logger.info(f"音频熔断器已初始化，配置: {self.config}")

    def _default_config(self) -> dict:
        return {
            # 基础参数
            'sr': 16000,
            'n_fft': 512,
            'hop_length': 256,

            # 历史惯性参数
            'history_size': 5,
            'history_threshold': 0.6,      # 60%历史为BGM则触发惯性

            # 决策阈值
            'music_score_threshold': 0.35,  # 音乐性评分阈值
            'reset_score': 0.15,           # 极度干净的阈值
        }

    def analyze_segment(
        self,
        audio_chunk: np.ndarray,
        segment_index: int
    ) -> CircuitBreakerDecision:
        """
        分析单个音频片段

        Args:
            audio_chunk: 音频数组（16kHz, float32）
            segment_index: 片段索引

        Returns:
            CircuitBreakerDecision: 决策结果
        """
        # 1. 计算频谱特征
        music_score = self._compute_spectral_score(audio_chunk)

        # 2. 极度干净判断（重置惯性）
        if music_score < self.config['reset_score']:
            self._update_history(False)
            return CircuitBreakerDecision(
                segment_index=segment_index,
                should_separate=False,
                confidence=1.0 - music_score,
                reason='clean_reset',
                spectral_score=music_score
            )

        # 3. 计算历史惯性
        history_confidence = self._get_history_confidence()

        # 4. 决策逻辑
        if len(self.history) >= 3 and history_confidence >= self.config['history_threshold']:
            # 惯性触发：历史显示有BGM
            should_separate = True
            reason = 'history_inertia'
            confidence = history_confidence
        else:
            # 无惯性：依赖当前检测
            should_separate = music_score > self.config['music_score_threshold']
            reason = 'fingerprint_detection'
            confidence = music_score

        self._update_history(should_separate)

        return CircuitBreakerDecision(
            segment_index=segment_index,
            should_separate=should_separate,
            confidence=confidence,
            reason=reason,
            spectral_score=music_score
        )

    def _compute_spectral_score(self, audio_chunk: np.ndarray) -> float:
        """
        计算频谱音乐性评分

        基于:
        - ZCR方差：音乐的ZCR变化通常比语音平滑
        - 谱质心标准差：音乐的频率中心波动较小

        Returns:
            float: 音乐性评分 (0.0-1.0)，越高越可能是音乐
        """
        if len(audio_chunk) < self.config['n_fft']:
            return 0.0

        y = audio_chunk + 1e-10  # 防止除零

        # 分帧计算频谱特征
        n_fft = self.config['n_fft']
        hop_length = self.config['hop_length']
        num_frames = (len(y) - n_fft) // hop_length + 1

        if num_frames <= 0:
            return 0.0

        frame_zcrs = []
        centroids = []

        # 采样策略：最多50帧
        step = max(1, num_frames // 50)

        for i in range(0, num_frames, step):
            start = i * hop_length
            end = start + n_fft
            if end > len(y):
                break

            frame = y[start:end] * np.hanning(n_fft)

            # 帧ZCR
            frame_zcr = np.sum(np.abs(np.diff(np.signbit(frame)))) / len(frame)
            frame_zcrs.append(frame_zcr)

            # 频谱质心
            spectrum = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(n_fft, 1/self.config['sr'])
            centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
            centroids.append(centroid)

        if not frame_zcrs:
            return 0.0

        # 计算统计量
        zcr_var = np.var(frame_zcrs)
        centroid_std = np.std(centroids)

        # 融合打分（音乐特征越明显，分数越高）
        score = 0.0

        # 判据1: ZCR极度稳定
        if zcr_var < 0.0015:
            score += 0.5
        elif zcr_var < 0.003:
            score += 0.2

        # 判据2: 频率中心稳定
        if centroid_std < 500:
            score += 0.4
        elif centroid_std < 900:
            score += 0.2

        return min(1.0, score)

    def _get_history_confidence(self) -> float:
        """计算历史惯性置信度"""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def _update_history(self, decision: bool):
        """更新历史"""
        self.history.append(decision)

    def reset(self):
        """重置历史（处理新视频时调用）"""
        self.history.clear()
        logger.debug("熔断器历史已重置")
```

---

## 五、文件3：熔断决策器

**路径**: `backend/app/services/fuse_breaker.py`

```python
"""
熔断决策器
协调人声分离与 ASR 补刀

核心原则：熔断升级优先于 ASR 补刀
"""
import logging
from typing import Dict, Optional

from ..models.circuit_breaker_models import (
    FuseAction,
    FuseState,
    FuseDecision,
    SeparationLevel,
    BGMLevel
)

logger = logging.getLogger(__name__)


class FuseBreaker:
    """
    熔断决策器

    决策原则：
    1. 熔断升级优先于ASR补刀
    2. 已升级后才允许Whisper补刀
    3. 避免在噪音环境下盲目补刀
    """

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        self.fuse_states: Dict[int, FuseState] = {}

        logger.info(f"熔断决策器已初始化，配置: {self.config}")

    def _default_config(self) -> dict:
        return {
            'confidence_threshold': 0.6,       # 置信度阈值
            'upgrade_threshold': 0.4,          # 低于此值触发模型升级
            'noise_event_tags': ['BGM', 'Noise', 'Cough', 'Sneeze'],
            'max_retry_count': 1,              # 最大重试次数
        }

    def decide_action(
        self,
        segment_id: int,
        confidence: float,
        event_tag: Optional[str],
        current_separation_level: SeparationLevel
    ) -> FuseDecision:
        """
        决策下一步动作

        Args:
            segment_id: 片段ID
            confidence: 转录置信度
            event_tag: 事件标签（BGM/Noise等）
            current_separation_level: 当前分离级别

        Returns:
            FuseDecision: 决策结果
        """
        # 获取或创建熔断状态
        if segment_id not in self.fuse_states:
            self.fuse_states[segment_id] = FuseState(
                segment_id=segment_id,
                separation_level=current_separation_level,
                bgm_level=BGMLevel.NONE,
                retry_count=0
            )

        state = self.fuse_states[segment_id]

        # 规则1: 高置信度，直接接受
        if confidence >= self.config['confidence_threshold']:
            logger.debug(
                f"片段 {segment_id}: 高置信度 {confidence:.2f}，接受结果"
            )
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason=f'high_confidence_{confidence:.2f}'
            )

        # 规则2: 检测到噪音事件且未升级，优先升级分离
        if event_tag in self.config['noise_event_tags']:
            if current_separation_level != SeparationLevel.MDX_EXTRA:
                logger.info(
                    f"片段 {segment_id}: 检测到噪音事件 {event_tag}，升级分离模型"
                )
                return FuseDecision(
                    action=FuseAction.UPGRADE_SEPARATION,
                    reason=f'noise_event_{event_tag}',
                    next_separation_level=self._get_next_level(current_separation_level)
                )

        # 规则3: 极低置信度且未升级，优先升级分离
        if confidence < self.config['upgrade_threshold']:
            if current_separation_level != SeparationLevel.MDX_EXTRA:
                logger.info(
                    f"片段 {segment_id}: 极低置信度 {confidence:.2f}，升级分离模型"
                )
                return FuseDecision(
                    action=FuseAction.UPGRADE_SEPARATION,
                    reason=f'low_confidence_{confidence:.2f}',
                    next_separation_level=self._get_next_level(current_separation_level)
                )

        # 规则4: 已升级或无需升级，且未超过重试次数，Whisper补刀
        if state.retry_count < self.config['max_retry_count']:
            logger.info(
                f"片段 {segment_id}: 置信度 {confidence:.2f}，使用 Whisper 补刀"
            )
            state.retry_count += 1
            return FuseDecision(
                action=FuseAction.WHISPER_RETRY,
                reason=f'low_confidence_retry_{state.retry_count}'
            )

        # 规则5: 超过重试次数，接受当前结果
        logger.warning(
            f"片段 {segment_id}: 超过重试次数，接受低质量结果 {confidence:.2f}"
        )
        return FuseDecision(
            action=FuseAction.ACCEPT,
            reason='max_retry_exceeded'
        )

    def _get_next_level(self, current: SeparationLevel) -> SeparationLevel:
        """获取下一个分离级别"""
        if current == SeparationLevel.NONE:
            return SeparationLevel.HTDEMUCS
        elif current == SeparationLevel.HTDEMUCS:
            return SeparationLevel.MDX_EXTRA
        else:
            return SeparationLevel.MDX_EXTRA

    def update_state(
        self,
        segment_id: int,
        new_separation_level: SeparationLevel
    ):
        """更新片段分离状态"""
        if segment_id in self.fuse_states:
            self.fuse_states[segment_id].separation_level = new_separation_level
            logger.debug(
                f"片段 {segment_id} 分离级别已更新: {new_separation_level.value}"
            )

    def reset(self):
        """重置所有状态（处理新视频时调用）"""
        self.fuse_states.clear()
        logger.debug("熔断决策器状态已重置")
```

---

## 六、文件4：动态权重计算

**修改文件**: `backend/app/core/config.py`

在 `ProjectConfig` 类中添加以下方法：

```python
def calculate_dynamic_weights(
    self,
    total_segments: int,
    segments_to_separate: int,
    segments_to_retry: int,
    engine: str = 'sensevoice'
) -> Dict[str, int]:
    """
    动态计算阶段权重

    Args:
        total_segments: 总片段数
        segments_to_separate: 需要分离的片段数
        segments_to_retry: 需要补刀的片段数（预估）
        engine: 转录引擎

    Returns:
        动态权重字典
    """
    # 基础权重
    base_weights = {
        'pending': 0,
        'extract': 5,
        'vad': 5,
        'bgm_detect': 2,
        'demucs': 0,
        'transcribe': 70,
        'retry': 0,
        'sentence_split': 3,
        'srt': 10,
        'complete': 0
    }

    if total_segments == 0:
        return base_weights

    # 计算分离比例
    sep_ratio = segments_to_separate / total_segments
    retry_ratio = segments_to_retry / total_segments

    # 动态调整
    base_weights['demucs'] = int(15 * sep_ratio)      # 最多15%
    base_weights['retry'] = int(20 * retry_ratio)     # 最多20%

    # 从转录阶段扣除
    used = base_weights['demucs'] + base_weights['retry']
    base_weights['transcribe'] = max(40, 70 - used)

    return base_weights
```

---

## 七、快速测试

创建测试脚本验证熔断逻辑：

**路径**: `backend/test_phase2.py`

```python
"""
Phase 2 快速测试脚本
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

    from app.core.config import config

    # 场景1: 纯净语音（无分离，无补刀）
    print("\n场景1: 纯净语音")
    weights = config.calculate_dynamic_weights(
        total_segments=100,
        segments_to_separate=0,
        segments_to_retry=0
    )
    print(f"  转录: {weights['transcribe']}%")
    print(f"  分离: {weights['demucs']}%")
    print(f"  补刀: {weights['retry']}%")

    # 场景2: 重度BGM（50%分离）
    print("\n场景2: 重度BGM")
    weights = config.calculate_dynamic_weights(
        total_segments=100,
        segments_to_separate=50,
        segments_to_retry=10
    )
    print(f"  转录: {weights['transcribe']}%")
    print(f"  分离: {weights['demucs']}%")
    print(f"  补刀: {weights['retry']}%")


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

---

## 十、下一步

完成 Phase 2 后，进入 [Phase 3: 转录服务重构](./03_Phase3_转录服务重构.md)
