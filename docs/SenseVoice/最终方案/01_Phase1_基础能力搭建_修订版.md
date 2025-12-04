# Phase 1: 基础能力搭建（修订版）

> 目标：实现 SenseVoice ONNX 服务和分句算法，复用现有硬件检测
>
> 工期：2-3天

---

## ⚠️ 重要修订

- ❌ **删除**：不再创建 `hardware_detector.py`（已存在）
- ✅ **复用**：使用现有的 `hardware_service.py`
- ✅ **扩展**：扩展现有的 `hardware_models.py`

---

## 一、任务清单

| 任务 | 文件 | 优先级 | 状态 |
|------|------|--------|------|
| SenseVoice ONNX 服务 | `sensevoice_onnx_service.py` | P0 | 新建 |
| 分句算法 | `sentence_splitter.py` | P0 | 新建 |
| 数据模型定义 | `sensevoice_models.py` | P0 | 新建 |
| 扩展硬件模型 | `hardware_models.py` | P1 | 修改 |

---

## 二、文件1：数据模型定义

**路径**: `backend/app/models/sensevoice_models.py`

```python
"""
SenseVoice 数据模型定义
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SenseVoiceONNXConfig:
    """SenseVoice ONNX 配置"""
    model_path: str = "models/sensevoice_small_int8.onnx"
    use_gpu: bool = True                      # 优先使用 GPU
    fallback_to_cpu: bool = True              # GPU 不可用时回退 CPU
    num_threads: int = 4                      # CPU 推理线程数
    batch_size: int = 1                       # 批处理大小
    quantization: str = "int8"                # int8 | fp16 | fp32
    enable_graph_optimization: bool = True
    optimization_level: int = 99              # ORT_ENABLE_ALL


@dataclass
class WordTimestamp:
    """字级时间戳"""
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class SenseVoiceResult:
    """SenseVoice 转录结果"""
    text: str                                 # 识别文本（原始）
    text_clean: str                           # 清理后文本（移除标签）
    confidence: float                         # 置信度 0.0-1.0
    words: List[WordTimestamp]                # 字级时间戳
    start: float                              # 段落开始时间
    end: float                                # 段落结束时间
    language: str = "auto"                    # 检测到的语言
    emotion: Optional[str] = None             # 情感标签
    event: Optional[str] = None               # 事件标签（BGM/Noise等）
    raw_result: Optional[dict] = None         # 原始结果


@dataclass
class SentenceSegment:
    """句级字幕段"""
    text: str
    start: float
    end: float
    words: List[WordTimestamp]
    confidence: float = 1.0
```

---

## 三、文件2：扩展硬件模型

**修改文件**: `backend/app/models/hardware_models.py`

在 `OptimizationConfig` 类中添加 SenseVoice 相关配置：

```python
@dataclass
class OptimizationConfig:
    """基于硬件的优化配置"""
    # 转录优化配置
    batch_size: int = 16
    concurrency: int = 1
    use_memory_mapping: bool = False
    cpu_affinity_cores: List[int] = field(default_factory=list)

    # 推荐设备选择
    recommended_device: str = "cpu"
    recommended_model: str = "medium"

    # ========== 新增：SenseVoice 相关配置 ==========
    enable_sensevoice: bool = True           # 是否启用 SenseVoice
    enable_demucs: bool = True               # 是否启用人声分离
    demucs_model: str = "htdemucs"           # 推荐的 Demucs 模型
    sensevoice_device: str = "cuda"          # SenseVoice 推荐设备
    note: str = ""                           # 配置说明

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "transcription": {
                "batch_size": self.batch_size,
                "concurrency": self.concurrency,
                "device": self.recommended_device,
                "recommended_model": self.recommended_model
            },
            "sensevoice": {
                "enabled": self.enable_sensevoice,
                "device": self.sensevoice_device,
                "enable_demucs": self.enable_demucs,
                "demucs_model": self.demucs_model
            },
            "system": {
                "use_memory_mapping": self.use_memory_mapping,
                "cpu_affinity_cores": self.cpu_affinity_cores,
                "process_priority": "normal",
                "note": self.note
            }
        }
```

---

## 四、文件3：扩展硬件服务

**修改文件**: `backend/app/services/hardware_service.py`

在 `CoreOptimizer` 类中添加 SenseVoice 相关配置生成：

```python
class CoreOptimizer:
    """基于硬件信息的核心优化决策器"""

    def get_optimization_config(self, hardware: HardwareInfo) -> OptimizationConfig:
        """根据硬件信息生成优化配置"""
        config = OptimizationConfig()

        # ... 现有代码 ...

        # ========== 新增：SenseVoice 配置决策 ==========
        config.sensevoice_device = self._get_sensevoice_device(hardware)
        config.enable_demucs = self._should_enable_demucs(hardware)
        config.demucs_model = self._get_demucs_model(hardware)
        config.note = self._generate_config_note(hardware)

        return config

    def _get_sensevoice_device(self, hardware: HardwareInfo) -> str:
        """决定 SenseVoice 使用的设备"""
        if hardware.cuda_available and hardware.gpu_memory_mb:
            return "cuda"
        return "cpu"

    def _should_enable_demucs(self, hardware: HardwareInfo) -> bool:
        """决定是否启用 Demucs"""
        if not hardware.cuda_available:
            # 无 GPU 默认禁用
            return False

        if hardware.gpu_memory_mb:
            max_gpu_memory = max(hardware.gpu_memory_mb)
            # 显存低于 4GB 默认禁用
            if max_gpu_memory < 4000:
                return False

        return True

    def _get_demucs_model(self, hardware: HardwareInfo) -> str:
        """推荐 Demucs 模型"""
        if not hardware.cuda_available or not hardware.gpu_memory_mb:
            return "htdemucs"

        max_gpu_memory = max(hardware.gpu_memory_mb)

        if max_gpu_memory >= 8000:
            return "mdx_extra"  # 高质量模型
        else:
            return "htdemucs"   # 轻量模型

    def _generate_config_note(self, hardware: HardwareInfo) -> str:
        """生成配置说明"""
        if not hardware.cuda_available:
            return "未检测到GPU，人声分离已禁用，转录使用CPU模式"

        if not hardware.gpu_memory_mb:
            return "GPU信息不完整，使用保守配置"

        max_gpu_memory = max(hardware.gpu_memory_mb)

        if max_gpu_memory < 4000:
            return f"低显存模式（{max_gpu_memory}MB），仅在检测到强BGM时使用htdemucs"
        elif max_gpu_memory < 8000:
            return f"标准配置（{max_gpu_memory}MB），支持人声分离（htdemucs）"
        else:
            return f"高性能配置（{max_gpu_memory}MB），支持高质量分离（mdx_extra）"
```

---

## 五、文件4：分句算法

**路径**: `backend/app/services/sentence_splitter.py`

（代码与原 Phase 1 相同，无需修改）

---

## 六、文件5：SenseVoice ONNX 服务

**路径**: `backend/app/services/sensevoice_onnx_service.py`

（代码与原 Phase 1 相同，无需修改）

---

## 七、快速测试（修订版）

创建测试脚本验证核心功能：

**路径**: `backend/test_phase1_revised.py`

```python
"""
Phase 1 快速测试脚本（修订版）
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 使用现有的硬件服务
from app.services.hardware_service import get_hardware_detector, get_hardware_optimizer
from app.services.sentence_splitter import SentenceSplitter
from app.services.sensevoice_onnx_service import get_sensevoice_service
from app.models.sensevoice_models import WordTimestamp


def test_hardware_detection():
    """测试硬件检测（使用现有服务）"""
    print("\n=== 测试硬件检测（现有服务）===")

    # 使用现有硬件检测器
    detector = get_hardware_detector()
    hardware_info = detector.detect()

    print(f"有GPU: {hardware_info.cuda_available}")
    print(f"GPU名称: {hardware_info.gpu_name}")
    print(f"GPU显存: {hardware_info.gpu_memory_mb}")
    print(f"CPU核心: {hardware_info.cpu_cores}")
    print(f"CPU名称: {hardware_info.cpu_name}")

    # 使用硬件优化器
    optimizer = get_hardware_optimizer()
    optimization_config = optimizer.get_optimization_config(hardware_info)

    print(f"\n优化配置:")
    print(f"  SenseVoice设备: {optimization_config.sensevoice_device}")
    print(f"  启用Demucs: {optimization_config.enable_demucs}")
    print(f"  Demucs模型: {optimization_config.demucs_model}")
    print(f"  说明: {optimization_config.note}")


def test_sentence_splitter():
    """测试分句算法"""
    print("\n=== 测试分句算法 ===")

    # 构造测试数据
    words = [
        WordTimestamp("今", 0.0, 0.2),
        WordTimestamp("天", 0.2, 0.4),
        WordTimestamp("天", 0.4, 0.6),
        WordTimestamp("气", 0.6, 0.8),
        WordTimestamp("很", 0.8, 1.0),
        WordTimestamp("好", 1.0, 1.2),
        WordTimestamp("。", 1.2, 1.3),
        WordTimestamp("我", 2.0, 2.2),  # 长停顿
        WordTimestamp("很", 2.2, 2.4),
        WordTimestamp("开", 2.4, 2.6),
        WordTimestamp("心", 2.6, 2.8),
        WordTimestamp("。", 2.8, 2.9),
    ]

    splitter = SentenceSplitter()
    sentences = splitter.split(words)

    print(f"输入: {len(words)}个字")
    print(f"输出: {len(sentences)}个句子")
    for i, sent in enumerate(sentences, 1):
        print(f"  句子{i}: {sent.text} ({sent.start:.1f}s - {sent.end:.1f}s)")


def test_sensevoice_service():
    """测试 SenseVoice 服务"""
    print("\n=== 测试 SenseVoice 服务 ===")

    service = get_sensevoice_service()

    # 加载模型
    print("加载模型...")
    success = service.load_model()
    print(f"加载结果: {'成功' if success else '失败'}")

    if success:
        print(f"模型信息: {service.get_model_info()}")

        # 卸载模型
        print("卸载模型...")
        service.unload_model()
        print("卸载完成")


if __name__ == "__main__":
    test_hardware_detection()
    test_sentence_splitter()
    test_sensevoice_service()

    print("\n=== Phase 1 测试完成 ===")
```

---

## 八、验收标准

- [ ] 硬件检测使用现有 `hardware_service.py`
- [ ] 硬件优化配置包含 SenseVoice 相关字段
- [ ] 分句算法可正确切分测试用例
- [ ] SenseVoice ONNX 服务可正常加载/卸载

---

## 九、注意事项

1. **复用现有服务**：
   - ✅ 使用 `get_hardware_detector()` 获取硬件信息
   - ✅ 使用 `get_hardware_optimizer()` 获取优化配置
   - ❌ 不要创建新的硬件检测器

2. **扩展而非替换**：
   - ✅ 在现有类中添加新字段
   - ✅ 在现有方法中添加新逻辑
   - ❌ 不要创建重复的类或函数

3. **向后兼容**：
   - 新增字段都有默认值
   - 不破坏现有功能

---

## 十、下一步

完成 Phase 1（修订版）后，进入 [Phase 2: 智能熔断集成](./02_Phase2_智能熔断集成.md)

（Phase 2 基本无需修改）
