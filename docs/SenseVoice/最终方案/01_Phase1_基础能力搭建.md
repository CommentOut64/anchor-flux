# Phase 1: 基础能力搭建

> 目标：实现 SenseVoice ONNX 服务、分句算法和硬件检测
>
> 工期：2-3天

---

## 一、任务清单

| 任务 | 文件 | 优先级 |
|------|------|--------|
| SenseVoice ONNX 服务 | `sensevoice_onnx_service.py` | P0 |
| 分句算法 | `sentence_splitter.py` | P0 |
| 硬件检测模块 | `hardware_detector.py` | P1 |
| 数据模型定义 | `sensevoice_models.py` | P0 |

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


@dataclass
class HardwareCapability:
    """硬件能力检测结果"""
    has_gpu: bool
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    recommended_config: dict = field(default_factory=dict)
```

---

## 三、文件2：硬件检测模块

**路径**: `backend/app/services/hardware_detector.py`

```python
"""
硬件能力检测模块
根据硬件条件生成推荐配置
"""
import logging
from typing import Dict
from ..models.sensevoice_models import HardwareCapability

logger = logging.getLogger(__name__)


class HardwareDetector:
    """硬件检测器"""

    @staticmethod
    def detect() -> HardwareCapability:
        """
        检测硬件能力

        Returns:
            HardwareCapability: 硬件能力信息
        """
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch 未安装，无法检测 GPU")
            return HardwareDetector._create_cpu_config()

        if not torch.cuda.is_available():
            logger.info("未检测到 GPU，使用 CPU 配置")
            return HardwareDetector._create_cpu_config()

        # 获取 GPU 信息
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_memory_gb = props.total_memory / (1024**3)

            logger.info(f"检测到 GPU: {gpu_name}, 显存: {gpu_memory_gb:.1f}GB")

            # 生成推荐配置
            recommended_config = HardwareDetector._generate_gpu_config(gpu_memory_gb)

            return HardwareCapability(
                has_gpu=True,
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_memory_gb,
                recommended_config=recommended_config
            )
        except Exception as e:
            logger.error(f"获取 GPU 信息失败: {e}")
            return HardwareDetector._create_cpu_config()

    @staticmethod
    def _create_cpu_config() -> HardwareCapability:
        """创建 CPU 配置"""
        return HardwareCapability(
            has_gpu=False,
            gpu_name="CPU",
            gpu_memory_gb=0.0,
            recommended_config={
                'sensevoice_device': 'cpu',
                'enable_demucs': False,          # 禁用人声分离
                'demucs_model': None,
                'max_concurrent_jobs': 1,
                'batch_size': 1,
                'note': '未检测到GPU，人声分离已禁用，转录使用CPU模式'
            }
        )

    @staticmethod
    def _generate_gpu_config(gpu_memory_gb: float) -> Dict:
        """
        根据 GPU 显存生成推荐配置

        Args:
            gpu_memory_gb: GPU 显存大小（GB）

        Returns:
            Dict: 推荐配置
        """
        if gpu_memory_gb < 4:
            # 低显存配置
            return {
                'sensevoice_device': 'cuda',
                'enable_demucs': 'on_demand',    # 仅强BGM时启用
                'demucs_model': 'htdemucs',      # 最轻量模型
                'max_concurrent_jobs': 1,
                'batch_size': 1,
                'note': '低显存模式，仅在检测到强BGM时使用htdemucs'
            }

        elif gpu_memory_gb < 8:
            # 中等显存配置
            return {
                'sensevoice_device': 'cuda',
                'enable_demucs': True,
                'demucs_model': 'htdemucs',
                'max_concurrent_jobs': 1,
                'batch_size': 2,
                'note': '标准配置，支持人声分离'
            }

        else:
            # 高显存配置
            return {
                'sensevoice_device': 'cuda',
                'enable_demucs': True,
                'demucs_model': 'mdx_extra',     # 高质量模型
                'max_concurrent_jobs': 2,
                'batch_size': 4,
                'note': '高性能配置，支持并发处理'
            }


# 全局单例
_hardware_capability: HardwareCapability = None


def get_hardware_capability() -> HardwareCapability:
    """获取硬件能力（单例）"""
    global _hardware_capability
    if _hardware_capability is None:
        _hardware_capability = HardwareDetector.detect()
    return _hardware_capability
```

---

## 四、文件3：分句算法

**路径**: `backend/app/services/sentence_splitter.py`

```python
"""
分句算法
将 SenseVoice 输出的字级时间戳切分为句级字幕
"""
import logging
from typing import List, Optional
from ..models.sensevoice_models import WordTimestamp, SentenceSegment

logger = logging.getLogger(__name__)


class SentenceSplitter:
    """
    分句算法 - 将字级时间戳切分为句级字幕

    切分条件（优先级从高到低）：
    1. 标点符号（。？！）必切
    2. 长停顿（>0.4s）必切
    3. 强制长度（5秒或30字）强制切分
    """

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()

        # 强制切分的标点符号
        self.terminal_punctuation = set('。？！?!')
        # 弱切分的标点符号（仅在句子过长时考虑）
        self.weak_punctuation = set('，,、；;：:')

        logger.info(f"分句算法已初始化，配置: {self.config}")

    def _default_config(self) -> dict:
        return {
            'pause_threshold': 0.4,       # 停顿切分阈值（秒）
            'max_duration': 5.0,          # 最大时长（秒）
            'max_chars': 30,              # 最大字符数
            'min_chars': 2,               # 最小字符数（避免单字成句）
        }

    def split(self, words: List[WordTimestamp]) -> List[SentenceSegment]:
        """
        将字级时间戳列表切分为句级字幕列表

        Args:
            words: SenseVoice 输出的字级时间戳列表

        Returns:
            List[SentenceSegment]: 句级字幕列表
        """
        if not words:
            return []

        sentences = []
        current_words = []
        current_start = words[0].start

        for i, word in enumerate(words):
            current_words.append(word)

            # 判断是否需要切分
            should_split, reason = self._should_split(
                current_words,
                word,
                words[i + 1] if i + 1 < len(words) else None
            )

            if should_split:
                # 生成句子
                sentence = self._create_sentence(current_words, current_start)
                if sentence:
                    sentences.append(sentence)
                    logger.debug(f"切分句子: {sentence.text} (原因: {reason})")

                # 重置
                current_words = []
                if i + 1 < len(words):
                    current_start = words[i + 1].start

        # 处理剩余的词
        if current_words:
            sentence = self._create_sentence(current_words, current_start)
            if sentence:
                sentences.append(sentence)

        logger.info(f"分句完成: {len(words)}个字 → {len(sentences)}个句子")
        return sentences

    def _should_split(
        self,
        current_words: List[WordTimestamp],
        current_word: WordTimestamp,
        next_word: Optional[WordTimestamp]
    ) -> tuple:
        """
        判断是否应该在当前词后切分

        Returns:
            (should_split: bool, reason: str)
        """
        text = ''.join(w.word for w in current_words)
        duration = current_word.end - current_words[0].start

        # 规则1: 终结标点符号必切
        if current_word.word in self.terminal_punctuation:
            return True, 'terminal_punctuation'

        # 规则2: 长停顿必切
        if next_word:
            pause = next_word.start - current_word.end
            if pause > self.config['pause_threshold']:
                return True, f'long_pause_{pause:.2f}s'

        # 规则3: 超过最大时长或字数，强制切分
        if duration >= self.config['max_duration']:
            return True, f'max_duration_{duration:.2f}s'

        if len(text) >= self.config['max_chars']:
            # 尝试在弱标点处切分
            for j in range(len(current_words) - 1, -1, -1):
                if current_words[j].word in self.weak_punctuation:
                    return True, 'max_chars_weak_punct'
            return True, 'max_chars_forced'

        return False, ''

    def _create_sentence(
        self,
        words: List[WordTimestamp],
        start: float
    ) -> Optional[SentenceSegment]:
        """创建句子段落"""
        if not words:
            return None

        text = ''.join(w.word for w in words)

        # 过滤过短的句子
        if len(text.strip()) < self.config['min_chars']:
            return None

        # 计算平均置信度
        avg_confidence = sum(w.confidence for w in words) / len(words)

        return SentenceSegment(
            text=text,
            start=start,
            end=words[-1].end,
            words=words,
            confidence=avg_confidence
        )
```

---

## 五、文件4：SenseVoice ONNX 服务

**路径**: `backend/app/services/sensevoice_onnx_service.py`

```python
"""
SenseVoice ONNX 推理服务
使用 ONNX Runtime 部署 SenseVoice (INT8 量化)
"""
import logging
import time
import gc
import numpy as np
from typing import Optional, List
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from ..models.sensevoice_models import (
    SenseVoiceONNXConfig,
    SenseVoiceResult,
    WordTimestamp
)

logger = logging.getLogger(__name__)


class SenseVoiceONNXService:
    """
    SenseVoice ONNX 推理服务

    功能:
    1. 模型加载/卸载
    2. ONNX 推理
    3. 结果解析
    4. 时间戳提取
    """

    def __init__(self, config: SenseVoiceONNXConfig = None):
        self.config = config or SenseVoiceONNXConfig()
        self.session = None
        self.device = None
        self.is_loaded = False

    def load_model(self) -> bool:
        """
        加载 ONNX 模型

        Returns:
            bool: 是否加载成功
        """
        if self.is_loaded:
            logger.warning("模型已加载，跳过")
            return True

        if ort is None:
            logger.error("onnxruntime 未安装，无法加载模型")
            return False

        try:
            start_time = time.time()

            # 检查模型文件
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                logger.error(f"模型文件不存在: {model_path}")
                return False

            # 配置 SessionOptions
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                if self.config.enable_graph_optimization
                else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            sess_options.intra_op_num_threads = self.config.num_threads

            # 选择执行提供者
            providers = self._get_providers()
            logger.info(f"尝试使用执行提供者: {providers}")

            # 创建 InferenceSession
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )

            self.device = self.session.get_providers()[0]
            self.is_loaded = True

            load_time = time.time() - start_time
            logger.info(
                f"SenseVoice ONNX 模型加载成功 "
                f"(设备: {self.device}, 耗时: {load_time:.2f}s)"
            )

            return True

        except Exception as e:
            logger.error(f"加载 SenseVoice ONNX 模型失败: {e}", exc_info=True)

            # 尝试 CPU fallback
            if self.config.fallback_to_cpu and self.config.use_gpu:
                logger.info("尝试 CPU fallback...")
                return self._load_cpu_fallback()

            return False

    def _get_providers(self) -> List[str]:
        """获取执行提供者列表"""
        providers = []

        if self.config.use_gpu:
            available = ort.get_available_providers()

            # 优先使用 CUDA
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
                logger.info("使用 CUDAExecutionProvider")

            # DirectML (Windows)
            elif 'DmlExecutionProvider' in available:
                providers.append('DmlExecutionProvider')
                logger.info("使用 DmlExecutionProvider")

        # 总是添加 CPU 作为 fallback
        providers.append('CPUExecutionProvider')

        return providers

    def _load_cpu_fallback(self) -> bool:
        """CPU 回退加载"""
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.config.num_threads

            self.session = ort.InferenceSession(
                self.config.model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )

            self.device = 'CPU'
            self.is_loaded = True

            logger.info("SenseVoice ONNX 模型已加载 (CPU模式)")
            return True

        except Exception as e:
            logger.error(f"CPU fallback 失败: {e}")
            return False

    def unload_model(self) -> None:
        """卸载模型释放资源"""
        if not self.is_loaded:
            return

        try:
            if self.session is not None:
                del self.session
                self.session = None

            self.is_loaded = False
            self.device = None

            # 垃圾回收
            gc.collect()

            logger.info("SenseVoice ONNX 模型已卸载")

        except Exception as e:
            logger.error(f"模型卸载失败: {e}")

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto"
    ) -> Optional[SenseVoiceResult]:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径
            language: 语言设置

        Returns:
            SenseVoiceResult: 转录结果
        """
        if not self.is_loaded:
            logger.error("模型未加载，请先调用 load_model()")
            return None

        try:
            # TODO: 实现 ONNX 推理
            # 1. 加载音频
            # 2. 预处理
            # 3. ONNX 推理
            # 4. 后处理

            logger.warning("ONNX 推理尚未实现，返回模拟结果")

            # 临时返回模拟结果
            return SenseVoiceResult(
                text="<|zh|><|Speech|><|NEUTRAL|>这是一个测试结果",
                text_clean="这是一个测试结果",
                confidence=0.85,
                words=[
                    WordTimestamp("这", 0.0, 0.2, 0.9),
                    WordTimestamp("是", 0.2, 0.4, 0.9),
                    WordTimestamp("一", 0.4, 0.6, 0.8),
                    WordTimestamp("个", 0.6, 0.8, 0.8),
                    WordTimestamp("测", 0.8, 1.0, 0.85),
                    WordTimestamp("试", 1.0, 1.2, 0.85),
                    WordTimestamp("结", 1.2, 1.4, 0.8),
                    WordTimestamp("果", 1.4, 1.6, 0.8),
                ],
                start=0.0,
                end=1.6,
                language="zh"
            )

        except Exception as e:
            logger.error(f"转录失败: {e}", exc_info=True)
            return None

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_path': self.config.model_path,
            'device': self.device,
            'is_loaded': self.is_loaded,
            'quantization': self.config.quantization
        }


# 全局单例
_sensevoice_service: Optional[SenseVoiceONNXService] = None


def get_sensevoice_service() -> SenseVoiceONNXService:
    """获取 SenseVoice 服务单例"""
    global _sensevoice_service
    if _sensevoice_service is None:
        _sensevoice_service = SenseVoiceONNXService()
    return _sensevoice_service
```

---

## 六、快速测试

创建测试脚本验证核心功能：

**路径**: `backend/test_phase1.py`

```python
"""
Phase 1 快速测试脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.hardware_detector import get_hardware_capability
from app.services.sentence_splitter import SentenceSplitter
from app.services.sensevoice_onnx_service import get_sensevoice_service
from app.models.sensevoice_models import WordTimestamp


def test_hardware_detector():
    """测试硬件检测"""
    print("\n=== 测试硬件检测 ===")
    hardware = get_hardware_capability()
    print(f"有GPU: {hardware.has_gpu}")
    print(f"GPU名称: {hardware.gpu_name}")
    print(f"GPU显存: {hardware.gpu_memory_gb:.1f}GB")
    print(f"推荐配置: {hardware.recommended_config}")


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
    test_hardware_detector()
    test_sentence_splitter()
    test_sensevoice_service()

    print("\n=== Phase 1 测试完成 ===")
```

---

## 七、验收标准

- [ ] 硬件检测可正确识别 GPU 状态和显存
- [ ] 分句算法可正确切分测试用例
- [ ] SenseVoice ONNX 服务可正常加载/卸载（即使模型文件不存在也能优雅处理）

---

## 八、注意事项

1. **ONNX 模型文件**：
   - 本阶段暂不实现真实的 ONNX 推理
   - `transcribe()` 方法返回模拟数据
   - Phase 3 时再完善推理逻辑

2. **硬件检测**：
   - 如果 PyTorch 未安装，会回退到 CPU 配置
   - 推荐配置仅供参考，可根据实际情况调整

3. **分句算法**：
   - 参数可通过 config 字典调整
   - 如果切分效果不理想，优先调整阈值参数

---

## 九、下一步

完成 Phase 1 后，进入 [Phase 2: 智能熔断集成](./02_Phase2_智能熔断集成.md)
