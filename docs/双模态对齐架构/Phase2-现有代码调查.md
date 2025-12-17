# Phase 2 现有代码调查报告

## 调查概述

本报告深入调查了现有的音频处理相关代码，为 Phase 2 双模态对齐架构实施做准备。调查重点关注 Demucs 服务、VAD 实现、音频处理流程和数据模型等关键组件。

## 1. Demucs 服务实现调查

### 1.1 当前实现方式

**文件位置**: `backend/app/services/demucs_service.py`

#### 核心特性
- **整轨分离支持**: ✅ 完全支持，通过 `separate_vocals()` 方法处理整个音频文件
- **按需分离**: ✅ 支持，通过 `separate_vocals_segment()` 方法处理指定时间段
- **模型管理**: ✅ 支持动态切换模型（htdemucs、htdemucs_ft、mdx_extra、mdx_extra_q）

#### 显存管理策略
```python
# 关键实现 - DemucsService 类
class DemucsService:
    _instance = None                    # 单例模式
    _model = None                       # 当前加载的模型
    _model_name_loaded = None          # 记录当前模型名称
    _model_lock = None                 # 线程锁保证并发安全
```

**显存优化机制**:
1. **懒加载**: 模型首次使用时才加载
2. **模型切换**: 支持动态卸载和切换模型
3. **缓存策略**: 自动清理 `.partial` 下载残留文件
4. **设备管理**: 支持 CUDA/CPU 自动切换

#### 输入输出接口

**全局分离接口**:
```python
def separate_vocals(
    self,
    audio_path: str,
    output_path: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> str:
    # 返回: 分离后的人声文件路径
```

**按需分离接口**:
```python
def separate_vocals_segment(
    self,
    audio_array: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    buffer_sec: float = None,
    shifts: int = 1
) -> np.ndarray:
    # 返回: 分离后的人声片段数组
```

**Chunk 分离接口** (Phase 2 专用):
```python
def separate_chunk(
    self,
    audio: np.ndarray,
    model: str = None,
    sr: int = 16000
) -> np.ndarray:
    # 专门用于熔断回溯场景的简化接口
```

#### 关键方法和参数

**配置参数**:
```python
@dataclass
class DemucsConfig:
    model_name: str = "htdemucs"           # 默认使用快速模型
    device: str = "cuda"                   # 设备选择
    shifts: int = 2                        # 增强次数
    overlap: float = 0.5                   # 分段重叠率
    segment_length: int = 10               # 每段处理长度（秒）
    segment_buffer_sec: float = 2.0        # 分离时前后缓冲
```

**BGM 检测参数**:
```python
bgm_sample_duration: float = 10.0      # 采样片段长度
bgm_light_threshold: float = 0.02      # 轻微BGM阈值
bgm_heavy_threshold: float = 0.15      # 强BGM阈值
```

## 2. VAD 相关实现调查

### 2.1 在 transcription_service.py 中的 VAD 使用

**VAD 模型支持**:
```python
class VADMethod(Enum):
    SILERO = "silero"      # 默认，无需认证，速度快
    PYANNOTE = "pyannote"  # 可选，需要HF Token，精度更高
```

#### Silero VAD 集成方式
**实现位置**: `transcription_service.py:2088-2187`

**关键特性**:
- 使用项目内置 ONNX 模型 (`backend/assets/silero/silero_vad.onnx`)
- 基于 onnxruntime 推理，跨平台兼容
- 内存占用低 (~2MB)
- 速度快

**核心实现**:
```python
def _vad_silero(
    self,
    audio_array: np.ndarray,
    sr: int,
    vad_config: VADConfig
) -> List[Dict]:
    from silero_vad import get_speech_timestamps
    from silero_vad.utils_vad import OnnxWrapper

    # 使用内置ONNX模型
    builtin_model_path = PathlibPath(__file__).parent.parent / "assets" / "silero" / "silero_vad.onnx"
    model = OnnxWrapper(str(builtin_model_path), force_onnx_cpu=False)

    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=sr,
        threshold=vad_config.onset,                    # 默认0.5
        min_speech_duration_ms=vad_config.min_speech_duration_ms,   # 默认500ms
        min_silence_duration_ms=vad_config.min_silence_duration_ms, # 默认500ms
        return_seconds=False
    )
```

#### VAD 切分逻辑
**分段策略**:
1. 检测语音段时间戳
2. 合并相邻段确保不超过 `chunk_size` (默认30秒)
3. 返回分段元数据列表

**返回数据结构**:
```python
[
    {"index": 0, "start": 0.0, "end": 30.5, "mode": "memory"},
    {"index": 1, "start": 30.5, "end": 58.2, "mode": "memory"},
    ...
]
```

#### VAD 配置参数
```python
@dataclass
class VADConfig:
    method: VADMethod = VADMethod.SILERO
    onset: float = 0.5                     # 语音开始阈值
    offset: float = 0.4                    # 语音结束阈值
    chunk_size: int = 30                   # 最大段长（秒）
    min_speech_duration_ms: int = 500      # 最小语音段长度
    min_silence_duration_ms: int = 500     # 最小静音长度
```

## 3. 音频提取和处理调查

### 3.1 音频提取逻辑

**主要实现位置**:
- `transcription_service.py:2900` - `_extract_audio()` 方法
- `transcription_service.py:3760` - `_extract_audio_with_array()` 方法
- `audio_extractor.py` - 专用音频提取工具类

#### FFmpeg 音频降采样实现
```python
def _extract_audio(self, input_file: str, audio_out: str) -> bool:
    cmd = [
        ffmpeg_cmd, '-y', '-i', input_file,
        '-vn',                    # 仅音频
        '-ac', '1',               # 单声道
        '-ar', '16000',           # 16kHz 采样率
        '-acodec', 'pcm_s16le',   # PCM 编码
        audio_out
    ]
```

**音频格式标准**:
- 采样率: 16kHz (与 Whisper/SenseVoice 一致)
- 声道: 单声道
- 编码: PCM 16-bit
- 格式: WAV

### 3.2 音频文件临时存储位置

**临时文件管理**:
```python
# 在 _extract_audio_with_array 中使用
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
    tmp_path = tmp_file.name
```

**缓存位置**:
- Demucs 分离缓存: `models/demucs/` 目录
- PyTorch 模型缓存: 用户 `.cache/torch/hub/checkpoints/`
- 预置模型: `backend/models/pretrained/`

### 3.3 音频格式转换

**AudioExtractor 工具类特性**:
```python
class AudioExtractor:
    async def extract_fast(self, video_path: Path, output_path: Path,
                           audio_codec: str = 'aac') -> Path:
        # 优先复制流避免重新编码
        if audio_codec in ['aac', 'mp3']:
            return await self._copy_audio_stream(video_path, output_path)
        # 其他编码转为 WAV（用于 Whisper）
        return await self._convert_to_wav(video_path, output_path)
```

**支持格式转换**:
- 视频 → 音频 (AAC/MP3/M4A/WAV)
- 音频重采样到 16kHz
- 流复制避免重编码（当可能时）

## 4. AudioChunk 数据模型调查

### 4.1 数据结构定义

**文件位置**: `backend/app/models/circuit_breaker_models.py`

#### ChunkProcessState 核心数据模型
```python
@dataclass
class ChunkProcessState:
    """Chunk 处理状态（核心：保留原始音频引用）"""
    chunk_index: int
    start_time: float                          # Chunk 起始时间（秒）
    end_time: float                            # Chunk 结束时间（秒）

    # ========== 音频引用（关键！）==========
    original_audio: np.ndarray = None          # 原始音频片段（分离前）
    current_audio: np.ndarray = None           # 当前使用的音频片段（可能已分离）
    sample_rate: int = 16000                   # 采样率（默认 16kHz）

    # ========== 分离状态 ==========
    separation_level: SeparationLevel = SeparationLevel.NONE
    separation_model_used: Optional[str] = None  # 实际使用的模型名

    # ========== 熔断状态 ==========
    fuse_retry_count: int = 0                  # 熔断重试次数
    max_fuse_retry: int = 1                    # 最大重试次数
```

### 4.2 字段说明

**时间信息**:
- `start_time/end_time`: Chunk 在原始音频中的绝对时间点
- `chunk_index`: Chunk 序列索引

**音频数据**:
- `original_audio`: 保留原始音频引用，用于熔断回溯
- `current_audio`: 当前处理的音频（可能已分离）
- `sample_rate`: 统一使用 16kHz 采样率

**状态管理**:
- `separation_level`: 分离级别 (NONE/HTDEMUCS/MDX_EXTRA)
- `fuse_retry_count`: 熔断重试计数
- `max_fuse_retry`: 最大重试限制

### 4.3 分离级别枚举
```python
class SeparationLevel(Enum):
    """人声分离级别"""
    NONE = "none"              # 未分离
    HTDEMUCS = "htdemucs"      # 轻度分离
    MDX_EXTRA = "mdx_extra"    # 重度分离（最高级别）
```

## 5. 发现的问题和改进建议

### 5.1 现有代码优势

#### ✅ Demucs 服务
1. **完整的模型管理**: 支持动态切换、懒加载、显存优化
2. **灵活的分离模式**: 支持全局分离、按需分离、Chunk分离
3. **策略解析器**: 完善的 BGM 检测和分离策略决策
4. **缓存机制**: 智能缓存避免重复分离

#### ✅ VAD 系统
1. **双模型支持**: Silero (快速) + Pyannote (高精度)
2. **内置模型**: Silero 使用内置 ONNX，无需下载
3. **参数可调**: 支持阈值、时长等详细配置
4. **降级机制**: VAD 失败时自动降级到能量检测

#### ✅ 音频处理
1. **统一采样率**: 全链路 16kHz 标准化
2. **多格式支持**: FFmpeg 支持各种音视频格式
3. **性能优化**: 流复制避免不必要重编码
4. **内存管理**: 支持内存模式和硬盘模式切换

### 5.2 需要改进的问题

#### ⚠️ 音频数据一致性
**问题**: 不同模块间音频格式转换可能引入不一致
- transcription_service.py 使用 whisper_load_audio()
- demucs_service.py 使用自定义音频处理
- audio_extractor.py 使用 librosa

**建议**: 统一音频加载和预处理接口

#### ⚠️ 错误处理机制
**问题**: 部分异常处理过于宽泛，可能掩盖具体问题
```python
except Exception as e:
    self.logger.error(f"VAD分段失败: {e}")
    return self._energy_based_split(audio_array, sr, vad_config.chunk_size)
```

**建议**: 细化异常类型，提供更精确的错误信息

#### ⚠️ 内存使用监控
**问题**: 大文件处理时缺少实时内存监控
**建议**: 添加内存使用阈值和自动降级机制

### 5.3 Phase 2 实施注意事项

#### 🔧 需要适配的接口
1. **ChunkProcessState**: 已完美支持 Phase 2 需求
2. **DemucsService.separate_chunk()**: 专为熔断回溯设计
3. **频谱分析**: AudioSpectrumClassifier 支持实时分诊

#### 🔧 配置整合
需要整合以下配置文件:
- `config/demucs_tiers.json` - Demucs 分级配置
- 频谱阈值配置 - 需要创建
- 熔断参数配置 - 已在代码中实现

## 6. 可复用代码和需要重构部分

### 6.1 ✅ 可直接复用的代码

#### Demucs 服务 (95% 可复用)
```python
# 完全适配 Phase 2
DemucsService.separate_chunk()          # ✅ 专为 Chunk 设计
DemucsService.separate_vocals_segment() # ✅ 按需分离
SeparationStrategyResolver              # ✅ 策略决策
```

#### VAD 系统 (90% 可复用)
```python
# 完全适配 Phase 2
TranscriptionService._vad_silero()      # ✅ 内存 VAD
VADConfig                              # ✅ 配置管理
VADMethod                              # ✅ 模型选择
```

#### 数据模型 (100% 可复用)
```python
# 完美适配 Phase 2
ChunkProcessState                      # ✅ 核心数据结构
SeparationLevel                        # ✅ 分离级别
SpectrumDiagnosis                      # ✅ 频谱分诊
```

### 6.2 ⚙️ 需要轻微重构的代码

#### 音频处理接口 (需要统一)
```python
# 建议创建统一接口
class AudioProcessor:
    @staticmethod
    def load_audio(path: str, sr: int = 16000) -> np.ndarray

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray

    @staticmethod
    def ensure_format(audio: np.ndarray, sr: int = 16000) -> np.ndarray
```

#### 缓存管理 (需要扩展)
```python
# 建议扩展 DemucsService 的缓存机制
def cache_chunk_result(chunk_key: str, separated_audio: np.ndarray) -> None
def get_cached_chunk(chunk_key: str) -> Optional[np.ndarray]
```

### 6.3 🆕 需要新增的功能

#### 双模态对齐接口
```python
class DualModalAligner:
    def align_speech_video(self, chunks: List[ChunkProcessState],
                          video_features: np.ndarray) -> AlignmentResult
        pass

    def extract_video_features(self, video_path: str) -> np.ndarray
        pass
```

#### 对齐质量评估
```python
class AlignmentQualityAssessor:
    def assess_alignment_quality(self, alignment_result: AlignmentResult) -> float
        pass

    def detect_misalignment(self, chunks: List[ChunkProcessState]) -> List[int]
        pass
```

## 7. 总结和建议

### 7.1 现有代码成熟度评估

| 组件 | 成熟度 | Phase 2 适配度 | 备注 |
|------|--------|---------------|------|
| Demucs 服务 | ⭐⭐⭐⭐⭐ | 95% | 完全适配，几乎无需修改 |
| VAD 系统 | ⭐⭐⭐⭐⭐ | 90% | 需要统一错误处理 |
| 音频处理 | ⭐⭐⭐⭐ | 80% | 需要统一接口 |
| 数据模型 | ⭐⭐⭐⭐⭐ | 100% | 完美设计 |
| 频谱分析 | ⭐⭐⭐⭐ | 85% | 需要扩展功能 |

### 7.2 Phase 2 实施建议

#### 优先级 1 (核心功能)
1. **统一音频处理接口** - 创建 AudioProcessor 类
2. **完善 ChunkProcessState 状态管理** - 添加对齐状态字段
3. **实现双模态特征提取** - 视频特征提取接口

#### 优先级 2 (质量保障)
1. **增强错误处理** - 细化异常类型和处理策略
2. **添加性能监控** - 内存、GPU、处理时间监控
3. **完善缓存机制** - Chunk 级别的智能缓存

#### 优先级 3 (优化扩展)
1. **配置管理整合** - 统一所有 Phase 2 相关配置
2. **自动化测试** - 端到端测试套件
3. **性能基准测试** - 建立性能基准和回归测试

### 7.3 技术债务清理

1. **代码注释更新** - 部分函数注释过时
2. **类型注解完善** - 添加完整的类型注解
3. **文档同步** - 更新架构文档反映最新实现
4. **单元测试** - 关键模块单元测试覆盖

---

**报告生成时间**: 2025-12-10
**调查范围**: backend/app/services/, backend/app/models/, backend/app/utils/
**调查深度**: 源码级详细分析
**Phase 2 就绪度**: 85% (大部分代码可直接复用，少量需要适配)