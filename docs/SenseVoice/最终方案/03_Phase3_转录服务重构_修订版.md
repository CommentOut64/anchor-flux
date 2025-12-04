# Phase 3: 转录服务重构（修订版）

> 目标：整合所有组件到转录服务，实现完整的 SenseVoice 转录流程
>
> 工期：3-4天

---

## ⚠️ 重要修订

- ❌ **删除**：不使用 `hardware_detector.py`（不存在）
- ✅ **复用**：使用现有的 `hardware_service.py`
- ✅ **修正**：SSE 推送使用 `get_sse_manager()` 和 `broadcast_sync()`
- ✅ **修正**：模型预加载使用现有 `_global_lock`

---

## 一、任务清单

| 任务 | 文件 | 优先级 |
|------|------|--------|
| VAD 物理切分重构 | `transcription_service.py` | P0 |
| SenseVoice 转录流程 | `transcription_service.py` | P0 |
| 分句集成 | `transcription_service.py` | P0 |
| 熔断逻辑集成 | `transcription_service.py` | P0 |
| 句级 SSE 推送 | `transcription_service.py` | P1 |
| Whisper 补刀机制 | `transcription_service.py` | P1 |
| 模型管理集成 | `model_preload_manager.py` | P1 |

---

## 二、整体流程

```
1. 音频提取
   ↓
2. VAD 物理切分（15-30s chunks）
   ↓
3. 频谱预判（AudioCircuitBreaker）
   ↓
4. 按需人声分离（Demucs）
   ↓
5. SenseVoice 转录
   ↓
6. 分句算法（SentenceSplitter）
   ↓
7. 句级 SSE 推送
   ↓
8. 置信度评估
   ↓
9. 熔断决策（FuseBreaker）
   ↓
10. [低置信度] → 升级分离 或 Whisper 补刀
   ↓
11. 生成字幕
```

---

## 三、核心修改：transcription_service.py

### 3.1 新增依赖导入（⚠️ 修订）

在文件顶部添加：

```python
from .sentence_splitter import SentenceSplitter
from .audio_circuit_breaker import AudioCircuitBreaker
from .fuse_breaker import FuseBreaker
from .sensevoice_onnx_service import get_sensevoice_service
# ❌ 错误：from .hardware_detector import get_hardware_capability
# ✅ 正确：使用现有硬件服务
from .hardware_service import get_hardware_detector, get_hardware_optimizer
from .sse_service import get_sse_manager  # ✅ 修正：使用 SSE 管理器

from ..models.sensevoice_models import SentenceSegment
from ..models.circuit_breaker_models import SeparationLevel
```

### 3.2 初始化新组件（⚠️ 修订）

在 `TranscriptionService.__init__()` 中添加：

```python
def __init__(self):
    # ... 现有初始化代码 ...

    # 新增：Phase 1-2 组件
    self.sentence_splitter = SentenceSplitter()
    self.circuit_breaker = AudioCircuitBreaker()
    self.fuse_breaker = FuseBreaker()

    # ❌ 错误：硬件检测
    # self.hardware_capability = get_hardware_capability()

    # ✅ 正确：使用现有硬件服务
    detector = get_hardware_detector()
    self.hardware_info = detector.detect()

    optimizer = get_hardware_optimizer()
    self.optimization_config = optimizer.get_optimization_config(self.hardware_info)

    self.logger.info(
        f"硬件能力: GPU={self.hardware_info.cuda_available}, "
        f"显存={max(self.hardware_info.gpu_memory_mb or [0])/1024:.1f}GB"
    )
```

### 3.3 主处理流程（SenseVoice 模式）

新增方法（与原版基本相同，仅修改硬件检测部分）：

```python
async def _process_video_sensevoice(self, job: JobState):
    """
    SenseVoice 主处理流程

    Args:
        job: 任务状态对象
    """
    try:
        # 1. 音频提取
        self._update_progress(job, 'extract', 0, '提取音频...')
        audio_path, audio_array = await self._extract_audio_with_array(job)
        self._update_progress(job, 'extract', 1, '音频提取完成')

        # 2. VAD 物理切分
        self._update_progress(job, 'vad', 0, 'VAD 物理切分...')
        vad_segments = await self._vad_physical_split(audio_path, audio_array, job)
        self._update_progress(job, 'vad', 1, f'VAD 切分完成，{len(vad_segments)} 个片段')

        # 3. 频谱预判
        self._update_progress(job, 'bgm_detect', 0, 'BGM 智能检测...')
        circuit_decisions = await self._spectral_analysis(vad_segments, audio_array, job)
        segments_to_separate = sum(1 for d in circuit_decisions if d.should_separate)
        self._update_progress(
            job, 'bgm_detect', 1,
            f'检测完成，{segments_to_separate}/{len(vad_segments)} 需分离'
        )

        # 4. 按需人声分离
        if segments_to_separate > 0:
            processed_segments = await self._smart_separation(
                vad_segments, circuit_decisions, audio_array, job
            )
        else:
            processed_segments = vad_segments

        # 5. SenseVoice 转录 + 分句
        self._update_progress(job, 'transcribe', 0, '开始转录...')
        transcript_results = await self._transcribe_with_sensevoice(
            processed_segments, audio_array, job
        )
        self._update_progress(job, 'transcribe', 1, f'转录完成，{len(transcript_results)} 个句子')

        # 6. 熔断检查 + Whisper 补刀
        final_results = await self._fuse_and_retry(
            transcript_results, processed_segments, audio_array, job
        )

        # 7. 生成字幕
        self._update_progress(job, 'srt', 0, '生成字幕...')
        await self._generate_subtitle_from_sentences(final_results, job)
        self._update_progress(job, 'srt', 1, '字幕生成完成')

        # 8. 完成
        job.status = 'completed'
        self._update_progress(job, 'complete', 1, '处理完成')

    except Exception as e:
        self.logger.error(f"SenseVoice 处理失败: {e}", exc_info=True)
        job.status = 'failed'
        job.error = str(e)
        raise
```

### 3.4 VAD 物理切分

（与原版相同，无需修改）

### 3.5 频谱预判

（与原版相同，无需修改）

### 3.6 智能分离（⚠️ 修订）

```python
async def _smart_separation(
    self,
    vad_segments: List[dict],
    circuit_decisions: List,
    audio_array: np.ndarray,
    job: JobState
) -> List[dict]:
    """
    智能分离（按需）

    Args:
        vad_segments: VAD 片段列表
        circuit_decisions: 频谱决策列表
        audio_array: 音频数组
        job: 任务状态

    Returns:
        List[dict]: 处理后的片段列表
    """
    from .demucs_service import get_demucs_service

    # 收集需要分离的片段
    segments_to_separate = [
        (i, seg) for i, (seg, decision) in enumerate(zip(vad_segments, circuit_decisions))
        if decision.should_separate
    ]

    if not segments_to_separate:
        self.logger.info("无需人声分离")
        return vad_segments

    # ❌ 错误：检查硬件配置
    # if not self.hardware_capability.has_gpu:

    # ✅ 正确：使用现有硬件信息
    if not self.hardware_info.cuda_available:
        self.logger.warning("无 GPU，跳过人声分离")
        return vad_segments

    # 加载 Demucs 模型
    demucs_service = get_demucs_service()

    # ❌ 错误：获取推荐模型
    # model_name = self.hardware_capability.recommended_config.get('demucs_model', 'htdemucs')

    # ✅ 正确：使用优化配置
    model_name = self.optimization_config.demucs_model

    if model_name is None or not self.optimization_config.enable_demucs:
        self.logger.warning("硬件配置禁用人声分离")
        return vad_segments

    demucs_service.set_model(model_name)

    self._update_progress(job, 'demucs', 0, f'人声分离 ({model_name})...')

    # 逐片段分离
    processed_segments = vad_segments.copy()

    for idx, (seg_idx, seg) in enumerate(segments_to_separate):
        # 分离人声
        separated_path = await self._separate_segment(
            seg, audio_array, demucs_service
        )

        # 更新片段路径
        processed_segments[seg_idx]['file'] = separated_path
        processed_segments[seg_idx]['separated'] = True

        # 进度更新
        progress = (idx + 1) / len(segments_to_separate)
        self._update_progress(
            job, 'demucs', progress,
            f'人声分离 {idx+1}/{len(segments_to_separate)}'
        )

    # 卸载 Demucs
    demucs_service.unload_model()

    return processed_segments
```

### 3.7 SenseVoice 转录 + 分句

（与原版相同，无需修改）

### 3.8 熔断 + 补刀

（与原版相同，无需修改）

### 3.9 Whisper 补刀

（与原版相同，无需修改）

### 3.10 句级 SSE 推送（⚠️ 重要修订）

```python
async def _push_sse_sentence(self, job: JobState, sentence: SentenceSegment):
    """
    推送句级 SSE 事件

    Args:
        job: 任务状态
        sentence: 句子段落
    """
    try:
        # ❌ 错误：使用不存在的 API
        # from .sse_service import get_sse_service
        # sse_service = get_sse_service()
        # await sse_service.push_event(...)

        # ✅ 正确：使用现有 SSE 管理器
        from .sse_service import get_sse_manager

        sse_manager = get_sse_manager()

        event_data = {
            'text': sentence.text,
            'start': sentence.start,
            'end': sentence.end,
            'confidence': sentence.confidence,
            'is_final': True,
            'source': 'sensevoice'
        }

        # ✅ 正确：使用 broadcast_sync（因为可能在后台线程调用）
        sse_manager.broadcast_sync(
            channel_id=f"job:{job.job_id}",
            event='sentence',
            data=event_data
        )

    except Exception as e:
        self.logger.error(f"SSE 推送失败: {e}")
```

### 3.11 生成字幕

（与原版相同，无需修改）

---

## 四、模型管理集成（⚠️ 重要修订）

修改 `backend/app/services/model_preload_manager.py`，新增 SenseVoice 管理：

```python
class ModelPreloadManager:
    def __init__(self, config=None):
        # ... 现有代码 ...

        # 新增：SenseVoice 缓存（单例模式）
        self._sensevoice_service = None

    def get_sensevoice_model(self):
        """
        获取 SenseVoice 模型（单例）

        ✅ 使用现有的全局锁保证线程安全
        """
        with self._global_lock:  # ✅ 使用现有锁
            if self._sensevoice_service is None:
                from .sensevoice_onnx_service import get_sensevoice_service
                self._sensevoice_service = get_sensevoice_service()

                if not self._sensevoice_service.is_loaded:
                    self._sensevoice_service.load_model()

                # 更新缓存版本
                self._preload_status["cache_version"] = int(time.time())
                self.logger.info("SenseVoice 模型已加载到缓存")

            return self._sensevoice_service

    def unload_sensevoice(self):
        """
        卸载 SenseVoice 模型

        ✅ 使用现有的全局锁保证线程安全
        """
        with self._global_lock:  # ✅ 使用现有锁
            if self._sensevoice_service is not None:
                self._sensevoice_service.unload_model()
                self._sensevoice_service = None

                # 更新缓存版本
                self._preload_status["cache_version"] = int(time.time())
                self.logger.info("SenseVoice 模型已卸载")

    def clear_cache(self):
        """清空所有缓存（扩展）"""
        with self._global_lock:
            # ... 现有清理代码 ...

            # 新增：清理 SenseVoice
            self.unload_sensevoice()

            self.logger.info("所有模型缓存已清空")
```

**重要说明**：
- ✅ 必须使用现有的 `self._global_lock` 而非创建新锁
- ✅ 必须更新 `self._preload_status["cache_version"]` 以保持统一
- ✅ 遵循现有的日志记录规范

---

## 五、进度更新方法修订（⚠️ 重要）

修改 `_update_progress` 方法以支持动态权重：

```python
def _update_progress(self, job, phase, phase_ratio, message=""):
    """
    更新进度（支持动态权重）

    ✅ 使用动态权重计算，适配 SenseVoice 引擎
    """
    from core.config import config

    # 动态获取引擎
    engine = getattr(job.settings, 'engine', 'whisperx')

    # 动态计算权重
    phase_weights = config.calculate_dynamic_weights(
        engine=engine,
        total_segments=getattr(job, 'total_segments', 100),
        segments_to_separate=getattr(job, 'segments_to_separate', 0),
        segments_to_retry=getattr(job, 'segments_to_retry', 0)
    )

    # 计算累计进度（使用动态权重）
    phase_order = ['pending', 'extract', 'bgm_detect', 'demucs_global', 'split', 'transcribe', 'align', 'srt', 'complete']

    try:
        current_phase_index = phase_order.index(phase)
        done_phases = phase_order[:current_phase_index]
    except ValueError:
        # 如果阶段不在标准列表中（如 'retry'），直接跳过
        done_phases = []

    done_weight = sum(phase_weights.get(p, 0) for p in done_phases)
    current_weight = phase_weights.get(phase, 0) * phase_ratio
    total_weight = 100

    job.progress = round((done_weight + current_weight) / total_weight * 100, 1)
    job.phase = phase
    job.phase_percent = round(phase_ratio * 100, 1)
    job.message = message

    # ✅ 正确：推送 SSE（使用现有管理器）
    from .sse_service import get_sse_manager

    sse_manager = get_sse_manager()
    sse_manager.broadcast_sync(
        channel_id=f"job:{job.job_id}",
        event="progress",
        data={
            "phase": phase,
            "progress": job.progress,
            "phase_percent": job.phase_percent,
            "message": message
        }
    )
```

---

## 六、快速测试（修订版）

创建端到端测试脚本：

**路径**: `backend/test_phase3_revised.py`

```python
"""
Phase 3 端到端测试脚本（修订版）
"""
import sys
from pathlib import Path
import asyncio

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.transcription_service import TranscriptionService
from app.models.job_models import JobState, JobSettings
# ✅ 测试现有硬件服务
from app.services.hardware_service import get_hardware_detector, get_hardware_optimizer


async def test_hardware_integration():
    """测试硬件服务集成"""
    print("\n=== 测试硬件服务集成 ===")

    detector = get_hardware_detector()
    hardware_info = detector.detect()

    print(f"GPU 可用: {hardware_info.cuda_available}")
    print(f"GPU 名称: {hardware_info.gpu_name}")
    print(f"GPU 显存: {hardware_info.gpu_memory_mb}")

    optimizer = get_hardware_optimizer()
    config = optimizer.get_optimization_config(hardware_info)

    print(f"\n优化配置:")
    print(f"  SenseVoice 设备: {config.sensevoice_device}")
    print(f"  启用 Demucs: {config.enable_demucs}")
    print(f"  Demucs 模型: {config.demucs_model}")
    print(f"  说明: {config.note}")


async def test_sensevoice_pipeline():
    """测试 SenseVoice 完整流程"""
    print("\n=== 测试 SenseVoice 完整流程 ===")

    # 创建测试任务
    job = JobState(
        job_id="test_001",
        input_path="test_data/test_video.mp4",
        output_path="test_data/test_output.srt",
        settings=JobSettings(
            engine='sensevoice'
        )
    )

    # 创建转录服务
    service = TranscriptionService()

    try:
        # 执行转录
        await service._process_video_sensevoice(job)

        print(f"\n任务状态: {job.status}")
        print(f"输出文件: {job.output_path}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_hardware_integration())
    asyncio.run(test_sensevoice_pipeline())

    print("\n=== Phase 3 测试完成 ===")
```

---

## 七、验收标准

- [ ] 完整流程可运行（即使部分功能未完全实现）
- [ ] VAD 物理切分正常工作
- [ ] 频谱预判可正确识别 BGM
- [ ] 分句算法输出句级粒度字幕
- [ ] 熔断机制正确触发
- [ ] 字幕文件成功生成
- [ ] **使用现有硬件服务，不创建重复代码**
- [ ] **SSE 推送使用正确的 API**
- [ ] **模型预加载使用现有全局锁**

---

## 八、注意事项

1. **渐进实现**：
   - 先实现核心流程框架
   - 部分功能可暂时用 TODO 标记
   - 确保主流程可运行

2. **显存管理**：
   - 确保模型严格串行化加载
   - Demucs 用完立即卸载
   - SenseVoice 和 Whisper 不同时加载

3. **错误处理**：
   - 每个步骤都要有异常捕获
   - 失败后更新任务状态
   - 记录详细日志

4. **⚠️ 代码复用原则**（关键）：
   - ✅ 使用 `get_hardware_detector()` 而非创建新检测器
   - ✅ 使用 `get_sse_manager()` 而非 `get_sse_service()`
   - ✅ 使用 `broadcast_sync()` 而非 `push_event()`
   - ✅ 扩展现有类而非创建新类

---

## 九、下一步

完成 Phase 3（修订版）后，进入 [Phase 4: 前端适配（修订版）](./04_Phase4_前端适配_修订版.md)
