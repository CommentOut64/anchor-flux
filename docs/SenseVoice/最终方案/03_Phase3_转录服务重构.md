# Phase 3: 转录服务重构

> 目标：整合所有组件到转录服务，实现完整的 SenseVoice 转录流程
>
> 工期：3-4天

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

### 3.1 新增依赖导入

在文件顶部添加：

```python
from .sentence_splitter import SentenceSplitter
from .audio_circuit_breaker import AudioCircuitBreaker
from .fuse_breaker import FuseBreaker
from .sensevoice_onnx_service import get_sensevoice_service
from .hardware_detector import get_hardware_capability
from ..models.sensevoice_models import SentenceSegment
from ..models.circuit_breaker_models import SeparationLevel
```

### 3.2 初始化新组件

在 `TranscriptionService.__init__()` 中添加：

```python
def __init__(self):
    # ... 现有初始化代码 ...

    # 新增：Phase 1-2 组件
    self.sentence_splitter = SentenceSplitter()
    self.circuit_breaker = AudioCircuitBreaker()
    self.fuse_breaker = FuseBreaker()

    # 硬件检测
    self.hardware_capability = get_hardware_capability()
    self.logger.info(
        f"硬件能力: GPU={self.hardware_capability.has_gpu}, "
        f"显存={self.hardware_capability.gpu_memory_gb:.1f}GB"
    )
```

### 3.3 主处理流程（SenseVoice 模式）

新增方法：

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

```python
async def _vad_physical_split(
    self,
    audio_path: str,
    audio_array: np.ndarray,
    job: JobState
) -> List[dict]:
    """
    VAD 物理切分

    职责：
    1. 防幻觉：过滤纯静音
    2. 显存保护：切分为 15-30s chunks
    3. 并行加速：提供 batch 原材料

    Args:
        audio_path: 音频文件路径
        audio_array: 音频数组
        job: 任务状态

    Returns:
        List[dict]: VAD 片段列表
    """
    from .vad_service import get_vad_service

    vad_service = get_vad_service()

    # VAD 配置（物理切分模式）
    vad_config = {
        'min_speech_duration_ms': 250,
        'min_silence_duration_ms': 100,
        'max_segment_duration_s': 30.0,      # 显存保护
        'target_segment_duration_s': 15.0,   # 目标长度
        'padding_ms': 200
    }

    segments = vad_service.detect_speech(
        audio_path,
        config=vad_config
    )

    self.logger.info(f"VAD 物理切分: {len(segments)} 个片段")

    return segments
```

### 3.5 频谱预判

```python
async def _spectral_analysis(
    self,
    vad_segments: List[dict],
    audio_array: np.ndarray,
    job: JobState
) -> List:
    """
    频谱预判

    Args:
        vad_segments: VAD 片段列表
        audio_array: 完整音频数组
        job: 任务状态

    Returns:
        List[CircuitBreakerDecision]: 决策列表
    """
    import soundfile as sf

    self.circuit_breaker.reset()
    decisions = []

    for i, seg in enumerate(vad_segments):
        # 提取片段音频
        start_sample = int(seg['start'] * 16000)
        end_sample = int(seg['end'] * 16000)
        audio_chunk = audio_array[start_sample:end_sample]

        # 频谱分析
        decision = self.circuit_breaker.analyze_segment(audio_chunk, i)
        decisions.append(decision)

        # 进度更新
        progress = (i + 1) / len(vad_segments)
        self._update_progress(
            job, 'bgm_detect', progress,
            f'频谱分析 {i+1}/{len(vad_segments)}'
        )

    return decisions
```

### 3.6 智能分离

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

    # 检查硬件配置
    if not self.hardware_capability.has_gpu:
        self.logger.warning("无 GPU，跳过人声分离")
        return vad_segments

    # 加载 Demucs 模型
    demucs_service = get_demucs_service()
    model_name = self.hardware_capability.recommended_config.get('demucs_model', 'htdemucs')

    if model_name is None:
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

```python
async def _transcribe_with_sensevoice(
    self,
    segments: List[dict],
    audio_array: np.ndarray,
    job: JobState
) -> List[SentenceSegment]:
    """
    SenseVoice 转录 + 分句

    Args:
        segments: 片段列表
        audio_array: 音频数组
        job: 任务状态

    Returns:
        List[SentenceSegment]: 句级字幕列表
    """
    sensevoice_service = get_sensevoice_service()

    # 加载模型
    if not sensevoice_service.is_loaded:
        sensevoice_service.load_model()

    all_sentences = []

    for i, seg in enumerate(segments):
        # 转录单个片段
        result = sensevoice_service.transcribe(
            seg.get('file', seg.get('audio_path')),
            language='auto'
        )

        if result is None:
            self.logger.warning(f"片段 {i} 转录失败")
            continue

        # 分句（关键！）
        sentences = self.sentence_splitter.split(result.words)

        # 逐句 SSE 推送
        for sentence in sentences:
            await self._push_sse_sentence(job, sentence)
            all_sentences.append(sentence)

        # 进度更新
        progress = (i + 1) / len(segments)
        self._update_progress(
            job, 'transcribe', progress,
            f'转录 {i+1}/{len(segments)}'
        )

    return all_sentences
```

### 3.8 熔断 + 补刀

```python
async def _fuse_and_retry(
    self,
    sentences: List[SentenceSegment],
    segments: List[dict],
    audio_array: np.ndarray,
    job: JobState
) -> List[SentenceSegment]:
    """
    熔断检查 + Whisper 补刀

    Args:
        sentences: 句级字幕列表
        segments: 原始片段列表
        audio_array: 音频数组
        job: 任务状态

    Returns:
        List[SentenceSegment]: 最终字幕列表
    """
    self.fuse_breaker.reset()

    retry_queue = []
    final_results = []

    for i, sentence in enumerate(sentences):
        # 熔断决策
        decision = self.fuse_breaker.decide_action(
            segment_id=i,
            confidence=sentence.confidence,
            event_tag=None,  # TODO: 从 SenseVoice 结果提取
            current_separation_level=SeparationLevel.NONE
        )

        if decision.action.value == 'accept':
            # 接受结果
            final_results.append(sentence)

        elif decision.action.value == 'upgrade_separation':
            # 升级分离模型（TODO: 实现）
            self.logger.info(f"句子 {i}: 需要升级分离模型")
            retry_queue.append((i, sentence))

        elif decision.action.value == 'whisper_retry':
            # Whisper 补刀
            self.logger.info(f"句子 {i}: 需要 Whisper 补刀")
            retry_queue.append((i, sentence))

    # 执行补刀
    if retry_queue:
        self._update_progress(job, 'retry', 0, f'Whisper 补刀 {len(retry_queue)} 个句子...')

        retried_sentences = await self._whisper_retry(
            retry_queue, segments, audio_array, job
        )

        final_results.extend(retried_sentences)

        self._update_progress(job, 'retry', 1, '补刀完成')

    # 按时间排序
    final_results.sort(key=lambda s: s.start)

    return final_results
```

### 3.9 Whisper 补刀

```python
async def _whisper_retry(
    self,
    retry_queue: List[tuple],
    segments: List[dict],
    audio_array: np.ndarray,
    job: JobState
) -> List[SentenceSegment]:
    """
    Whisper 补刀

    Args:
        retry_queue: 需要补刀的句子队列
        segments: 原始片段列表
        audio_array: 音频数组
        job: 任务状态

    Returns:
        List[SentenceSegment]: 补刀后的句子列表
    """
    # TODO: 实现 Whisper 补刀逻辑
    # 1. 加载 Whisper 模型
    # 2. 根据句子时间范围提取音频
    # 3. Whisper 转录
    # 4. 返回新句子

    self.logger.warning("Whisper 补刀尚未实现，返回原句子")

    return [sentence for _, sentence in retry_queue]
```

### 3.10 句级 SSE 推送

```python
async def _push_sse_sentence(self, job: JobState, sentence: SentenceSegment):
    """
    推送句级 SSE 事件

    Args:
        job: 任务状态
        sentence: 句子段落
    """
    try:
        from .sse_service import get_sse_service

        sse_service = get_sse_service()

        event_data = {
            'text': sentence.text,
            'start': sentence.start,
            'end': sentence.end,
            'confidence': sentence.confidence,
            'is_final': True,
            'source': 'sensevoice'
        }

        await sse_service.push_event(
            job_id=job.job_id,
            event='sentence',
            data=event_data
        )

    except Exception as e:
        self.logger.error(f"SSE 推送失败: {e}")
```

### 3.11 生成字幕

```python
async def _generate_subtitle_from_sentences(
    self,
    sentences: List[SentenceSegment],
    job: JobState
):
    """
    从句级字幕生成 SRT 文件

    Args:
        sentences: 句级字幕列表
        job: 任务状态
    """
    from .subtitle_generator import SubtitleGenerator

    generator = SubtitleGenerator()

    # 转换格式
    subtitle_items = []
    for i, sentence in enumerate(sentences, 1):
        subtitle_items.append({
            'index': i,
            'start': sentence.start,
            'end': sentence.end,
            'text': sentence.text
        })

    # 生成 SRT
    output_path = job.output_path
    generator.generate_srt(subtitle_items, output_path)

    self.logger.info(f"字幕已生成: {output_path}")
```

---

## 四、模型管理集成

修改 `backend/app/services/model_preload_manager.py`，新增 SenseVoice 管理：

```python
class ModelPreloadManager:
    def __init__(self, config=None):
        # ... 现有代码 ...

        # 新增：SenseVoice 缓存
        self._sensevoice_service = None

    def get_sensevoice_model(self):
        """获取 SenseVoice 模型"""
        if self._sensevoice_service is None:
            from .sensevoice_onnx_service import get_sensevoice_service
            self._sensevoice_service = get_sensevoice_service()

            if not self._sensevoice_service.is_loaded:
                self._sensevoice_service.load_model()

        return self._sensevoice_service

    def unload_sensevoice(self):
        """卸载 SenseVoice 模型"""
        if self._sensevoice_service is not None:
            self._sensevoice_service.unload_model()
            self._sensevoice_service = None

    def clear_all(self):
        """清空所有缓存"""
        # ... 现有代码 ...

        # 卸载 SenseVoice
        self.unload_sensevoice()
```

---

## 五、快速测试

创建端到端测试脚本：

**路径**: `backend/test_phase3.py`

```python
"""
Phase 3 端到端测试脚本
"""
import sys
from pathlib import Path
import asyncio

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.transcription_service import TranscriptionService
from app.models.job_models import JobState, JobSettings


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
    asyncio.run(test_sensevoice_pipeline())

    print("\n=== Phase 3 测试完成 ===")
```

---

## 六、验收标准

- [ ] 完整流程可运行（即使部分功能未完全实现）
- [ ] VAD 物理切分正常工作
- [ ] 频谱预判可正确识别 BGM
- [ ] 分句算法输出句级粒度字幕
- [ ] 熔断机制正确触发
- [ ] 字幕文件成功生成

---

## 七、注意事项

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

---

## 八、下一步

完成 Phase 3 后，进入 [Phase 4: 前端适配](./04_Phase4_前端适配.md)
