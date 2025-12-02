# Demucs人声分离集成 - 分段实现计划

本文档详细规划Demucs音频分离功能的分阶段实现路径，确保每个阶段都可以独立开发、测试和验证。

---

## 总体架构与原则

### 设计原则

1. **渐进式集成**：每个阶段都是可独立测试的功能单元
2. **向后兼容**：不影响现有功能，Demucs功能可开关
3. **故障降级**：Demucs失败时自动降级为原始音频
4. **代码复用**：最大化利用现有的checkpoint、SSE、配置机制
5. **性能优先**：内存模式优先，合理管理GPU显存

### 集成点位置

在现有的5阶段流程中插入Demucs处理：

```
【现有流程】
1. extract (5%) → 音频提取 → audio.wav
2. split (5%)   → VAD分段
3. transcribe (60%) → 逐段转录
4. align (20%)  → 批次对齐
5. srt (10%)    → 生成字幕

【新流程】
1. extract (5%) → 音频提取 → audio.wav
2. [NEW] bgm_detect (2%) → BGM检测（采样检测）
3. [NEW] demucs_global (8%) → 全局人声分离（可选）
4. split (5%) → VAD分段（使用原始或分离后的音频）
5. transcribe (50%) → 逐段转录（支持按需分离+熔断）
6. align (20%) → 批次对齐
7. srt (10%) → 生成字幕
```

---

## 阶段划分

### Phase 1: 基础服务层 (DemucsService)

**目标**: 创建独立的Demucs服务，提供三种核心功能

**工作量估计**: 1-2天

#### 1.1 创建DemucsService基础框架

**文件**: `backend/app/services/demucs_service.py`

**功能清单**:
- [ ] 基础类结构
  - [ ] 单例模式
  - [ ] 懒加载模型
  - [ ] 线程锁
  - [ ] 日志配置
- [ ] 配置数据类
  - [ ] `DemucsConfig`: 模型配置（model_name, device, shifts等）
  - [ ] `BGMLevel`: 枚举（NONE, LIGHT, HEAVY）
- [ ] 模型管理
  - [ ] `_load_model()`: 加载htdemucs模型
  - [ ] `unload_model()`: 卸载模型释放显存
  - [ ] 模型缓存管理

**依赖安装**:
```python
# requirements.txt
demucs>=4.0.0
```

**测试验证**:
```python
def test_demucs_service_init():
    """测试服务初始化"""
    from services.demucs_service import get_demucs_service
    demucs = get_demucs_service()
    assert demucs is not None
    assert demucs.config.model_name == "htdemucs"
```

#### 1.2 实现全局人声分离功能

**方法**: `separate_vocals(audio_path, output_path, progress_callback)`

**功能清单**:
- [ ] 音频加载（使用demucs.audio.AudioFile）
- [ ] 人声分离处理（apply_model）
- [ ] 提取vocals源（sources[vocals_idx]）
- [ ] 保存分离结果
- [ ] 缓存机制（基于文件hash + mtime）
- [ ] 进度回调支持

**测试验证**:
```python
def test_separate_vocals():
    """测试全局人声分离"""
    demucs = get_demucs_service()

    # 使用测试音频
    output = demucs.separate_vocals("test_audio.wav")

    # 验证输出文件
    assert os.path.exists(output)

    # 验证音频质量（采样率、声道数）
    import soundfile as sf
    data, sr = sf.read(output)
    assert sr == 44100 or sr == 16000
```

#### 1.3 实现按需分离功能（内存模式）

**方法**: `separate_vocals_segment(audio_array, sr, start_sec, end_sec, buffer_sec)`

**功能清单**:
- [ ] 时间范围计算（含缓冲区）
- [ ] 音频切片提取
- [ ] 双声道处理（mono → stereo）
- [ ] 采样率转换（如需要）
- [ ] 人声分离处理
- [ ] 去除缓冲区，返回精确片段
- [ ] 转为单声道（Whisper要求）

**关键特性**:
- 支持内存数组输入（避免磁盘I/O）
- 自动添加前后缓冲（减少边界失真）
- Zero-copy切片（性能优化）

**测试验证**:
```python
def test_separate_vocals_segment():
    """测试按需分离"""
    import numpy as np
    demucs = get_demucs_service()

    # 模拟10秒音频
    sr = 16000
    audio = np.random.randn(160000).astype(np.float32)

    # 分离2-5秒
    vocals = demucs.separate_vocals_segment(
        audio, sr=sr, start_sec=2.0, end_sec=5.0
    )

    # 验证长度（约3秒）
    assert abs(len(vocals) - 48000) < 1600
```

#### 1.4 实现BGM检测功能（分位数采样）

**方法**: `detect_background_music_level(audio_path, audio_array, sr, duration_sec)`

**功能清单**:
- [ ] 分位数采样点计算（15%, 50%, 85%）
- [ ] 逐采样点分离+能量计算
- [ ] `_calculate_bgm_ratio()`: 计算BGM能量占比
  - RMS能量计算
  - 比例公式: `1.0 - (rms_vocal / rms_original)`
- [ ] 决策逻辑
  - max_ratio > 0.6 → HEAVY
  - max_ratio > 0.2 → LIGHT
  - 否则 → NONE
- [ ] 异常处理（采样失败、音频过短）

**测试验证**:
```python
def test_bgm_detection():
    """测试BGM检测"""
    from services.demucs_service import BGMLevel
    demucs = get_demucs_service()

    # 测试纯对话
    level, ratios = demucs.detect_background_music_level("dialogue.wav")
    assert level == BGMLevel.NONE
    assert len(ratios) == 3  # 三个采样点

    # 测试MV
    level, ratios = demucs.detect_background_music_level("music_video.wav")
    assert level == BGMLevel.HEAVY
    assert max(ratios) > 0.6
```

---

### Phase 2: 配置扩展与数据结构

**目标**: 扩展现有配置和数据结构，支持Demucs选项

**工作量估计**: 0.5-1天

#### 2.1 更新全局配置（config.py）

**文件**: `backend/app/core/config.py`

**修改点**:
```python
class ProjectConfig:
    def __init__(self):
        # ... 现有代码 ...

        # ========== Demucs配置 ==========
        self.DEMUCS_ENABLED = True  # 全局开关
        self.DEMUCS_MODEL = "htdemucs"  # 模型名称
        self.DEMUCS_DEVICE = "cuda"  # 默认设备

        # ========== 进度权重配置（更新） ==========
        self.PHASE_WEIGHTS = {
            "pending": 0,
            "extract": 5,
            "bgm_detect": 2,      # 【新增】BGM检测
            "demucs_global": 8,   # 【新增】全局分离
            "split": 5,
            "transcribe": 50,     # 【调整】从60降低
            "align": 20,
            "translate": 0,
            "proofread": 0,
            "srt": 10,
            "complete": 0
        }
```

**测试验证**:
```python
def test_phase_weights():
    """验证权重总和"""
    from core.config import config
    total = sum(config.PHASE_WEIGHTS.values())
    assert total == 100
```

#### 2.2 扩展JobSettings

**文件**: `backend/app/models/job_models.py`

**修改点**:
```python
@dataclass
class DemucsSettings:
    """Demucs配置"""
    enabled: bool = True
    mode: str = "auto"  # auto/always/never/on_demand
    retry_threshold_logprob: float = -0.8
    retry_threshold_no_speech: float = 0.6
    circuit_breaker_enabled: bool = True
    consecutive_threshold: int = 3
    ratio_threshold: float = 0.2

@dataclass
class JobSettings:
    model: str = "medium"
    compute_type: str = "float16"
    device: str = "cuda"
    batch_size: int = 16
    word_timestamps: bool = False
    cpu_affinity: Optional[CPUAffinityConfig] = None
    demucs: DemucsSettings = field(default_factory=DemucsSettings)  # 【新增】
```

#### 2.3 扩展Checkpoint结构

**修改点**: `TranscriptionService._save_checkpoint()`

**新增字段**:
```python
checkpoint_data = {
    "job_id": job.job_id,
    "phase": "...",
    "processing_mode": "...",

    # 【新增】Demucs相关字段
    "demucs": {
        "enabled": True,
        "mode": "auto",
        "bgm_level": "light",  # none/light/heavy
        "bgm_ratios": [0.15, 0.12, 0.18],
        "global_separation_done": False,
        "vocals_path": None,  # 分离后的音频路径
        "circuit_breaker": {
            "consecutive_retries": 0,
            "total_retries": 2,
            "processed_segments": 10
        },
        "retry_triggered": False
    },

    # 现有字段...
    "total_segments": len(segments),
    "processed_indices": list(processed_indices),
    "segments": segments,
    "unaligned_results": unaligned_results
}
```

**兼容性处理**:
- 旧checkpoint没有demucs字段时，使用默认配置
- 支持平滑升级

---

### Phase 3: TranscriptionService集成（基础）

**目标**: 在转录流程中集成Demucs，实现基础的全局分离功能

**工作量估计**: 2-3天

#### 3.1 修改_run_pipeline插入Demucs阶段

**文件**: `backend/app/services/transcription_service.py`

**修改流程**:

```python
def _run_pipeline(self, job: JobState):
    """执行转录处理管道（支持Demucs）"""

    # ... 现有前置代码 ...

    # 阶段1: 音频提取
    if not audio_path.exists() or checkpoint is None:
        self._extract_audio(...)

    # 【新增】阶段2: BGM检测（可选）
    bgm_level = BGMLevel.NONE
    bgm_ratios = []
    demucs_settings = job.settings.demucs

    if demucs_settings.enabled and demucs_settings.mode in ["auto", "always"]:
        if not checkpoint or 'demucs' not in checkpoint:
            # 执行BGM检测
            bgm_level, bgm_ratios = self._detect_bgm(audio_path, job)
            self.logger.info(f"BGM检测结果: {bgm_level.value}, ratios={bgm_ratios}")
        else:
            # 从checkpoint恢复
            demucs_state = checkpoint['demucs']
            bgm_level = BGMLevel(demucs_state.get('bgm_level', 'none'))
            bgm_ratios = demucs_state.get('bgm_ratios', [])

    # 【新增】阶段3: 全局人声分离（条件执行）
    use_vocals = False
    vocals_path = None

    # 决策：何时执行全局分离
    should_separate_global = (
        demucs_settings.enabled and (
            demucs_settings.mode == "always" or
            (demucs_settings.mode == "auto" and bgm_level == BGMLevel.HEAVY)
        )
    )

    if should_separate_global:
        if not checkpoint or not checkpoint.get('demucs', {}).get('global_separation_done'):
            # 执行全局分离
            vocals_path = self._separate_vocals_global(audio_path, job)
            use_vocals = True
            self.logger.info(f"全局人声分离完成: {vocals_path}")
        else:
            # 从checkpoint恢复
            vocals_path = checkpoint['demucs']['vocals_path']
            use_vocals = True
            self.logger.info(f"跳过全局分离，使用已有结果: {vocals_path}")

    # 决定后续使用哪个音频文件
    active_audio_path = vocals_path if use_vocals else audio_path

    # 阶段4: VAD分段（使用active_audio_path）
    if not current_segments:
        current_segments = self._split_audio(active_audio_path, processing_mode, ...)

    # 阶段5: 转录处理（暂时不加按需分离，Phase 4再实现）
    for seg in todo_segments:
        result = self._transcribe_segment(seg, model, job, audio_array)
        # ...

    # 后续阶段保持不变...
```

#### 3.2 实现辅助方法

**新增方法列表**:

1. **`_detect_bgm(audio_path, job)`**
   ```python
   def _detect_bgm(self, audio_path: str, job: JobState) -> Tuple[BGMLevel, List[float]]:
       """执行BGM检测，更新进度"""
       self._update_progress(job, 'bgm_detect', 0, 'BGM检测中...')

       from services.demucs_service import get_demucs_service
       demucs = get_demucs_service()

       level, ratios = demucs.detect_background_music_level(audio_path)

       self._update_progress(job, 'bgm_detect', 1, f'BGM检测完成: {level.value}')

       # 推送SSE事件
       self._push_sse_bgm_detected(job, level, ratios)

       return level, ratios
   ```

2. **`_separate_vocals_global(audio_path, job)`**
   ```python
   def _separate_vocals_global(self, audio_path: str, job: JobState) -> str:
       """执行全局人声分离，更新进度"""
       self._update_progress(job, 'demucs_global', 0, '人声分离中...')

       from services.demucs_service import get_demucs_service
       demucs = get_demucs_service()

       def progress_callback(p, msg):
           self._update_progress(job, 'demucs_global', p, msg)

       vocals_path = demucs.separate_vocals(
           audio_path,
           progress_callback=progress_callback
       )

       self._update_progress(job, 'demucs_global', 1, '人声分离完成')

       return vocals_path
   ```

3. **`_push_sse_bgm_detected(job, level, ratios)`**
   ```python
   def _push_sse_bgm_detected(self, job: JobState, level: BGMLevel, ratios: List[float]):
       """推送BGM检测结果"""
       try:
           from services.sse_service import get_sse_manager
           sse_manager = get_sse_manager()

           channel_id = f"job:{job.job_id}"
           sse_manager.broadcast_sync(
               channel_id,
               "bgm_detected",
               {
                   "level": level.value,
                   "ratios": ratios,
                   "max_ratio": max(ratios) if ratios else 0,
                   "recommendation": "全局分离" if level == BGMLevel.HEAVY else "按需分离"
               }
           )
       except Exception as e:
           self.logger.debug(f"SSE推送失败（非致命）: {e}")
   ```

#### 3.3 Checkpoint保存与恢复

**修改 `_save_checkpoint()`**:
```python
def _save_checkpoint(self, job_dir: Path, data: dict, job: JobState):
    """保存checkpoint（扩展Demucs字段）"""

    # 确保demucs字段存在
    if 'demucs' not in data:
        data['demucs'] = {
            "enabled": job.settings.demucs.enabled,
            "mode": job.settings.demucs.mode,
            "bgm_level": "none",
            "bgm_ratios": [],
            "global_separation_done": False,
            "vocals_path": None,
            "circuit_breaker": None,
            "retry_triggered": False
        }

    # 原有保存逻辑...
    checkpoint_path = job_dir / "checkpoint.json"
    temp_path = checkpoint_path.with_suffix(".tmp")

    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    temp_path.replace(checkpoint_path)
```

**修改 `_run_pipeline()` 中的checkpoint恢复**:
```python
if checkpoint:
    self.logger.info(f"发现检查点，从 {checkpoint.get('phase')} 阶段恢复")

    # 恢复Demucs状态
    demucs_state = checkpoint.get('demucs', {})
    if demucs_state:
        bgm_level = BGMLevel(demucs_state.get('bgm_level', 'none'))
        bgm_ratios = demucs_state.get('bgm_ratios', [])
        use_vocals = demucs_state.get('global_separation_done', False)
        vocals_path = demucs_state.get('vocals_path')

        self.logger.info(f"恢复Demucs状态: BGM={bgm_level.value}, 全局分离={'已完成' if use_vocals else '未完成'}")
```

#### 3.4 测试验证

**测试场景**:

1. **全局分离模式（mode=always）**
   ```python
   def test_global_separation_always():
       """测试强制全局分离"""
       settings = JobSettings(
           demucs=DemucsSettings(enabled=True, mode="always")
       )
       job = service.create_job("test.mp4", "/path/to/test.mp4", settings)
       service._run_pipeline(job)

       # 验证vocals.wav存在
       vocals_path = Path(job.dir) / "audio_vocals.wav"
       assert vocals_path.exists()
   ```

2. **自动模式（mode=auto）**
   ```python
   def test_auto_mode_heavy_bgm():
       """测试重BGM自动触发全局分离"""
       # 使用带BGM的测试音频
       # 预期: 检测到HEAVY，执行全局分离
   ```

3. **断点续传**
   ```python
   def test_checkpoint_with_demucs():
       """测试Demucs断点续传"""
       # 1. 执行到分离阶段后暂停
       # 2. 重启恢复
       # 3. 验证跳过已完成的分离
   ```

---

### Phase 4: 按需分离与熔断机制

**目标**: 实现智能的按需分离和动态熔断，应对突发BGM

**工作量估计**: 2-3天

#### 4.1 新增熔断器数据类

**文件**: `backend/app/services/transcription_service.py`

**代码**:
```python
class BreakToGlobalSeparation(Exception):
    """熔断异常：触发时需要升级为全局人声分离模式"""
    pass


@dataclass
class CircuitBreakerState:
    """熔断器状态（用于跟踪转录过程中的重试情况）"""
    consecutive_retries: int = 0
    total_retries: int = 0
    total_segments: int = 0
    processed_segments: int = 0

    def record_retry(self):
        """记录一次重试"""
        self.consecutive_retries += 1
        self.total_retries += 1

    def record_success(self):
        """记录一次成功（重置连续计数）"""
        self.consecutive_retries = 0
        self.processed_segments += 1

    def should_break(self, config: DemucsSettings) -> bool:
        """判断是否应该触发熔断"""
        if not config.circuit_breaker_enabled:
            return False

        # 条件1: 连续重试次数
        if self.consecutive_retries >= config.consecutive_threshold:
            return True

        # 条件2: 总重试比例（至少处理5个segment后才检查）
        if self.processed_segments >= 5:
            retry_ratio = self.total_retries / self.processed_segments
            if retry_ratio >= config.ratio_threshold:
                return True

        return False

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "consecutive_retries": self.consecutive_retries,
            "total_retries": self.total_retries,
            "processed_segments": self.processed_segments,
            "retry_ratio": self.total_retries / max(1, self.processed_segments)
        }
```

#### 4.2 实现带重试的转录方法

**新增方法**: `_transcribe_segment_with_retry()`

**代码**:
```python
def _transcribe_segment_with_retry(
    self,
    seg_meta: Dict,
    model,
    job: JobState,
    audio_array: Optional[np.ndarray] = None,
    demucs_settings: Optional[DemucsSettings] = None,
    circuit_breaker: Optional[CircuitBreakerState] = None
) -> Optional[Dict]:
    """
    带重试的转录方法（支持Demucs人声分离重试 + 动态熔断）

    流程：
    1. 首次转录（使用原始音频）
    2. 检查置信度
    3. 如果置信度低，使用Demucs分离人声后重试
    4. 更新熔断器状态
    5. 检查是否触发熔断
    6. 返回置信度更高的结果

    Raises:
        BreakToGlobalSeparation: 当触发熔断条件时抛出
    """
    if demucs_settings is None:
        demucs_settings = job.settings.demucs

    # 首次转录
    result = self._transcribe_segment(seg_meta, model, job, audio_array)

    if not result or not demucs_settings.enabled:
        if circuit_breaker:
            circuit_breaker.record_success()
        return result

    # 检查是否需要重试
    needs_retry = self._check_transcription_confidence(
        result,
        demucs_settings.retry_threshold_logprob,
        demucs_settings.retry_threshold_no_speech
    )

    if not needs_retry:
        # 不需要重试，记录成功
        if circuit_breaker:
            circuit_breaker.record_success()
        return result

    # ========== 需要重试的逻辑 ==========
    self.logger.info(f"段落 {seg_meta['index']} 置信度低，尝试人声分离重试")

    # 更新熔断器状态
    if circuit_breaker:
        circuit_breaker.record_retry()

        # 检查是否触发熔断
        if circuit_breaker.should_break(demucs_settings):
            stats = circuit_breaker.get_stats()
            self.logger.warning(
                f"触发熔断！连续重试={stats['consecutive_retries']}, "
                f"总重试比例={stats['retry_ratio']:.1%}"
            )
            raise BreakToGlobalSeparation(
                f"连续{stats['consecutive_retries']}段需要Demucs重试，"
                f"建议升级为全局人声分离模式"
            )

    # 尝试按需分离
    try:
        from services.demucs_service import get_demucs_service
        demucs = get_demucs_service()

        start_sec = seg_meta['start']
        end_sec = seg_meta['end']

        if audio_array is not None:
            # 内存模式：分离人声
            vocals = demucs.separate_vocals_segment(
                audio_array, sr=16000,
                start_sec=start_sec, end_sec=end_sec
            )

            # 构造临时seg_meta
            retry_seg = seg_meta.copy()
            retry_seg['start'] = 0
            retry_seg['end'] = len(vocals) / 16000

            # 重新转录
            retry_result = self._transcribe_segment_in_memory(
                vocals, retry_seg, model, job, is_vocals=True
            )
        else:
            # 硬盘模式：暂不支持
            self.logger.warning("硬盘模式暂不支持Demucs重试")
            return result

        if retry_result:
            # 校正时间偏移
            original_start = seg_meta['start']
            for seg in retry_result.get('segments', []):
                seg['start'] += original_start
                seg['end'] += original_start

            # 比较两次结果，返回更好的
            if self._is_better_result(retry_result, result):
                self.logger.info(f"段落 {seg_meta['index']} 重试成功，使用分离后的结果")
                retry_result['used_demucs'] = True
                return retry_result

    except Exception as e:
        self.logger.warning(f"Demucs重试失败: {e}")

    return result


def _check_transcription_confidence(
    self,
    result: Dict,
    logprob_threshold: float,
    no_speech_threshold: float
) -> bool:
    """检查转录结果的置信度"""
    segments = result.get('segments', [])

    if not segments:
        return True  # 没有识别出内容，需要重试

    # 计算平均置信度
    total_logprob = 0
    total_no_speech = 0
    count = 0

    for seg in segments:
        if 'avg_logprob' in seg:
            total_logprob += seg['avg_logprob']
            count += 1
        if 'no_speech_prob' in seg:
            total_no_speech += seg['no_speech_prob']

    if count == 0:
        return False

    avg_logprob = total_logprob / count
    avg_no_speech = total_no_speech / count if count > 0 else 0

    # 判断是否需要重试
    if avg_logprob < logprob_threshold:
        self.logger.debug(f"avg_logprob={avg_logprob:.2f} < {logprob_threshold}, 需要重试")
        return True

    if avg_no_speech > no_speech_threshold:
        self.logger.debug(f"no_speech_prob={avg_no_speech:.2f} > {no_speech_threshold}, 需要重试")
        return True

    return False


def _is_better_result(self, new_result: Dict, old_result: Dict) -> bool:
    """比较两个转录结果，判断新结果是否更好"""
    new_segments = new_result.get('segments', [])
    old_segments = old_result.get('segments', [])

    if not new_segments:
        return False
    if not old_segments:
        return True

    # 比较平均logprob
    def get_avg_logprob(segments):
        logprobs = [s.get('avg_logprob', -1) for s in segments if 'avg_logprob' in s]
        return np.mean(logprobs) if logprobs else -1

    new_logprob = get_avg_logprob(new_segments)
    old_logprob = get_avg_logprob(old_segments)

    return new_logprob > old_logprob
```

#### 4.3 修改转录主循环支持熔断

**修改 `_run_pipeline()` 中的转录循环**:

```python
# 阶段5: 转录处理（带熔断机制）
max_global_retries = 1
global_retry_count = 0

while global_retry_count <= max_global_retries:
    try:
        # 初始化熔断器
        circuit_breaker = CircuitBreakerState(total_segments=len(current_segments))

        # 转录循环
        for seg_meta in todo_segments:
            if job.canceled:
                raise RuntimeError('任务已取消')
            if job.paused:
                raise RuntimeError('任务已暂停')

            # 使用带重试的转录方法
            result = self._transcribe_segment_with_retry(
                seg_meta,
                model,
                job,
                audio_array=audio_array,
                demucs_settings=job.settings.demucs,
                circuit_breaker=circuit_breaker
            )

            if result:
                unaligned_results.append(result)
                processed_indices.add(seg_meta['index'])
                job.processed = len(processed_indices)

                # 保存checkpoint
                self._save_checkpoint(job_dir, {
                    "job_id": job.job_id,
                    "phase": "transcribe",
                    "processing_mode": processing_mode.value,
                    "total_segments": len(current_segments),
                    "processed_indices": list(processed_indices),
                    "segments": current_segments,
                    "unaligned_results": unaligned_results,
                    "demucs": {
                        # 包含熔断器状态
                        "circuit_breaker": circuit_breaker.get_stats()
                    }
                }, job)

        # 正常完成
        stats = circuit_breaker.get_stats()
        self.logger.info(
            f"转录完成: {stats['processed_segments']}段, "
            f"重试{stats['total_retries']}次 ({stats['retry_ratio']:.1%})"
        )
        break  # 退出while循环

    except BreakToGlobalSeparation as e:
        global_retry_count += 1
        self.logger.warning(f"熔断触发 (第{global_retry_count}次): {e}")

        if global_retry_count > max_global_retries:
            self.logger.error("已达到最大全局重试次数，使用当前结果")
            break

        # ========== 升级为全局分离模式 ==========
        self.logger.info("升级为全局人声分离模式，重新处理...")

        # 1. 丢弃已转录内容
        unaligned_results = []
        processed_indices = set()

        # 2. 执行全局人声分离
        from services.demucs_service import get_demucs_service
        demucs = get_demucs_service()

        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_array, 16000)

        try:
            vocals_path = demucs.separate_vocals(
                temp_path,
                progress_callback=lambda p, m: self._update_progress(
                    job, 'demucs_global', p, f'全局人声分离: {m}'
                )
            )

            # 3. 加载分离后的人声
            import whisperx
            audio_array = whisperx.load_audio(vocals_path)

            # 4. 重新VAD分段（使用纯人声）
            current_segments = self._split_audio_in_memory(audio_array, sr=16000)

            self.logger.info(f"全局分离完成，重新分段: {len(current_segments)}段")

            # 5. 更新checkpoint标记
            checkpoint_data = {
                "job_id": job.job_id,
                "phase": "split_complete",
                "processing_mode": processing_mode.value,
                "total_segments": len(current_segments),
                "processed_indices": [],
                "segments": current_segments,
                "unaligned_results": [],
                "demucs": {
                    "enabled": True,
                    "mode": "always",
                    "bgm_level": "heavy",
                    "global_separation_done": True,
                    "vocals_path": vocals_path,
                    "retry_triggered": True,
                    "circuit_breaker": None
                }
            }
            self._save_checkpoint(job_dir, checkpoint_data, job)

            # 6. 禁用后续的按需分离
            job.settings.demucs.enabled = False

        finally:
            os.unlink(temp_path)

        # 继续while循环，使用分离后的音频重新转录
        continue
```

#### 4.4 SSE事件推送

**新增事件类型**:
```python
def _push_sse_circuit_breaker_triggered(
    self,
    job: JobState,
    stats: Dict,
    reason: str
):
    """推送熔断触发事件"""
    try:
        from services.sse_service import get_sse_manager
        sse_manager = get_sse_manager()

        channel_id = f"job:{job.job_id}"
        sse_manager.broadcast_sync(
            channel_id,
            "circuit_breaker_triggered",
            {
                "reason": reason,
                "stats": stats,
                "action": "升级为全局人声分离模式",
                "estimated_extra_time": 120  # 预估额外耗时
            }
        )
    except Exception as e:
        self.logger.debug(f"SSE推送失败: {e}")
```

#### 4.5 测试验证

**测试场景**:

1. **按需分离成功**
   ```python
   def test_on_demand_separation():
       """测试按需分离"""
       # 使用部分段落有BGM的音频
       # 预期: 只对低置信度段落进行分离
   ```

2. **熔断触发（连续重试）**
   ```python
   def test_circuit_breaker_consecutive():
       """测试连续重试触发熔断"""
       # 使用强BGM音频
       # 预期: 连续3段触发重试后熔断
   ```

3. **熔断触发（总比例）**
   ```python
   def test_circuit_breaker_ratio():
       """测试重试比例触发熔断"""
       # 预期: 20%段落需要重试后熔断
   ```

---

### Phase 5: 前端适配与用户界面

**目标**: 添加前端UI支持，允许用户配置Demucs选项

**工作量估计**: 1-2天

#### 5.1 更新API接口

**文件**: `backend/app/routes/*.py`

**新增/修改接口**:

1. **创建任务接口**（支持demucs配置）
   ```python
   @router.post("/transcribe")
   async def create_transcription_job(request: TranscriptionRequest):
       settings = JobSettings(
           model=request.model,
           # ...
           demucs=DemucsSettings(
               enabled=request.demucs_enabled,
               mode=request.demucs_mode,
               # ...
           )
       )
       # ...
   ```

2. **获取默认配置接口**
   ```python
   @router.get("/config/demucs")
   async def get_demucs_config():
       return {
           "enabled": config.DEMUCS_ENABLED,
           "model": config.DEMUCS_MODEL,
           "modes": ["auto", "always", "never", "on_demand"],
           "default_mode": "auto"
       }
   ```

#### 5.2 前端进度条适配

**文件**: `frontend/src/components/TaskProgress.vue`

**修改点**:

1. **阶段中文名称映射**
   ```typescript
   const PHASE_NAMES: Record<string, string> = {
       pending: '等待中',
       extract: '提取音频',
       bgm_detect: '检测背景音乐',      // 新增
       demucs_global: '分离人声',        // 新增
       split: '音频分段',
       transcribe: '转录中',
       align: '对齐时间轴',
       srt: '生成字幕',
       complete: '完成'
   };
   ```

2. **进度条颜色**
   ```typescript
   const PHASE_COLORS: Record<string, string> = {
       // ...
       bgm_detect: 'purple',
       demucs_global: 'violet',
       // ...
   };
   ```

3. **熔断提示**
   ```typescript
   function handleCircuitBreakerTriggered(data: ProgressData) {
       showToast({
           type: 'warning',
           message: '检测到强背景音乐干扰，正在切换为全局人声分离模式...',
           duration: 5000
       });

       // 进度回退动画
       animateProgressTo(15);
   }
   ```

#### 5.3 设置界面（可选）

**文件**: `frontend/src/views/Settings.vue`

**UI组件**:
- Demucs开关
- 模式选择（auto/always/never/on_demand）
- 高级选项（阈值、熔断配置）

---

### Phase 6: 测试与优化

**目标**: 全面测试各种场景，优化性能和用户体验

**工作量估计**: 2-3天

#### 6.1 单元测试

**文件**: `backend/tests/test_demucs_service.py`

**测试用例**:
- [ ] 模型加载与卸载
- [ ] 全局人声分离
- [ ] 按需分离（内存模式）
- [ ] BGM检测（三种级别）
- [ ] 能量比计算
- [ ] 缓存机制

#### 6.2 集成测试

**文件**: `backend/tests/test_transcription_with_demucs.py`

**测试场景**:
- [ ] 纯对话视频（mode=auto, 跳过Demucs）
- [ ] 轻BGM视频（mode=auto, 按需分离）
- [ ] 重BGM视频（mode=auto, 全局分离）
- [ ] MV视频（mode=always）
- [ ] 熔断触发与恢复
- [ ] 断点续传（各阶段）
- [ ] 错误降级

#### 6.3 性能优化

**优化点**:
- [ ] 模型预加载（启动时或首次使用前）
- [ ] 缓存管理（LRU, 自动清理）
- [ ] 内存监控（防止OOM）
- [ ] GPU显存管理（与Whisper共享）
- [ ] 进度推送频率控制（避免SSE风暴）

#### 6.4 用户体验优化

**改进点**:
- [ ] 进度消息优化（更友好的提示）
- [ ] 错误提示优化（明确原因和建议）
- [ ] 预估时间（基于历史数据）
- [ ] 日志输出优化（降低噪音）

---

## 实施顺序建议

```
Week 1:
├─ Day 1-2: Phase 1 (DemucsService)
│  ├─ 1.1-1.2: 基础框架 + 全局分离
│  └─ 1.3-1.4: 按需分离 + BGM检测
│
├─ Day 3: Phase 2 (配置扩展)
│  ├─ 2.1: config.py
│  ├─ 2.2: JobSettings
│  └─ 2.3: Checkpoint
│
└─ Day 4-5: Phase 3 (TranscriptionService基础集成)
   ├─ 3.1: _run_pipeline修改
   ├─ 3.2: 辅助方法
   ├─ 3.3: Checkpoint保存/恢复
   └─ 3.4: 测试验证

Week 2:
├─ Day 1-3: Phase 4 (按需分离+熔断)
│  ├─ 4.1: 熔断器数据类
│  ├─ 4.2: _transcribe_segment_with_retry
│  ├─ 4.3: 修改转录主循环
│  ├─ 4.4: SSE事件
│  └─ 4.5: 测试验证
│
├─ Day 4-5: Phase 5 (前端适配)
│  ├─ 5.1: API接口
│  ├─ 5.2: 进度条
│  └─ 5.3: 设置界面
│
└─ Week 3:
   └─ Phase 6 (测试优化)
      ├─ 6.1: 单元测试
      ├─ 6.2: 集成测试
      ├─ 6.3: 性能优化
      └─ 6.4: 用户体验优化
```

---

## 风险与应对

### 风险1: 显存不足（Whisper + Demucs同时运行）

**应对**:
- 检测显存使用率，超过阈值时卸载Demucs模型
- 提供CPU降级模式
- 用户配置选项：禁用Demucs或使用CPU

### 风险2: 性能开销过大

**应对**:
- 默认模式为`auto`（智能决策）
- 提供`never`模式（完全跳过）
- 优化按需分离（只处理低置信度段落）

### 风险3: 模型下载失败（国内网络）

**应对**:
- 使用HF镜像源
- 提供离线模型包
- 明确的错误提示和手动下载指引

### 风险4: 断点续传兼容性

**应对**:
- 兼容旧格式checkpoint
- 新字段全部可选（有默认值）
- 版本号标记

---

## 验收标准

### 功能验收

- [ ] 支持三种模式（auto/always/never）
- [ ] BGM检测准确率 > 80%
- [ ] 按需分离成功触发
- [ ] 熔断机制正常工作
- [ ] 断点续传兼容新旧格式
- [ ] 错误降级不影响转录

### 性能验收

- [ ] 纯对话视频无额外开销
- [ ] 轻BGM视频额外耗时 < 20%
- [ ] 重BGM视频额外耗时 < 100%
- [ ] 显存峰值 < 8GB（medium模型）

### 用户体验验收

- [ ] 进度条实时更新
- [ ] 熔断提示清晰友好
- [ ] 错误提示明确原因
- [ ] 设置界面易于理解

---

## 文档输出

完成后需要更新的文档：

1. **API文档**: 新增Demucs相关接口
2. **用户手册**: Demucs功能说明和配置指引
3. **架构文档**: 更新流程图和数据结构
4. **性能测试报告**: 各场景下的性能数据
5. **llmdoc更新**: 使用tr:recorder agent更新项目文档系统

---

## 总结

本计划将Demucs集成分为6个清晰的阶段，每个阶段都有明确的：
- 目标和范围
- 工作量估计
- 代码实现细节
- 测试验证方法

遵循此计划，可以确保：
1. **渐进式开发**：每阶段可独立测试
2. **风险可控**：及早发现问题
3. **向后兼容**：不影响现有功能
4. **质量保证**：充分测试覆盖

预计总工作量：**2-3周**（按每天6-8小时计算）
