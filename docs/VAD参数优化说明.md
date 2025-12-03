# VAD参数优化说明

## 问题描述

在字幕与音频对齐时，经常出现以下情况：
- 字幕开始时间提前了
- 段落前面包含大量空白或背景音乐
- VAD可能错误地将低音量背景音（如背景音乐）纳入语音段

## 根本原因分析

### 1. VAD检测流程

```
原始音频 → Silero VAD检测 → 获取语音时间戳 → WhisperX对齐 → 最终字幕时间戳
```

### 2. 参数影响

| 参数 | 原值 | 当前值 | 影响 |
|------|------|------|------|
| **onset** | 0.5 | **0.70** | 更严格的语音开始判定，过滤背景音乐残余 |
| **offset** | 0.363 | **0.50** | 对应onset的调整，保持比例关系 |
| **min_speech_duration_ms** | 250ms | **500ms** | 避免误检碎片音和残余BGM |
| **min_silence_duration_ms** | 100ms | **500ms** | 更长的静音间隔，更好地分割段落 |

> **注意**：当使用Demucs分离后，vocals仍可能有轻微BGM残余。提高VAD阈值可以更好地过滤这些残余。

### 3. 参数含义

#### onset（语音开始阈值）
- **范围**：0.0 - 1.0
- **含义**：Silero VAD判定为"语音开始"的概率阈值
- **0.5**：对低音量敏感，会误判背景音乐
- **0.65**：更严格，只识别主要语音
- **建议**：0.60 - 0.70

#### offset（语音结束阈值）
- **范围**：0.0 - 1.0
- **含义**：Silero VAD判定为"语音结束"的概率阈值
- **通常**：为onset的70%左右
- **当onset=0.65时**：offset应为 0.65 × 0.7 ≈ 0.45

#### min_speech_duration_ms（最小语音段长度）
- **含义**：低于此长度的语音会被过滤
- **250ms**：容易误检碎片音
- **400ms**：更好的平衡
- **建议**：300 - 500ms

#### min_silence_duration_ms（最小静音长度）
- **含义**：低于此长度的静音会被合并
- **100ms**：太短，背景音乐的间隙会被认为是"静音"
- **400ms**：充分过滤背景音乐
- **建议**：300 - 500ms

## 修改内容

### VADConfig 数据类更新

```python
@dataclass
class VADConfig:
    # ... 详细说明注释 ...
    onset: float = 0.65                    # 提升至0.65
    offset: float = 0.45                   # 调整为0.45
    chunk_size: int = 30                   # 保持不变
    min_speech_duration_ms: int = 400      # 新增参数：默认400ms
    min_silence_duration_ms: int = 400     # 新增参数：默认400ms
```

### 使用位置更新

`_vad_silero()` 方法现在直接使用 VADConfig 中的参数：

```python
speech_timestamps = get_speech_timestamps(
    audio_tensor,
    model,
    sampling_rate=sr,
    threshold=vad_config.onset,                           # 使用config值
    min_speech_duration_ms=vad_config.min_speech_duration_ms,    # 使用config值
    min_silence_duration_ms=vad_config.min_silence_duration_ms,  # 使用config值
    return_seconds=False
)
```

## 预期效果

✅ **优化后的效果**
- 字幕不再提前，准确对应实际语音
- 背景音乐不再被包含在字幕时间范围内
- VAD分段更加准确，减少误检

## 调试指南

如果调整后仍有问题，可以尝试以下参数组合：

### 保守方案（最严格，用于背景音乐较强）
```python
onset: 0.70
offset: 0.50
min_speech_duration_ms: 500
min_silence_duration_ms: 500
```

### 平衡方案（推荐，当前方案）
```python
onset: 0.65
offset: 0.45
min_speech_duration_ms: 400
min_silence_duration_ms: 400
```

### 激进方案（较宽松，用于几乎没有背景音乐）
```python
onset: 0.55
offset: 0.40
min_speech_duration_ms: 300
min_silence_duration_ms: 300
```

## 用户配置建议

如果要让用户可以调整这些参数，建议在前端添加：

```json
{
  "vad_settings": {
    "method": "silero",
    "onset": 0.65,
    "offset": 0.45,
    "min_speech_duration_ms": 400,
    "min_silence_duration_ms": 400
  }
}
```

## 相关代码位置

- **VADConfig定义**：`backend/app/services/transcription_service.py` 第37-56行
- **Silero VAD使用**：`backend/app/services/transcription_service.py` 第1617行（`_vad_silero` 方法）
- **配置初始化**：`backend/app/services/transcription_service.py` 第1173行、1548行

## 参考资源

- Silero VAD 文档：https://github.com/snakers4/silero-vad
- 官方推荐参数：onset=0.5, offset=0.363（保守）

