# Demucs 人声分离使用指南

## 概述

Demucs 是一个高质量的音频源分离模型，本项目集成了多种 Demucs 模型用于提升背景音乐环境下的转录质量。

## 支持的模型

| 模型 | 说明 | 推荐场景 |
|------|------|---------|
| `mdx_extra` | **默认推荐**。MDX架构，质量与速度平衡好 | 大多数场景 |
| `htdemucs` | Hybrid Transformer架构，经典模型 | 一般用途 |
| `htdemucs_ft` | htdemucs的微调版本，针对人声优化 | 人声提取优先 |
| `mdx_extra_q` | mdx_extra的量化版本，更快但质量略低 | 追求速度 |

> **注意**：从 v2.4 版本开始，默认模型从 `htdemucs` 更改为 `mdx_extra`，以获得更好的人声分离效果。

## 功能特性

### 1. 四种工作模式

| 模式 | 说明 | 使用场景 |
|------|------|---------|
| `auto` | 自动决策 | **推荐**。根据BGM检测结果自动选择策略 |
| `always` | 始终分离 | 确定有背景音的场景，如音乐MV、演唱会 |
| `never` | 禁用分离 | 纯人声场景，如播客、访谈 |
| `on_demand` | 按需分离 | 只对低置信度段落分离（实验性） |

### 2. 智能 BGM 检测

**分位数采样策略**：在音频的 15%、50%、85% 位置各采样 10 秒

**检测结果**：
- `NONE`: 无背景音乐 → 无需分离
- `LIGHT`: 轻微背景音 → **全局分离**（v2.4+优化：即使轻微BGM也会影响VAD精度）
- `HEAVY`: 强背景音 → 全局分离

> **注意**：从 v2.4 版本开始，`auto` 模式下 `LIGHT` 和 `HEAVY` 级别都会触发全局人声分离，以获得更好的转录效果。

**BGM检测阈值配置**（位于 `demucs_service.py`）：
- `bgm_light_threshold`: 0.02 (2%) - 轻微BGM阈值
- `bgm_heavy_threshold`: 0.15 (15%) - 强BGM阈值

### 3. 按需分离与熔断机制

**按需分离**：
- 首次用原始音频转录
- 检测置信度（avg_logprob 和 no_speech_prob）
- 如果置信度低，对该段使用 Demucs 分离后重试
- 比较两次结果，选择更好的

**熔断机制**：
- 监控重试情况
- 触发条件（满足任一）：
  - 连续 N 个段落需要重试（默认 N=3）
  - 总重试比例超过阈值（默认 20%）
- 触发后自动升级为全局分离模式

## API 使用

### 启动转录任务（带 Demucs 配置）

**请求示例**：

```python
import requests
import json

# Demucs 配置
demucs_config = {
    "enabled": True,
    "mode": "auto",  # auto/always/never/on_demand
    "retry_threshold_logprob": -0.8,
    "retry_threshold_no_speech": 0.6,
    "circuit_breaker_enabled": True,
    "consecutive_threshold": 3,
    "ratio_threshold": 0.2
}

# 完整设置
settings = {
    "model": "medium",
    "compute_type": "float16",
    "device": "cuda",
    "batch_size": 16,
    "word_timestamps": False,
    "demucs": demucs_config
}

# 启动任务
response = requests.post(
    "http://localhost:8000/api/start",
    data={
        "job_id": "your_job_id",
        "settings": json.dumps(settings)
    }
)

print(response.json())
```

### SSE 事件监听

**连接 SSE 流**：

```python
import requests

job_id = "your_job_id"
url = f"http://localhost:8000/api/stream/{job_id}"

with requests.get(url, stream=True) as response:
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('event:'):
                event_type = line.split(':', 1)[1].strip()
            elif line.startswith('data:'):
                data = line.split(':', 1)[1].strip()
                handle_event(event_type, data)
```

**新增事件类型**：

1. **bgm_detected** - BGM 检测结果
```json
{
  "level": "light",
  "ratios": [0.15, 0.18, 0.12],
  "max_ratio": 0.18,
  "recommendation": "按需分离"
}
```

2. **circuit_breaker_triggered** - 熔断触发
```json
{
  "triggered": true,
  "reason": "转录质量低，触发熔断升级",
  "stats": {
    "consecutive_retries": 3,
    "total_retries": 5,
    "processed_segments": 12,
    "retry_ratio": 0.417
  },
  "action": "升级为全局人声分离模式"
}
```

3. **segment** - 单段转录完成（已有）
```json
{
  "index": 0,
  "text": "转录文本",
  "start": 0.0,
  "end": 5.5,
  "used_demucs": true
}
```

## 配置参数详解

### DemucsSettings

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 Demucs |
| `mode` | string | `"auto"` | 工作模式 |
| `retry_threshold_logprob` | float | `-0.8` | 重试阈值（logprob 低于此值重试） |
| `retry_threshold_no_speech` | float | `0.6` | 重试阈值（no_speech_prob 高于此值重试） |
| `circuit_breaker_enabled` | bool | `true` | 是否启用熔断机制 |
| `consecutive_threshold` | int | `3` | 连续重试触发熔断的次数 |
| `ratio_threshold` | float | `0.2` | 重试比例触发熔断的阈值 |

### BGM 检测参数（DemucsConfig）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bgm_sample_duration` | `10.0` | 每个采样片段时长（秒） |
| `bgm_light_threshold` | `0.2` | 轻微BGM阈值 |
| `bgm_heavy_threshold` | `0.6` | 强BGM阈值 |

## 使用场景推荐

### 场景 1: 音乐MV字幕生成

**特点**：背景音乐强，人声混合

**推荐配置**：
```json
{
  "enabled": true,
  "mode": "always",
  "circuit_breaker_enabled": false
}
```

**说明**：直接全局分离，跳过检测环节

### 场景 2: 访谈/播客

**特点**：纯人声或背景音乐很弱

**推荐配置**：
```json
{
  "enabled": true,
  "mode": "auto",
  "circuit_breaker_enabled": true
}
```

**说明**：自动检测，大概率判断为 NONE，不执行分离

### 场景 3: 直播录制

**特点**：背景音乐不确定，需要灵活处理

**推荐配置**：
```json
{
  "enabled": true,
  "mode": "auto",
  "circuit_breaker_enabled": true,
  "consecutive_threshold": 5,
  "ratio_threshold": 0.3
}
```

**说明**：使用 auto 模式 + 宽松的熔断阈值

### 场景 4: 高质量要求

**特点**：不在乎处理时间，追求最高质量

**推荐配置**：
```json
{
  "enabled": true,
  "mode": "always",
  "circuit_breaker_enabled": false
}
```

**说明**：始终使用分离后的音频

## 工作流程图

### Auto 模式流程

```
开始
  ↓
提取音频 (5%)
  ↓
BGM检测 (2%)
  ├─→ HEAVY → 全局分离 (8%) → 继续
  ├─→ LIGHT → 跳过全局分离 → 继续
  └─→ NONE  → 跳过全局分离 → 继续
  ↓
音频分段 (5%)
  ↓
转录循环 (50%)
  ├─→ 首次转录
  ├─→ 检查置信度
  │     ├─→ 高 → 继续下一段
  │     └─→ 低 → Demucs按需分离重试
  │            ├─→ 更新熔断器
  │            ├─→ 检查熔断条件
  │            │     ├─→ 未触发 → 继续
  │            │     └─→ 触发 → 抛出异常
  │            └─→ 选择更好结果
  └─→ (所有段落完成)
  ↓
批次对齐 (20%)
  ↓
生成SRT (10%)
  ↓
完成
```

### 熔断触发后

```
触发熔断
  ↓
推送 SSE 事件
  ↓
执行全局人声分离 (8%)
  ↓
重新加载音频
  ↓
继续处理剩余段落（使用分离后音频）
  ↓
完成
```

## 断点续传支持

**Checkpoint 结构**：

```json
{
  "job_id": "xxx",
  "phase": "transcribe",
  "processing_mode": "memory",
  "demucs": {
    "enabled": true,
    "mode": "auto",
    "bgm_level": "light",
    "bgm_ratios": [0.15, 0.18, 0.12],
    "global_separation_done": false,
    "vocals_path": null,
    "circuit_breaker": {
      "consecutive_retries": 2,
      "total_retries": 3,
      "processed_segments": 10
    },
    "retry_triggered": false
  },
  "segments": [...],
  "processed_indices": [0, 1, 2, ...],
  "unaligned_results": [...]
}
```

**恢复逻辑**：
- BGM 检测结果会被保存，恢复时跳过检测
- 如果已完成全局分离，直接使用分离后的音频
- 熔断器状态会被恢复，避免重复触发

## 性能考量

### 内存占用

- **Demucs 模型**：
  - `mdx_extra`：~100MB（默认推荐）
  - `htdemucs`：~80MB
  - `htdemucs_ft`：~80MB
  - `mdx_extra_q`：~50MB（量化版本）
- **音频数组（内存模式）**：
  - 1分钟：~4MB
  - 5分钟：~20MB
  - 10分钟：~40MB

### 处理时间估算

假设 10 分钟音频，GPU: RTX 3080

| 操作 | 时间 |
|------|------|
| BGM 检测（3次采样） | ~30s |
| 全局人声分离 | ~60s |
| 按需分离（单段30s） | ~10s |

**模式对比**：

- `never`: 基准时间
- `auto + NONE`: +30s (仅检测)
- `auto + HEAVY`: +90s (检测+全局分离)
- `always`: +60s (仅全局分离，跳过检测)
- `on_demand`: +30s + N×10s (N=需要重试的段落数)

## 故障排查

### 问题 1: Demucs 未生效

**检查清单**：
1. 确认 `demucs_enabled = True`
2. 检查 `mode` 不是 `"never"`
3. 查看日志中的 BGM 检测结果
4. 确认 GPU 可用（CPU 模式会很慢）

### 问题 2: 转录仍然质量低

**可能原因**：
1. BGM 检测判断为 NONE，但实际有背景音
   - **解决**：改用 `mode="always"`
2. 熔断未触发（阈值过高）
   - **解决**：降低 `consecutive_threshold` 或 `ratio_threshold`
3. 音频本身质量差（非BGM问题）
   - **解决**：Demucs 无法解决

### 问题 3: 处理时间过长

**优化建议**：
1. 使用 GPU（`device="cuda"`）
2. 降低 `shifts` 值（牺牲质量换速度）
3. 对于纯人声场景，使用 `mode="never"`

### 问题 4: 内存不足

**解决方案**：
1. 系统会自动降级到硬盘模式
2. 手动限制并发任务数
3. 增加系统内存

## 最佳实践

1. **首次使用**：使用 `mode="auto"`，观察 BGM 检测结果
2. **批量处理相似内容**：根据首次检测结果调整配置
3. **生产环境**：启用熔断机制，设置合理阈值
4. **关键任务**：使用 `mode="always"` 确保质量
5. **监控 SSE 事件**：实时了解 Demucs 工作状态

## 依赖安装

```bash
# 安装 Demucs
pip install demucs>=4.0.0

# 安装 librosa（用于音频重采样）
pip install librosa
```

**首次运行**会自动下载所选模型（50-100MB）

## 模型切换

默认使用 `mdx_extra` 模型。如需切换模型，修改 `backend/app/services/demucs_service.py` 中的 `DemucsConfig`：

```python
@dataclass
class DemucsConfig:
    model_name: str = "mdx_extra"  # 可选: htdemucs, htdemucs_ft, mdx_extra_q
    # ...
```

**可用模型列表**：
- `mdx_extra` - 默认推荐，质量最佳
- `htdemucs` - 经典 Hybrid Transformer 模型
- `htdemucs_ft` - 针对人声微调的版本
- `mdx_extra_q` - 量化版本，速度更快

## 常见问题

**Q: Demucs 会降低转录速度吗？**

A: 会有一定影响。`auto` 模式下，轻微 BGM 影响很小（只多 30s 检测时间）。强 BGM 会多花 1-2 分钟进行全局分离。

**Q: 可以只对某些段落使用 Demucs 吗？**

A: 可以。使用 `mode="on_demand"` + 熔断机制，系统会自动判断。

**Q: BGM 检测的准确率如何？**

A: 采用分位数采样策略，准确率约 90%。对于判断不准的情况，可以手动指定 `mode="always"` 或 `"never"`。

**Q: 支持哪些音频格式？**

A: 所有 FFmpeg 支持的格式（mp3, wav, flac, m4a, ogg 等）

**Q: 能否自定义 BGM 检测参数？**

A: 可以。修改 `DemucsConfig` 类中的参数（需要修改代码）。未来可能开放到配置文件。

---

## 更新日志

### v2.4.0 (2025-01)

- ✅ 默认模型更换为 `mdx_extra`，提升分离质量
- ✅ 支持多模型选择（mdx_extra, htdemucs, htdemucs_ft, mdx_extra_q）
- ✅ `auto` 模式下 LIGHT 级别 BGM 也会触发分离
- ✅ 提高 `shifts` 默认值到 2，增强分离质量
- ✅ 提高 `overlap` 默认值到 0.5，减少边界伪影
- ✅ 优化 BGM 检测阈值（light=0.02, heavy=0.15）

### v1.0.0 (2024-12-02)

- ✅ 初始版本发布
- ✅ 支持 4 种工作模式
- ✅ 智能 BGM 检测
- ✅ 按需分离与熔断机制
- ✅ 断点续传支持
- ✅ SSE 实时事件推送
