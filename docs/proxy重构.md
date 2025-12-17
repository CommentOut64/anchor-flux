好的，wgh，我直接在这里完整输出开发文档。

---

# Proxy 视频加载系统重构方案

## 一、现状分析

### 1.1 系统架构概述

当前系统采用渐进式视频加载策略：
- **360p 预览** → **720p 高清** → **原始视频**（如兼容）
- 通过 SSE (Server-Sent Events) 推送转码进度
- 前端使用 Vue 3 Composable (`useVideoStatus.js`) 管理状态

### 1.2 核心文件

| 模块          | 文件路径                                              | 职责                            |
| ------------- | ----------------------------------------------------- | ------------------------------- |
| 后端转码服务  | `backend/app/services/media_prep_service.py`          | FFmpeg 转码、任务队列、进度推送 |
| 后端路由      | `backend/app/api/routes/media_routes.py`              | API 端点、格式判断、触发转码    |
| 媒体分析      | `backend/app/utils/media_analyzer.py`                 | FFprobe 分析、编解码器检测      |
| 前端 SSE 管理 | `frontend/src/services/sseChannelManager.js`          | SSE 连接、重连、事件分发        |
| 前端状态管理  | `frontend/src/composables/useVideoStatus.js`          | 视频状态、URL 切换逻辑          |
| 视频播放器    | `frontend/src/components/editor/VideoStage/index.vue` | 播放器 UI、转码占位符           |

### 1.3 当前数据流

```
用户上传视频
    ↓
media_routes.py 分析格式
    ↓ (需要转码)
media_prep_service.py 入队
    ↓
FFmpeg 转码 (360p → 720p)
    ↓
SSE 推送进度事件
    ↓
sseChannelManager.js 接收
    ↓
useVideoStatus.js 更新状态
    ↓
VideoStage/index.vue 渲染
```

---

## 二、问题深度分析

### 2.1 问题一：前端无法获取真实转码进度

**现象**：
- 进度条不准确或卡顿
- 刷新后进度丢失

**根因**：
1. `media_prep_service.py:489-534` 中 SSE 推送使用 `except: pass` 吞掉错误
2. FFmpeg 进度解析依赖 `out_time_ms`，某些视频无法获取总时长
3. 前端刷新后需重新订阅 SSE，但后端任务可能已在进行中

**相关代码**：
```python
# media_prep_service.py:520-525
try:
    sse_manager.broadcast_sync(job_id, event_type, progress_data)
except:
    pass  # 错误被静默吞掉
```

### 2.2 问题二：刷新后恢复连接困难

**现象**：
- 刷新后视频状态不一致
- 无法智能判断当前转码阶段

**根因**：
1. `useVideoStatus.js` 状态存储在内存，刷新即丢失
2. `get_progressive_status` API 返回的状态与前端状态定义不一致
3. SSE 重连逻辑在 `sseChannelManager.js:315-328` 会重建连接，但历史进度无法恢复

**相关代码**：
```javascript
// sseChannelManager.js:315-328
resubscribeJobChannel(jobId, handlers) {
  // 关闭旧连接，创建新连接
  // 问题：历史进度丢失
}
```

### 2.3 问题三：状态管理复杂

**现象**：
- 调用链过长，难以追踪
- `projectStore.meta.videoStatus` 与 `useVideoStatus` 状态重复

**根因**：
1. `useVideoStatus.js` 维护内部状态（309行）
2. `projectStore` 也维护 `videoStatus`
3. `EditorView.vue` 作为中间层传递大量 props

**调用链**：
```
SSE Event → sseChannelManager → EditorView (中转) → useVideoStatus → VideoStage
```

### 2.4 问题四：缺少零转码优先（容器重封装）

**现象**：
- H.264+AAC 的 MKV/MOV 文件被完全转码
- 浪费 CPU 资源和时间

**根因**：
1. `media_routes.py:28-34` 仅按容器格式判断是否转码
2. 未实现 "编解码器兼容但容器不兼容" 的检测
3. FFmpeg 命令始终使用 `-c:v libx264` 重编码

**当前判断逻辑**：
```python
# media_routes.py:28-34
BROWSER_COMPATIBLE_FORMATS = {'.mp4', '.webm'}
NEED_TRANSCODE_FORMATS = {'.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v'}
NEED_TRANSCODE_CODECS = {'hevc', 'h265', 'vp9', 'av1'}
```

**缺失逻辑**：
```
IF 容器不兼容 AND 编解码器兼容 (H.264+AAC):
    使用 -c:v copy -c:a copy 重封装  # 零转码
ELSE:
    执行完整转码
```

### 2.5 问题五：FFmpeg 参数未优化

**现象**：
- GOP 硬编码为 30
- 参数分散在代码中，难以调整

**根因**：
1. `media_prep_service.py:232-245` (360p) 和 `:331-345` (720p) 硬编码参数
2. 未从 `config.py` 读取配置
3. 缺少针对快速预览的极速参数

**当前参数**：
```python
# 360p (lines 232-245)
'-preset', 'ultrafast',
'-crf', '28',
'-g', '30',  # GOP 硬编码

# 720p (lines 331-345)
'-preset', 'fast',
'-crf', '23',
'-g', '30',  # GOP 硬编码
```

### 2.6 问题六：360p→720p 切换不流畅

**现象**：
- 切换时有明显卡顿/黑屏
- 播放进度可能跳跃

**根因**：
1. `VideoStage/index.vue:309-381` 切换逻辑复杂
2. 需要保存/恢复播放状态（时间、播放状态）
3. 浏览器视频缓冲策略影响

**相关代码**：
```javascript
// VideoStage/index.vue:309-381
watch(() => props.currentResolution, async (newRes, oldRes) => {
  // 保存当前时间 → 切换源 → 恢复时间
  // 问题：时序控制复杂，容易出现竞态
})
```

### 2.7 问题七：视频未加载时缺少操作拦截

**现象**：
- 用户可以点击播放/暂停、拖动进度条
- 操作无效果或导致错误

**根因**：
1. `VideoStage/index.vue` 未根据加载状态禁用控件
2. 缺少统一的 "视频就绪" 状态判断

---

## 三、重构方案

### 3.1 架构设计

#### 3.1.1 新架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              useProxyVideo (新 Composable)               │   │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  │   │
│  │  │ 状态机  │  │ SSE订阅  │  │ URL管理 │  │ 错误处理 │  │   │
│  │  └────┬────┘  └────┬─────┘  └────┬────┘  └────┬─────┘  │   │
│  │       └────────────┴─────────────┴────────────┘         │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                    │
│  ┌─────────────────────────▼───────────────────────────────┐   │
│  │                   VideoStage.vue                         │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │   │
│  │  │ 播放器核心  │  │ 控件拦截层   │  │ 进度显示       │  │   │
│  │  └─────────────┘  └──────────────┘  └────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   sseChannelManager.js                          │
│           (proxy 事件自动路由到 useProxyVideo)                  │
└─────────────────────────────────────────────────────────────────┘
                              │ SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Backend                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              MediaPrepService (重构)                     │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│  │  │ 智能转码决策 │  │ 容器重封装   │  │ 完整转码      │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │   │
│  │         │                 │                   │          │   │
│  │         └─────────────────┴───────────────────┘          │   │
│  │                           │                              │   │
│  │  ┌────────────────────────▼────────────────────────┐    │   │
│  │  │         统一进度推送 (带错误重试)                │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              /api/media/proxy-status (新接口)            │   │
│  │  返回: 当前阶段、进度、可用URL、错误信息                 │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 状态机设计

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
┌──────┐  分析  ┌─────────┐  需要转码  ┌───────────────┐      │
│ IDLE │ ────→ │ANALYZING│ ─────────→ │TRANSCODING_360│──┐   │
└──────┘       └─────────┘            └───────────────┘  │   │
                    │                        │           │   │
                    │ 可直接播放             │ 完成      │   │
                    │                        ▼           │   │
                    │                 ┌─────────────┐    │   │
                    │                 │ READY_360P  │    │   │
                    │                 └──────┬──────┘    │   │
                    │                        │           │   │
                    │                        │ 自动开始  │   │
                    │                        ▼           │   │
                    │                 ┌───────────────┐  │   │
                    │                 │TRANSCODING_720│  │   │
                    │                 └───────┬───────┘  │   │
                    │                         │          │   │
                    │                         │ 完成     │   │
                    │                         ▼          │   │
                    │                 ┌─────────────┐    │   │
                    └───────────────→ │ READY_720P  │ ←──┘   │
                                      └──────┬──────┘        │
                                             │               │
                                             │ 错误          │
                                             ▼               │
                                      ┌─────────────┐        │
                                      │   ERROR     │ ───────┘
                                      └─────────────┘   重试
```

**状态枚举**：
```javascript
const ProxyState = {
  IDLE: 'idle',                     // 初始状态
  ANALYZING: 'analyzing',           // 分析中
  TRANSCODING_360: 'transcoding_360', // 360p转码中
  READY_360P: 'ready_360p',         // 360p就绪，可播放
  TRANSCODING_720: 'transcoding_720', // 720p转码中
  READY_720P: 'ready_720p',         // 720p就绪
  REMUXING: 'remuxing',             // 容器重封装中
  DIRECT_PLAY: 'direct_play',       // 原始文件可直接播放
  ERROR: 'error'                    // 错误状态
}
```

### 3.2 后端重构

#### 3.2.1 统一配置管理

**文件**: `backend/app/core/config.py`

新增配置项：
```python
# Proxy 视频配置
PROXY_CONFIG = {
    # 360p 预览参数
    "preview_360p": {
        "scale": 360,
        "preset": "ultrafast",
        "crf": 28,
        "gop": 30,
        "audio": False,  # 无音频
    },
    # 720p 高清参数
    "proxy_720p": {
        "scale": 720,
        "preset": "fast",
        "crf": 23,
        "gop": 30,
        "audio_bitrate": "128k",
    },
    # 容器重封装配置
    "remux": {
        "enabled": True,
        "compatible_codecs": {"h264", "aac", "mp3"},
        "target_container": "mp4",
    },
    # SSE 推送配置
    "sse": {
        "progress_interval": 0.5,  # 秒
        "retry_count": 3,
        "retry_delay": 0.1,  # 秒
    }
}

# 浏览器兼容性配置
BROWSER_COMPATIBILITY = {
    "compatible_containers": {".mp4", ".webm"},
    "compatible_video_codecs": {"h264", "vp8"},
    "compatible_audio_codecs": {"aac", "mp3", "opus", "vorbis"},
    "need_transcode_codecs": {"hevc", "h265", "vp9", "av1"},
}
```

#### 3.2.2 智能转码决策器

**文件**: `backend/app/services/media_prep_service.py`

新增方法：
```python
class TranscodeDecision(Enum):
    """转码决策类型"""
    DIRECT_PLAY = "direct_play"      # 直接播放
    REMUX_ONLY = "remux_only"        # 仅重封装
    TRANSCODE_VIDEO = "transcode_video"  # 仅转码视频
    TRANSCODE_AUDIO = "transcode_audio"  # 仅转码音频
    TRANSCODE_FULL = "transcode_full"    # 完整转码

def analyze_transcode_decision(self, video_info: dict) -> TranscodeDecision:
    """
    智能分析转码决策
  
    决策优先级：
    1. 容器和编解码器都兼容 → DIRECT_PLAY
    2. 容器不兼容但编解码器兼容 → REMUX_ONLY
    3. 视频编解码器不兼容 → TRANSCODE_VIDEO 或 TRANSCODE_FULL
    4. 仅音频不兼容 → TRANSCODE_AUDIO
    """
    container = video_info.get('container', '').lower()
    video_codec = video_info.get('video_codec', '').lower()
    audio_codec = video_info.get('audio_codec', '').lower()
  
    config = settings.BROWSER_COMPATIBILITY
  
    container_ok = f".{container}" in config['compatible_containers']
    video_ok = video_codec in config['compatible_video_codecs']
    audio_ok = audio_codec in config['compatible_audio_codecs'] or not audio_codec
  
    if container_ok and video_ok and audio_ok:
        return TranscodeDecision.DIRECT_PLAY
  
    if video_ok and audio_ok:
        # 编解码器兼容，仅需重封装
        return TranscodeDecision.REMUX_ONLY
  
    if not video_ok:
        return TranscodeDecision.TRANSCODE_FULL
  
    if not audio_ok:
        return TranscodeDecision.TRANSCODE_AUDIO
  
    return TranscodeDecision.TRANSCODE_FULL
```

#### 3.2.3 容器重封装实现

**文件**: `backend/app/services/media_prep_service.py`

新增方法：
```python
async def remux_container(
    self,
    job_id: str,
    video_path: Path,
    output_path: Path,
    progress_callback: Optional[Callable] = None
) -> bool:
    """
    容器重封装（零转码）
  
    使用 -c:v copy -c:a copy 直接复制流
    速度极快，通常几秒内完成
    """
    ffmpeg_cmd = self._get_ffmpeg_cmd()
  
    cmd = [
        ffmpeg_cmd,
        '-i', str(video_path),
        '-c:v', 'copy',           # 视频流直接复制
        '-c:a', 'copy',           # 音频流直接复制
        '-movflags', '+faststart', # 优化网络播放
        '-progress', 'pipe:1',
        '-y',
        str(output_path)
    ]
  
    # 执行并推送进度...
    return await self._execute_ffmpeg(job_id, cmd, "remux", progress_callback)
```

#### 3.2.4 统一进度推送（带重试）

**文件**: `backend/app/services/media_prep_service.py`

重构方法：
```python
def _broadcast_progress(
    self,
    job_id: str,
    event_type: str,
    data: dict,
    retry_count: int = 3
) -> bool:
    """
    带重试的进度推送
  
    Args:
        job_id: 任务ID
        event_type: 事件类型
        data: 事件数据
        retry_count: 重试次数
  
    Returns:
        是否推送成功
    """
    config = settings.PROXY_CONFIG['sse']
  
    for attempt in range(retry_count):
        try:
            sse_manager.broadcast_sync(job_id, event_type, data)
            return True
        except Exception as e:
            logger.warning(
                f"SSE推送失败 (尝试 {attempt + 1}/{retry_count}): "
                f"job={job_id}, event={event_type}, error={e}"
            )
            if attempt < retry_count - 1:
                time.sleep(config['retry_delay'])
  
    logger.error(f"SSE推送最终失败: job={job_id}, event={event_type}")
    return False
```

#### 3.2.5 新增状态查询接口

**文件**: `backend/app/api/routes/media_routes.py`

新增端点：
```python
@router.get("/proxy-status/{job_id}")
async def get_proxy_status(job_id: str):
    """
    获取 Proxy 视频完整状态
  
    用于前端刷新后恢复状态
  
    Returns:
        {
            "state": "transcoding_720",  # 当前状态
            "progress": 45.5,            # 当前进度百分比
            "decision": "transcode_full", # 转码决策
            "urls": {
                "360p": "/api/media/.../preview_360p.mp4",
                "720p": null,
                "source": "/api/media/.../source.mp4"
            },
            "error": null,               # 错误信息
            "started_at": "2024-01-01T00:00:00Z",
            "estimated_remaining": 30    # 预估剩余秒数
        }
    """
    # 从 MediaPrepService 获取任务状态
    task_status = media_prep_service.get_task_status(job_id)
  
    # 检查文件存在性
    urls = {
        "360p": _get_url_if_exists(job_id, "preview_360p.mp4"),
        "720p": _get_url_if_exists(job_id, "proxy_720p.mp4"),
        "source": _get_source_url(job_id)
    }
  
    return {
        "state": task_status.state,
        "progress": task_status.progress,
        "decision": task_status.decision,
        "urls": urls,
        "error": task_status.error,
        "started_at": task_status.started_at,
        "estimated_remaining": task_status.estimated_remaining
    }
```

### 3.3 前端重构

#### 3.3.1 新建 useProxyVideo Composable

**文件**: `frontend/src/composables/useProxyVideo.js`

```javascript
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { sseChannelManager } from '@/services/sseChannelManager'
import { mediaApi } from '@/api/media'

/**
 * Proxy 视频状态管理 Composable
 * 
 * 职责：
 * 1. 管理 Proxy 视频完整生命周期
 * 2. 自动订阅/取消订阅 SSE 事件
 * 3. 处理刷新后状态恢复
 * 4. 提供统一的 isReady 状态
 */

// 状态枚举
export const ProxyState = {
  IDLE: 'idle',
  ANALYZING: 'analyzing',
  TRANSCODING_360: 'transcoding_360',
  READY_360P: 'ready_360p',
  TRANSCODING_720: 'transcoding_720',
  READY_720P: 'ready_720p',
  REMUXING: 'remuxing',
  DIRECT_PLAY: 'direct_play',
  ERROR: 'error'
}

export function useProxyVideo(jobId) {
  // 响应式状态
  const state = ref(ProxyState.IDLE)
  const progress = ref(0)
  const error = ref(null)
  const urls = ref({
    preview360p: null,
    proxy720p: null,
    source: null
  })

  // 计算属性
  const isReady = computed(() => {
    return [
      ProxyState.READY_360P,
      ProxyState.READY_720P,
      ProxyState.DIRECT_PLAY
    ].includes(state.value)
  })

  const isTranscoding = computed(() => {
    return [
      ProxyState.ANALYZING,
      ProxyState.TRANSCODING_360,
      ProxyState.TRANSCODING_720,
      ProxyState.REMUXING
    ].includes(state.value)
  })

  const currentUrl = computed(() => {
    // 优先级：720p > 360p > source
    if (urls.value.proxy720p) return urls.value.proxy720p
    if (urls.value.preview360p) return urls.value.preview360p
    if (state.value === ProxyState.DIRECT_PLAY) return urls.value.source
    return null
  })

  const currentResolution = computed(() => {
    if (urls.value.proxy720p) return '720p'
    if (urls.value.preview360p) return '360p'
    if (state.value === ProxyState.DIRECT_PLAY) return 'source'
    return null
  })

  // SSE 事件处理器
  const sseHandlers = {
    onAnalyzeComplete: (data) => {
      if (data.decision === 'direct_play') {
        state.value = ProxyState.DIRECT_PLAY
        urls.value.source = data.source_url
      } else if (data.decision === 'remux_only') {
        state.value = ProxyState.REMUXING
      } else {
        state.value = ProxyState.TRANSCODING_360
      }
    },
  
    onRemuxProgress: (data) => {
      progress.value = data.progress
    },
  
    onRemuxComplete: (data) => {
      state.value = ProxyState.READY_720P
      urls.value.proxy720p = data.url
      progress.value = 100
    },
  
    onPreview360pProgress: (data) => {
      progress.value = data.progress
    },
  
    onPreview360pComplete: (data) => {
      state.value = ProxyState.READY_360P
      urls.value.preview360p = data.url
      progress.value = 0 // 重置进度，准备720p
      // 自动开始 720p
      state.value = ProxyState.TRANSCODING_720
    },
  
    onProxyProgress: (data) => {
      progress.value = data.progress
    },
  
    onProxyComplete: (data) => {
      state.value = ProxyState.READY_720P
      urls.value.proxy720p = data.url
      progress.value = 100
    },
  
    onProxyError: (data) => {
      state.value = ProxyState.ERROR
      error.value = data.message
    }
  }

  // 初始化：恢复状态
  async function initialize() {
    if (!jobId.value) return
  
    try {
      // 从后端获取当前状态
      const status = await mediaApi.getProxyStatus(jobId.value)
    
      // 恢复状态
      state.value = status.state
      progress.value = status.progress || 0
      urls.value = {
        preview360p: status.urls?.['360p'] || null,
        proxy720p: status.urls?.['720p'] || null,
        source: status.urls?.source || null
      }
      error.value = status.error
    
      // 如果正在转码，订阅 SSE
      if (isTranscoding.value) {
        subscribeSSE()
      }
    } catch (e) {
      console.error('恢复 Proxy 状态失败:', e)
      // 失败时重新开始
      state.value = ProxyState.ANALYZING
      subscribeSSE()
    }
  }

  // 订阅 SSE
  function subscribeSSE() {
    if (!jobId.value) return
  
    sseChannelManager.subscribeJobChannel(jobId.value, {
      ...sseHandlers,
      // 其他非 proxy 相关的 handler 可以留空或传入
    })
  }

  // 取消订阅
  function unsubscribeSSE() {
    if (!jobId.value) return
    sseChannelManager.unsubscribeJobChannel(jobId.value)
  }

  // 重试
  function retry() {
    if (state.value !== ProxyState.ERROR) return
  
    error.value = null
    state.value = ProxyState.ANALYZING
    // 触发后端重新分析
    mediaApi.retryProxy(jobId.value)
    subscribeSSE()
  }

  // 生命周期
  onMounted(() => {
    initialize()
  })

  onUnmounted(() => {
    unsubscribeSSE()
  })

  // 监听 jobId 变化
  watch(jobId, (newId, oldId) => {
    if (oldId) unsubscribeSSE()
    if (newId) initialize()
  })

  return {
    // 状态
    state,
    progress,
    error,
    urls,
  
    // 计算属性
    isReady,
    isTranscoding,
    currentUrl,
    currentResolution,
  
    // 方法
    retry,
    subscribeSSE,
    unsubscribeSSE
  }
}
```

#### 3.3.2 VideoStage 控件拦截层

**文件**: `frontend/src/components/editor/VideoStage/index.vue`

新增拦截逻辑：
```vue
<template>
  <div class="video-stage" :class="{ 'video-not-ready': !isVideoReady }">
    <!-- 转码进度遮罩 -->
    <div v-if="showTranscodingOverlay" class="transcoding-overlay">
      <div class="transcoding-info">
        <div class="transcoding-status">{{ transcodingStatusText }}</div>
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: `${progress}%` }"></div>
        </div>
        <div class="progress-text">{{ progress.toFixed(1) }}%</div>
      </div>
    </div>
  
    <!-- 错误提示 -->
    <div v-if="showError" class="error-overlay">
      <div class="error-info">
        <div class="error-message">{{ errorMessage }}</div>
        <button @click="$emit('retry')">重试</button>
      </div>
    </div>
  
    <!-- 视频播放器 -->
    <video
      ref="videoRef"
      :src="currentUrl"
      @click="handleVideoClick"
      @timeupdate="handleTimeUpdate"
    />
  
    <!-- 自定义控制栏 -->
    <div class="video-controls" :class="{ disabled: !isVideoReady }">
      <button 
        class="play-btn" 
        @click="handlePlayPause"
        :disabled="!isVideoReady"
      >
        {{ isPlaying ? '暂停' : '播放' }}
      </button>
    
      <div 
        class="progress-slider"
        @mousedown="handleSeekStart"
        @click="handleSeekClick"
        :class="{ disabled: !isVideoReady }"
      >
        <div class="progress-track">
          <div class="progress-played" :style="{ width: `${playedPercent}%` }"></div>
        </div>
      </div>
    
      <div class="time-display">
        {{ formatTime(currentTime) }} / {{ formatTime(duration) }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { ProxyState } from '@/composables/useProxyVideo'

const props = defineProps({
  state: String,
  progress: Number,
  currentUrl: String,
  currentResolution: String,
  error: String
})

const emit = defineEmits(['retry', 'play', 'pause', 'seek'])

const videoRef = ref(null)
const isPlaying = ref(false)
const currentTime = ref(0)
const duration = ref(0)

// 视频是否就绪
const isVideoReady = computed(() => {
  return [
    ProxyState.READY_360P,
    ProxyState.READY_720P,
    ProxyState.DIRECT_PLAY
  ].includes(props.state)
})

// 显示转码遮罩
const showTranscodingOverlay = computed(() => {
  return [
    ProxyState.ANALYZING,
    ProxyState.TRANSCODING_360,
    ProxyState.TRANSCODING_720,
    ProxyState.REMUXING
  ].includes(props.state)
})

// 显示错误
const showError = computed(() => {
  return props.state === ProxyState.ERROR
})

// 转码状态文本
const transcodingStatusText = computed(() => {
  const texts = {
    [ProxyState.ANALYZING]: '分析视频中...',
    [ProxyState.REMUXING]: '容器重封装中...',
    [ProxyState.TRANSCODING_360]: '生成 360p 预览...',
    [ProxyState.TRANSCODING_720]: '生成 720p 高清...'
  }
  return texts[props.state] || '处理中...'
})

// 控件事件处理（带拦截）
function handlePlayPause() {
  if (!isVideoReady.value) {
    console.warn('视频未就绪，操作被拦截')
    return
  }

  if (isPlaying.value) {
    videoRef.value?.pause()
    emit('pause')
  } else {
    videoRef.value?.play()
    emit('play')
  }
  isPlaying.value = !isPlaying.value
}

function handleSeekStart(e) {
  if (!isVideoReady.value) {
    e.preventDefault()
    e.stopPropagation()
    console.warn('视频未就绪，拖动被拦截')
    return
  }
  // 正常处理拖动...
}

function handleSeekClick(e) {
  if (!isVideoReady.value) {
    console.warn('视频未就绪，跳转被拦截')
    return
  }
  // 正常处理点击跳转...
}

// 分辨率切换时保持播放状态
watch(() => props.currentUrl, async (newUrl, oldUrl) => {
  if (!newUrl || !oldUrl || !videoRef.value) return

  // 保存当前状态
  const wasPlaying = isPlaying.value
  const savedTime = videoRef.value.currentTime

  // 等待新源加载
  await new Promise(resolve => {
    videoRef.value.addEventListener('loadedmetadata', resolve, { once: true })
  })

  // 恢复状态
  videoRef.value.currentTime = savedTime
  if (wasPlaying) {
    videoRef.value.play()
  }
})
</script>

<style scoped>
.video-stage.video-not-ready {
  cursor: not-allowed;
}

.video-controls.disabled {
  opacity: 0.5;
  pointer-events: none;
}

.progress-slider.disabled {
  cursor: not-allowed;
}

.transcoding-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
}

.progress-bar {
  width: 300px;
  height: 8px;
  background: #333;
  border-radius: 4px;
  overflow: hidden;
  margin: 16px 0;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #4CAF50, #8BC34A);
  transition: width 0.3s ease;
}
</style>
```

#### 3.3.3 SSE 管理器扩展

**文件**: `frontend/src/services/sseChannelManager.js`

扩展事件类型：
```javascript
// 在 setupJobEventHandlers 中添加新事件
const jobEventMap = {
  // ... 现有事件 ...

  // Proxy 相关事件（扩展）
  analyze_complete: (data) => {
    console.log(`[SSE Job ${jobId}] 分析完成:`, data.decision)
    handlers.onAnalyzeComplete?.(data)
  },

  remux_progress: (data) => {
    console.log(`[SSE Job ${jobId}] 重封装进度:`, data.progress)
    handlers.onRemuxProgress?.(data)
  },

  remux_complete: (data) => {
    console.log(`[SSE Job ${jobId}] 重封装完成`)
    handlers.onRemuxComplete?.(data)
  },

  proxy_error: (data) => {
    console.error(`[SSE Job ${jobId}] Proxy 错误:`, data.message)
    handlers.onProxyError?.(data)
  },

  // 保留现有事件
  preview_360p_progress: (data) => {
    handlers.onPreview360pProgress?.(data)
  },
  preview_360p_complete: (data) => {
    handlers.onPreview360pComplete?.(data)
  },
  proxy_progress: (data) => {
    handlers.onProxyProgress?.(data)
  },
  proxy_complete: (data) => {
    handlers.onProxyComplete?.(data)
  }
}
```

### 3.4 FFmpeg 参数优化

#### 3.4.1 360p 预览（极速模式）

```python
# 目标：最快生成可预览的视频
cmd = [
    ffmpeg_cmd,
    '-i', str(video_path),
    # 输入优化
    '-threads', '0',              # 自动线程
    # 视频处理
    '-vf', f'scale=-2:{config["scale"]}',
    '-c:v', 'libx264',
    '-preset', config['preset'],  # ultrafast
    '-tune', 'fastdecode',        # 优化解码速度
    '-crf', str(config['crf']),   # 28
    # GOP 设置（关键）
    '-g', str(config['gop']),     # 30
    '-keyint_min', str(config['gop'] // 2),  # 15
    '-sc_threshold', '0',         # 禁用场景检测，保证 GOP 稳定
    # 无音频
    '-an',
    # 输出优化
    '-movflags', '+faststart',    # 元数据前置
    '-progress', 'pipe:1',
    '-y',
    str(output_path)
]
```

#### 3.4.2 720p 高清

```python
# 目标：平衡质量和速度
cmd = [
    ffmpeg_cmd,
    '-i', str(video_path),
    '-threads', '0',
    # 视频
    '-vf', f'scale=-2:{config["scale"]}',
    '-c:v', 'libx264',
    '-preset', config['preset'],  # fast
    '-crf', str(config['crf']),   # 23
    # GOP 设置
    '-g', str(config['gop']),     # 30
    '-keyint_min', str(config['gop'] // 2),
    '-sc_threshold', '0',
    # 音频
    '-c:a', 'aac',
    '-b:a', config['audio_bitrate'],  # 128k
    '-ar', '44100',               # 采样率
    # 输出
    '-movflags', '+faststart',
    '-progress', 'pipe:1',
    '-y',
    str(output_path)
]
```

#### 3.4.3 容器重封装

```python
# 目标：零转码，直接复制流
cmd = [
    ffmpeg_cmd,
    '-i', str(video_path),
    '-c:v', 'copy',               # 视频直接复制
    '-c:a', 'copy',               # 音频直接复制
    '-movflags', '+faststart',
    '-progress', 'pipe:1',
    '-y',
    str(output_path)
]
```

---

## 四、实现步骤

### 阶段一：后端基础重构

1. **统一配置** - 将所有硬编码参数移至 `config.py`
2. **智能决策器** - 实现 `analyze_transcode_decision()` 方法
3. **容器重封装** - 实现 `remux_container()` 方法
4. **进度推送重试** - 重构 `_broadcast_progress()` 添加重试机制
5. **状态查询接口** - 新增 `/api/media/proxy-status/{job_id}` 端点

### 阶段二：前端状态管理重构

1. **新建 Composable** - 创建 `useProxyVideo.js`
2. **SSE 事件扩展** - 扩展 `sseChannelManager.js` 支持新事件
3. **状态恢复逻辑** - 实现 `initialize()` 从后端恢复状态

### 阶段三：UI 交互优化

1. **控件拦截层** - 修改 `VideoStage/index.vue` 添加操作拦截
2. **进度显示** - 优化转码进度 UI
3. **错误处理** - 添加错误提示和重试按钮

### 阶段四：无感切换

1. **分辨率切换** - 实现 360p→720p 平滑切换
2. **状态保持** - 切换时保持播放进度和状态

### 阶段五：测试与优化

1. **单元测试** - 测试各决策分支
2. **集成测试** - 测试完整流程
3. **性能测试** - 测试重封装 vs 完整转码速度对比

---

## 五、测试用例

### 5.1 后端测试

| 测试场景       | 输入       | 预期结果          |
| -------------- | ---------- | ----------------- |
| H.264+AAC+MP4  | video.mp4  | `DIRECT_PLAY`     |
| H.264+AAC+MKV  | video.mkv  | `REMUX_ONLY`      |
| H.264+AAC+MOV  | video.mov  | `REMUX_ONLY`      |
| HEVC+AAC+MP4   | hevc.mp4   | `TRANSCODE_FULL`  |
| H.264+FLAC+MKV | video.mkv  | `TRANSCODE_AUDIO` |
| VP9+Opus+WebM  | video.webm | `TRANSCODE_FULL`  |

### 5.2 前端测试

| 测试场景   | 操作         | 预期结果               |
| ---------- | ------------ | ---------------------- |
| 视频未加载 | 点击播放     | 操作被拦截，控制台警告 |
| 视频未加载 | 拖动进度条   | 操作被拦截             |
| 转码中     | 刷新页面     | 自动恢复状态和进度     |
| 360p就绪   | 等待720p完成 | 自动切换，保持播放进度 |
| 转码失败   | 点击重试     | 重新开始转码           |

---

## 六、风险与缓解

| 风险               | 影响           | 缓解措施                             |
| ------------------ | -------------- | ------------------------------------ |
| FFprobe 分析不准确 | 错误的转码决策 | 添加后备逻辑，分析失败时默认完整转码 |
| 重封装后不可播放   | 用户无法观看   | 播放失败时自动降级到完整转码         |
| SSE 连接不稳定     | 进度丢失       | 定期轮询 `/proxy-status` 作为后备    |
| 并发任务过多       | 系统过载       | 保持现有优先队列机制，限制并发数     |

---

## 七、总结

本重构方案通过以下改进解决了全部 7 个问题：

1. **真实进度** - 带重试的 SSE 推送 + 后备轮询
2. **刷新恢复** - `/proxy-status` 接口 + `useProxyVideo` 自动初始化
3. **简化状态** - 单一 Composable 管理所有状态
4. **零转码优先** - 智能决策器 + 容器重封装
5. **FFmpeg 优化** - 配置化参数 + 强制 GOP
6. **无感切换** - 保存/恢复播放状态
7. **操作拦截** - `isVideoReady` 计算属性 + 控件禁用

---

wgh，以上就是完整的开发文档。需要我开始实现哪个部分吗？