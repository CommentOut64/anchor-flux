# 前端 API 服务层设计文档

## 1. 后端 API 接口完整清单

### 1.1 转录任务相关 (`/api/*`)

| 方法 | 路径 | 功能 | 返回 |
|------|------|------|------|
| POST | `/api/upload` | 上传文件并自动创建转录任务 | `{job_id, filename, message, queue_position}` |
| POST | `/api/create-job` | 为本地input文件创建任务 | `{job_id, filename}` |
| POST | `/api/start` | 启动转录任务（加入队列） | `{job_id, started, queue_position}` |
| POST | `/api/cancel/{job_id}` | 取消任务 | `{job_id, canceled, data_deleted}` |
| POST | `/api/pause/{job_id}` | 暂停任务 | `{job_id, paused}` |
| POST | `/api/prioritize/{job_id}` | 任务插队 | `{job_id, prioritized, mode, queue_position}` |
| GET | `/api/status/{job_id}` | 获取任务详细状态 | 完整任务对象 + `media_status` |
| GET | `/api/download/{job_id}` | 下载SRT文件 | FileResponse |
| GET | `/api/queue-status` | 获取队列状态摘要 | `{queue, running, interrupted, jobs}` |
| GET | `/api/incomplete-jobs` | 获取未完成任务列表 | `{jobs, count}` |
| POST | `/api/restore-job/{job_id}` | 从checkpoint恢复任务 | 任务对象 |
| GET | `/api/check-resume/{job_id}` | 检查是否可恢复 | `{can_resume, progress, ...}` |

### 1.2 SSE 事件流 (`/api/*`)

| 方法 | 路径 | 功能 | 事件类型 |
|------|------|------|---------|
| GET | `/api/stream/{job_id}` | 单个任务进度推送 | `progress`, `signal`, `ping` |
| GET | `/api/events/global` | 全局任务状态推送 | `initial_state`, `queue_update`, `job_status`, `job_progress` |

### 1.3 媒体资源相关 (`/api/media/*`)

| 方法 | 路径 | 功能 | 返回 |
|------|------|------|------|
| GET | `/api/media/{job_id}/video` | 获取视频（支持Range，自动Proxy） | Video Stream (206) |
| GET | `/api/media/{job_id}/audio` | 获取音频WAV文件 | Audio Stream (206) |
| GET | `/api/media/{job_id}/peaks` | 获取波形峰值数据 | `{peaks, duration, method}` |
| GET | `/api/media/{job_id}/thumbnails` | 获取缩略图/Sprite图 | `{sprite, thumb_width, timestamps}` |
| GET | `/api/media/{job_id}/srt` | 获取SRT字幕内容 | `{job_id, filename, content}` |
| POST | `/api/media/{job_id}/srt` | 保存编辑后的SRT | `{success, message}` |
| GET | `/api/media/{job_id}/info` | 获取媒体信息摘要 | `{video, audio, peaks, srt}` |
| GET | `/api/media/{job_id}/proxy-status` | 检查Proxy视频生成状态 | `{exists, generating, progress}` |
| POST | `/api/media/{job_id}/post-process` | 转录后处理（预生成资源） | `{peaks, thumbnails, proxy}` |

### 1.4 模型管理相关 (`/api/models/*`)

| 方法 | 路径 | 功能 | 返回 |
|------|------|------|------|
| GET | `/api/models/whisper` | 列出Whisper模型 | `[{id, name, size, status}]` |
| GET | `/api/models/align` | 列出对齐模型 | `[{language, status}]` |
| POST | `/api/models/whisper/{model_id}/download` | 下载模型 | `{success, message}` |
| DELETE | `/api/models/whisper/{model_id}` | 删除模型 | `{success, message}` |
| GET | `/api/models/stream` | SSE模型下载进度 | SSE Stream |

### 1.5 文件管理相关 (`/api/*`)

| 方法 | 路径 | 功能 | 返回 |
|------|------|------|------|
| GET | `/api/files` | 列出input目录文件 | `{files, input_dir}` |
| DELETE | `/api/files/{filename}` | 删除input文件 | `{success, message}` |

---

## 2. 前端 API 服务层架构设计

### 2.1 设计原则

1. **模块化分层** - 按功能模块划分（transcription, media, models等）
2. **统一错误处理** - 集中处理HTTP错误和网络异常
3. **请求拦截器** - 统一配置baseURL、超时、请求头等
4. **SSE独立管理** - 单独的SSE服务类，支持自动重连
5. **类型安全** - 明确的接口返回类型（通过JSDoc）
6. **Promise化** - 所有API返回Promise，便于async/await
7. **单例模式** - 确保全局只有一个HTTP客户端和SSE管理器
8. **缓存策略** - 媒体资源URL缓存，减少重复请求

### 2.2 目录结构

```
frontend/src/services/
├── api/
│   ├── index.js              # API服务总入口，导出所有服务
│   ├── client.js             # Axios实例配置，请求/响应拦截器
│   ├── transcriptionApi.js   # 转录任务相关API
│   ├── mediaApi.js           # 媒体资源相关API
│   ├── modelApi.js           # 模型管理相关API
│   └── fileApi.js            # 文件管理相关API
├── sseService.js             # SSE事件流管理（增强现有）
└── cacheService.js           # 已存在，用于缓存
```

### 2.3 核心模块设计

#### 2.3.1 `client.js` - HTTP客户端基础配置

**职责**：
- 创建配置好的Axios实例
- 添加请求/响应拦截器
- 统一错误处理和转换
- 支持请求取消

**接口**：
```javascript
// 创建并导出axios实例
export const apiClient

// 请求拦截器 - 添加baseURL、token等
apiClient.interceptors.request.use(config => {
  // 设置baseURL
  // 添加认证token（如需要）
  return config
})

// 响应拦截器 - 统一错误处理
apiClient.interceptors.response.use(
  response => response.data,
  error => {
    // 处理HTTP错误
    // 处理网络错误
    // 抛出统一格式的错误
  }
)
```

#### 2.3.2 `transcriptionApi.js` - 转录任务API

**职责**：管理转录任务的完整生命周期

**类设计**：
```javascript
class TranscriptionAPI {
  /**
   * 上传文件并创建转录任务
   * @param {File} file - 视频文件对象
   * @param {Function} onProgress - 上传进度回调 (percent) => void
   * @returns {Promise<{job_id, filename, queue_position}>}
   */
  async uploadFile(file, onProgress)

  /**
   * 启动转录任务（加入队列）
   * @param {string} jobId - 任务ID
   * @param {Object} settings - 转录设置 {model, compute_type, device, batch_size}
   * @returns {Promise<{job_id, started, queue_position}>}
   */
  async startJob(jobId, settings)

  /**
   * 取消任务
   * @param {string} jobId - 任务ID
   * @param {boolean} deleteData - 是否删除任务数据
   * @returns {Promise<{canceled, data_deleted}>}
   */
  async cancelJob(jobId, deleteData = false)

  /**
   * 暂停任务
   * @param {string} jobId - 任务ID
   * @returns {Promise<{paused}>}
   */
  async pauseJob(jobId)

  /**
   * 任务插队
   * @param {string} jobId - 任务ID
   * @param {string} mode - 'gentle' | 'force'
   * @returns {Promise<{prioritized, queue_position}>}
   */
  async prioritizeJob(jobId, mode = 'gentle')

  /**
   * 获取任务状态
   * @param {string} jobId - 任务ID
   * @param {boolean} includeMedia - 是否包含媒体状态
   * @returns {Promise<Object>} 完整任务对象
   */
  async getJobStatus(jobId, includeMedia = true)

  /**
   * 获取队列状态
   * @returns {Promise<{queue, running, interrupted, jobs}>}
   */
  async getQueueStatus()

  /**
   * 下载SRT文件
   * @param {string} jobId - 任务ID
   * @returns {Promise<Blob>} SRT文件Blob
   */
  async downloadResult(jobId)

  /**
   * 获取所有未完成任务
   * @returns {Promise<{jobs, count}>}
   */
  async getIncompleteJobs()

  /**
   * 检查任务是否可恢复
   * @param {string} jobId - 任务ID
   * @returns {Promise<{can_resume, progress, message}>}
   */
  async checkResume(jobId)

  /**
   * 恢复任务
   * @param {string} jobId - 任务ID
   * @returns {Promise<Object>} 任务对象
   */
  async restoreJob(jobId)
}

export default new TranscriptionAPI()
```

#### 2.3.3 `mediaApi.js` - 媒体资源API

**职责**：管理编辑器所需的所有媒体资源

**类设计**：
```javascript
class MediaAPI {
  /**
   * 获取视频URL（会自动处理Proxy）
   * @param {string} jobId - 任务ID
   * @returns {string} 视频URL
   */
  getVideoUrl(jobId)

  /**
   * 获取音频URL
   * @param {string} jobId - 任务ID
   * @returns {string} 音频URL
   */
  getAudioUrl(jobId)

  /**
   * 获取波形峰值数据
   * @param {string} jobId - 任务ID
   * @param {number} samples - 采样点数（默认2000）
   * @returns {Promise<{peaks, duration, method}>}
   */
  async getPeaks(jobId, samples = 2000)

  /**
   * 获取视频缩略图
   * @param {string} jobId - 任务ID
   * @param {number} count - 缩略图数量
   * @param {boolean} sprite - 是否使用Sprite图
   * @returns {Promise<{sprite, thumb_width, timestamps}>}
   */
  async getThumbnails(jobId, count = 10, sprite = true)

  /**
   * 获取SRT字幕内容
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id, filename, content}>}
   */
  async getSRTContent(jobId)

  /**
   * 保存编辑后的SRT内容
   * @param {string} jobId - 任务ID
   * @param {string} content - SRT文本内容
   * @returns {Promise<{success, message, filename}>}
   */
  async saveSRTContent(jobId, content)

  /**
   * 获取媒体信息摘要（一次性获取所有资源状态）
   * @param {string} jobId - 任务ID
   * @returns {Promise<{video, audio, peaks, srt, thumbnails}>}
   */
  async getMediaInfo(jobId)

  /**
   * 检查Proxy视频生成状态
   * @param {string} jobId - 任务ID
   * @returns {Promise<{exists, generating, progress, status}>}
   */
  async getProxyStatus(jobId)

  /**
   * 触发转录后处理（预生成所有资源）
   * @param {string} jobId - 任务ID
   * @returns {Promise<{peaks, thumbnails, proxy}>}
   */
  async postProcess(jobId)
}

export default new MediaAPI()
```

#### 2.3.4 `sseService.js` - SSE事件流管理（增强）

**职责**：管理所有SSE连接，支持自动重连和事件分发

**类设计**：
```javascript
class SSEService {
  /**
   * 订阅全局事件流（所有任务状态变化）
   * @param {Object} handlers - 事件处理器
   *   {
   *     onInitialState: (state) => void,
   *     onQueueUpdate: (queue) => void,
   *     onJobStatus: (jobId, status) => void,
   *     onJobProgress: (jobId, progress) => void
   *   }
   * @returns {Function} 取消订阅函数
   */
  subscribeGlobal(handlers)

  /**
   * 订阅单个任务事件流
   * @param {string} jobId - 任务ID
   * @param {Object} handlers - 事件处理器
   *   {
   *     onProgress: (data) => void,
   *     onComplete: (data) => void,
   *     onFailed: (data) => void,
   *     onProxyProgress: (progress) => void
   *   }
   * @returns {Function} 取消订阅函数
   */
  subscribeJob(jobId, handlers)

  /**
   * 取消订阅
   * @param {string} channel - 频道ID
   */
  unsubscribe(channel)

  /**
   * 取消所有订阅
   */
  unsubscribeAll()

  /**
   * 重连指定频道
   * @param {string} channel - 频道ID
   */
  reconnect(channel)
}

export default new SSEService()
```

#### 2.3.5 `modelApi.js` - 模型管理API

**职责**：管理Whisper和对齐模型的下载、删除

**类设计**：
```javascript
class ModelAPI {
  /**
   * 获取Whisper模型列表
   * @returns {Promise<Array>} 模型列表
   */
  async listWhisperModels()

  /**
   * 获取对齐模型列表
   * @returns {Promise<Array>} 对齐模型列表
   */
  async listAlignModels()

  /**
   * 下载Whisper模型
   * @param {string} modelId - 模型ID (tiny, base, small, medium, large-v2, large-v3)
   * @returns {Promise<{success, message}>}
   */
  async downloadWhisperModel(modelId)

  /**
   * 下载对齐模型
   * @param {string} language - 语言代码 (zh, en, ja, etc.)
   * @returns {Promise<{success, message}>}
   */
  async downloadAlignModel(language)

  /**
   * 删除Whisper模型
   * @param {string} modelId - 模型ID
   * @returns {Promise<{success, message}>}
   */
  async deleteWhisperModel(modelId)
}

export default new ModelAPI()
```

#### 2.3.6 `index.js` - API服务总入口

**职责**：统一导出所有API服务

```javascript
export { default as transcriptionApi } from './transcriptionApi'
export { default as mediaApi } from './mediaApi'
export { default as modelApi } from './modelApi'
export { default as fileApi } from './fileApi'
export { default as sseService } from '../sseService'
export { apiClient } from './client'
```

---

## 3. 完整流水线流程对接

### 3.1 Uploading -> Transcribing

**前端实现**：
1. 用户选择文件
2. 调用 `transcriptionApi.uploadFile(file, onProgress)`
3. 获得 `job_id`
4. 调用 `transcriptionApi.startJob(jobId, settings)` 启动任务
5. 订阅 `sseService.subscribeJob(jobId, handlers)` 监听进度

**Store集成**：
```javascript
// TaskListView.vue
async function handleUpload() {
  const { job_id } = await transcriptionApi.uploadFile(selectedFile.value, (progress) => {
    // 更新上传进度
  })

  await transcriptionApi.startJob(job_id, defaultSettings)

  // 添加到 unifiedTaskStore
  taskStore.addTask({
    job_id,
    filename: selectedFile.value.name,
    status: 'queued',
    phase: 'transcribing'
  })
}
```

### 3.2 Transcribing -> (Auto Jump)

**前端实现**：
1. 监听全局SSE流 `sseService.subscribeGlobal()`
2. 接收 `job_status` 事件，检测 `status === 'finished'`
3. 自动调用 `router.push(/editor/${jobId})`

**Store集成**：
```javascript
// App.vue 或 main.js
sseService.subscribeGlobal({
  onJobStatus(jobId, status) {
    const task = taskStore.getTask(jobId)
    if (task && status === 'finished' && task.phase === 'transcribing') {
      // 自动跳转到编辑器
      router.push(`/editor/${jobId}`)
      // 更新阶段
      taskStore.updateTask(jobId, { phase: 'editing' })
    }
  }
})
```

### 3.3 Editing -> Exporting

**前端实现**：
1. 编辑器加载时调用 `mediaApi.getMediaInfo(jobId)` 获取所有资源
2. 加载SRT: `mediaApi.getSRTContent(jobId)`
3. 编辑后保存: `mediaApi.saveSRTContent(jobId, content)`
4. 导出: 使用前端生成的SRT内容直接下载

---

## 4. 错误处理策略

### 4.1 HTTP错误处理

```javascript
// client.js 响应拦截器
apiClient.interceptors.response.use(
  response => response.data,
  error => {
    if (error.response) {
      // HTTP错误 (4xx, 5xx)
      const { status, data } = error.response
      if (status === 404) {
        // 资源未找到
      } else if (status === 500) {
        // 服务器错误
      }
      throw new APIError(status, data.detail || '请求失败')
    } else if (error.request) {
      // 网络错误
      throw new NetworkError('网络连接失败')
    } else {
      throw error
    }
  }
)
```

### 4.2 SSE断线重连

```javascript
// sseService.js
class SSEService {
  _setupAutoReconnect(channel) {
    // 监听error事件
    eventSource.onerror = () => {
      this.reconnect(channel)
    }
  }

  reconnect(channel, maxRetries = 3) {
    // 指数退避重连
    setTimeout(() => {
      if (retries < maxRetries) {
        this.subscribe(channel, handlers)
      }
    }, Math.pow(2, retries) * 1000)
  }
}
```

---

## 5. 使用示例

### 5.1 上传并启动任务

```javascript
import { transcriptionApi, sseService } from '@/services/api'

// 上传文件
const { job_id } = await transcriptionApi.uploadFile(file, (progress) => {
  console.log(`上传进度: ${progress}%`)
})

// 启动任务
await transcriptionApi.startJob(job_id, {
  model: 'medium',
  compute_type: 'float16',
  device: 'cuda',
  batch_size: 16
})

// 监听进度
const unsubscribe = sseService.subscribeJob(job_id, {
  onProgress(data) {
    console.log(`转录进度: ${data.percent}%`)
  },
  onComplete(data) {
    console.log('转录完成！')
    unsubscribe()
  }
})
```

### 5.2 加载编辑器资源

```javascript
import { mediaApi } from '@/services/api'

// 获取媒体信息摘要
const mediaInfo = await mediaApi.getMediaInfo(jobId)

if (mediaInfo.video.exists) {
  // 加载视频
  videoElement.src = mediaApi.getVideoUrl(jobId)
}

if (mediaInfo.audio.exists) {
  // 加载波形
  const { peaks, duration } = await mediaApi.getPeaks(jobId)
  wavesurfer.load(peaks, duration)
}

if (mediaInfo.srt.exists) {
  // 加载字幕
  const { content } = await mediaApi.getSRTContent(jobId)
  projectStore.importSRT(content, metadata)
}
```

---

## 6. 总结

### 优势：
1. **模块化清晰** - 每个API模块职责单一
2. **易于测试** - 单例类便于mock和单元测试
3. **统一错误处理** - 减少重复代码
4. **类型安全** - JSDoc提供良好的IDE提示
5. **SSE独立管理** - 自动重连，事件分发
6. **便于维护** - 后端API变更只需修改对应模块

### 下一步：
1. 实现 `client.js` 基础HTTP客户端
2. 实现 `transcriptionApi.js`
3. 实现 `mediaApi.js`
4. 增强 `sseService.js`
5. 集成到现有组件和Store中
