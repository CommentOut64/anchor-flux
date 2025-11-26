已完成的修改总结

  1. EditorView.vue - 编辑器视图重构

  文件: frontend/src/views/EditorView.vue

  核心改动:

  | 功能     | 修改内容                                               |
  |--------|----------------------------------------------------|
  | 流式加载   | 任务创建后即可进入编辑器，无需等待转录完成                              |
  | 实时字幕更新 | 订阅 SSE 流实时接收转录进度，每5秒轮询刷新 segments                  |
  | 状态感知   | 根据任务状态（finished/processing/queued/created）采取不同加载策略 |
  | 进度显示   | 头部显示转录状态和进度百分比                                     |
  | 保存功能   | 调用后端 API 保存 SRT 内容                                 |

  关键函数:
  - loadProject() - 根据任务状态决定加载方式
  - loadCompletedSRT() - 加载已完成的 SRT 文件
  - loadTranscribingSegments() - 从 checkpoint 获取已转录的 segments
  - subscribeSSE() - 订阅实时进度推送
  - startProgressPolling() - 定期轮询刷新字幕数据

  ---
  2. TaskListView.vue - 任务列表修复

  文件: frontend/src/views/TaskListView.vue

  改动:

  | 问题     | 修复                                                      |
  |--------|---------------------------------------------------------|
  | 字段名不一致 | task.jobId → task.job_id, task.fileName → task.filename |
  | 状态值不一致 | completed → finished, transcribing → processing         |
  | 任务点击限制 | 移除状态限制，所有状态都可进入编辑器                                      |
  | 删除功能   | 添加后端 API 调用 transcriptionApi.cancelJob(jobId, true)     |
  | 状态文本映射 | 更新 getStatusText() 匹配后端枚举                               |

  ---
  3. transcriptionApi.js - API 修复

  文件: frontend/src/services/api/transcriptionApi.js

  改动:
  - startJob() 添加 Content-Type: multipart/form-data header，修复 422 错误

  ---
  未完成的工作

  以下是需要继续完成的任务：

  1. App.vue 全局 SSE 订阅

  目的: 在任务列表页面实时更新任务状态，转录完成后可自动跳转

  需要实现:
  // App.vue 或 main.js 中
  import { sseService } from '@/services/sseService'
  import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'

  // 连接全局事件流
  sseService.connect('/api/events/global')

  // 监听任务状态更新
  sseService.on('job_status', (data) => {
    const taskStore = useUnifiedTaskStore()
    taskStore.updateTask(data.job_id, {
      status: data.status,
      progress: data.progress
    })
  })

  // 监听任务完成，可选择自动跳转
  sseService.on('job_complete', (data) => {
    // 如果用户在任务列表页，显示通知或自动跳转
  })

  2. unifiedTaskStore 同步增强

  目的: 启动时从后端同步任务状态，确保数据一致

  需要实现:
  // unifiedTaskStore.js 中添加
  async function syncTasksFromBackend() {
    const queueStatus = await transcriptionApi.getQueueStatus()
    // 遍历更新本地任务状态
    for (const [jobId, jobData] of Object.entries(queueStatus.jobs)) {
      updateTask(jobId, {
        status: jobData.status,
        progress: jobData.progress
      })
    }
  }

  // 初始化时调用
  restoreTasks()
  syncTasksFromBackend()

  3. 视频预览组件检查

  需要验证:
  - VideoStage 组件是否正确构建视频 URL
  - 后端 /api/media/{job_id}/video 是否支持 Range 请求
  - 是否需要等待 Proxy 视频生成

  4. 波形组件检查

  需要验证:
  - WaveformTimeline 组件是否正确加载峰值数据
  - 后端 /api/media/{job_id}/peaks 是否正常工作
  - 音频文件是否可用

  5. 完整流程测试

  测试清单:
  1. 上传视频文件
  2. 任务加入队列
  3. 进入编辑器查看转录进度
  4. 实时更新字幕列表
  5. 转录完成后加载完整 SRT
  6. 编辑字幕
  7. 保存到后端
  8. 导出 SRT/VTT 文件