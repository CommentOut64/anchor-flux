# UnifiedTaskStore - 统一任务状态管理

## 概述

UnifiedTaskStore 是整个应用的核心状态管理器，负责管理所有任务的生命周期，从上传、转录到编辑、导出的完整流程。它实现了任务状态的统一管理，确保转录和编辑工作流的无缝衔接。

## 功能特性

1. **任务生命周期管理**
   - 任务阶段状态：uploading → transcribing → editing ��� exporting → completed
   - 自动状态转换和路由跳转
   - 任务进度跟踪

2. **任务队列管理**
   - 使用 Map 数据结构提高查找性能
   - 支持任务优先级调整
   - 实时状态更新

3. **数据持久化**
   - localStorage 任务列表持久化
   - 任务状态恢复机制
   - 防止刷新丢失

4. **实时通信**
   - SSE (Server-Sent Events) 集成
   - 实时进度更新
   - 任务完成通知

## 技术依赖

```json
{
  "pinia": "^3.0.0",
  "vue": "^3.4.0",
  "vue-router": "^4.0.0"
}
```

## 数据结构

```javascript
// 任务阶段枚举
const TaskPhase = {
  UPLOADING: 'uploading',      // 上传中
  TRANSCRIBING: 'transcribing', // 转录中
  EDITING: 'editing',          // 编辑中
  EXPORTING: 'exporting',      // 导出中
  COMPLETED: 'completed'       // 已完成
}

// 任务状态枚举（与后端一致）
const TaskStatus = {
  CREATED: 'created',
  QUEUED: 'queued',
  PROCESSING: 'processing',
  PAUSED: 'paused',
  FINISHED: 'finished',
  FAILED: 'failed',
  CANCELED: 'canceled'
}

// 任务数据结构
interface Task {
  job_id: string
  filename: string
  file_path?: string
  status: TaskStatus
  phase: TaskPhase
  progress: number
  message: string
  settings?: JobSettings
  createdAt: number
  updatedAt: number
  isDirty?: boolean
}
```

## 核心实现

```javascript
// stores/unifiedTaskStore.js
import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'

export const useUnifiedTaskStore = defineStore('unifiedTask', () => {
  const router = useRouter()

  // 核心数据
  const tasks = ref(new Map())
  const activeTaskId = ref(null)
  const currentTask = ref(null)

  // 自动进入编辑模式
  watch(() => tasks.value, (newTasks) => {
    for (const [id, task] of newTasks) {
      if (task.phase === TaskPhase.TRANSCRIBING && task.progress === 100) {
        // 转录完成，自动切换到编辑阶段
        task.phase = TaskPhase.EDITING
        // 自动跳转到编辑器
        router.push(`/editor/${id}`)
      }
    }
  }, { deep: true })

  // Actions
  function addTask(taskData) {
    const task = {
      ...taskData,
      createdAt: Date.now(),
      updatedAt: Date.now()
    }
    tasks.value.set(task.job_id, task)
    saveTasks()
  }

  function updateTaskStatus(jobId, status) {
    const task = tasks.value.get(jobId)
    if (task) {
      task.status = status
      task.updatedAt = Date.now()
      saveTasks()
    }
  }

  function updateTaskProgress(jobId, progress, status) {
    const task = tasks.value.get(jobId)
    if (task) {
      task.progress = progress
      if (status) task.status = status
      task.updatedAt = Date.now()
    }
  }

  async function loadTask(jobId) {
    const task = tasks.value.get(jobId)
    if (!task || task.status !== TaskStatus.FINISHED) return false

    currentTask.value = task
    activeTaskId.value = jobId

    if (router.currentRoute.value.path !== `/editor/${jobId}`) {
      router.push(`/editor/${jobId}`)
    }
    return true
  }

  async function saveCurrentTask() {
    if (!currentTask.value) return

    const projectStore = useProjectStore()
    await projectStore.saveProject()
    currentTask.value.isDirty = false
  }

  // 持久化
  function saveTasks() {
    const tasksArray = Array.from(tasks.value.values())
    localStorage.setItem('task-list', JSON.stringify(tasksArray))
  }

  function restoreTasks() {
    const saved = localStorage.getItem('task-list')
    if (saved) {
      try {
        const tasksArray = JSON.parse(saved)
        tasks.value = new Map(tasksArray.map(t => [t.job_id, t]))
      } catch (e) {
        console.error('恢复任务列表失败:', e)
      }
    }
  }

  // 初始化时恢复
  restoreTasks()

  // 监听任务变化，自动保存
  watch(tasks, saveTasks, { deep: true })

  return {
    TaskPhase,
    TaskStatus,
    tasks: computed(() => Array.from(tasks.value.values())),
    activeTaskId,
    currentTask,
    addTask,
    updateTaskStatus,
    updateTaskProgress,
    loadTask,
    saveCurrentTask,
    getTask: (jobId) => tasks.value.get(jobId)
  }
})
```

## API 接口

### 添加任务
```javascript
addTask(taskData: {
  job_id: string
  filename: string
  status: TaskStatus
  // ...其他字段
})
```

### 更新任务状态
```javascript
updateTaskStatus(jobId: string, status: TaskStatus)
```

### 更新任务进度
```javascript
updateTaskProgress(jobId: string, progress: number, status?: TaskStatus)
```

### 加载任务到编辑器
```javascript
loadTask(jobId: string): Promise<boolean>
```

### 保存当前任务
```javascript
saveCurrentTask(): Promise<void>
```

### 获取任务
```javascript
getTask(jobId: string): Task | undefined
```

## 与其他组件的关系

### 依赖关系
- 依赖 `ProjectStore` 进行实际的项目数据保存
- 依赖 `vue-router` 进行页面跳转

### 被依赖关系
- `TaskMonitor` 组件使用此 Store 获取和更新任务状态
- `EditorView` 组件使用此 Store 管理当前编辑任务
- 所有涉及任务操作的组件都依赖此 Store

## 使用示例

```vue
<script setup>
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'

const taskStore = useUnifiedTaskStore()

// 添加新任务
function handleFileUpload(file) {
  const task = {
    job_id: generateId(),
    filename: file.name,
    status: 'uploaded',
    phase: 'uploading',
    progress: 0,
    message: '文件已上传'
  }
  taskStore.addTask(task)
}

// 监听任务状态
watch(() => taskStore.currentTask, (task) => {
  if (task?.phase === 'editing') {
    console.log('进入编辑模式')
  }
})
</script>
```

## 配置选项

```javascript
// 可配置的常量
const MAX_TASKS_IN_MEMORY = 100  // 内存中最多保存的任务数
const AUTO_SAVE_INTERVAL = 3000  // 自动保存间隔（毫秒）
const TASK_CLEANUP_AGE = 7 * 24 * 60 * 60 * 1000  // 任务清理时间（7天）
```

## 错误处理

```javascript
// 任务加载失败处理
async function loadTaskWithFallback(jobId) {
  try {
    const success = await taskStore.loadTask(jobId)
    if (!success) {
      ElMessage.error('任务未找到或状态异常')
    }
  } catch (error) {
    console.error('加载任务失败:', error)
    ElMessage.error('任务加载失败，请重试')
  }
}
```

## 性能优化

1. **使用 Map 而非数组**
   - Map 的查找性能为 O(1)，优于数组的 O(n)

2. **防抖保存**
   - 使用 watch 的 throttle 选项减少 localStorage 写入

3. **任务清理**
   - 定期清理过期任务，避免内存泄漏

## 测试要点

1. 任��状态转换正确性
2. 持久化和恢复功能
3. 并发任务处理
4. 内存泄漏检测
5. 路由跳转时机

## 未来扩展

1. 任务优先级队列实现
2. 任务批量操作
3. 任务模板功能
4. 任务历史记录追踪
5. WebSocket 替代 SSE