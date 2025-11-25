# TaskMonitor - 任务监控组件

## 概述

TaskMonitor 是一个非侵入式的任务监控弹窗组件，设计灵感来自 Edge 浏览器的下载管理界面。它提供了实时的任务状态监控、进度显示和任务控制功能，支持暂停、恢复、取消等操作。

## 功能特性

1. **实时任务监控**
   - SSE (Server-Sent Events) 实时更新
   - 自动刷新任务状态
   - 进度条可视化

2. **任务控制**
   - 暂停/恢复处理中的任务
   - 取消未完成的任务
   - 任务优先级调整（预留）

3. **交互设计**
   - 弹窗式非侵入设计
   - 最多显示8个最近任务
   - 一键查看所有任务

4. **状态管理**
   - 与 UnifiedTaskStore 深度集成
   - 任务状态持久化
   - 未保存内容提醒

## 技术依赖

```json
{
  "vue": "^3.4.0",
  "element-plus": "^2.11.0",
  "axios": "^1.0.0",
  "@element-plus/icons-vue": "^2.0.0"
}
```

## 组件属性

```typescript
// Props
interface TaskMonitorProps {
  maxRecentTasks?: number    // 最多显示的任务数，默认8
  pollInterval?: number       // 轮询间隔（毫秒），默认3000
  position?: string          // 弹窗位置，默认'bottom-end'
}

// Emits
interface TaskMonitorEmits {
  'task-click': (task: Task) => void
  'task-edit': (jobId: string) => void
  'open-manager': () => void
}
```

## 核心实现

```vue
<!-- components/TaskMonitor/index.vue -->
<template>
  <el-popover
    placement="bottom-end"
    :width="380"
    trigger="click"
    popper-class="task-popover-custom"
  >
    <template #reference>
      <div class="task-trigger-btn">
        <el-badge :value="activeCount" :hidden="activeCount === 0" type="primary">
          <el-button circle>
            <el-icon :size="20">
              <Loading v-if="hasRunningTask" />
              <List v-else />
            </el-icon>
          </el-button>
        </el-badge>
      </div>
    </template>

    <div class="task-list-container">
      <!-- Header -->
      <div class="popover-header">
        <span>任务列表</span>
        <div class="header-actions">
          <span class="task-count">{{ activeCount }} 个进行中</span>
          <el-button
            v-if="activeCount > 0"
            link
            size="small"
            @click="pauseAll"
          >
            全部暂停
          </el-button>
        </div>
      </div>

      <!-- Task List -->
      <el-scrollbar max-height="400px">
        <div v-if="recentTasks.length === 0" class="empty-state">
          <el-icon :size="48"><Folder /></el-icon>
          <p>暂无任务</p>
        </div>

        <div
          v-for="task in recentTasks"
          :key="task.job_id"
          class="task-item"
          @click="handleTaskClick(task)"
          :class="{
            'is-active': currentTaskId === task.job_id,
            'clickable': task.status === 'finished'
          }"
        >
          <!-- Task Icon -->
          <div class="task-icon">
            <el-icon v-if="task.status === 'finished'" color="#67C23A">
              <CircleCheckFilled />
            </el-icon>
            <el-icon v-else-if="task.status === 'failed'" color="#F56C6C">
              <CircleCloseFilled />
            </el-icon>
            <el-icon v-else-if="task.status === 'paused'" color="#E6A23C">
              <VideoPause />
            </el-icon>
            <el-icon v-else class="is-loading">
              <Loading />
            </el-icon>
          </div>

          <!-- Task Info -->
          <div class="task-info">
            <div class="task-name" :title="task.filename">
              {{ task.filename }}
            </div>

            <div class="task-status">
              <div v-if="task.status === 'processing'" class="progress-wrapper">
                <el-progress
                  :percentage="task.progress || 0"
                  :stroke-width="3"
                  :show-text="false"
                />
                <span class="progress-text">{{ task.progress || 0 }}%</span>
              </div>
              <div v-else class="status-text">
                {{ getStatusText(task) }}
              </div>
            </div>
          </div>

          <!-- Task Actions -->
          <div class="task-actions" @click.stop>
            <el-button
              v-if="canPause(task)"
              link
              size="small"
              @click="pauseTask(task)"
              title="暂停"
            >
              <el-icon><VideoPause /></el-icon>
            </el-button>

            <el-button
              v-else-if="canResume(task)"
              link
              size="small"
              @click="resumeTask(task)"
              title="继续"
            >
              <el-icon><VideoPlay /></el-icon>
            </el-button>

            <el-button
              v-if="canCancel(task)"
              type="danger"
              link
              size="small"
              @click="cancelTask(task)"
              title="取消"
            >
              <el-icon><Close /></el-icon>
            </el-button>

            <el-button
              v-if="task.status === 'finished'"
              type="primary"
              link
              size="small"
              @click="editTask(task)"
              title="编辑字幕"
            >
              <el-icon><Edit /></el-icon>
            </el-button>
          </div>
        </div>
      </el-scrollbar>

      <!-- Footer -->
      <div class="popover-footer" @click="openTaskManager">
        <span>查看所有任务</span>
        <el-icon><ArrowRight /></el-icon>
      </div>
    </div>
  </el-popover>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

const router = useRouter()
const taskStore = useUnifiedTaskStore()

// Props
const props = defineProps({
  maxRecentTasks: {
    type: Number,
    default: 8
  },
  pollInterval: {
    type: Number,
    default: 3000
  }
})

// 最近任务列表
const recentTasks = computed(() => {
  return taskStore.tasks
    .slice()
    .sort((a, b) => b.createdAt - a.createdAt)
    .slice(0, props.maxRecentTasks)
})

// 活跃任务数量
const activeCount = computed(() =>
  taskStore.tasks.filter(t =>
    ['queued', 'processing'].includes(t.status)
  ).length
)

// 是否有正在运行的任务
const hasRunningTask = computed(() =>
  taskStore.tasks.some(t => t.status === 'processing')
)

// 当前编辑器打开的任务ID
const currentTaskId = computed(() => taskStore.activeTaskId)

// SSE 连接管理
let eventSource = null

onMounted(() => {
  connectSSE()
  startPolling()
})

onUnmounted(() => {
  disconnectSSE()
  stopPolling()
})

// SSE 连接
function connectSSE() {
  eventSource = new EventSource('/api/tasks/stream')

  eventSource.addEventListener('task_progress', (e) => {
    const data = JSON.parse(e.data)
    taskStore.updateTaskProgress(data.job_id, data.progress, data.status)
  })

  eventSource.addEventListener('task_complete', (e) => {
    const data = JSON.parse(e.data)
    taskStore.updateTaskStatus(data.job_id, 'finished')
    ElMessage.success(`任务 ${data.filename} 已完成`)
  })

  eventSource.addEventListener('error', () => {
    console.error('SSE连接错误，将在5秒后重连')
    setTimeout(connectSSE, 5000)
  })
}

function disconnectSSE() {
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
}

// 定时轮询（备用方案）
let pollTimer = null

function startPolling() {
  pollTimer = setInterval(async () => {
    if (!hasRunningTask.value) return

    try {
      const response = await axios.get('/api/tasks/status')
      response.data.forEach(task => {
        taskStore.updateTaskStatus(task.job_id, task.status)
        if (task.progress !== undefined) {
          taskStore.updateTaskProgress(task.job_id, task.progress)
        }
      })
    } catch (error) {
      console.error('轮询失败:', error)
    }
  }, props.pollInterval)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

// 任务操作
async function pauseTask(task) {
  try {
    await axios.post(`/api/pause/${task.job_id}`)
    taskStore.updateTaskStatus(task.job_id, 'paused')
    ElMessage.success('任务已暂停')
  } catch (error) {
    ElMessage.error('暂停失败')
  }
}

async function resumeTask(task) {
  try {
    const formData = new FormData()
    formData.append('job_id', task.job_id)
    formData.append('settings', JSON.stringify(task.settings))

    await axios.post('/api/start', formData)
    taskStore.updateTaskStatus(task.job_id, 'queued')
    ElMessage.success('任务已恢复')
  } catch (error) {
    ElMessage.error('恢复失败')
  }
}

async function cancelTask(task) {
  try {
    await ElMessageBox.confirm(
      '确定要取消此任务吗？',
      '确认取消',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    await axios.post(`/api/cancel/${task.job_id}`)
    taskStore.updateTaskStatus(task.job_id, 'canceled')
    ElMessage.success('任务已取消')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('取消失败')
    }
  }
}

function editTask(task) {
  router.push(`/editor/${task.job_id}`)
}

async function pauseAll() {
  const activeTasks = taskStore.tasks.filter(t =>
    ['queued', 'processing'].includes(t.status)
  )

  for (const task of activeTasks) {
    await pauseTask(task)
  }
}

// 任务点击处理
async function handleTaskClick(task) {
  if (task.status !== 'finished') return

  // 检查未保存内容
  if (taskStore.currentTask?.isDirty) {
    await ElMessageBox.confirm(
      '当前任务有未保存的修改，是否保存？',
      '切换任务',
      {
        confirmButtonText: '保���并切换',
        cancelButtonText: '直接切换',
        distinguishCancelAndClose: true,
        type: 'warning'
      }
    ).then(() => {
      taskStore.saveCurrentTask()
    }).catch((action) => {
      if (action !== 'close') {
        // 用户选择直接切换
      }
    })
  }

  editTask(task)
}

function openTaskManager() {
  router.push('/tasks')
}

// 状态判断
function canPause(task) {
  return ['processing', 'queued'].includes(task.status)
}

function canResume(task) {
  return task.status === 'paused'
}

function canCancel(task) {
  return !['finished', 'canceled'].includes(task.status)
}

function getStatusText(status) {
  const statusMap = {
    'created': '已创建',
    'uploaded': '已上传',
    'queued': '排队中',
    'processing': '处理中',
    'finished': '已完成',
    'failed': '失败',
    'canceled': '已取消',
    'paused': '已暂停'
  }
  return statusMap[status] || status
}
</script>
```

## 样式定制

```scss
// 主题变量
$popover-bg: var(--bg-secondary);
$popover-border: var(--border-color);
$item-hover-bg: rgba(255, 255, 255, 0.05);
$item-active-bg: rgba(64, 158, 255, 0.2);

// 尺寸变量
$popover-width: 380px;
$max-height: 400px;
$item-height: 80px;

// 自定义弹窗样式
.task-popover-custom {
  padding: 0 !important;
  background: $popover-bg !important;
  border: 1px solid $popover-border !important;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3) !important;
  border-radius: 12px !important;
}
```

## API 接口

### 后端接口要求

```typescript
// 获取任务状态
GET /api/tasks/status
Response: Task[]

// SSE 事件流
GET /api/tasks/stream
Events: task_progress, task_complete, task_failed

// 暂停任务
POST /api/pause/:jobId

// 恢复任务
POST /api/start
Body: { job_id, settings }

// 取消任务
POST /api/cancel/:jobId
```

## 与其他组件的关系

### 依赖关系
- 强依赖 `UnifiedTaskStore` 进行任务状态管理
- 依赖 `vue-router` 进行页面跳转
- 依赖 `axios` 进行API调用

### 被依赖关系
- 被 `EditorView` 组件包含使用

## 使用示例

```vue
<template>
  <div class="editor-header">
    <!-- 任务监控按钮 -->
    <TaskMonitor
      :max-recent-tasks="10"
      :poll-interval="5000"
      @task-edit="handleTaskEdit"
    />
  </div>
</template>

<script setup>
import TaskMonitor from '@/components/TaskMonitor'

function handleTaskEdit(jobId) {
  console.log('编辑任务:', jobId)
}
</script>
```

## 性能优化

1. **虚拟列表**
   - 大量任务时考虑使用虚拟滚动

2. **防抖更新**
   - 频繁进度更新使用节流

3. **SSE 重连策略**
   - 指数退避算法避免频繁重连

## 测试要点

1. SSE 连接稳定性
2. 任务状态切换正确性
3. 并发操作处理
4. 内存泄漏（事件监听器清理）
5. 错误恢复机制

## 未来扩展

1. 任务分组显示
2. 任务搜索过滤
3. 批量任务操作
4. 任务统计图表
5. 任务通知推送