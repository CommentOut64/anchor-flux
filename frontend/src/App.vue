<script setup>
/**
 * App.vue - 应用根组件
 *
 * 职责：
 * - 启动时同步后端任务列表（修复幽灵任务）
 * - 全局 SSE 事件监听
 * - 自动跳转到编辑器（转录完成时）
 * - 全局任务状态同步
 */
import { onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import sseChannelManager from '@/services/sseChannelManager'

const router = useRouter()
const route = useRoute()
const taskStore = useUnifiedTaskStore()

let unsubscribeGlobal = null

onMounted(async () => {
  console.log('[App] 应用已挂载，执行初始化')

  // 第一步：从后端同步任务列表（第一阶段修复：数据同步）
  console.log('[App] 步骤 1: 从后端同步任务列表...')
  try {
    const syncSuccess = await taskStore.syncTasksFromBackend()
    if (syncSuccess) {
      console.log('[App] 任务列表同步成功')
    } else {
      console.warn('[App] 任务列表同步失败，将使用本地 localStorage 数据')
    }
  } catch (error) {
    console.error('[App] 任务列表同步异常:', error)
  }

  // 第二步：订阅全局 SSE 事件流（用于实时更新）
  console.log('[App] 步骤 2: 订阅全局 SSE 事件流...')
  unsubscribeGlobal = sseChannelManager.subscribeGlobal({
    onInitialState(state) {
      console.log('[App] 全局初始状态:', state)

      // 同步任务列表到 store（第二阶段修复：实时更新）
      if (state.jobs && Array.isArray(state.jobs)) {
        state.jobs.forEach(job => {
          // 检查 store 中是否已有此任务
          const existingTask = taskStore.getTask(job.id)
          if (!existingTask) {
            // 添加新任务（包含完整信息）
            taskStore.addTask({
              job_id: job.id,
              filename: job.filename,
              status: job.status,
              progress: job.progress,
              message: job.message,
              phase: job.phase || (job.status === 'finished' ? 'editing' : 'transcribing'),
              createdAt: job.created_time || Date.now()
            })
          } else {
            // 更新现有任务
            taskStore.updateTask(job.id, {
              status: job.status,
              progress: job.progress,
              message: job.message,
              phase: job.phase || existingTask.phase,
              createdAt: job.created_time || existingTask.createdAt
            })
          }
        })
      }
    },

    onQueueUpdate(queue) {
      console.log('[App] 队列更新:', queue)
      // 可以在这里更新队列顺序
    },

    onJobStatus(jobId, status, data) {
      console.log(`[App] 任务 ${jobId} 状态变化:`, status, data)

      // 更新 store 中的任务状态
      const task = taskStore.getTask(jobId)
      if (task) {
        taskStore.updateTask(jobId, {
          status,
          message: data.message || ''
        })

        // 转录完成自动跳转到编辑器
        if (status === 'finished' && task.phase === 'transcribing') {
          console.log(`[App] 任务 ${jobId} 转录完成，准备跳转到编辑器`)

          // 更新阶段为编辑
          taskStore.updateTask(jobId, {
            phase: 'editing'
          })

          // 自动跳转到编辑器（仅在当前不在编辑器页面时）
          if (!route.path.startsWith('/editor/')) {
            console.log(`[App] 自动跳转到编辑器: /editor/${jobId}`)
            router.push(`/editor/${jobId}`)
          }
        }
      }
    },

    onJobProgress(jobId, percent, data) {
      console.log(`[App] 任务 ${jobId} 进度:`, percent)

      // 更新 store 中的任务进度（实时更新卡片）
      taskStore.updateTaskProgress(jobId, percent, data.status)

      // 更新消息
      if (data.message) {
        taskStore.updateTaskMessage(jobId, data.message)
      }
    },

    onConnected(data) {
      console.log('[App] 全局 SSE 连接成功:', data)
    }
  })
})

onUnmounted(() => {
  console.log('[App] 应用卸载，关闭 SSE 连接')

  // 取消全局订阅
  if (unsubscribeGlobal) {
    unsubscribeGlobal()
  }
})
</script>

<template>
  <router-view />
</template>

<style scoped>
/* 无作用域样式 */
</style>
