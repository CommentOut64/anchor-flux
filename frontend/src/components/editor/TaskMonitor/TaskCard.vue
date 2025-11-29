<template>
  <div
    class="task-card"
    :class="[
      `variant-${variant}`,
      { 'is-draggable': draggable }
    ]"
  >
    <!-- 拖动手柄 -->
    <div v-if="draggable" class="drag-handle" title="拖动排序">
      <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M11 18c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm-2-8c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0-6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm6 4c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
      </svg>
    </div>

    <!-- 任务信息 -->
    <div class="task-info">
      <div class="task-header">
        <span class="task-name" :title="task.title || task.filename">
          {{ task.title || task.filename }}
        </span>
        <span
          class="task-phase"
          :style="{
            background: getPhaseStyle(task).bgColor,
            color: getPhaseStyle(task).color
          }"
        >
          {{ getPhaseLabel(task) }}
        </span>
      </div>

      <!-- 进度条 -->
      <div v-if="showProgress" class="task-progress">
        <div class="progress-bar">
          <div
            class="progress-fill"
            :style="{ width: task.progress + '%' }"
          ></div>
        </div>
        <span class="progress-text">{{ formatProgress(task.progress) }}%</span>
      </div>

      <!-- 错误信息 -->
      <div v-if="task.status === 'failed'" class="task-error">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
        <span>{{ task.message || '转录失败' }}</span>
      </div>

      <!-- 完成时间 -->
      <div v-if="task.status === 'finished'" class="task-meta">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/>
        </svg>
        <span>{{ formatTime(task.completed_at || task.updatedAt) }}</span>
      </div>
    </div>

    <!-- 操作按钮 -->
    <div class="task-actions">
      <button
        v-if="task.status === 'processing'"
        class="action-btn"
        @click="pauseTask"
        title="暂停"
      >
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
        </svg>
      </button>

      <button
        v-if="task.status === 'paused'"
        class="action-btn action-btn--success"
        @click="resumeTask"
        title="恢复"
      >
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M8 5v14l11-7z"/>
        </svg>
      </button>

      <button
        v-if="['processing', 'queued', 'paused'].includes(task.status)"
        class="action-btn action-btn--danger"
        @click="cancelTask"
        title="取消"
      >
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
        </svg>
      </button>

      <button
        v-if="task.status === 'finished'"
        class="action-btn action-btn--primary"
        @click="openEditor"
        title="打开编辑器"
      >
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { transcriptionApi } from '@/services/api'
import { PHASE_CONFIG, STATUS_CONFIG, formatProgress } from '@/constants/taskPhases'

const props = defineProps({
  task: { type: Object, required: true },
  variant: { type: String, default: 'default' },
  draggable: { type: Boolean, default: false }
})

const router = useRouter()

const showProgress = computed(() =>
  ['processing', 'queued', 'paused'].includes(props.task.status)
)

// 获取阶段样式
function getPhaseStyle(task) {
  if (task.status === 'failed') return STATUS_CONFIG.failed
  if (task.status === 'processing' && task.phase) {
    return PHASE_CONFIG[task.phase] || PHASE_CONFIG.pending
  }
  return STATUS_CONFIG[task.status] || STATUS_CONFIG.created
}

// 获取阶段标签
function getPhaseLabel(task) {
  if (task.status === 'failed') return '失败'
  if (task.status === 'processing' && task.phase) {
    return PHASE_CONFIG[task.phase]?.label || '处理中'
  }
  return STATUS_CONFIG[task.status]?.label || task.status
}

// 暂停任务
async function pauseTask() {
  await transcriptionApi.pauseJob(props.task.job_id)
}

// 恢复任务
async function resumeTask() {
  await transcriptionApi.resumeJob(props.task.job_id)
}

// 取消任务
async function cancelTask() {
  if (!confirm('确定要取消这个任务吗?')) return
  await transcriptionApi.cancelJob(props.task.job_id, false)
}

// 打开编辑器
function openEditor() {
  router.push(`/editor/${props.task.job_id}`)
}

// 格式化时间
function formatTime(timestamp) {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date

  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
  return date.toLocaleDateString('zh-CN')
}
</script>

<style lang="scss" scoped>
.task-card {
  display: flex;
  gap: 12px;
  padding: 12px;
  background: var(--bg-primary);
  border: 1px solid var(--border-default);
  border-radius: 6px;
  margin-bottom: 8px;
  transition: all 0.2s;

  &:hover {
    background: var(--bg-elevated);
  }

  &.is-draggable {
    cursor: move;
  }
}

.drag-handle {
  width: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  cursor: grab;
  opacity: 0.5;

  &:hover {
    opacity: 1;
  }

  &:active {
    cursor: grabbing;
  }

  svg {
    width: 16px;
    height: 16px;
  }
}

.task-info {
  flex: 1;
  min-width: 0;
}

.task-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
  gap: 8px;
}

.task-name {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-phase {
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  white-space: nowrap;
}

.task-progress {
  display: flex;
  align-items: center;
  gap: 8px;
}

.progress-bar {
  flex: 1;
  height: 4px;
  background: var(--border-muted);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--primary);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-muted);
  min-width: 35px;
  text-align: right;
}

.task-error {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--danger);

  svg {
    width: 14px;
    height: 14px;
  }
}

.task-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-muted);

  svg {
    width: 14px;
    height: 14px;
    color: var(--success);
  }
}

.task-actions {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.action-btn {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
  border-radius: 4px;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.2s;

  svg {
    width: 16px;
    height: 16px;
  }

  &:hover {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  &--primary {
    color: var(--primary);
    &:hover {
      background: rgba(88, 166, 255, 0.15);
    }
  }

  &--success {
    color: var(--success);
    &:hover {
      background: rgba(63, 185, 80, 0.15);
    }
  }

  &--danger:hover {
    background: rgba(248, 81, 73, 0.15);
    color: var(--danger);
  }
}

// 拖动占位符样式
.task-ghost {
  opacity: 0.5;
  background: var(--bg-tertiary);
  border: 2px dashed var(--border-default);
}
</style>
