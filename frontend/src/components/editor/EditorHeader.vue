<template>
  <header class="editor-header">
    <!-- 左侧：返回 + 任务信息堆叠 -->
    <div class="header-left">
      <router-link to="/tasks" class="nav-back" title="返回任务列表">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/>
        </svg>
      </router-link>

      <div class="divider-vertical"></div>

      <div class="task-info-stack">
        <h1 class="task-name" :title="taskName">{{ taskName }}</h1>
        <div class="task-meta">
          <span class="status-dot" :class="statusClass"></span>
          <span class="meta-text">{{ metaText }}</span>
        </div>
      </div>
    </div>

    <!-- 中间：动态进度区 -->
    <div class="header-center">
      <!-- 场景1: 当前任务转录中或已暂停 -->
      <el-popover
        v-if="showCurrentTaskProgress"
        trigger="hover"
        :width="240"
        popper-class="control-popover-dark"
        :show-after="200"
        placement="bottom"
      >
        <template #reference>
          <div class="progress-capsule" :class="{ paused: isPaused }">
            <div class="progress-track">
              <div class="progress-fill" :style="{ width: currentTaskProgress + '%' }"></div>
            </div>
            <!-- 显示阶段标签和进度 -->
            <span class="capsule-text">
              <span
                class="phase-tag"
                :style="{
                  background: phaseStyle.bgColor,
                  color: phaseStyle.color
                }"
              >
                {{ phaseLabel }}
              </span>
              <span class="progress-percent">{{ formatProgress(currentTaskProgress) }}%</span>
            </span>
          </div>
        </template>

        <div class="hover-controls">
          <div class="label">当前任务控制</div>
          <div class="btn-group">
            <!-- 暂停/恢复按钮 -->
            <el-button v-if="!isPaused" circle size="small" @click="$emit('pause')">
              <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
              </svg>
            </el-button>
            <el-button v-else circle size="small" type="success" @click="$emit('resume')">
              <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                <path d="M8 5v14l11-7z"/>
              </svg>
            </el-button>
            <!-- 取消按钮 -->
            <el-button circle size="small" type="danger" plain @click="$emit('cancel')">
              <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
              </svg>
            </el-button>
          </div>
        </div>
      </el-popover>

      <!-- 场景2: 当前任务完成，显示队列总进度 -->
      <div v-else class="queue-progress">
        <div class="progress-track">
          <div class="progress-fill complete" :style="{ width: queueProgressPercent + '%' }"></div>
        </div>
        <span class="progress-text">
          {{ queueCompleted }}/{{ queueTotal }} 任务完成
          <span v-if="queueProgressPercent === 100" class="complete-check">
            <svg viewBox="0 0 24 24" fill="currentColor" width="12" height="12">
              <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/>
            </svg>
          </span>
        </span>
      </div>
    </div>

    <!-- 右侧：操作按钮 -->
    <div class="header-right">
      <!-- 任务监控器 -->
      <el-popover
        trigger="click"
        :width="400"
        popper-class="task-monitor-popover"
        placement="bottom-end"
      >
        <template #reference>
          <div class="monitor-trigger">
            <button class="icon-btn" title="任务监控">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M20 19V8H4v11h16m0-14a2 2 0 012 2v12a2 2 0 01-2 2H4a2 2 0 01-2-2V7a2 2 0 012-2h16M6 10h2v6H6v-6m4-1h2v7h-2V9m4 4h2v3h-2v-3z"/>
              </svg>
            </button>
            <span v-if="activeTasks > 0" class="badge">{{ activeTasks }}</span>
          </div>
        </template>
        <TaskMonitor :current-job-id="jobId" />
      </el-popover>

      <div class="divider-vertical"></div>

      <!-- 撤销/重做 -->
      <el-tooltip content="撤销 (Ctrl+Z)" placement="bottom">
        <button class="icon-btn" :class="{ disabled: !canUndo }" @click="$emit('undo')">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"/>
          </svg>
        </button>
      </el-tooltip>
      <el-tooltip content="重做 (Ctrl+Y)" placement="bottom">
        <button class="icon-btn" :class="{ disabled: !canRedo }" @click="$emit('redo')">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M18.4 10.6C16.55 8.99 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z"/>
          </svg>
        </button>
      </el-tooltip>

      <div class="divider-vertical"></div>

      <!-- 导出按钮 -->
      <el-dropdown trigger="click" @command="handleExport">
        <button class="export-btn">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
          </svg>
          <span>导出</span>
          <svg class="arrow" viewBox="0 0 24 24" fill="currentColor">
            <path d="M7 10l5 5 5-5z"/>
          </svg>
        </button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="srt">SRT 格式</el-dropdown-item>
            <el-dropdown-item command="vtt">WebVTT 格式</el-dropdown-item>
            <el-dropdown-item command="txt">纯文本</el-dropdown-item>
            <el-dropdown-item command="json">JSON 格式</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>
  </header>
</template>

<script setup>
/**
 * EditorHeader - 编辑器顶栏组件
 *
 * 职责：
 * - 导航控制（返回按钮）
 * - 任务信息展示（名称、状态）
 * - 动态进度显示（当前任务 / 队列总进度）
 * - 全局操作入口（任务监控、撤销/重做、导出）
 */
import { computed } from 'vue'
import TaskMonitor from './TaskMonitor/index.vue'
import { PHASE_CONFIG, STATUS_CONFIG, formatProgress } from '@/constants/taskPhases'

const props = defineProps({
  jobId: { type: String, required: true },
  taskName: { type: String, default: '未命名项目' },
  currentTaskStatus: { type: String, default: 'idle' },      // 'processing', 'queued', 'paused', 'finished', etc.
  currentTaskPhase: { type: String, default: 'pending' },    // 任务阶段（transcribe, align, etc.）
  currentTaskProgress: { type: Number, default: 0 },         // 0-100
  queueCompleted: { type: Number, default: 0 },              // 已完成任务数
  queueTotal: { type: Number, default: 0 },                  // 总任务数
  canUndo: { type: Boolean, default: false },
  canRedo: { type: Boolean, default: false },
  activeTasks: { type: Number, default: 0 },                 // 正在进行的任务数
  lastSaved: { type: [Number, null], default: null }         // 上次保存时间戳
})

defineEmits(['undo', 'redo', 'export', 'pause', 'resume', 'cancel'])

// 是否暂停状态
const isPaused = computed(() => props.currentTaskStatus === 'paused')

// 是否显示当前任务进度（转录中、排队中、或已暂停）
const showCurrentTaskProgress = computed(() =>
  ['processing', 'queued', 'paused'].includes(props.currentTaskStatus)
)

// 队列进度百分比
const queueProgressPercent = computed(() =>
  props.queueTotal > 0 ? Math.round((props.queueCompleted / props.queueTotal) * 100) : 0
)

// 状态点样式
const statusClass = computed(() => {
  if (showCurrentTaskProgress.value) return 'processing'
  if (queueProgressPercent.value === 100 && props.queueTotal > 0) return 'complete'
  return 'idle'
})

// 元信息文字
const metaText = computed(() => {
  if (isPaused.value) {
    return `已暂停 ${props.currentTaskProgress}%`
  }
  if (showCurrentTaskProgress.value) {
    return `转录中 ${props.currentTaskProgress}%`
  }
  if (props.lastSaved) {
    const date = new Date(props.lastSaved)
    const time = date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
    return `自动保存于 ${time}`
  }
  return '准备就绪'
})

// 获取阶段样式（与TaskMonitor保持一致）
const phaseStyle = computed(() => {
  // 如果任务暂停，使用暂停状态样式
  if (isPaused.value) {
    return STATUS_CONFIG.paused || {
      bgColor: 'rgba(210, 153, 34, 0.15)',
      color: '#d29922',
      label: '已暂停'
    }
  }
  // 如果任务正在处理且有阶段信息，使用阶段样式
  if (props.currentTaskStatus === 'processing' && props.currentTaskPhase) {
    return PHASE_CONFIG[props.currentTaskPhase] || PHASE_CONFIG.pending
  }
  // 其他情况使用状态样式
  return STATUS_CONFIG[props.currentTaskStatus] || STATUS_CONFIG.created
})

// 阶段标签
const phaseLabel = computed(() => {
  // 如果任务暂停，显示"已暂停"
  if (isPaused.value) {
    return '已暂停'
  }
  // 如果任务正在处理且有阶段信息，显示阶段标签
  if (props.currentTaskStatus === 'processing' && props.currentTaskPhase) {
    return PHASE_CONFIG[props.currentTaskPhase]?.label || '处理中'
  }
  // 其他情况显示状态标签
  return STATUS_CONFIG[props.currentTaskStatus]?.label || props.currentTaskStatus
})

// 处理导出
function handleExport(format) {
  // 向父组件发送 export 事件
  // 由父组件处理实际导出逻辑
  const event = new CustomEvent('header-export', { detail: format })
  window.dispatchEvent(event)
}
</script>

<style lang="scss" scoped>
$header-h: 56px;

.editor-header {
  height: $header-h;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-default);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  position: relative;
  flex-shrink: 0;
}

// 左侧堆叠布局
.header-left {
  display: flex;
  align-items: center;
  gap: 12px;

  .nav-back {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    transition: all 0.2s;

    svg {
      width: 20px;
      height: 20px;
    }

    &:hover {
      background: var(--bg-tertiary);
      color: var(--text-primary);
    }
  }

  .task-info-stack {
    display: flex;
    flex-direction: column;
    gap: 2px;

    .task-name {
      margin: 0;
      font-size: 14px;
      font-weight: 500;
      color: var(--text-primary);
      max-width: 300px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .task-meta {
      display: flex;
      align-items: center;
      gap: 5px;

      .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;

        &.processing {
          background: var(--primary);
          animation: pulse 1.5s infinite;
        }
        &.complete { background: var(--success); }
        &.idle { background: var(--text-muted); }
      }

      .meta-text {
        font-size: 11px;
        color: var(--text-muted);
      }
    }
  }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

// 中间绝对定位居中
.header-center {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
}

// 当前任务进度胶囊
.progress-capsule {
  background: var(--bg-elevated);
  border-radius: 12px;
  padding: 4px 14px;
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  transition: all 0.2s;

  &:hover {
    background: var(--bg-tertiary);
    transform: scale(1.02);
  }

  &.paused {
    opacity: 0.85;
  }

  .progress-track {
    width: 120px;
    height: 4px;
    background: var(--border-muted);
    border-radius: 2px;
    overflow: hidden;

    .progress-fill {
      height: 100%;
      background: var(--primary);
      transition: width 0.3s ease;
    }
  }

  .capsule-text {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    white-space: nowrap;

    .phase-tag {
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
      white-space: nowrap;
    }

    .progress-percent {
      color: var(--text-secondary);
      font-family: var(--font-mono);
    }
  }
}

// 队列总进度
.queue-progress {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 16px;
  background: var(--bg-elevated);
  border-radius: 10px;

  .progress-track {
    width: 150px;
    height: 4px;
    background: var(--border-muted);
    border-radius: 2px;
    overflow: hidden;

    .progress-fill {
      height: 100%;
      background: var(--primary);
      transition: width 0.5s ease;

      &.complete {
        background: var(--success);
      }
    }
  }

  .progress-text {
    font-size: 12px;
    color: var(--text-secondary);
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 6px;

    .complete-check {
      color: var(--success);
      display: flex;
      align-items: center;
    }
  }
}

// 悬浮控制面板
.hover-controls {
  .label {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 10px;
  }

  .btn-group {
    display: flex;
    gap: 8px;
    justify-content: center;
  }
}

// 右侧按钮组
.header-right {
  display: flex;
  align-items: center;
  gap: 8px;

  .monitor-trigger {
    position: relative;
    display: flex;
    align-items: center;

    .badge {
      position: absolute;
      top: 0;
      right: 0;
      background: var(--primary);
      color: #fff;
      font-size: 10px;
      padding: 0 4px;
      min-width: 16px;
      height: 16px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
      transform: translate(25%, -25%);
    }
  }

  .icon-btn {
    width: 36px;
    height: 36px;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;

    svg {
      width: 18px;
      height: 18px;
    }

    &:hover:not(.disabled) {
      color: var(--text-primary);
      background: var(--bg-tertiary);
    }

    &.disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }
  }

  .export-btn {
    height: 32px;
    padding: 0 14px;
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: var(--radius-md);
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s;

    svg {
      width: 16px;
      height: 16px;
    }

    .arrow {
      width: 14px;
      height: 14px;
      margin-left: -2px;
    }

    &:hover {
      background: var(--primary-hover);
    }
  }
}

.divider-vertical {
  width: 1px;
  height: 20px;
  background: var(--border-muted);
  margin: 0 4px;
}
</style>

<style lang="scss">
// 全局样式：Popover 深色主题
.control-popover-dark {
  background: var(--bg-elevated) !important;
  border-color: var(--border-default) !important;

  .el-popper__arrow::before {
    background: var(--bg-elevated) !important;
    border-color: var(--border-default) !important;
  }
}

.task-monitor-popover {
  background: var(--bg-secondary) !important;
  border-color: var(--border-default) !important;
  padding: 0 !important;

  .el-popper__arrow::before {
    background: var(--bg-secondary) !important;
    border-color: var(--border-default) !important;
  }
}
</style>
