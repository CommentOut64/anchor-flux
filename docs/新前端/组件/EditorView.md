# EditorView - 编辑器视图组件

## 概述

EditorView 是字幕编辑器的主容器组件，负责协调和管理所有子组件，实现完整的字幕编辑工作流。它采用灵活的布局系统，整合视频播放、波形时间轴、字幕列表、验证面板等核心功能模块。

## 功能特性

1. **布局管理**
   - 响应式多面板布局
   - 可调整面板大小
   - 布局持久化存储
   - 全屏模式支持

2. **组件协调**
   - 统一的数据流管理
   - 组件间通信协调
   - 状态同步控制
   - 错误边界处理

3. **工作流集成**
   - 文件导入导出
   - 自动保存管理
   - 撤销/重做控制
   - 快捷键统一管理

4. **用户体验**
   - 加载状态管理
   - 错误提示处理
   - 操作确认对话框
   - 离开页面提醒

## 技术依���

```json
{
  "vue": "^3.4.0",
  "pinia": "^3.0.0",
  "vue-router": "^4.0.0",
  "splitpanes": "^3.1.0",
  "@vueuse/core": "^10.0.0"
}
```

## 组件属性

```typescript
interface EditorViewProps {
  jobId: string              // 任务ID（必需）

  // 布局配置
  layout?: 'horizontal' | 'vertical' | 'compact'  // 默认'horizontal'
  sidebarPosition?: 'left' | 'right'             // 默认'left'
  defaultSplitSizes?: number[]                   // 面板默认比例

  // 功能开关
  enableAutoSave?: boolean       // 自动保存，默认true
  autoSaveInterval?: number      // 自动保存间隔（秒），默认30
  enableShortcuts?: boolean      // 键盘快捷键，默认true
  confirmOnLeave?: boolean       // 离开确认，默认true
}

interface EditorViewEmits {
  'save': (content: string) => void
  'export': (format: string) => void
  'layout-change': (layout: string) => void
  'error': (error: Error) => void
}
```

## 核心实现

```vue
<!-- views/EditorView.vue -->
<template>
  <div class="editor-view" :class="`layout-${layout}`">
    <!-- 顶部工具栏 -->
    <header class="editor-header">
      <div class="header-left">
        <router-link to="/tasks" class="back-link">
          <el-button :icon="ArrowLeft" link>返回任务列表</el-button>
        </router-link>
        <h2 class="project-name">{{ projectName }}</h2>
        <el-tag v-if="isDirty" type="warning">未保存</el-tag>
      </div>

      <div class="header-center">
        <PlaybackControls
          :compact="true"
          @play="handlePlay"
          @pause="handlePause"
          @seek="handleSeek"
        />
      </div>

      <div class="header-right">
        <!-- 撤销/重做 -->
        <el-button-group>
          <el-button
            @click="undo"
            :disabled="!canUndo"
            :icon="RefreshLeft"
            title="撤销 (Ctrl+Z)"
          />
          <el-button
            @click="redo"
            :disabled="!canRedo"
            :icon="RefreshRight"
            title="重做 (Ctrl+Y)"
          />
        </el-button-group>

        <!-- 保存/导出 -->
        <el-button
          @click="saveProject"
          type="primary"
          :loading="saving"
          :icon="DocumentChecked"
        >
          保存
        </el-button>

        <el-dropdown @command="handleExport">
          <el-button :icon="Download">
            导出<el-icon><ArrowDown /></el-icon>
          </el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="srt">SRT格式</el-dropdown-item>
              <el-dropdown-item command="vtt">WebVTT格式</el-dropdown-item>
              <el-dropdown-item command="txt">纯文本</el-dropdown-item>
              <el-dropdown-item command="json">JSON格式</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>

        <!-- 布局切换 -->
        <el-dropdown @command="changeLayout">
          <el-button :icon="Grid" circle />
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="horizontal">横向布局</el-dropdown-item>
              <el-dropdown-item command="vertical">纵向布局</el-dropdown-item>
              <el-dropdown-item command="compact">紧凑布局</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>

        <!-- 任务监控 -->
        <TaskMonitor />
      </div>
    </header>

    <!-- 主编辑区 -->
    <splitpanes class="editor-body" :horizontal="layout === 'horizontal'">
      <!-- 左侧/上侧面板 -->
      <pane :size="splitSizes[0]" min-size="20">
        <splitpanes horizontal>
          <!-- 视频播放器 -->
          <pane :size="videoSize" min-size="30">
            <VideoStage
              :job-id="jobId"
              :show-subtitle="true"
              @loaded="handleVideoLoaded"
              @error="handleVideoError"
            />
          </pane>

          <!-- 波形时间轴 -->
          <pane :size="100 - videoSize" min-size="20">
            <WaveformTimeline
              :job-id="jobId"
              :show-thumbnails="true"
              @region-update="handleRegionUpdate"
              @region-click="handleRegionClick"
            />
          </pane>
        </splitpanes>
      </pane>

      <!-- 右侧/下侧面板 -->
      <pane :size="splitSizes[1]" min-size="30">
        <el-tabs v-model="activeTab" class="editor-tabs">
          <!-- 字幕列表标签 -->
          <el-tab-pane label="字幕列表" name="subtitles">
            <SubtitleList
              :auto-scroll="true"
              :editable="true"
              @subtitle-click="handleSubtitleClick"
              @subtitle-edit="handleSubtitleEdit"
              @subtitle-delete="handleSubtitleDelete"
            />
          </el-tab-pane>

          <!-- 验证面板标签 -->
          <el-tab-pane name="validation">
            <template #label>
              <span>
                问题检查
                <el-badge
                  :value="validationErrorCount"
                  :hidden="validationErrorCount === 0"
                  type="danger"
                />
              </span>
            </template>
            <ValidationPanel
              :collapsible="false"
              :default-expanded="true"
              @issue-click="handleIssueClick"
              @fix-issue="handleIssueFix"
            />
          </el-tab-pane>

          <!-- AI 助手标签 -->
          <el-tab-pane label="AI 助手" name="assistant">
            <AIAssistant
              :context="currentSubtitle"
              @apply-suggestion="handleApplySuggestion"
            />
          </el-tab-pane>
        </el-tabs>
      </pane>
    </splitpanes>

    <!-- 状态栏 -->
    <footer class="editor-footer">
      <div class="footer-left">
        <span>共 {{ totalSubtitles }} 条字幕</span>
        <span v-if="currentSubtitle">
          当前：#{{ currentSubtitleIndex + 1 }}
        </span>
      </div>

      <div class="footer-center">
        <span v-if="lastSaved">
          最后保存：{{ formatTime(lastSaved) }}
        </span>
      </div>

      <div class="footer-right">
        <el-button
          v-if="hasProblems"
          @click="activeTab = 'validation'"
          type="danger"
          link
          size="small"
        >
          {{ validationErrorCount }} 个问题
        </el-button>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, provide } from 'vue'
import { useRoute, useRouter, onBeforeRouteLeave } from 'vue-router'
import { Splitpanes, Pane } from 'splitpanes'
import 'splitpanes/dist/splitpanes.css'

import { useProjectStore } from '@/stores/projectStore'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { ElMessage, ElMessageBox } from 'element-plus'

import VideoStage from '@/components/VideoStage'
import WaveformTimeline from '@/components/WaveformTimeline'
import SubtitleList from '@/components/SubtitleList'
import PlaybackControls from '@/components/PlaybackControls'
import ValidationPanel from '@/components/ValidationPanel'
import TaskMonitor from '@/components/TaskMonitor'
import AIAssistant from '@/components/AIAssistant'

// Props & Emits
const props = defineProps({
  jobId: {
    type: String,
    required: true
  },
  layout: {
    type: String,
    default: 'horizontal',
    validator: v => ['horizontal', 'vertical', 'compact'].includes(v)
  },
  sidebarPosition: {
    type: String,
    default: 'left'
  },
  defaultSplitSizes: {
    type: Array,
    default: () => [60, 40]
  },
  enableAutoSave: {
    type: Boolean,
    default: true
  },
  autoSaveInterval: {
    type: Number,
    default: 30
  },
  enableShortcuts: {
    type: Boolean,
    default: true
  },
  confirmOnLeave: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['save', 'export', 'layout-change', 'error'])

// Router
const route = useRoute()
const router = useRouter()

// Stores
const projectStore = useProjectStore()
const taskStore = useUnifiedTaskStore()

// State
const layout = ref(props.layout)
const splitSizes = ref([...props.defaultSplitSizes])
const videoSize = ref(layout.value === 'compact' ? 100 : 60)
const activeTab = ref('subtitles')
const saving = ref(false)
const lastSaved = ref(null)

// Provide editor context
provide('editorContext', {
  jobId: props.jobId,
  layout,
  saving
})

// Computed
const projectName = computed(() => projectStore.meta.filename || 'Untitled')
const isDirty = computed(() => projectStore.isDirty)
const totalSubtitles = computed(() => projectStore.totalSubtitles)
const currentSubtitle = computed(() => projectStore.currentSubtitle)
const currentSubtitleIndex = computed(() => {
  if (!currentSubtitle.value) return -1
  return projectStore.subtitles.findIndex(s => s.id === currentSubtitle.value.id)
})

const canUndo = computed(() => projectStore.canUndo)
const canRedo = computed(() => projectStore.canRedo)

const validationErrorCount = computed(() =>
  projectStore.validationErrors.filter(e => e.severity === 'error').length
)
const hasProblems = computed(() => projectStore.validationErrors.length > 0)

// Lifecycle
onMounted(async () => {
  await loadProject()
  setupAutoSave()
  setupKeyboardShortcuts()
})

onUnmounted(() => {
  clearAutoSave()
  removeKeyboardShortcuts()
})

// Load project data
async function loadProject() {
  try {
    // Try to restore from cache first
    const restored = await projectStore.restoreProject(props.jobId)

    if (!restored) {
      // Load from backend
      const response = await fetch(`/api/projects/${props.jobId}`)
      const data = await response.json()

      projectStore.importSRT(data.srtContent, {
        jobId: props.jobId,
        filename: data.filename,
        videoPath: data.videoPath,
        audioPath: data.audioPath,
        peaksPath: data.peaksPath,
        duration: data.duration
      })
    }

    // Update task store
    taskStore.setActiveTask(props.jobId)
  } catch (error) {
    console.error('加载项目失败:', error)
    ElMessage.error('项目加载失败')
    emit('error', error)
  }
}

// Save project
async function saveProject() {
  if (saving.value) return

  saving.value = true
  try {
    const srtContent = projectStore.generateSRT()

    const response = await fetch(`/api/projects/${props.jobId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ srtContent })
    })

    if (!response.ok) throw new Error('保存失败')

    projectStore.saveProject()
    lastSaved.value = Date.now()
    ElMessage.success('保存成功')
    emit('save', srtContent)
  } catch (error) {
    console.error('保存失败:', error)
    ElMessage.error('保存失败')
  } finally {
    saving.value = false
  }
}

// Auto save
let autoSaveTimer = null

function setupAutoSave() {
  if (!props.enableAutoSave) return

  autoSaveTimer = setInterval(() => {
    if (isDirty.value) {
      saveProject()
    }
  }, props.autoSaveInterval * 1000)
}

function clearAutoSave() {
  if (autoSaveTimer) {
    clearInterval(autoSaveTimer)
    autoSaveTimer = null
  }
}

// Export functions
async function handleExport(format) {
  let content = ''
  let filename = projectName.value.replace(/\.[^/.]+$/, '')

  switch (format) {
    case 'srt':
      content = projectStore.generateSRT()
      filename += '.srt'
      break
    case 'vtt':
      content = generateVTT()
      filename += '.vtt'
      break
    case 'txt':
      content = generateText()
      filename += '.txt'
      break
    case 'json':
      content = JSON.stringify(projectStore.subtitles, null, 2)
      filename += '.json'
      break
  }

  // Download file
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)

  ElMessage.success(`导出${format.toUpperCase()}格式成功`)
  emit('export', format)
}

function generateVTT() {
  let vtt = 'WEBVTT\n\n'
  projectStore.subtitles.forEach((sub, i) => {
    vtt += `${i + 1}\n`
    vtt += `${formatVTTTime(sub.start)} --> ${formatVTTTime(sub.end)}\n`
    vtt += `${sub.text}\n\n`
  })
  return vtt
}

function generateText() {
  return projectStore.subtitles.map(s => s.text).join('\n')
}

function formatVTTTime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = (seconds % 60).toFixed(3)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.padStart(6, '0')}`
}

// Layout management
function changeLayout(newLayout) {
  layout.value = newLayout
  if (newLayout === 'compact') {
    videoSize.value = 100
    splitSizes.value = [40, 60]
  } else {
    videoSize.value = 60
    splitSizes.value = props.defaultSplitSizes
  }
  emit('layout-change', newLayout)
}

// Event handlers
function handleVideoLoaded(duration) {
  console.log('视频加载完成，时长:', duration)
}

function handleVideoError(error) {
  console.error('视频加载错误:', error)
  ElMessage.error('视频加载失败')
}

function handleRegionUpdate(region) {
  console.log('时间轴区域更新:', region)
}

function handleRegionClick(region) {
  console.log('时间轴区域点击:', region)
}

function handleSubtitleClick(subtitle) {
  console.log('字幕点击:', subtitle)
}

function handleSubtitleEdit(id, field, value) {
  console.log('字幕编辑:', id, field, value)
}

function handleSubtitleDelete(id) {
  console.log('字幕删除:', id)
}

function handleIssueClick(issue) {
  activeTab.value = 'subtitles'
}

function handleIssueFix(issue) {
  console.log('修复问题:', issue)
}

function handleApplySuggestion(suggestion) {
  console.log('应用AI建议:', suggestion)
}

function handlePlay() {
  console.log('播放')
}

function handlePause() {
  console.log('暂停')
}

function handleSeek(time) {
  console.log('跳转到:', time)
}

// Undo/Redo
function undo() {
  projectStore.undo()
  ElMessage.success('已撤销')
}

function redo() {
  projectStore.redo()
  ElMessage.success('已重做')
}

// Keyboard shortcuts
function setupKeyboardShortcuts() {
  if (!props.enableShortcuts) return
  document.addEventListener('keydown', handleKeydown)
}

function removeKeyboardShortcuts() {
  document.removeEventListener('keydown', handleKeydown)
}

function handleKeydown(e) {
  // Ignore if typing in input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

  if (e.ctrlKey || e.metaKey) {
    switch (e.key) {
      case 's':
        e.preventDefault()
        saveProject()
        break
      case 'z':
        e.preventDefault()
        if (e.shiftKey) {
          redo()
        } else {
          undo()
        }
        break
      case 'y':
        e.preventDefault()
        redo()
        break
    }
  }
}

// Format time helper
function formatTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString()
}

// Route guard
onBeforeRouteLeave((to, from) => {
  if (props.confirmOnLeave && isDirty.value) {
    return ElMessageBox.confirm(
      '当前项目有未保存的修改，确定要离开吗？',
      '离开确认',
      {
        confirmButtonText: '保存并离开',
        cancelButtonText: '直接离开',
        distinguishCancelAndClose: true,
        type: 'warning'
      }
    ).then(() => {
      return saveProject()
    }).catch(action => {
      return action !== 'close'
    })
  }
  return true
})
</script>
```

## 样式定义

```scss
.editor-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg-primary);

  // 头部工具栏
  .editor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);

    .header-left,
    .header-center,
    .header-right {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .project-name {
      font-size: 18px;
      font-weight: 500;
      margin: 0;
    }

    .back-link {
      text-decoration: none;
    }
  }

  // 主编辑区
  .editor-body {
    flex: 1;
    overflow: hidden;

    .splitpanes__pane {
      display: flex;
      flex-direction: column;
    }

    .editor-tabs {
      height: 100%;
      display: flex;
      flex-direction: column;

      .el-tabs__content {
        flex: 1;
        overflow: hidden;
      }

      .el-tab-pane {
        height: 100%;
        overflow: auto;
      }
    }
  }

  // 状态栏
  .editor-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 20px;
    background: var(--bg-tertiary);
    border-top: 1px solid var(--border-color);
    font-size: 13px;
    color: var(--text-muted);

    .footer-left,
    .footer-center,
    .footer-right {
      display: flex;
      align-items: center;
      gap: 16px;
    }
  }

  // 布局变体
  &.layout-vertical {
    .editor-body {
      flex-direction: column;
    }
  }

  &.layout-compact {
    .editor-header {
      padding: 8px 12px;
    }

    .editor-footer {
      padding: 4px 12px;
      font-size: 12px;
    }
  }
}

// Splitpanes 样式覆盖
.splitpanes--horizontal > .splitpanes__splitter {
  height: 7px;
  background: var(--border-color);
  cursor: row-resize;

  &:hover {
    background: var(--primary-color);
  }
}

.splitpanes--vertical > .splitpanes__splitter {
  width: 7px;
  background: var(--border-color);
  cursor: col-resize;

  &:hover {
    background: var(--primary-color);
  }
}
```

## 与其他组件的��系

### 依赖关系
- 包含所有核心编辑组件
- 依赖 `ProjectStore` 和 `UnifiedTaskStore`
- 使用 `splitpanes` 实现可调整布局

### 被依赖关系
- 作为主��图被路由系统加载
- 为子组件提供编辑器上下文

## 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+S | 保存项目 |
| Ctrl+Z | 撤销 |
| Ctrl+Y / Ctrl+Shift+Z | 重做 |
| 空格 | 播放/暂停 |
| ← / → | 快退/快进 |

## 性能优化

1. **懒加载组件**
   - AI助手等非核心组件按需加载

2. **防抖自动保存**
   - 避免频繁保存操作

3. **布局缓存**
   - 用户布局偏好持久化

## 测试要点

1. 组件集成协调
2. 数据流同步
3. 自动保存可靠性
4. 布局切换稳定性
5. 离开页面确认

## 未来扩展

1. 多标签编辑
2. 协作编辑模式
3. 自定义工具栏
4. 插件系统支持
5. 更多导出格式