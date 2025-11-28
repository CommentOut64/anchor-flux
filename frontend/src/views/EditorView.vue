<template>
  <div class="editor-view">
    <!-- 顶部导航栏 -->
    <header class="editor-header">
      <div class="header-left">
        <router-link to="/tasks" class="back-btn" title="返回任务列表">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/>
          </svg>
        </router-link>
        <div class="project-info">
          <h1 class="project-name">{{ projectName }}</h1>
          <span v-if="isDirty" class="unsaved-badge">未保存</span>
          <!-- 转录状态指示器 -->
          <span v-if="isTranscribing" class="transcribing-badge">
            <span class="spinner-small"></span>
            转录中 {{ taskProgress }}%
          </span>
          <span v-else-if="taskStatus === 'queued'" class="queued-badge">排队中</span>
        </div>
      </div>

      <div class="header-center">
        <PlaybackControls :compact="true" />
      </div>

      <div class="header-right">
        <!-- 撤销/重做 -->
        <div class="btn-group">
          <button class="header-btn" :class="{ disabled: !canUndo }" @click="undo" title="撤销 (Ctrl+Z)">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"/>
            </svg>
          </button>
          <button class="header-btn" :class="{ disabled: !canRedo }" @click="redo" title="重做 (Ctrl+Y)">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M18.4 10.6C16.55 8.99 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z"/>
            </svg>
          </button>
        </div>

        <!-- 保存按钮 -->
        <button class="header-btn header-btn--primary" @click="saveProject" :class="{ loading: saving }" title="保存 (Ctrl+S)">
          <svg v-if="!saving" viewBox="0 0 24 24" fill="currentColor">
            <path d="M17 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm-5 16c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3zm3-10H5V5h10v4z"/>
          </svg>
          <span v-if="saving" class="spinner"></span>
          <span>保存</span>
        </button>

        <!-- 导出下拉菜单 -->
        <div class="dropdown" ref="exportDropdownRef">
          <button class="header-btn" @click="toggleExportMenu">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
            </svg>
            <span>导出</span>
            <svg class="arrow" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 10l5 5 5-5z"/>
            </svg>
          </button>
          <div v-show="showExportMenu" class="dropdown-menu">
            <button @click="handleExport('srt')">SRT 格式</button>
            <button @click="handleExport('vtt')">WebVTT 格式</button>
            <button @click="handleExport('txt')">纯文本</button>
            <button @click="handleExport('json')">JSON 格式</button>
          </div>
        </div>
      </div>
    </header>

    <!-- 加载状态 -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <span>加载项目中...</span>
    </div>

    <!-- 错误状态 -->
    <div v-else-if="loadError" class="error-overlay">
      <svg viewBox="0 0 24 24" fill="currentColor" class="error-icon">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
      </svg>
      <h3>加载失败</h3>
      <p>{{ loadError }}</p>
      <button class="retry-btn" @click="loadProject">重试</button>
      <router-link to="/tasks" class="back-link">返回任务列表</router-link>
    </div>

    <!-- 主编辑区域 -->
    <main v-else class="editor-main">
      <!-- 左侧面板：视频 + 波形 -->
      <div class="panel panel-left" :style="{ width: leftPanelWidth + 'px' }">
        <div class="video-section">
          <VideoStage ref="videoStageRef" :job-id="jobId" :show-subtitle="true" :enable-keyboard="false" @loaded="handleVideoLoaded" @error="handleVideoError" />
        </div>
        <div class="timeline-section">
          <WaveformTimeline ref="waveformRef" :job-id="jobId" @ready="handleWaveformReady" @region-update="handleRegionUpdate" @region-click="handleRegionClick" />
        </div>
      </div>

      <!-- 调整分隔条 -->
      <div class="resizer" @mousedown="startResize"></div>

      <!-- 右侧面板：字幕列表 + 其他 -->
      <div class="panel panel-right">
        <!-- 标签页导航 -->
        <div class="tab-nav">
          <button class="tab-btn" :class="{ active: activeTab === 'subtitles' }" @click="activeTab = 'subtitles'">字幕列表</button>
          <button class="tab-btn" :class="{ active: activeTab === 'validation' }" @click="activeTab = 'validation'">
            问题检查
            <span v-if="errorCount > 0" class="badge">{{ errorCount }}</span>
          </button>
          <button class="tab-btn" :class="{ active: activeTab === 'assistant' }" @click="activeTab = 'assistant'">AI 助手</button>
        </div>

        <!-- 标签页内容 -->
        <div class="tab-content">
          <div v-show="activeTab === 'subtitles'" class="tab-pane">
            <SubtitleList :auto-scroll="true" :editable="true" @subtitle-click="handleSubtitleClick" @subtitle-edit="handleSubtitleEdit" />
          </div>

          <div v-show="activeTab === 'validation'" class="tab-pane">
            <div class="placeholder-panel">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
              <h3>问题检查</h3>
              <p v-if="errorCount === 0">没有发现问题</p>
              <p v-else>发现 {{ errorCount }} 个问题</p>
              <div class="error-list" v-if="errorCount > 0">
                <div v-for="error in validationErrors" :key="`${error.type}-${error.index}`" class="error-item" :class="error.severity" @click="jumpToError(error)">
                  <span class="error-index">#{{ error.index + 1 }}</span>
                  <span class="error-message">{{ error.message }}</span>
                </div>
              </div>
            </div>
          </div>

          <div v-show="activeTab === 'assistant'" class="tab-pane">
            <div class="placeholder-panel">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M21 10.5h-6.5V4h-5v6.5H3v5h6.5V22h5v-6.5H21v-5z"/>
              </svg>
              <h3>AI 助手</h3>
              <p>智能字幕优化功能</p>
              <p class="coming-soon">即将推出</p>
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- 底部状态栏 -->
    <footer class="editor-footer">
      <div class="footer-left">
        <span>{{ totalSubtitles }} 条字幕</span>
        <span v-if="currentSubtitle" class="divider">|</span>
        <span v-if="currentSubtitle">当前: #{{ currentSubtitleIndex + 1 }}</span>
      </div>
      <div class="footer-center">
        <span v-if="lastSaved">最后保存: {{ formatLastSaved(lastSaved) }}</span>
      </div>
      <div class="footer-right">
        <span v-if="errorCount > 0" class="error-indicator" @click="activeTab = 'validation'">{{ errorCount }} 个问题</span>
      </div>
    </footer>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, provide, watch } from 'vue'
import { useRouter, onBeforeRouteLeave } from 'vue-router'
import { useProjectStore } from '@/stores/projectStore'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { mediaApi, transcriptionApi } from '@/services/api'
import { sseService } from '@/services/sseService'
import { useShortcuts } from '@/hooks/useShortcuts'
import PlaybackControls from '@/components/editor/PlaybackControls/index.vue'
import VideoStage from '@/components/editor/VideoStage/index.vue'
import SubtitleList from '@/components/editor/SubtitleList/index.vue'
import WaveformTimeline from '@/components/editor/WaveformTimeline/index.vue'

const props = defineProps({ jobId: { type: String, required: true } })
const router = useRouter()
const projectStore = useProjectStore()
const taskStore = useUnifiedTaskStore()
const exportDropdownRef = ref(null)
const activeTab = ref('subtitles')
const saving = ref(false)
const lastSaved = ref(null)
const showExportMenu = ref(false)
const leftPanelWidth = ref(600)
const isResizing = ref(false)
// 加载状态
const isLoading = ref(true)
const loadError = ref(null)
// 任务状态（用于实时转录显示）
const taskStatus = ref(null)     // 'created' | 'queued' | 'processing' | 'finished' | 'failed'
const taskProgress = ref(0)      // 转录进度 0-100
const isTranscribing = ref(false) // 是否正在转录中
let sseUnsubscribe = null         // SSE 取消订阅函数
let progressPollTimer = null      // 进度轮询定时器

// 子组件引用（用于快捷键调用组件方法）
const videoStageRef = ref(null)
const waveformRef = ref(null)

provide('editorContext', { jobId: props.jobId, saving })

const projectName = computed(() => {
  // 优先显示用户自定义的 title，否则显示文件名（去除扩展名）
  if (projectStore.meta.title) {
    return projectStore.meta.title
  }
  const filename = projectStore.meta.filename || '未命名项目'
  const lastDotIndex = filename.lastIndexOf('.')
  if (lastDotIndex > 0) {
    return filename.substring(0, lastDotIndex)
  }
  return filename
})
const isDirty = computed(() => projectStore.isDirty)
const totalSubtitles = computed(() => projectStore.totalSubtitles)
const currentSubtitle = computed(() => projectStore.currentSubtitle)
const currentSubtitleIndex = computed(() => currentSubtitle.value ? projectStore.subtitles.findIndex(s => s.id === currentSubtitle.value.id) : -1)
const canUndo = computed(() => projectStore.canUndo)
const canRedo = computed(() => projectStore.canRedo)
const validationErrors = computed(() => projectStore.validationErrors)
const errorCount = computed(() => validationErrors.value.filter(e => e.severity === 'error').length)

// 从后端加载项目数据（支持流式加载）
async function loadProject() {
  isLoading.value = true
  loadError.value = null

  try {
    // 1. 获取任务状态
    console.log('[EditorView] 获取任务状态:', props.jobId)
    const jobStatus = await transcriptionApi.getJobStatus(props.jobId, true)
    console.log('[EditorView] 任务状态:', jobStatus)

    taskStatus.value = jobStatus.status
    taskProgress.value = jobStatus.progress || 0

    // 设置元数据
    projectStore.meta.jobId = props.jobId
    projectStore.meta.filename = jobStatus.filename || '未知文件'
    projectStore.meta.title = jobStatus.title || ''  // 加载用户自定义名称
    projectStore.meta.videoPath = mediaApi.getVideoUrl(props.jobId)
    projectStore.meta.audioPath = mediaApi.getAudioUrl(props.jobId)
    projectStore.meta.duration = jobStatus.media_status?.video?.duration || 0

    // 2. 尝试从本地存储恢复（优先使用本地编辑的数据）
    const restored = await projectStore.restoreProject(props.jobId)
    if (restored && projectStore.subtitles.length > 0) {
      console.log('[EditorView] 从本地存储恢复成功，字幕数量:', projectStore.subtitles.length)
      isTranscribing.value = false
      return  // 使用本地数据，不再从后端加载
    }

    // 3. 本地无数据，根据任务状态从后端加载字幕数据
    if (jobStatus.status === 'finished') {
      // 任务已完成，加载完整 SRT
      await loadCompletedSRT()
    } else if (['processing', 'queued'].includes(jobStatus.status)) {
      // 正在转录，加载已有的 segments 并订阅 SSE
      isTranscribing.value = true
      await loadTranscribingSegments()
      subscribeSSE()
      startProgressPolling()
    } else if (jobStatus.status === 'created') {
      // 任务刚创建，等待开始
      isTranscribing.value = true
      subscribeSSE()
    } else if (jobStatus.status === 'failed') {
      // 任务失败，尝试加载已有 segments
      await loadTranscribingSegments()
    }
    // paused, canceled 状态也尝试加载已有内容

    console.log('[EditorView] 项目加载完成')
  } catch (error) {
    console.error('[EditorView] 加载项目失败:', error)

    // 第一阶段修复：处理 404 错误，清理无效任务
    if (error.response?.status === 404) {
      console.warn(`[EditorView] 任务已在后端删除: ${props.jobId}，正在清理本地记录...`)
      try {
        // 删除本地任务记录
        await taskStore.deleteTask(props.jobId)
        loadError.value = '任务不存在（已被删除），本地记录已清理'
        // 2秒后返回任务列表
        setTimeout(() => {
          router.push('/tasks')
        }, 2000)
      } catch (deleteError) {
        console.error('[EditorView] 删除本地任务记录失败:', deleteError)
        loadError.value = '任务不存在，且清理本地记录失败，请刷新页面'
      }
    } else {
      loadError.value = error.message || '加载失败'
    }
  } finally {
    isLoading.value = false
  }
}

// 加载已完成的 SRT 文件
async function loadCompletedSRT() {
  try {
    const srtData = await mediaApi.getSRTContent(props.jobId)
    console.log('[EditorView] SRT 数据:', srtData)

    projectStore.importSRT(srtData.content, {
      jobId: props.jobId,
      filename: srtData.filename || projectStore.meta.filename,
      duration: projectStore.meta.duration,
      videoPath: projectStore.meta.videoPath,
      audioPath: projectStore.meta.audioPath
    })
    isTranscribing.value = false
  } catch (error) {
    console.warn('[EditorView] 加载 SRT 失败，尝试加载 segments:', error)
    await loadTranscribingSegments()
  }
}

// 加载正在转录的 segments（从 checkpoint）
async function loadTranscribingSegments() {
  try {
    const textData = await transcriptionApi.getTranscriptionText(props.jobId)
    console.log('[EditorView] 转录文字数据:', textData)

    if (textData.segments && textData.segments.length > 0) {
      // 将 segments 转换为 SRT 格式
      const srtContent = segmentsToSRT(textData.segments)
      projectStore.importSRT(srtContent, {
        jobId: props.jobId,
        filename: projectStore.meta.filename,
        duration: projectStore.meta.duration,
        videoPath: projectStore.meta.videoPath,
        audioPath: projectStore.meta.audioPath
      })
      taskProgress.value = textData.progress?.percentage || 0
    }
  } catch (error) {
    console.warn('[EditorView] 加载转录文字失败:', error)
    // 没有数据也不报错，等待 SSE 推送
  }
}

// 将 segments 转换为 SRT 格式字符串
function segmentsToSRT(segments) {
  if (!segments || segments.length === 0) return ''

  return segments.map((seg, idx) => {
    const start = formatSRTTime(seg.start)
    const end = formatSRTTime(seg.end)
    return `${idx + 1}\n${start} --> ${end}\n${seg.text || ''}\n`
  }).join('\n')
}

// 格式化为 SRT 时间格式
function formatSRTTime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  const ms = Math.round((seconds % 1) * 1000)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`
}

// 订阅 SSE 实时进度
function subscribeSSE() {
  const sseUrl = `/api/stream/${props.jobId}`
  console.log('[EditorView] 订阅 SSE:', sseUrl)

  // 连接 SSE
  sseService.connect(sseUrl)

  // 监听进度事件
  const handleProgress = (data) => {
    console.log('[EditorView] SSE progress:', data)
    taskProgress.value = data.percent || 0
    taskStatus.value = data.status
  }

  // 监听信号事件（完成/失败）
  const handleSignal = async (data) => {
    console.log('[EditorView] SSE signal:', data)
    if (data.signal === 'job_complete') {
      // 转录完成，加载完整 SRT
      isTranscribing.value = false
      taskStatus.value = 'finished'
      await loadCompletedSRT()
      stopProgressPolling()
    } else if (data.signal === 'job_failed') {
      taskStatus.value = 'failed'
      isTranscribing.value = false
      stopProgressPolling()
    }
  }

  // 监听消息
  const handleMessage = (msg) => {
    if (msg.type === 'progress') handleProgress(msg.data)
    if (msg.type === 'signal') handleSignal(msg.data)
  }

  sseService.on('progress', handleProgress)
  sseService.on('signal', handleSignal)
  sseService.on('message', handleMessage)

  // 保存取消订阅函数
  sseUnsubscribe = () => {
    sseService.off('progress', handleProgress)
    sseService.off('signal', handleSignal)
    sseService.off('message', handleMessage)
    sseService.disconnect()
  }
}

// 开始进度轮询（定期刷新 segments）
function startProgressPolling() {
  stopProgressPolling()
  progressPollTimer = setInterval(async () => {
    if (!isTranscribing.value) {
      stopProgressPolling()
      return
    }
    try {
      await loadTranscribingSegments()
    } catch (e) {
      console.warn('[EditorView] 轮询刷新 segments 失败:', e)
    }
  }, 5000) // 每5秒刷新一次
}

// 停止进度轮询
function stopProgressPolling() {
  if (progressPollTimer) {
    clearInterval(progressPollTimer)
    progressPollTimer = null
  }
}

// 保存项目到后端
async function saveProject() {
  if (saving.value) return
  saving.value = true
  try {
    // 生成 SRT 内容
    const srtContent = projectStore.generateSRT()
    // 调用后端 API 保存
    await mediaApi.saveSRTContent(props.jobId, srtContent)
    // 更新本地状态
    await projectStore.saveProject()
    lastSaved.value = Date.now()
    console.log('[EditorView] 项目保存成功')
  } catch (error) {
    console.error('[EditorView] 保存失败:', error)
    alert('保存失败: ' + (error.message || '未知错误'))
  } finally {
    saving.value = false
  }
}

function undo() { if (canUndo.value) projectStore.undo() }
function redo() { if (canRedo.value) projectStore.redo() }
function toggleExportMenu() { showExportMenu.value = !showExportMenu.value }

function handleExport(format) {
  showExportMenu.value = false
  let content = ''
  let filename = projectName.value.replace(/\.[^/.]+$/, '')

  switch (format) {
    case 'srt': content = projectStore.generateSRT(); filename += '.srt'; break
    case 'vtt': content = generateVTT(); filename += '.vtt'; break
    case 'txt': content = projectStore.subtitles.map(s => s.text).join('\n'); filename += '.txt'; break
    case 'json': content = JSON.stringify(projectStore.subtitles, null, 2); filename += '.json'; break
  }

  downloadFile(content, filename)
}

function generateVTT() {
  let vtt = 'WEBVTT\n\n'
  projectStore.subtitles.forEach((sub, i) => {
    const start = formatVTTTime(sub.start)
    const end = formatVTTTime(sub.end)
    vtt += `${i + 1}\n${start} --> ${end}\n${sub.text}\n\n`
  })
  return vtt
}

function formatVTTTime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = (seconds % 60).toFixed(3)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.padStart(6, '0')}`
}

function downloadFile(content, filename) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function handleVideoLoaded(duration) { console.log('视频加载完成:', duration) }
function handleVideoError(error) { console.error('视频加载错误:', error) }
function handleWaveformReady() { console.log('波形加载完成') }
function handleRegionUpdate(region) { console.log('区域更新:', region) }
function handleRegionClick(region) { console.log('区域点击:', region) }
function handleSubtitleClick(subtitle) { console.log('字幕点击:', subtitle) }
function handleSubtitleEdit(id, field, value) { console.log('字幕编辑:', id, field, value) }

function jumpToError(error) {
  const subtitle = projectStore.subtitles[error.index]
  if (subtitle) {
    projectStore.view.selectedSubtitleId = subtitle.id
    projectStore.seekTo(subtitle.start)
    activeTab.value = 'subtitles'
  }
}

function startResize(e) {
  isResizing.value = true
  document.addEventListener('mousemove', onResize)
  document.addEventListener('mouseup', stopResize)
}

function onResize(e) {
  if (!isResizing.value) return
  const newWidth = e.clientX
  leftPanelWidth.value = Math.max(400, Math.min(window.innerWidth - 400, newWidth))
}

function stopResize() {
  isResizing.value = false
  document.removeEventListener('mousemove', onResize)
  document.removeEventListener('mouseup', stopResize)
}

function handleClickOutside(e) {
  if (exportDropdownRef.value && !exportDropdownRef.value.contains(e.target)) {
    showExportMenu.value = false
  }
}

// ========== 快捷键操作函数 ==========

// 播放/暂停切换
function togglePlay() {
  projectStore.player.isPlaying = !projectStore.player.isPlaying
}

// 上一帧（步进后退）
function stepBackward() {
  // TODO: 实现逐帧后退功能（需要根据帧率计算）
  const frameTime = 1 / 30 // 假设30fps
  const newTime = Math.max(0, projectStore.player.currentTime - frameTime)
  projectStore.seekTo(newTime)
}

// 下一帧（步进前进）
function stepForward() {
  // TODO: 实现逐帧前进功能（需要根据帧率计算）
  const frameTime = 1 / 30 // 假设30fps
  const newTime = Math.min(projectStore.meta.duration, projectStore.player.currentTime + frameTime)
  projectStore.seekTo(newTime)
}

// 快退5秒
function seekBackward() {
  const newTime = Math.max(0, projectStore.player.currentTime - 5)
  projectStore.seekTo(newTime)
}

// 快进5秒
function seekForward() {
  const newTime = Math.min(projectStore.meta.duration, projectStore.player.currentTime + 5)
  projectStore.seekTo(newTime)
}

// 跳转到开头
function seekToStart() {
  projectStore.seekTo(0)
}

// 跳转到结尾
function seekToEnd() {
  projectStore.seekTo(projectStore.meta.duration)
}

// 波形放大
function zoomInWave() {
  // TODO: 需要 WaveformTimeline 组件暴露 zoomIn 方法
  console.log('波形放大功能待实现')
}

// 波形缩小
function zoomOutWave() {
  // TODO: 需要 WaveformTimeline 组件暴露 zoomOut 方法
  console.log('波形缩小功能待实现')
}

// 波形适应屏幕
function fitWave() {
  // TODO: 需要 WaveformTimeline 组件暴露 fitToScreen 方法
  console.log('波形适应屏幕功能待实现')
}

// 视频画面放大
function zoomInVideo() {
  // TODO: 需要实现视频画面缩放功能
  console.log('视频画面放大功能待实现')
}

// 视频画面缩小
function zoomOutVideo() {
  // TODO: 需要实现视频���面缩放功能
  console.log('视频画面缩小功能待实现')
}

// 画面适应窗口
function fitVideo() {
  // TODO: 需要实现视频画面自适应功能
  console.log('画面适应窗口功能待实现')
}

// 字体变大
function fontSizeUp() {
  // TODO: 需要实现字幕字体大小调整功能
  console.log('字体变大功能待实现')
}

// 字体变小
function fontSizeDown() {
  // TODO: 需要实现字幕字体大小调整功能
  console.log('字体变小功能待实现')
}

// 分割字幕
function splitSubtitle() {
  // TODO: 需要实现在当前播放位置分割字幕的功能
  console.log('分割字幕功能待实现')
}

// 合并字幕
function mergeSubtitle() {
  // TODO: 需要实现合并选中字幕的功能
  console.log('合并字幕功能待实现')
}

// 导出（显示导出菜单）
function exportSubtitle() {
  showExportMenu.value = !showExportMenu.value
}

// 打开任务监控
function openTaskMonitor() {
  // TODO: 需要实现任务监控面板功能
  console.log('打开任务监控功能待实现')
}

// 使用快捷键系统
useShortcuts({
  // 播放与导航
  togglePlay,           // Space: 播放/暂停
  stepBackward,         // Left: 上一帧
  stepForward,          // Right: 下一帧
  seekBackward,         // Shift+Left: 快退5秒
  seekForward,          // Shift+Right: 快进5秒
  seekToStart,          // Home: 跳转到开头
  seekToEnd,            // End: 跳转到结尾

  // 视图缩放
  zoomInWave,           // =: 波形放大
  zoomOutWave,          // -: 波形缩小
  fitWave,              // \: 波形适应屏幕
  zoomInVideo,          // .: 视频画面放大
  zoomOutVideo,         // ,: 视频画面缩小
  fitVideo,             // Shift+Z: 画面适应窗口

  // 字幕编辑
  fontSizeUp,           // Alt+]: 字体变大
  fontSizeDown,         // Alt+[: 字体变小
  splitSubtitle,        // Ctrl+K: 分割字幕
  mergeSubtitle,        // Ctrl+J: 合并字幕

  // 全局操作
  save: saveProject,    // Ctrl+S: 保存
  undo,                 // Ctrl+Z: 撤销
  redo,                 // Ctrl+Shift+Z 或 Ctrl+Y: 重做
  export: exportSubtitle, // Ctrl+E: 导出
  openTaskMonitor,      // Ctrl+M: 打开任务监控
})

function formatLastSaved(timestamp) {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

onBeforeRouteLeave(async (to, from) => {
  // 如果有未保存的修改，自动保存后再离开
  if (isDirty.value) {
    try {
      await projectStore.saveProject()
      console.log('[EditorView] 离开前自动保存成功')
    } catch (error) {
      console.error('[EditorView] 离开前保存失败:', error)
      // 保存失败时询问用户是否仍要离开
      const answer = window.confirm('保存失败，确定要离开吗? 未保存的修改可能会丢失。')
      if (!answer) return false
    }
  }
})

onMounted(() => {
  loadProject()
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
  // ���理 SSE 订阅和轮询
  if (sseUnsubscribe) sseUnsubscribe()
  stopProgressPolling()
})
</script>

<style lang="scss" scoped>
@use '@/styles/variables' as *;
@use '@/styles/mixins' as *;

.editor-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg-base);
  color: var(--text-normal);
}

// 加载状态
.loading-overlay {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  color: var(--text-muted);

  .loading-spinner {
    width: 48px;
    height: 48px;
    border: 3px solid var(--border-default);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
}

// 错误状态
.error-overlay {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  color: var(--text-muted);
  text-align: center;

  .error-icon {
    width: 64px;
    height: 64px;
    color: var(--danger);
  }

  h3 {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  p {
    font-size: 14px;
    max-width: 400px;
    margin: 0;
  }

  .retry-btn {
    padding: 10px 24px;
    background: var(--primary);
    color: white;
    border-radius: var(--radius-md);
    font-size: 14px;
    transition: background var(--transition-fast);
    &:hover { background: var(--primary-hover); }
  }

  .back-link {
    font-size: 13px;
    color: var(--text-secondary);
    text-decoration: underline;
    &:hover { color: var(--text-primary); }
  }
}

@keyframes spin { to { transform: rotate(360deg); } }

.editor-header {
  @include flex-between;
  height: 56px;
  padding: 0 16px;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-default);
  flex-shrink: 0;

  .header-left, .header-center, .header-right { display: flex; align-items: center; gap: 12px; }
  .header-center { flex: 1; justify-content: center; max-width: 600px; margin: 0 24px; }

  .back-btn {
    @include flex-center;
    width: 36px;
    height: 36px;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    transition: all var(--transition-fast);
    svg { width: 20px; height: 20px; }
    &:hover { background: var(--bg-tertiary); color: var(--text-primary); }
  }

  .project-name { font-size: 16px; font-weight: 600; color: var(--text-primary); margin: 0; }
  .unsaved-badge { padding: 2px 8px; background: rgba(210, 153, 34, 0.15); color: var(--warning); font-size: 11px; border-radius: var(--radius-full); }
  .transcribing-badge {
    display: flex; align-items: center; gap: 6px;
    padding: 2px 10px; background: rgba(88, 166, 255, 0.15); color: var(--primary);
    font-size: 11px; border-radius: var(--radius-full);
    .spinner-small { width: 12px; height: 12px; border: 2px solid rgba(88, 166, 255, 0.3); border-top-color: var(--primary); border-radius: 50%; animation: spin 1s linear infinite; }
  }
  .queued-badge { padding: 2px 8px; background: rgba(139, 148, 158, 0.15); color: var(--text-muted); font-size: 11px; border-radius: var(--radius-full); }

  .btn-group { display: flex; background: var(--bg-secondary); border-radius: var(--radius-md); overflow: hidden; }

  .header-btn {
    height: 36px;
    padding: 0 12px;
    @include flex-center;
    gap: 6px;
    background: var(--bg-secondary);
    border-radius: var(--radius-md);
    color: var(--text-normal);
    font-size: 13px;
    transition: all var(--transition-fast);
    svg { width: 18px; height: 18px; }
    .arrow { width: 14px; height: 14px; margin-left: -2px; }
    &:hover { background: var(--bg-tertiary); }
    &.disabled { opacity: 0.4; pointer-events: none; }
    &--primary { background: var(--primary); color: white; &:hover { background: var(--primary-hover); } }
    &.loading { pointer-events: none; }
    .spinner { width: 14px; height: 14px; border: 2px solid rgba(255, 255, 255, 0.3); border-top-color: white; border-radius: 50%; animation: spin 0.8s linear infinite; }
  }

  .dropdown { position: relative; }
  .dropdown-menu {
    position: absolute; top: 100%; right: 0; margin-top: 4px;
    background: var(--bg-elevated); border: 1px solid var(--border-default); border-radius: var(--radius-md);
    box-shadow: var(--shadow-lg); padding: 4px; z-index: 100; min-width: 140px;
    button { display: block; width: 100%; padding: 8px 12px; text-align: left; font-size: 13px; color: var(--text-normal); border-radius: var(--radius-sm); transition: background var(--transition-fast); &:hover { background: var(--bg-tertiary); } }
  }
}

.editor-main { flex: 1; display: flex; overflow: hidden; }
.panel { display: flex; flex-direction: column; overflow: hidden; }
.panel-left { min-width: 400px; .video-section { flex: 1; min-height: 200px; padding: 12px; } .timeline-section { height: 240px; padding: 0 12px 12px; } }
.resizer { width: 4px; background: var(--border-default); cursor: col-resize; transition: background var(--transition-fast); &:hover { background: var(--primary); } }
.panel-right { flex: 1; min-width: 350px; display: flex; flex-direction: column; background: var(--bg-primary); }

.tab-nav {
  display: flex; padding: 0 12px; background: var(--bg-secondary); border-bottom: 1px solid var(--border-default);
  .tab-btn {
    position: relative; padding: 12px 16px; font-size: 13px; color: var(--text-secondary); transition: color var(--transition-fast);
    &::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: transparent; transition: background var(--transition-fast); }
    &:hover { color: var(--text-normal); }
    &.active { color: var(--primary); &::after { background: var(--primary); } }
    .badge { @include flex-center; min-width: 18px; height: 18px; margin-left: 6px; padding: 0 5px; background: var(--danger); color: white; font-size: 11px; border-radius: var(--radius-full); }
  }
}

.tab-content { flex: 1; overflow: hidden; }
.tab-pane { height: 100%; overflow: auto; }

.placeholder-panel {
  @include flex-center;
  @include flex-column;
  padding: 48px 24px; text-align: center; color: var(--text-muted);
  svg { width: 48px; height: 48px; margin-bottom: 16px; opacity: 0.5; }
  h3 { font-size: 16px; font-weight: 600; color: var(--text-normal); margin-bottom: 8px; }
  p { font-size: 13px; margin-bottom: 4px; }
  .coming-soon { color: var(--primary); font-style: italic; }
  .error-list { width: 100%; max-width: 400px; margin-top: 16px; text-align: left; }
  .error-item {
    display: flex; align-items: center; gap: 8px; padding: 8px 12px; margin-bottom: 4px;
    background: var(--bg-secondary); border-radius: var(--radius-sm); cursor: pointer; transition: background var(--transition-fast);
    &:hover { background: var(--bg-tertiary); }
    &.error { border-left: 3px solid var(--danger); }
    &.warning { border-left: 3px solid var(--warning); }
    .error-index { font-family: var(--font-mono); font-size: 12px; color: var(--text-muted); }
    .error-message { font-size: 13px; color: var(--text-normal); }
  }
}

.editor-footer {
  @include flex-between;
  height: 28px; padding: 0 16px;
  background: var(--bg-secondary); border-top: 1px solid var(--border-default);
  font-size: 12px; color: var(--text-muted); flex-shrink: 0;
  .footer-left, .footer-center, .footer-right { display: flex; align-items: center; gap: 8px; }
  .divider { color: var(--border-default); }
  .error-indicator { color: var(--danger); cursor: pointer; &:hover { text-decoration: underline; } }
}
</style>
