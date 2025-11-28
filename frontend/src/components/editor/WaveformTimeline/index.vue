<template>
  <div class="waveform-timeline" ref="containerRef">
    <!-- 缩放控制栏 -->
    <div class="timeline-header">
      <div class="zoom-controls">
        <button class="zoom-btn" @click="zoomOut" title="缩小">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 13H5v-2h14v2z"/>
          </svg>
        </button>
        <div class="zoom-slider">
          <input
            type="range"
            :value="zoomLevel"
            min="20"
            max="500"
            step="10"
            @input="handleZoomInput"
          />
        </div>
        <button class="zoom-btn" @click="zoomIn" title="放大">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
          </svg>
        </button>
        <span class="zoom-label">{{ zoomLevel }}%</span>
        <button class="fit-btn" @click="fitToScreen" title="适应屏幕">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 5v4h2V5h4V3H5c-1.1 0-2 .9-2 2zm2 10H3v4c0 1.1.9 2 2 2h4v-2H5v-4zm14 4h-4v2h4c1.1 0 2-.9 2-2v-4h-2v4zm0-16h-4v2h4v4h2V5c0-1.1-.9-2-2-2z"/>
          </svg>
        </button>
      </div>

      <div class="time-indicator">
        <span class="current-time">{{ formatTime(currentTime) }}</span>
        <span class="separator">/</span>
        <span class="total-time">{{ formatTime(duration) }}</span>
      </div>
    </div>

    <!-- 波形容器 -->
    <div class="waveform-wrapper">
      <div id="waveform" ref="waveformRef"></div>

      <!-- 加载状态 -->
      <div v-if="isLoading" class="waveform-loading">
        <div class="loading-spinner"></div>
        <span>加载波形中...</span>
      </div>

      <!-- 错误状态 -->
      <div v-if="hasError" class="waveform-error">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
        </svg>
        <span>{{ errorMessage }}</span>
        <button @click="retryLoad">重试</button>
      </div>
    </div>

    <!-- 时间轴刻度 -->
    <div id="timeline" ref="timelineRef"></div>

    <!-- 操作提示 -->
    <div class="timeline-tips">
      <span class="tip"><kbd>拖拽</kbd> 调整字幕时间</span>
      <span class="tip"><kbd>点击</kbd> 跳转播放</span>
      <span class="tip"><kbd>Ctrl+滚轮</kbd> 缩放</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useProjectStore } from '@/stores/projectStore'

// Props
const props = defineProps({
  audioUrl: String,
  peaksUrl: String,
  jobId: String,
  waveColor: { type: String, default: '#58a6ff' },
  progressColor: { type: String, default: '#238636' },
  cursorColor: { type: String, default: '#f85149' },
  height: { type: Number, default: 128 },
  regionColor: { type: String, default: 'rgba(88, 166, 255, 0.25)' },
  dragEnabled: { type: Boolean, default: true },
  resizeEnabled: { type: Boolean, default: true }
})

const emit = defineEmits(['ready', 'region-update', 'region-click', 'seek', 'zoom'])

// Store
const projectStore = useProjectStore()

// Refs
const containerRef = ref(null)
const waveformRef = ref(null)
const timelineRef = ref(null)

// State
const zoomLevel = ref(100)
const isLoading = ref(true)
const hasError = ref(false)
const errorMessage = ref('')
const isReady = ref(false)
const isUpdatingRegions = ref(false)

// Wavesurfer实例
let wavesurfer = null
let regionsPlugin = null
let regionUpdateTimer = null

// Computed
const audioSource = computed(() => {
  if (props.audioUrl) return props.audioUrl
  if (props.jobId) return `/api/media/${props.jobId}/audio`
  return projectStore.meta.audioPath || ''
})

const peaksSource = computed(() => {
  if (props.peaksUrl) return props.peaksUrl
  if (props.jobId) return `/api/media/${props.jobId}/peaks?samples=2000`
  return projectStore.meta.peaksPath || ''
})

const currentTime = computed(() => projectStore.player.currentTime)
const duration = computed(() => projectStore.meta.duration || 0)

// 初始化 Wavesurfer
async function initWavesurfer() {
  if (!waveformRef.value) return

  try {
    // 动态导入 wavesurfer
    const WaveSurfer = (await import('wavesurfer.js')).default
    const RegionsPlugin = (await import('wavesurfer.js/dist/plugins/regions.js')).default
    const TimelinePlugin = (await import('wavesurfer.js/dist/plugins/timeline.js')).default

    // 创建插件
    regionsPlugin = RegionsPlugin.create()

    const timelinePlugin = TimelinePlugin.create({
      container: timelineRef.value,
      primaryLabelInterval: 10,
      secondaryLabelInterval: 5,
      primaryColor: '#6e7681',
      secondaryColor: '#484f58',
      primaryFontColor: '#8b949e',
      secondaryFontColor: '#6e7681'
    })

    // 创建实例
    wavesurfer = WaveSurfer.create({
      container: waveformRef.value,
      waveColor: props.waveColor,
      progressColor: props.progressColor,
      cursorColor: props.cursorColor,
      height: props.height,
      normalize: true,
      backend: 'MediaElement',
      plugins: [regionsPlugin, timelinePlugin],
      minPxPerSec: 50,
      scrollParent: true,
      fillParent: true,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      // 静音波形音频，避免与视频声音重叠
      media: document.createElement('audio'),
    })
    
    // 确保 WaveSurfer 静音（音频由视频播放器控制）
    wavesurfer.setMuted(true)

    // 设置事件监听
    setupWavesurferEvents()
    setupRegionEvents()

    // 加载数据
    await loadAudioData()
  } catch (error) {
    console.error('初始化波形失败:', error)
    hasError.value = true
    errorMessage.value = '波形组件加载失败'
    isLoading.value = false
  }
}

// 设置 Wavesurfer 事件
function setupWavesurferEvents() {
  if (!wavesurfer) return

  wavesurfer.on('ready', () => {
    isLoading.value = false
    isReady.value = true
    renderSubtitleRegions()
    emit('ready')
  })

  // 注意：不监听 wavesurfer 的 play/pause 事件来修改 Store
  // WaveSurfer 只作为视觉组件，跟随 VideoStage 的状态
  // Store.isPlaying 由 VideoStage 和用户操作统一管理

  wavesurfer.on('timeupdate', (time) => {
    projectStore.player.currentTime = time
  })

  wavesurfer.on('interaction', (time) => {
    projectStore.seekTo(time)
    emit('seek', time)
  })

  wavesurfer.on('zoom', (minPxPerSec) => {
    const newZoom = Math.round((minPxPerSec / 50) * 100)
    zoomLevel.value = newZoom
    emit('zoom', newZoom)
  })

  wavesurfer.on('error', (error) => {
    console.error('Wavesurfer error:', error)
    hasError.value = true
    errorMessage.value = '波形加载失败'
    isLoading.value = false
  })
}

// 设置 Region 事件
function setupRegionEvents() {
  if (!regionsPlugin) return

  // WaveSurfer.js 7.x 使用 'region-updated' 事件
  regionsPlugin.on('region-updated', (region) => {
    if (isUpdatingRegions.value) return
    projectStore.updateSubtitle(region.id, {
      start: region.start,
      end: region.end
    })
    emit('region-update', region)
  })

  regionsPlugin.on('region-clicked', (region, e) => {
    e.stopPropagation()
    projectStore.view.selectedSubtitleId = region.id
    projectStore.seekTo(region.start)
    if (wavesurfer) wavesurfer.play()
    emit('region-click', region)
  })

  regionsPlugin.on('region-in', (region) => {
    region.setOptions({ color: 'rgba(88, 166, 255, 0.4)' })
  })

  regionsPlugin.on('region-out', (region) => {
    region.setOptions({ color: props.regionColor })
  })
}

// 加载音频数据
async function loadAudioData() {
  if (!audioSource.value) {
    isLoading.value = false
    return
  }

  try {
    // 尝试加载峰值数据
    if (peaksSource.value) {
      const response = await fetch(peaksSource.value)
      if (response.ok) {
        const data = await response.json()
        wavesurfer.load(audioSource.value, data.peaks, data.duration)
        return
      }
    }

    // 降级：直接加载音频
    wavesurfer.load(audioSource.value)
  } catch (error) {
    console.error('加载音频失败:', error)
    // 尝试直接加载音频
    wavesurfer.load(audioSource.value)
  }
}

// 渲染字幕区域
function renderSubtitleRegions() {
  if (!isReady.value || !regionsPlugin) return

  isUpdatingRegions.value = true
  regionsPlugin.clearRegions()

  projectStore.subtitles.forEach(subtitle => {
    const isSelected = subtitle.id === projectStore.view.selectedSubtitleId
    regionsPlugin.addRegion({
      id: subtitle.id,
      start: subtitle.start,
      end: subtitle.end,
      color: isSelected ? 'rgba(163, 113, 247, 0.35)' : props.regionColor,
      drag: props.dragEnabled,
      resize: props.resizeEnabled
    })
  })

  setTimeout(() => {
    isUpdatingRegions.value = false
  }, 100)
}

// 缩放控制
function handleZoomInput(e) {
  const value = parseInt(e.target.value)
  setZoom(value)
}

function setZoom(value) {
  if (!wavesurfer) return
  zoomLevel.value = value
  const minPxPerSec = (value / 100) * 50
  wavesurfer.zoom(minPxPerSec)
  projectStore.view.zoomLevel = value
}

function zoomIn() {
  setZoom(Math.min(500, zoomLevel.value + 20))
}

function zoomOut() {
  setZoom(Math.max(20, zoomLevel.value - 20))
}

function fitToScreen() {
  if (!wavesurfer || !containerRef.value) return
  const containerWidth = containerRef.value.offsetWidth - 32
  const audioDuration = wavesurfer.getDuration()
  if (audioDuration > 0) {
    const minPxPerSec = containerWidth / audioDuration
    wavesurfer.zoom(minPxPerSec)
    zoomLevel.value = Math.round((minPxPerSec / 50) * 100)
  }
}

// 重试加载
function retryLoad() {
  hasError.value = false
  errorMessage.value = ''
  isLoading.value = true
  loadAudioData()
}

// 格式化时间
function formatTime(seconds) {
  if (!seconds || isNaN(seconds)) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

// 监听字幕变化
watch(
  () => [...projectStore.subtitles],
  () => {
    if (isReady.value && !isUpdatingRegions.value) {
      clearTimeout(regionUpdateTimer)
      regionUpdateTimer = setTimeout(() => {
        renderSubtitleRegions()
      }, 100)
    }
  },
  { deep: true }
)

// 监听播放状态
watch(() => projectStore.player.isPlaying, (playing) => {
  if (!wavesurfer || !isReady.value) return
  if (playing) {
    wavesurfer.play()
  } else {
    wavesurfer.pause()
  }
})

// 监听时间变化
watch(() => projectStore.player.currentTime, (newTime) => {
  if (!wavesurfer || !isReady.value) return
  const currentWsTime = wavesurfer.getCurrentTime()
  if (Math.abs(currentWsTime - newTime) > 0.1) {
    wavesurfer.seekTo(newTime / wavesurfer.getDuration())
  }
})

// 监听选中字幕变化
watch(() => projectStore.view.selectedSubtitleId, () => {
  if (isReady.value) {
    renderSubtitleRegions()
  }
})

// 鼠标滚轮缩放
function handleWheel(e) {
  if (!e.ctrlKey) return
  e.preventDefault()
  const delta = e.deltaY < 0 ? 20 : -20
  setZoom(Math.max(20, Math.min(500, zoomLevel.value + delta)))
}

onMounted(async () => {
  await nextTick()
  await initWavesurfer()
  containerRef.value?.addEventListener('wheel', handleWheel, { passive: false })
})

onUnmounted(() => {
  containerRef.value?.removeEventListener('wheel', handleWheel)
  clearTimeout(regionUpdateTimer)
  if (wavesurfer) {
    wavesurfer.destroy()
    wavesurfer = null
  }
})
</script>

<style lang="scss" scoped>
.waveform-timeline {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-secondary);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

// 头部控制栏
.timeline-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: var(--bg-tertiary);
  border-bottom: 1px solid var(--border-default);
}

.zoom-controls {
  display: flex;
  align-items: center;
  gap: 8px;

  .zoom-btn, .fit-btn {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    transition: all var(--transition-fast);

    svg { width: 16px; height: 16px; }

    &:hover {
      background: var(--bg-elevated);
      color: var(--text-primary);
    }
  }

  .zoom-slider {
    width: 100px;

    input[type="range"] {
      -webkit-appearance: none;
      width: 100%;
      height: 4px;
      background: var(--bg-elevated);
      border-radius: 2px;
      cursor: pointer;

      &::-webkit-slider-thumb {
        -webkit-appearance: none;
        width: 12px;
        height: 12px;
        background: var(--primary);
        border-radius: 50%;
        cursor: pointer;
      }
    }
  }

  .zoom-label {
    min-width: 50px;
    font-size: 12px;
    font-family: var(--font-mono);
    color: var(--text-muted);
    text-align: center;
  }
}

.time-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono);
  font-size: 13px;

  .current-time {
    color: var(--primary);
    font-weight: 600;
  }

  .separator {
    color: var(--text-muted);
  }

  .total-time {
    color: var(--text-secondary);
  }
}

// 波形容器
.waveform-wrapper {
  flex: 1;
  position: relative;
  min-height: 128px;

  #waveform {
    height: 100%;

    :deep(.wavesurfer-region) {
      border-radius: 2px;
      transition: background-color 0.2s;

      &:hover {
        background-color: rgba(88, 166, 255, 0.4) !important;
      }
    }

    :deep(.wavesurfer-handle) {
      background: var(--primary) !important;
      width: 4px !important;
      border-radius: 2px;
    }
  }
}

// 加载状态
.waveform-loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: var(--bg-secondary);
  color: var(--text-muted);

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-default);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

// 错误状态
.waveform-error {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: var(--bg-secondary);
  color: var(--text-muted);

  svg {
    width: 40px;
    height: 40px;
    color: var(--danger);
  }

  button {
    padding: 6px 16px;
    background: var(--primary);
    color: white;
    border-radius: var(--radius-md);
    font-size: 13px;
    &:hover { background: var(--primary-hover); }
  }
}

// 时间轴
#timeline {
  height: 24px;
  padding: 0 16px;
  background: var(--bg-tertiary);
  border-top: 1px solid var(--border-default);
}

// 操作提示
.timeline-tips {
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 8px 16px;
  background: var(--bg-tertiary);
  border-top: 1px solid var(--border-default);

  .tip {
    font-size: 11px;
    color: var(--text-muted);

    kbd {
      display: inline-block;
      padding: 2px 6px;
      margin-right: 4px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-default);
      border-radius: 3px;
      font-family: var(--font-mono);
      font-size: 10px;
    }
  }
}
</style>
