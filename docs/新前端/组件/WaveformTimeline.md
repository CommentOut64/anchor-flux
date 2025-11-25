# WaveformTimeline - 波形时间轴组件

## 概述

WaveformTimeline 是字幕编辑器的核心时间轴组件，基于 wavesurfer.js v7 实现。它提供了音频波形可视化、字幕区域管理、时间轴拖拽编辑和视频缩略图预览等功能，是实现精确字幕时间调整的关键组件。

## 功能特性

1. **波形可视化**
   - 基于后端预计算的峰值数据快速渲染
   - 支持缩放和平移
   - 实时播放进度指示

2. **字幕区域管理**
   - 可拖拽调整开始/结束时间
   - 区域大小可调整
   - 点击区域跳转播放

3. **视频缩略图**
   - 时间轴下方显示视频预览
   - 点击缩略图快速定位
   - 自动从后端获取

4. **双向数据同步**
   - 与 ProjectStore 字幕数据双向绑定
   - 与播放器状态实时同步
   - 防循环更新机制

## 技术依赖

```json
{
  "wavesurfer.js": "^7.0.0",
  "vue": "^3.4.0",
  "pinia": "^3.0.0",
  "element-plus": "^2.11.0"
}
```

## 组件属性

```typescript
interface WaveformTimelineProps {
  // 音频源配置
  audioUrl?: string          // 音频URL
  peaksUrl?: string         // 峰值数据URL
  jobId?: string           // 任务ID（自动构建URL）

  // 波形配置
  waveColor?: string       // 波形颜色，默认'#4a9eff'
  progressColor?: string   // 进度颜色，默认'#1e90ff'
  cursorColor?: string     // 光标颜色，默认'#ff0000'
  height?: number         // 波形高度，默认128

  // Region配置
  regionColor?: string     // 字幕区域颜色
  regionAlpha?: number    // 透明度，默认0.3
  dragEnabled?: boolean   // 允许拖拽，默认true
  resizeEnabled?: boolean // 允许调整大小，默认true

  // 缩略图配置
  showThumbnails?: boolean    // 显示缩略图，默认true
  thumbnailCount?: number     // 缩略图数量，默认20
  thumbnailHeight?: number    // 缩略图高度，默认60
}

interface WaveformTimelineEmits {
  'ready': () => void
  'region-update': (region: Region) => void
  'region-click': (region: Region) => void
  'seek': (time: number) => void
  'zoom': (level: number) => void
}
```

## 核心实现

```vue
<!-- components/editor/Timeline/index.vue -->
<template>
  <div class="waveform-timeline" ref="containerRef">
    <!-- 缩放控制 -->
    <div class="zoom-controls">
      <el-slider
        v-model="zoomLevel"
        :min="10"
        :max="1000"
        :step="10"
        :show-tooltip="false"
        @change="handleZoom"
      />
      <span class="zoom-label">{{ zoomLevel }}%</span>
      <el-button @click="fitToScreen" size="small">适应屏幕</el-button>
    </div>

    <!-- 波形容器 -->
    <div id="waveform" ref="waveformRef"></div>

    <!-- 时间轴 -->
    <div id="timeline" ref="timelineRef"></div>

    <!-- 视频缩略图轨道 -->
    <div
      v-if="showThumbnails"
      class="thumbnail-track"
      ref="thumbnailRef"
      :style="{ height: `${thumbnailHeight}px` }"
    >
      <div class="thumbnails-container">
        <img
          v-for="(thumb, index) in thumbnails"
          :key="index"
          :src="thumb"
          :style="{ left: `${(index / thumbnails.length) * 100}%` }"
          class="thumbnail-image"
          :title="`${formatTime(thumbnailTimestamps[index])}`"
          @click="seekToThumbnail(index)"
        />
      </div>
      <div v-if="loadingThumbnails" class="thumbnails-loading">
        <el-icon class="is-loading"><Loading /></el-icon>
        <span>加载缩略图...</span>
      </div>
    </div>

    <!-- 操作提示 -->
    <div class="timeline-tips">
      <span><kbd>拖拽</kbd> 调整字幕时间</span>
      <span><kbd>点击</kbd> 跳转播放</span>
      <span><kbd>滚轮</kbd> 缩放时间轴</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.js'
import TimelinePlugin from 'wavesurfer.js/dist/plugins/timeline.js'
import { useProjectStore } from '@/stores/projectStore'
import { ElMessage } from 'element-plus'

// Props & Emits
const props = defineProps({
  audioUrl: String,
  peaksUrl: String,
  jobId: String,
  waveColor: {
    type: String,
    default: '#4a9eff'
  },
  progressColor: {
    type: String,
    default: '#1e90ff'
  },
  cursorColor: {
    type: String,
    default: '#ff0000'
  },
  height: {
    type: Number,
    default: 128
  },
  regionColor: {
    type: String,
    default: 'rgba(0, 123, 255, 0.3)'
  },
  regionAlpha: {
    type: Number,
    default: 0.3
  },
  dragEnabled: {
    type: Boolean,
    default: true
  },
  resizeEnabled: {
    type: Boolean,
    default: true
  },
  showThumbnails: {
    type: Boolean,
    default: true
  },
  thumbnailCount: {
    type: Number,
    default: 20
  },
  thumbnailHeight: {
    type: Number,
    default: 60
  }
})

const emit = defineEmits(['ready', 'region-update', 'region-click', 'seek', 'zoom'])

// Store
const projectStore = useProjectStore()

// Refs
const containerRef = ref(null)
const waveformRef = ref(null)
const timelineRef = ref(null)
const thumbnailRef = ref(null)

// State
const zoomLevel = ref(100)
const thumbnails = ref([])
const thumbnailTimestamps = ref([])
const loadingThumbnails = ref(false)
const isReady = ref(false)
const isUpdatingRegions = ref(false)

// Instances
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

// Initialize
onMounted(async () => {
  await initWavesurfer()
  await loadPeaksData()
  if (props.showThumbnails) {
    await loadThumbnails()
  }
})

onUnmounted(() => {
  if (wavesurfer) {
    wavesurfer.destroy()
  }
})

async function initWavesurfer() {
  // Create plugins
  regionsPlugin = RegionsPlugin.create()

  const timelinePlugin = TimelinePlugin.create({
    container: timelineRef.value
  })

  // Create wavesurfer instance
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
    scrollParent: true
  })

  // Setup event handlers
  setupWavesurferEvents()
  setupRegionEvents()
}

function setupWavesurferEvents() {
  wavesurfer.on('play', () => {
    projectStore.player.isPlaying = true
  })

  wavesurfer.on('pause', () => {
    projectStore.player.isPlaying = false
  })

  wavesurfer.on('timeupdate', (currentTime) => {
    projectStore.player.currentTime = currentTime
  })

  wavesurfer.on('ready', () => {
    isReady.value = true
    renderSubtitleRegions()
    emit('ready')
  })

  wavesurfer.on('zoom', (minPxPerSec) => {
    const newZoom = Math.round((minPxPerSec / 50) * 100)
    zoomLevel.value = newZoom
    emit('zoom', newZoom)
  })

  wavesurfer.on('interaction', (newTime) => {
    emit('seek', newTime)
  })
}

function setupRegionEvents() {
  // Region更新结束事件
  regionsPlugin.on('region-update-end', (region) => {
    if (isUpdatingRegions.value) return

    const subtitleId = region.id
    projectStore.updateSubtitle(subtitleId, {
      start: region.start,
      end: region.end
    })

    emit('region-update', region)
  })

  // Region点击事件
  regionsPlugin.on('region-clicked', (region) => {
    projectStore.view.selectedSubtitleId = region.id
    projectStore.seekTo(region.start)
    wavesurfer.play()
    emit('region-click', region)
  })

  // Region进入/离开事件（高亮当前播放）
  regionsPlugin.on('region-in', (region) => {
    region.setOptions({ color: 'rgba(0, 123, 255, 0.5)' })
  })

  regionsPlugin.on('region-out', (region) => {
    region.setOptions({ color: props.regionColor })
  })
}

async function loadPeaksData() {
  if (!peaksSource.value) return

  try {
    const response = await fetch(peaksSource.value)
    const data = await response.json()

    // 使用后端峰值数据加载
    wavesurfer.load(
      audioSource.value,
      data.peaks,
      data.duration
    )
  } catch (error) {
    console.error('加载波形数据失败:', error)
    ElMessage.error('波形数据加载失败')

    // 降级：直接加载音频（会慢一些）
    wavesurfer.load(audioSource.value)
  }
}

async function loadThumbnails() {
  if (!props.jobId) return

  loadingThumbnails.value = true

  try {
    const response = await fetch(
      `/api/media/${props.jobId}/thumbnails?count=${props.thumbnailCount}`
    )
    const data = await response.json()

    thumbnails.value = data.thumbnails
    thumbnailTimestamps.value = data.timestamps
  } catch (error) {
    console.error('加载缩略图失败:', error)
    // 缩略图不是必需的，失败时静默处理
  } finally {
    loadingThumbnails.value = false
  }
}

function renderSubtitleRegions() {
  if (!isReady.value) return

  // 标记正在更新，避免循环
  isUpdatingRegions.value = true

  // 清除旧Regions
  regionsPlugin.clearRegions()

  // 添加新Regions
  projectStore.subtitles.forEach(subtitle => {
    const region = regionsPlugin.addRegion({
      id: subtitle.id,
      start: subtitle.start,
      end: subtitle.end,
      color: props.regionColor,
      drag: props.dragEnabled,
      resize: props.resizeEnabled
    })

    // 高亮选中的字幕
    if (subtitle.id === projectStore.view.selectedSubtitleId) {
      region.setOptions({ color: 'rgba(0, 123, 255, 0.5)' })
    }
  })

  setTimeout(() => {
    isUpdatingRegions.value = false
  }, 100)
}

// Watch store changes
watch(() => projectStore.subtitles, () => {
  if (isReady.value && !isUpdatingRegions.value) {
    // 防抖更新
    clearTimeout(regionUpdateTimer)
    regionUpdateTimer = setTimeout(() => {
      renderSubtitleRegions()
    }, 300)
  }
}, { deep: true })

// 同步播放状态
watch(() => projectStore.player.isPlaying, (playing) => {
  if (!wavesurfer || !isReady.value) return

  if (playing) {
    wavesurfer.play()
  } else {
    wavesurfer.pause()
  }
})

// 同步播放时间
watch(() => projectStore.player.currentTime, (newTime) => {
  if (!wavesurfer || !isReady.value) return

  const currentTime = wavesurfer.getCurrentTime()
  if (Math.abs(currentTime - newTime) > 0.1) {
    wavesurfer.seekTo(newTime / wavesurfer.getDuration())
  }
})

// 同步缩放级别
watch(() => projectStore.view.zoomLevel, (level) => {
  if (!wavesurfer || !isReady.value) return

  const minPxPerSec = (level / 100) * 50
  wavesurfer.zoom(minPxPerSec)
})

// Methods
function handleZoom(value) {
  if (!wavesurfer) return

  const minPxPerSec = (value / 100) * 50
  wavesurfer.zoom(minPxPerSec)
  projectStore.view.zoomLevel = value
}

function fitToScreen() {
  if (!wavesurfer) return

  const containerWidth = containerRef.value.offsetWidth
  const duration = wavesurfer.getDuration()
  const minPxPerSec = containerWidth / duration

  wavesurfer.zoom(minPxPerSec)
  zoomLevel.value = Math.round((minPxPerSec / 50) * 100)
}

function seekToThumbnail(index) {
  const timestamp = thumbnailTimestamps.value[index]
  if (timestamp !== undefined && wavesurfer) {
    projectStore.seekTo(timestamp)
    wavesurfer.seekTo(timestamp / wavesurfer.getDuration())
  }
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

// 鼠标滚轮缩放
function handleWheel(event) {
  if (!wavesurfer || !event.ctrlKey) return

  event.preventDefault()

  const delta = event.deltaY < 0 ? 10 : -10
  const newZoom = Math.max(10, Math.min(1000, zoomLevel.value + delta))

  handleZoom(newZoom)
}

onMounted(() => {
  containerRef.value?.addEventListener('wheel', handleWheel, { passive: false })
})

onUnmounted(() => {
  containerRef.value?.removeEventListener('wheel', handleWheel)
})
</script>
```

## 样式定义

```scss
.waveform-timeline {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-secondary);
  padding: 16px;

  // 缩放控制
  .zoom-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;

    .el-slider {
      width: 200px;
    }

    .zoom-label {
      min-width: 50px;
      color: var(--text-muted);
      font-size: 14px;
    }
  }

  // 波形容器
  #waveform {
    flex: 1;
    min-height: 128px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    overflow: hidden;
  }

  // 时间轴
  #timeline {
    height: 24px;
    margin-top: 8px;
    color: var(--text-muted);
  }

  // 缩略图轨道
  .thumbnail-track {
    position: relative;
    margin-top: 8px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 4px;
    overflow: hidden;

    .thumbnails-container {
      position: relative;
      height: 100%;
    }

    .thumbnail-image {
      position: absolute;
      height: 100%;
      width: auto;
      cursor: pointer;
      opacity: 0.8;
      transition: all 0.2s;

      &:hover {
        transform: scale(1.2);
        opacity: 1;
        z-index: 10;
      }
    }

    .thumbnails-loading {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.5);
      color: var(--text-muted);
    }
  }

  // 操作提示
  .timeline-tips {
    display: flex;
    gap: 20px;
    margin-top: 8px;
    font-size: 12px;
    color: var(--text-muted);

    kbd {
      padding: 2px 6px;
      background: var(--bg-primary);
      border: 1px solid var(--border-color);
      border-radius: 3px;
    }
  }
}
```

## 性能优化

1. **后端峰值数据**
   - 预计算避免前端解析
   - 2000采样点适合大部分场景
   - 缓存机制减少重复请求

2. **防抖更新**
   - Region更新使用300ms防抖
   - 避免频繁重绘

3. **虚拟化渲染**
   - wavesurfer.js内置虚拟化
   - 只渲染可见区域

4. **内存管理**
   - 组件销毁时清理实例
   - 清理所有事件监听器

## 与其他组件的关系

### 依赖关系
- 强依赖 `ProjectStore` 的字幕数据和播放状态
- 依赖后端媒体API提供峰值数据和缩略图

### 被依赖关系
- 被 `EditorView` 包含使用
- 与 `VideoStage` 播放状态同步
- 与 `SubtitleList` 字幕数据同步

## 测试要点

1. 波形加载和渲染
2. Region拖拽精确度
3. 双向数据同步
4. 大量字幕性能
5. 缩放功能
6. 缩略图加载和交互

## 未来扩展

1. 多轨道支持
2. 标记点功能
3. 批量区域操作
4. 音频频谱显��
5. 自定义Region样式