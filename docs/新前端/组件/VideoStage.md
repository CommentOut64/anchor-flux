# VideoStage - 视频播放器组件

## 概述

VideoStage 是字幕编辑器的核心视频播放组件，负责视频的播放控制、字幕实时显示和与全局播放状态的双向同步。它基于 HTML5 video 元素，提供了完整的播放控制功能和字幕覆盖显示。

## 功能特性

1. **视频播放控制**
   - 播放/暂停控制
   - 进度拖拽定位
   - 倍速播放（0.5x - 4.0x）
   - 音量控制

2. **字幕实时显示**
   - 字幕覆盖层
   - 自动跟随播放时间
   - 样式自定义

3. **状态同步**
   - 与 ProjectStore 播放器状态双向绑定
   - 时间同步防抖处理
   - 播放状态实时更新

4. **格式兼容处理**
   - 支持 Proxy 视频自动加载
   - 浏览器兼容性检测
   - 错误降级处理

## 技术依赖

```json
{
  "vue": "^3.4.0",
  "pinia": "^3.0.0",
  "@vueuse/core": "^10.0.0"
}
```

## 组件属性

```typescript
interface VideoStageProps {
  // 视频源配置
  videoUrl?: string           // 视频URL，不传则自动构建
  jobId?: string             // 任务ID，用于构建URL
  autoPlay?: boolean         // 自动播放，默认false
  muted?: boolean           // 静音启动，默认false

  // 字幕显示配置
  showSubtitle?: boolean     // 显示字幕，默认true
  subtitleStyle?: {
    fontSize?: string
    color?: string
    background?: string
    padding?: string
  }

  // 控制配置
  enableKeyboard?: boolean   // 键盘快捷键，默认true
  seekStep?: number         // 快进/快退步长（秒），默认5
}

interface VideoStageEmits {
  'loaded': (duration: number) => void
  'error': (error: Error) => void
  'play': () => void
  'pause': () => void
  'timeupdate': (currentTime: number) => void
  'ended': () => void
}
```

## 核心实现

```vue
<!-- components/editor/VideoStage/index.vue -->
<template>
  <div class="video-stage" :class="{ fullscreen: isFullscreen }">
    <!-- 视频容器 -->
    <div class="video-container" ref="containerRef">
      <!-- HTML5 视频元素 -->
      <video
        ref="videoRef"
        :src="videoSource"
        :muted="muted"
        @loadedmetadata="onMetadataLoaded"
        @timeupdate="onTimeUpdate"
        @play="onPlay"
        @pause="onPause"
        @ended="onEnded"
        @error="onError"
        @waiting="isLoading = true"
        @canplay="isLoading = false"
      />

      <!-- 字幕覆盖层 -->
      <div
        v-if="showSubtitle && currentSubtitleText"
        class="subtitle-overlay"
        :style="computedSubtitleStyle"
      >
        {{ currentSubtitleText }}
      </div>

      <!-- 加载指示器 -->
      <div v-if="isLoading" class="loading-overlay">
        <el-icon class="is-loading" :size="48">
          <Loading />
        </el-icon>
      </div>

      <!-- 错误提示 -->
      <div v-if="hasError" class="error-overlay">
        <el-icon :size="48"><WarningFilled /></el-icon>
        <p>{{ errorMessage }}</p>
        <el-button
          v-if="canRetry"
          @click="retryLoad"
          type="primary"
        >
          重试
        </el-button>
      </div>

      <!-- 播放按钮（大） -->
      <transition name="fade">
        <div
          v-if="showBigPlayButton"
          class="big-play-button"
          @click="togglePlay"
        >
          <el-icon :size="64">
            <VideoPlay v-if="!isPlaying" />
            <VideoPause v-else />
          </el-icon>
        </div>
      </transition>
    </div>

    <!-- 视频信息栏 -->
    <div class="video-info">
      <span class="time-display">
        {{ formatTime(currentTime) }} / {{ formatTime(duration) }}
      </span>
      <span class="filename">{{ filename }}</span>
      <el-button
        v-if="supportsFullscreen"
        @click="toggleFullscreen"
        circle
        size="small"
      >
        <el-icon>
          <FullScreen v-if="!isFullscreen" />
          <Close v-else />
        </el-icon>
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { useKeyModifier, useFullscreen } from '@vueuse/core'
import { ElMessage } from 'element-plus'

// Props & Emits
const props = defineProps({
  videoUrl: String,
  jobId: String,
  autoPlay: {
    type: Boolean,
    default: false
  },
  muted: {
    type: Boolean,
    default: false
  },
  showSubtitle: {
    type: Boolean,
    default: true
  },
  subtitleStyle: Object,
  enableKeyboard: {
    type: Boolean,
    default: true
  },
  seekStep: {
    type: Number,
    default: 5
  }
})

const emit = defineEmits([
  'loaded', 'error', 'play', 'pause', 'timeupdate', 'ended'
])

// Store
const projectStore = useProjectStore()

// Refs
const videoRef = ref(null)
const containerRef = ref(null)

// State
const isLoading = ref(false)
const hasError = ref(false)
const errorMessage = ref('')
const canRetry = ref(false)
const showBigPlayButton = ref(false)
const lastSyncTime = ref(0)

// Fullscreen
const { isFullscreen, toggle: toggleFullscreen } = useFullscreen(containerRef)
const supportsFullscreen = computed(() => document.fullscreenEnabled)

// Computed
const videoSource = computed(() => {
  if (props.videoUrl) return props.videoUrl
  if (props.jobId) return `/api/media/${props.jobId}/video`
  return projectStore.meta.videoPath || ''
})

const filename = computed(() => projectStore.meta.filename)
const duration = computed(() => projectStore.meta.duration)
const currentTime = computed(() => projectStore.player.currentTime)
const isPlaying = computed(() => projectStore.player.isPlaying)

const currentSubtitleText = computed(() => {
  return projectStore.currentSubtitle?.text || ''
})

const computedSubtitleStyle = computed(() => ({
  fontSize: props.subtitleStyle?.fontSize || '24px',
  color: props.subtitleStyle?.color || 'white',
  background: props.subtitleStyle?.background || 'rgba(0, 0, 0, 0.8)',
  padding: props.subtitleStyle?.padding || '8px 16px',
  ...props.subtitleStyle
}))

// Watch Store State
watch(() => projectStore.player.isPlaying, (playing) => {
  if (!videoRef.value) return

  if (playing) {
    videoRef.value.play().catch(error => {
      console.error('播放失败:', error)
      projectStore.player.isPlaying = false
    })
  } else {
    videoRef.value.pause()
  }
})

watch(() => projectStore.player.currentTime, (newTime) => {
  if (!videoRef.value) return

  // 防抖：避免循环触发
  const now = Date.now()
  if (now - lastSyncTime.value < 100) return

  // 差值大于0.1秒才同步
  if (Math.abs(videoRef.value.currentTime - newTime) > 0.1) {
    videoRef.value.currentTime = newTime
    lastSyncTime.value = now
  }
})

watch(() => projectStore.player.playbackRate, (rate) => {
  if (videoRef.value) {
    videoRef.value.playbackRate = rate
  }
})

watch(() => projectStore.player.volume, (volume) => {
  if (videoRef.value) {
    videoRef.value.volume = volume
  }
})

// Event Handlers
function onMetadataLoaded() {
  const video = videoRef.value
  projectStore.meta.duration = video.duration

  // 应用初始设置
  video.playbackRate = projectStore.player.playbackRate
  video.volume = projectStore.player.volume

  emit('loaded', video.duration)

  if (props.autoPlay) {
    togglePlay()
  }
}

function onTimeUpdate() {
  const video = videoRef.value
  projectStore.player.currentTime = video.currentTime
  emit('timeupdate', video.currentTime)
}

function onPlay() {
  projectStore.player.isPlaying = true
  showBigPlayButton.value = false
  emit('play')
}

function onPause() {
  projectStore.player.isPlaying = false
  emit('pause')
}

function onEnded() {
  projectStore.player.isPlaying = false
  emit('ended')
}

function onError(event) {
  const video = videoRef.value
  hasError.value = true

  // 分析错误类型
  const error = video?.error
  if (error) {
    switch (error.code) {
      case 1:
        errorMessage.value = '视频加载被中止'
        break
      case 2:
        errorMessage.value = '网络错误'
        canRetry.value = true
        break
      case 3:
        errorMessage.value = '视频解码失败'
        checkForProxy()
        break
      case 4:
        errorMessage.value = '不支持的视频格式'
        checkForProxy()
        break
      default:
        errorMessage.value = '未知错误'
    }
  }

  emit('error', new Error(errorMessage.value))
}

// Proxy视频处理
async function checkForProxy() {
  if (!props.jobId) return

  try {
    const response = await fetch(`/api/media/${props.jobId}/proxy-status`)
    const data = await response.json()

    if (data.exists) {
      // Proxy已存在，尝试加载
      videoRef.value.src = `/api/media/${props.jobId}/video?proxy=true&t=${Date.now()}`
      hasError.value = false
      canRetry.value = true
    } else {
      // 需要生成Proxy
      errorMessage.value = '视频格式不兼容，正在生成预览版本...'
      pollProxyStatus()
    }
  } catch (error) {
    console.error('检查Proxy状态失败:', error)
  }
}

function pollProxyStatus() {
  const timer = setInterval(async () => {
    try {
      const response = await fetch(`/api/media/${props.jobId}/proxy-status`)
      const data = await response.json()

      if (data.exists) {
        clearInterval(timer)
        retryLoad()
      }
    } catch (error) {
      clearInterval(timer)
    }
  }, 3000)

  // 最多等待5分钟
  setTimeout(() => clearInterval(timer), 300000)
}

function retryLoad() {
  hasError.value = false
  errorMessage.value = ''
  videoRef.value.load()
}

// Controls
function togglePlay() {
  projectStore.player.isPlaying = !projectStore.player.isPlaying
}

function seek(seconds) {
  const video = videoRef.value
  if (!video) return

  const newTime = Math.max(0, Math.min(video.duration, video.currentTime + seconds))
  projectStore.seekTo(newTime)
}

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)

  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }
  return `${m}:${s.toString().padStart(2, '0')}`
}

// Keyboard Shortcuts
function handleKeyboard(event) {
  if (!props.enableKeyboard || !videoRef.value) return

  switch (event.code) {
    case 'Space':
      event.preventDefault()
      togglePlay()
      break
    case 'ArrowLeft':
      event.preventDefault()
      seek(-props.seekStep)
      break
    case 'ArrowRight':
      event.preventDefault()
      seek(props.seekStep)
      break
    case 'ArrowUp':
      event.preventDefault()
      projectStore.player.volume = Math.min(1, projectStore.player.volume + 0.1)
      break
    case 'ArrowDown':
      event.preventDefault()
      projectStore.player.volume = Math.max(0, projectStore.player.volume - 0.1)
      break
    case 'KeyF':
      if (!event.ctrlKey && !event.metaKey) {
        event.preventDefault()
        toggleFullscreen()
      }
      break
  }
}

// Lifecycle
onMounted(() => {
  document.addEventListener('keydown', handleKeyboard)

  // 鼠标悬停显示大播放按钮
  containerRef.value?.addEventListener('mouseenter', () => {
    if (!isPlaying.value) {
      showBigPlayButton.value = true
    }
  })

  containerRef.value?.addEventListener('mouseleave', () => {
    showBigPlayButton.value = false
  })
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyboard)
})
</script>
```

## 样式定义

```scss
.video-stage {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #000;

  .video-container {
    position: relative;
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;

    video {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
    }

    // 字幕覆盖层
    .subtitle-overlay {
      position: absolute;
      bottom: 60px;
      left: 50%;
      transform: translateX(-50%);
      border-radius: 4px;
      max-width: 80%;
      text-align: center;
      pointer-events: none;
      z-index: 10;
    }

    // 加载/错误覆盖层
    .loading-overlay,
    .error-overlay {
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.8);
      color: white;
      z-index: 20;
    }

    // 大播放按钮
    .big-play-button {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.3);
      cursor: pointer;
      z-index: 15;
      transition: opacity 0.3s;

      &:hover {
        background: rgba(0, 0, 0, 0.5);
      }
    }
  }

  // 全屏模式
  &.fullscreen {
    position: fixed;
    inset: 0;
    z-index: 9999;

    .video-container video {
      width: 100%;
      height: 100%;
    }
  }
}
```

## 与其他组件的关系

### 依赖关系
- 强依赖 `ProjectStore` 的播放器状态
- 依赖 Element Plus 的 UI 组件

### 被依赖关系
- 被 `EditorView` 包含使用
- 与 `WaveformTimeline` 协同工作
- 与 `PlaybackControls` 共享控制

## 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| 空格 | 播放/暂停 |
| ← | 后退5秒 |
| → | 前进5秒 |
| ↑ | 增加音量 |
| ↓ | 减少音量 |
| F | 全屏切换 |

## 性能优化

1. **防抖同步**
   - 时间同步使用100ms防抖
   - 避免Store和video元素循环触发

2. **懒加载**
   - Proxy视频按需生成
   - 错误时才检查替代方案

3. **内存管理**
   - 组件销毁时清理事件监听器

## 测试要点

1. 视频格式兼容性
2. 播放状态同步
3. 字幕显示正确性
4. 键盘快捷键功能
5. 全屏模式切换
6. 错误恢复机制

## 未来扩展

1. 画中画模式
2. 视频截图功能
3. 帧级精确控制
4. 多音轨切换
5. 视频滤镜效果