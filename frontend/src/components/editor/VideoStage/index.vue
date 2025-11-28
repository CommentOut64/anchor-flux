<template>
  <div class="video-stage" :class="{ 'is-fullscreen': isFullscreen }">
    <!-- 视频容器 -->
    <div class="video-container" ref="containerRef" @click="handleContainerClick" @dblclick="toggleFullscreen">
      <!-- HTML5 视频元素 -->
      <video
        ref="videoRef"
        :src="videoSource"
        :muted="muted"
        preload="metadata"
        @loadedmetadata="onMetadataLoaded"
        @timeupdate="onTimeUpdate"
        @play="onPlay"
        @pause="onPause"
        @ended="onEnded"
        @error="onError"
        @waiting="isBuffering = true"
        @canplay="isBuffering = false"
        @progress="onProgress"
      />

      <!-- 字幕覆盖层 -->
      <div v-if="showSubtitle && currentSubtitleText" class="subtitle-overlay">
        <span class="subtitle-text">{{ currentSubtitleText }}</span>
      </div>

      <!-- 加载指示器 -->
      <transition name="fade">
        <div v-if="isBuffering" class="video-overlay loading-overlay">
          <div class="loading-spinner"></div>
          <span>加载中...</span>
        </div>
      </transition>

      <!-- 错误提示 -->
      <transition name="fade">
        <div v-if="hasError" class="video-overlay error-overlay">
          <svg class="error-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
          </svg>
          <p class="error-message">{{ errorMessage }}</p>
          <button v-if="canRetry" class="retry-btn" @click="retryLoad">重试</button>
        </div>
      </transition>

      <!-- 播放/暂停状态提示（短暂显示） -->
      <transition name="pop">
        <div v-if="showStateHint" class="state-hint">
          <svg v-if="stateHintType === 'play'" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z"/>
          </svg>
          <svg v-else-if="stateHintType === 'pause'" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
          </svg>
          <svg v-else-if="stateHintType === 'volume'" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
          </svg>
          <span v-if="stateHintText">{{ stateHintText }}</span>
        </div>
      </transition>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useProjectStore } from '@/stores/projectStore'

// Props
const props = defineProps({
  videoUrl: String,
  jobId: String,
  autoPlay: { type: Boolean, default: false },
  muted: { type: Boolean, default: false },
  showSubtitle: { type: Boolean, default: true },
  enableKeyboard: { type: Boolean, default: true },
  seekStep: { type: Number, default: 5 }
})

const emit = defineEmits(['loaded', 'error', 'play', 'pause', 'timeupdate', 'ended'])

// Store
const projectStore = useProjectStore()

// Refs
const videoRef = ref(null)
const containerRef = ref(null)

// State
const isBuffering = ref(false)
const hasError = ref(false)
const errorMessage = ref('')
const canRetry = ref(false)
const isFullscreen = ref(false)
const lastSyncTime = ref(0)

// 状态提示（短暂显示播放/暂停图标）
const showStateHint = ref(false)
const stateHintType = ref('')
const stateHintText = ref('')
let stateHintTimer = null

// Computed
const videoSource = computed(() => {
  if (props.videoUrl) return props.videoUrl
  if (props.jobId) return `/api/media/${props.jobId}/video`
  return projectStore.meta.videoPath || ''
})

const currentSubtitleText = computed(() => projectStore.currentSubtitle?.text || '')
const isPlaying = computed(() => projectStore.player.isPlaying)

// 显示状态提示
function showHint(type, text = '') {
  stateHintType.value = type
  stateHintText.value = text
  showStateHint.value = true
  clearTimeout(stateHintTimer)
  stateHintTimer = setTimeout(() => {
    showStateHint.value = false
  }, 800)
}

// 监听 Store 播放状态（单向：Store → Video）
watch(() => projectStore.player.isPlaying, async (playing) => {
  if (!videoRef.value) return

  const isPaused = videoRef.value.paused

  if (playing && isPaused) {
    // 需要播放且当前是暂停状态
    try {
      await videoRef.value.play()
    } catch (error) {
      console.error('[VideoStage] 播放失败:', error)
      // 播放失败时回滚状态
      projectStore.player.isPlaying = false
    }
  } else if (!playing && !isPaused) {
    // 需要暂停且当前是播放状态
    videoRef.value.pause()
  }
})

// 监听 Store 时间变化（外部跳转）
watch(() => projectStore.player.currentTime, (newTime) => {
  if (!videoRef.value) return
  const now = Date.now()
  if (now - lastSyncTime.value < 100) return
  if (Math.abs(videoRef.value.currentTime - newTime) > 0.1) {
    videoRef.value.currentTime = newTime
    lastSyncTime.value = now
  }
})

// 监听播放速度
watch(() => projectStore.player.playbackRate, (rate) => {
  if (videoRef.value) videoRef.value.playbackRate = rate
})

// 监听音量
watch(() => projectStore.player.volume, (volume) => {
  if (videoRef.value) videoRef.value.volume = volume
})

// 事件处理
function onMetadataLoaded() {
  const video = videoRef.value
  projectStore.meta.duration = video.duration
  video.playbackRate = projectStore.player.playbackRate
  video.volume = projectStore.player.volume
  emit('loaded', video.duration)
  if (props.autoPlay) togglePlay()
}

function onTimeUpdate() {
  const video = videoRef.value
  projectStore.player.currentTime = video.currentTime
  emit('timeupdate', video.currentTime)
}

function onPlay() {
  // 显示播放状态提示
  showHint('play')
  emit('play')
}

function onPause() {
  // 显示暂停状态提示
  showHint('pause')
  emit('pause')
}

function onEnded() {
  // 视频播放结束，需要同步到 Store
  projectStore.player.isPlaying = false
  emit('ended')
}

function onProgress() {
  // 可以计算缓冲进度
}

function onError() {
  const video = videoRef.value
  hasError.value = true
  const error = video?.error
  if (error) {
    switch (error.code) {
      case 1: errorMessage.value = '视频加载被中止'; break
      case 2: errorMessage.value = '网络错误'; canRetry.value = true; break
      case 3: errorMessage.value = '视频解码失败'; break
      case 4: errorMessage.value = '不支持的视频格式'; break
      default: errorMessage.value = '未知错误'
    }
  }
  emit('error', new Error(errorMessage.value))
}

function retryLoad() {
  hasError.value = false
  errorMessage.value = ''
  videoRef.value?.load()
}

// 控制方法
function togglePlay() {
  projectStore.player.isPlaying = !projectStore.player.isPlaying
}

function seek(seconds) {
  const video = videoRef.value
  if (!video) return
  const newTime = Math.max(0, Math.min(video.duration, video.currentTime + seconds))
  projectStore.seekTo(newTime)
}

function toggleFullscreen() {
  if (!containerRef.value) return
  if (!document.fullscreenElement) {
    containerRef.value.requestFullscreen()
    isFullscreen.value = true
  } else {
    document.exitFullscreen()
    isFullscreen.value = false
  }
}

// 键盘快捷键
function handleKeyboard(e) {
  if (!props.enableKeyboard || !videoRef.value) return
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

  switch (e.code) {
    case 'Space':
      e.preventDefault()
      togglePlay()
      break
    case 'ArrowLeft':
      e.preventDefault()
      seek(-props.seekStep)
      break
    case 'ArrowRight':
      e.preventDefault()
      seek(props.seekStep)
      break
    case 'KeyF':
      if (!e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        toggleFullscreen()
      }
      break
  }
}

// 全屏变化监听
function handleFullscreenChange() {
  isFullscreen.value = !!document.fullscreenElement
}

// 处理视频容器点击（切换播放暂停）
let clickTimer = null
function handleContainerClick(e) {
  // 清除之前的延迟
  if (clickTimer) {
    clearTimeout(clickTimer)
    clickTimer = null
    return // 这是双击的第二次点击，忽略
  }

  // 延迟执行，如果在延迟期间发生双击，则取消单击操作
  clickTimer = setTimeout(() => {
    clickTimer = null
    // 执行切换播放暂停
    togglePlay()
  }, 200) // 200ms 延迟，足够检测双击
}

onMounted(() => {
  document.addEventListener('keydown', handleKeyboard)
  document.addEventListener('fullscreenchange', handleFullscreenChange)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyboard)
  document.removeEventListener('fullscreenchange', handleFullscreenChange)
  clearTimeout(stateHintTimer)
  if (clickTimer) clearTimeout(clickTimer)
})
</script>

<style lang="scss" scoped>
.video-stage {
  width: 100%;
  height: 100%;
  background: var(--bg-base);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.video-container {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #000;

  video {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
  }
}

// 字幕覆盖层
.subtitle-overlay {
  position: absolute;
  bottom: 48px;
  left: 50%;
  transform: translateX(-50%);
  max-width: 80%;
  z-index: 10;
  pointer-events: none;

  .subtitle-text {
    display: inline-block;
    padding: 8px 20px;
    font-size: 20px;
    line-height: 1.4;
    color: #fff;
    background: rgba(0, 0, 0, 0.75);
    border-radius: var(--radius-sm);
    text-align: center;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
  }
}

// 通用覆盖层
.video-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 20;
}

// 加载状态
.loading-overlay {
  background: rgba(0, 0, 0, 0.6);
  color: var(--text-secondary);
  gap: 12px;

  .loading-spinner {
    width: 40px;
    height: 40px;
    border: 3px solid rgba(255, 255, 255, 0.1);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

// 错误状态
.error-overlay {
  background: rgba(0, 0, 0, 0.85);
  color: var(--text-secondary);
  gap: 16px;

  .error-icon {
    width: 48px;
    height: 48px;
    color: var(--danger);
  }

  .error-message {
    font-size: 14px;
  }

  .retry-btn {
    padding: 8px 24px;
    background: var(--primary);
    color: white;
    border-radius: var(--radius-md);
    font-size: 14px;
    transition: background var(--transition-fast);

    &:hover { background: var(--primary-hover); }
  }
}

// 状态提示（短暂显示播放/暂停图标）
.state-hint {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 16px 24px;
  background: rgba(0, 0, 0, 0.7);
  border-radius: var(--radius-lg);
  color: white;
  z-index: 30;
  pointer-events: none;

  svg {
    width: 32px;
    height: 32px;
  }

  span {
    font-size: 16px;
    font-weight: 500;
  }
}

// 全屏模式
.is-fullscreen {
  .subtitle-overlay {
    bottom: 80px;

    .subtitle-text {
      font-size: 28px;
      padding: 12px 28px;
    }
  }
}

// 动画
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.pop-enter-active {
  animation: pop-in 0.3s ease;
}
.pop-leave-active {
  animation: pop-out 0.2s ease;
}

@keyframes pop-in {
  0% { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
  100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
}

@keyframes pop-out {
  0% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
  100% { transform: translate(-50%, -50%) scale(1.2); opacity: 0; }
}
</style>
