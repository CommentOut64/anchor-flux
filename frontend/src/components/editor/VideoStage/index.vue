<template>
  <div class="video-stage" :class="{ 'is-fullscreen': isFullscreen }">
    <!-- 视频容器 -->
    <div class="video-container" ref="containerRef" @click="handleContainerClick" @dblclick="toggleFullscreen">
      <!-- 视频转码中的占位符 -->
      <transition name="fade">
        <div v-if="showTranscodingPlaceholder" class="video-overlay transcoding-overlay">
          <div class="transcoding-spinner"></div>
          <h3>视频画面正在解码</h3>
          <p>{{ transcodingMessage }}</p>
          <div v-if="transcodingProgress > 0 && transcodingProgress < 100" class="progress-bar">
            <div class="progress-fill" :style="{ width: transcodingProgress + '%' }"></div>
            <span>{{ transcodingProgress }}%</span>
          </div>
        </div>
      </transition>

      <!-- HTML5 视频元素 -->
      <video
        ref="videoRef"
        :src="effectiveVideoSource"
        :muted="muted"
        :preload="preloadStrategy"
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

      <!-- 渐进式加载状态指示器 -->
      <transition name="fade">
        <div v-if="showProgressiveHint" class="progressive-indicator">
          <span class="resolution-badge" :class="resolutionClass">
            {{ currentResolutionLabel }}
          </span>
          <span v-if="isUpgrading" class="upgrade-progress">
            HD {{ Math.round(upgradeProgress) }}%
          </span>
        </div>
      </transition>

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
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useProjectStore } from '@/stores/projectStore'

// Props
const props = defineProps({
  videoUrl: String,
  jobId: String,
  autoPlay: { type: Boolean, default: false },
  muted: { type: Boolean, default: false },
  showSubtitle: { type: Boolean, default: true },
  enableKeyboard: { type: Boolean, default: true },
  seekStep: { type: Number, default: 5 },
  // 渐进式加载相关
  progressiveUrl: String,           // 从外部传入的渐进式 URL
  currentResolution: String,        // 当前分辨率 ('360p', '720p', 'source')
  isUpgrading: { type: Boolean, default: false },  // 是否正在升级
  upgradeProgress: { type: Number, default: 0 }    // 升级进度
})

const emit = defineEmits(['loaded', 'error', 'play', 'pause', 'timeupdate', 'ended', 'resolution-change'])

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
const retryCount = ref(0)
const maxRetries = 3
const showProgressiveHint = ref(false)
let progressiveHintTimer = null

// 状态提示（短暂显示播放/暂停图标）
const showStateHint = ref(false)
const stateHintType = ref('')
const stateHintText = ref('')
let stateHintTimer = null
let timeUpdateTimer = null  // 时间更新防抖定时器

// 防抖的时间更新函数
function debouncedTimeUpdate(time) {
  if (timeUpdateTimer) clearTimeout(timeUpdateTimer)
  timeUpdateTimer = setTimeout(() => {
    // 只有当时间差较大时才更新（避免循环触发）
    if (Math.abs(projectStore.player.currentTime - time) > 0.2) {
      lastSyncTime.value = Date.now()  // 设置同步标记
      projectStore.player.currentTime = time
    }
  }, 200)  // 增加到200ms防抖，减少更新频率
}

// Computed
const videoSource = computed(() => {
  if (props.videoUrl) return props.videoUrl
  if (props.jobId) return `/api/media/${props.jobId}/video`
  return projectStore.meta.videoPath || ''
})

// 实际使用的视频源（支持渐进式加载）
const effectiveVideoSource = computed(() => {
  // 优先使用渐进式 URL
  if (props.progressiveUrl) return props.progressiveUrl
  return videoSource.value
})

// 动态 preload 策略（根据视频时长决定）
const preloadStrategy = computed(() => {
  const duration = projectStore.meta.duration
  if (duration < 180) return 'auto'       // 3分钟内：自动预加载（避免过度预加载）
  if (duration < 1800) return 'metadata'  // 30分钟内：仅元数据
  return 'metadata'                       // 超长视频：也使用metadata而不是none，保证基本性能
})

// 分辨率标签
const currentResolutionLabel = computed(() => {
  switch (props.currentResolution) {
    case '360p': return '360P'
    case '720p': return 'HD'
    case 'source': return 'SRC'
    default: return ''
  }
})

// 分辨率样式类
const resolutionClass = computed(() => {
  return {
    'preview': props.currentResolution === '360p',
    'hd': props.currentResolution === '720p',
    'source': props.currentResolution === 'source'
  }
})

const currentSubtitleText = computed(() => projectStore.currentSubtitle?.text || '')
const isPlaying = computed(() => projectStore.player.isPlaying)

// 转码占位符相关
const showTranscodingPlaceholder = computed(() => {
  // 当视频源为空或正在升级时显示（且没有严重错误）
  return (!effectiveVideoSource.value || props.isUpgrading) && !hasError.value
})

const transcodingMessage = computed(() => {
  if (props.currentResolution === '360p') {
    return '正在生成高清视频 (720p)...'
  } else {
    return '正在优化视频以提升拖动性能...'
  }
})

const transcodingProgress = computed(() => {
  return props.upgradeProgress || 0
})

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

// 显示分辨率提示（视频源变更时）
function showResolutionHint() {
  showProgressiveHint.value = true
  clearTimeout(progressiveHintTimer)
  progressiveHintTimer = setTimeout(() => {
    showProgressiveHint.value = false
  }, 3000)
}

// 监听 Store 播放状态（单向：Store → Video）
watch(() => projectStore.player.isPlaying, async (playing) => {
  if (!videoRef.value) return

  const isPaused = videoRef.value.paused

  if (playing && isPaused) {
    try {
      await videoRef.value.play()
    } catch (error) {
      console.error('[VideoStage] 播放失败:', error)
      projectStore.player.isPlaying = false
    }
  } else if (!playing && !isPaused) {
    videoRef.value.pause()
  }
})

// 监听 Store 时间变化（外部跳转）
watch(() => projectStore.player.currentTime, (newTime) => {
  if (!videoRef.value) return
  const now = Date.now()

  // 防止循环同步：如果刚刚由video更新了Store，则跳过
  if (now - lastSyncTime.value < 300) return

  // 只有当时间差较大时才seek（避免小抖动）
  const timeDiff = Math.abs(videoRef.value.currentTime - newTime)
  if (timeDiff > 0.5) {
    videoRef.value.currentTime = newTime
    lastSyncTime.value = now
    console.log(`[VideoStage] 外部跳转: ${newTime.toFixed(2)}s (diff: ${timeDiff.toFixed(2)}s)`)
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

// 监听视频源变化（渐进式加载升级时）
watch(() => props.progressiveUrl, async (newUrl, oldUrl) => {
  if (newUrl && newUrl !== oldUrl) {
    console.log('[VideoStage] 视频源变更:', newUrl, 'resolution:', props.currentResolution)

    const video = videoRef.value
    if (!video) return

    // 保存当前播放状态
    const currentTime = video.currentTime || 0
    const wasPlaying = !video.paused
    const currentVolume = video.volume
    const currentRate = video.playbackRate

    // 清除错误状态
    hasError.value = false
    errorMessage.value = ''
    retryCount.value = 0

    // 显示分辨率提示
    showResolutionHint()
    emit('resolution-change', props.currentResolution)

    // 等待下一帧，确保src已更新
    await nextTick()

    try {
      // 强制重新加载视频
      video.load()

      // 等待元数据加载
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('加载超时')), 10000)

        const onLoaded = () => {
          clearTimeout(timeout)
          video.removeEventListener('loadedmetadata', onLoaded)
          video.removeEventListener('error', onError)
          resolve()
        }

        const onError = () => {
          clearTimeout(timeout)
          video.removeEventListener('loadedmetadata', onLoaded)
          video.removeEventListener('error', onError)
          reject(new Error('加载失败'))
        }

        video.addEventListener('loadedmetadata', onLoaded, { once: true })
        video.addEventListener('error', onError, { once: true })
      })

      // 恢复播放状态
      video.currentTime = currentTime
      video.volume = currentVolume
      video.playbackRate = currentRate

      if (wasPlaying) {
        await video.play()
      }

      console.log('[VideoStage] 视频源切换成功，已恢复播放状态')
    } catch (error) {
      console.error('[VideoStage] 视频源切换失败:', error)
    }
  }
})

// 事件处理
function onMetadataLoaded() {
  const video = videoRef.value
  projectStore.meta.duration = video.duration
  video.playbackRate = projectStore.player.playbackRate
  video.volume = projectStore.player.volume
  retryCount.value = 0
  emit('loaded', video.duration)
  if (props.autoPlay) togglePlay()
}

function onTimeUpdate() {
  const video = videoRef.value
  // 使用防抖更新 Store 时间（减少高频更新）
  debouncedTimeUpdate(video.currentTime)
  emit('timeupdate', video.currentTime)
}

function onPlay() {
  showHint('play')
  emit('play')
}

function onPause() {
  showHint('pause')
  emit('pause')
}

function onEnded() {
  projectStore.player.isPlaying = false
  emit('ended')
}

function onProgress() {
  // 可以计算缓冲进度
}

function onError() {
  const video = videoRef.value
  const error = video?.error

  // 如果视频正在转码中，不显示错误（显示转码占位符）
  if (!effectiveVideoSource.value || props.isUpgrading) {
    console.log('[VideoStage] 视频正在转码中，跳过错误提示')
    hasError.value = false
    return
  }

  // 如果视频源为空，也不显示错误（等待视频生成）
  if (!effectiveVideoSource.value) {
    console.log('[VideoStage] 视频源为空，等待生成')
    hasError.value = false
    return
  }

  hasError.value = true

  if (error) {
    switch (error.code) {
      case 1: errorMessage.value = '视频加载被中止'; canRetry.value = true; break
      case 2: errorMessage.value = '网络错误'; canRetry.value = true; break
      case 3: errorMessage.value = '视频解码失败'; canRetry.value = true; break
      case 4: errorMessage.value = '视频加载失败'; canRetry.value = true; break  // 改为可重试，因为可能是转码中
      default: errorMessage.value = '未知错误'; canRetry.value = true
    }
  }

  console.error('[VideoStage] 视频加载错误:', error?.code, errorMessage.value)

  // 自动重试机制
  if (canRetry.value && retryCount.value < maxRetries) {
    retryCount.value++
    console.log(`[VideoStage] 视频加载失败，自动重试 ${retryCount.value}/${maxRetries}`)
    errorMessage.value = `${errorMessage.value}，正在重试 (${retryCount.value}/${maxRetries})...`

    setTimeout(() => {
      hasError.value = false
      videoRef.value?.load()
    }, 2000)
  } else if (retryCount.value >= maxRetries) {
    console.error('[VideoStage] 达到最大重试次数')
    errorMessage.value = `${errorMessage.value}，请手动重试`
  }

  emit('error', new Error(errorMessage.value))
}

function retryLoad() {
  hasError.value = false
  errorMessage.value = ''
  retryCount.value = 0
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
  if (clickTimer) {
    clearTimeout(clickTimer)
    clickTimer = null
    return
  }

  clickTimer = setTimeout(() => {
    clickTimer = null
    togglePlay()
  }, 200)
}

onMounted(() => {
  document.addEventListener('keydown', handleKeyboard)
  document.addEventListener('fullscreenchange', handleFullscreenChange)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyboard)
  document.removeEventListener('fullscreenchange', handleFullscreenChange)
  clearTimeout(stateHintTimer)
  clearTimeout(progressiveHintTimer)
  clearTimeout(timeUpdateTimer)  // 清理时间更新防抖定时器
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

// 转码中状态
.transcoding-overlay {
  background: rgba(0, 0, 0, 0.9);
  color: var(--text-secondary);
  gap: 16px;

  .transcoding-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid rgba(255, 255, 255, 0.1);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  h3 {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-normal);
    margin: 0;
  }

  p {
    font-size: 14px;
    color: var(--text-muted);
    margin: 0;
  }

  .progress-bar {
    width: 300px;
    height: 24px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-md);
    overflow: hidden;
    position: relative;

    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--primary), var(--primary-hover));
      transition: width 0.3s ease;
    }

    span {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 12px;
      font-weight: 600;
      color: white;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
    }
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

// 渐进式加载指示器
.progressive-indicator {
  position: absolute;
  top: 16px;
  right: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  z-index: 25;
  pointer-events: none;

  .resolution-badge {
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 4px;
    letter-spacing: 0.5px;

    &.preview {
      background: rgba(255, 193, 7, 0.9);
      color: #000;
    }

    &.hd {
      background: rgba(76, 175, 80, 0.9);
      color: #fff;
    }

    &.source {
      background: rgba(33, 150, 243, 0.9);
      color: #fff;
    }
  }

  .upgrade-progress {
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 500;
    background: rgba(0, 0, 0, 0.7);
    color: rgba(76, 175, 80, 1);
    border-radius: 4px;
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
