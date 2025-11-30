/**
 * 视频状态管理器 - 管理渐进式加载的视频状态
 *
 * 提供：
 * - 当前可用的视频 URL
 * - 视频生成进度
 * - 质量切换逻辑
 *
 * 注意：SSE事件订阅由父组件统一管理，通过调用handleProxyProgress/handleProxyComplete更新状态
 */
import { ref, computed, watch } from 'vue'
import { mediaApi } from '@/services/api'

/**
 * 视频加载阶段
 */
export const VIDEO_STAGES = {
  NONE: 'none',           // 无可用视频
  LOADING: 'loading',     // 加载中
  PREVIEW: 'preview',     // 360p 预览
  HIGH_QUALITY: 'high',   // 720p 高质量
  SOURCE: 'source'        // 原始视频
}

export function useVideoStatus(jobId) {
  // 状态
  const currentStage = ref(VIDEO_STAGES.LOADING)
  const currentUrl = ref(null)
  const currentResolution = ref(null)
  const isUpgrading = ref(false)        // 是否正在升级到更高质量
  const upgradeProgress = ref(0)        // 升级进度
  const needsTranscode = ref(false)     // 是否需要转码
  const transcodeReason = ref('')       // 转码原因

  // 各阶段状态
  const preview360p = ref({
    exists: false,
    url: null,
    progress: 0
  })

  const proxy720p = ref({
    exists: false,
    url: null,
    progress: 0
  })

  const source = ref({
    exists: false,
    compatible: false,
    url: null
  })

  // 计算属性
  const bestAvailableUrl = computed(() => {
    // 优先使用 720p
    if (proxy720p.value.exists && proxy720p.value.url) {
      return proxy720p.value.url
    }
    // 其次使用 360p 预览
    if (preview360p.value.exists && preview360p.value.url) {
      return preview360p.value.url
    }
    // 最后使用兼容的原始视频
    if (source.value.exists && source.value.compatible && source.value.url) {
      return source.value.url
    }
    return null
  })

  const hasHighQualityAvailable = computed(() => {
    return proxy720p.value.exists || (source.value.exists && source.value.compatible)
  })

  const isGenerating = computed(() => {
    return (preview360p.value.progress > 0 && preview360p.value.progress < 100) ||
           (proxy720p.value.progress > 0 && proxy720p.value.progress < 100)
  })

  // 方法
  /**
   * 获取渐进式加载状态
   */
  async function fetchProgressiveStatus() {
    try {
      const response = await mediaApi.getProgressiveStatus(jobId.value || jobId)
      const data = response.data || response

      // 更新状态
      needsTranscode.value = data.needs_transcode
      transcodeReason.value = data.transcode_reason || ''

      // 更新 360p 预览状态
      if (data.preview_360p) {
        preview360p.value.exists = data.preview_360p.exists
        preview360p.value.url = data.preview_360p.url
      }

      // 更新 720p 代理状态
      if (data.proxy_720p) {
        proxy720p.value.exists = data.proxy_720p.exists
        proxy720p.value.url = data.proxy_720p.url
        if (data.proxy_720p.status) {
          proxy720p.value.progress = data.proxy_720p.status.progress || 0
        }
      }

      // 更新原始视频状态
      if (data.source) {
        source.value.exists = data.source.exists
        source.value.compatible = data.source.compatible
        source.value.url = data.source.url
      }

      // 更新当前使用的 URL
      if (data.recommended_url) {
        currentUrl.value = data.recommended_url
        currentResolution.value = data.current_resolution
      }

      // 更新阶段
      updateStage()

      return data
    } catch (error) {
      console.error('[useVideoStatus] 获取状态失败:', error)
      return null
    }
  }

  /**
   * 更新当前阶段
   */
  function updateStage() {
    if (proxy720p.value.exists) {
      currentStage.value = VIDEO_STAGES.HIGH_QUALITY
    } else if (preview360p.value.exists) {
      currentStage.value = VIDEO_STAGES.PREVIEW
      isUpgrading.value = proxy720p.value.progress > 0
      upgradeProgress.value = proxy720p.value.progress
    } else if (source.value.exists && source.value.compatible) {
      currentStage.value = VIDEO_STAGES.SOURCE
    } else if (isGenerating.value) {
      currentStage.value = VIDEO_STAGES.LOADING
    } else {
      currentStage.value = VIDEO_STAGES.NONE
    }
  }

  /**
   * 订阅 SSE 事件（已废弃，改由父组件统一管理SSE）
   * @deprecated 请在父组件的SSE订阅中调用handleProxyProgress/handleProxyComplete方法
   */
  function subscribeEvents() {
    console.warn('[useVideoStatus] subscribeEvents已废弃，请在父组件中统一处理SSE事件')
  }

  /**
   * 处理Proxy进度更新（由父组件调用）
   */
  function handleProxyProgress(data) {
    proxy720p.value.progress = data.progress || 0
    isUpgrading.value = true
    upgradeProgress.value = data.progress || 0
    console.log('[useVideoStatus] Proxy进度:', data.progress)
  }

  /**
   * 处理Proxy完成事件（由父组件调用）
   */
  function handleProxyComplete(data) {
    proxy720p.value.exists = true
    proxy720p.value.url = data.video_url || currentUrl.value
    proxy720p.value.progress = 100
    isUpgrading.value = false

    // 切换到高质量视频
    currentUrl.value = proxy720p.value.url
    currentResolution.value = '720p'
    currentStage.value = VIDEO_STAGES.HIGH_QUALITY

    console.log('[useVideoStatus] 720p 高清视频已就绪:', proxy720p.value.url)
  }

  /**
   * 手动触发预览生成
   */
  async function triggerPreviewGeneration() {
    try {
      const id = jobId.value || jobId
      await mediaApi.generatePreview(id)
      console.log('[useVideoStatus] 预览生成已触发')
    } catch (error) {
      console.error('[useVideoStatus] 触发预览生成失败:', error)
    }
  }

  // 监听 jobId 变化，自动获取状态
  watch(
    () => jobId.value || jobId,
    (newId) => {
      if (newId) {
        fetchProgressiveStatus()
      }
    },
    { immediate: true }
  )

  return {
    // 状态
    currentStage,
    currentUrl,
    currentResolution,
    isUpgrading,
    upgradeProgress,
    needsTranscode,
    transcodeReason,
    preview360p,
    proxy720p,
    source,

    // 计算属性
    bestAvailableUrl,
    hasHighQualityAvailable,
    isGenerating,

    // 方法
    fetchProgressiveStatus,
    triggerPreviewGeneration,
    handleProxyProgress,      // 新增：处理Proxy进度（由父组件调用）
    handleProxyComplete,      // 新增：处理Proxy完成（由父组件调用）

    // 常量
    VIDEO_STAGES
  }
}
