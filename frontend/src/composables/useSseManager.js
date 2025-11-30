/**
 * SSE 管理器 - 统一管理视频渐进式加载的 SSE 事件订阅
 *
 * 用于监听视频生成进度和完成事件
 */
import { ref, onUnmounted } from 'vue'

export function useSseManager() {
  const eventSource = ref(null)
  const listeners = new Map()  // event_type -> [callbacks]
  const currentChannel = ref(null)
  const isConnected = ref(false)
  const reconnectAttempts = ref(0)
  const MAX_RECONNECT_ATTEMPTS = 5
  const RECONNECT_DELAY = 5000

  /**
   * 订阅 SSE 事件
   * @param {string} channel - 频道 ID（如 'job:xxx'）
   * @param {string} eventType - 事件类型
   * @param {function} callback - 回调函数
   */
  function subscribe(channel, eventType, callback) {
    const key = `${channel}:${eventType}`

    if (!listeners.has(key)) {
      listeners.set(key, [])
    }

    listeners.get(key).push(callback)

    // 确保 EventSource 已连接
    ensureConnection(channel)

    console.log(`[SSE] 订阅事件: ${key}`)
  }

  /**
   * 取消订阅
   */
  function unsubscribe(channel, eventType, callback) {
    const key = `${channel}:${eventType}`

    if (listeners.has(key)) {
      const callbacks = listeners.get(key)
      const index = callbacks.indexOf(callback)
      if (index > -1) {
        callbacks.splice(index, 1)
      }

      if (callbacks.length === 0) {
        listeners.delete(key)
      }
    }
  }

  /**
   * 确保 SSE 连接已建立
   */
  function ensureConnection(channel) {
    // 如果已连接到同一频道，无需重连
    if (eventSource.value && currentChannel.value === channel && isConnected.value) {
      return
    }

    // 关闭旧连接
    if (eventSource.value) {
      eventSource.value.close()
      eventSource.value = null
    }

    currentChannel.value = channel

    // 根据频道类型选择正确的端点
    let url
    if (channel === 'global') {
      url = '/api/events/global'
    } else if (channel.startsWith('job:')) {
      const jobId = channel.replace('job:', '')
      url = `/api/stream/${jobId}`
    } else {
      console.error(`[SSE] 不支持的频道类型: ${channel}`)
      return
    }

    eventSource.value = new EventSource(url)

    eventSource.value.onopen = () => {
      console.log('[SSE] 连接已建立')
      isConnected.value = true
      reconnectAttempts.value = 0
    }

    eventSource.value.onerror = (error) => {
      console.error('[SSE] 连接错误:', error)
      isConnected.value = false

      // 重连逻辑
      if (reconnectAttempts.value < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts.value++
        console.log(`[SSE] 尝试重连 (${reconnectAttempts.value}/${MAX_RECONNECT_ATTEMPTS})...`)
        setTimeout(() => {
          if (eventSource.value) {
            eventSource.value.close()
            eventSource.value = null
            ensureConnection(channel)
          }
        }, RECONNECT_DELAY)
      } else {
        console.error('[SSE] 达到最大重连次数，停止重连')
      }
    }

    // 监听所有消息
    eventSource.value.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const eventType = data.type || 'message'
        const key = `${channel}:${eventType}`

        // 触发回调
        if (listeners.has(key)) {
          listeners.get(key).forEach(callback => {
            callback(data.data || data)
          })
        }

        // 同时触发通配符监听器
        const wildcardKey = `${channel}:*`
        if (listeners.has(wildcardKey)) {
          listeners.get(wildcardKey).forEach(callback => {
            callback(data)
          })
        }
      } catch (error) {
        console.error('[SSE] 消息解析失败:', error)
      }
    }
  }

  /**
   * 订阅视频生成进度事件
   * @param {string} jobId - 任务 ID
   * @param {object} handlers - 事件处理器
   */
  function subscribeVideoProgress(jobId, handlers = {}) {
    const channel = `job:${jobId}`

    // 360p 预览进度
    if (handlers.onPreview360pProgress) {
      subscribe(channel, 'preview_360p_progress', handlers.onPreview360pProgress)
    }

    // 360p 预览完成
    if (handlers.onPreview360pComplete) {
      subscribe(channel, 'preview_360p_complete', handlers.onPreview360pComplete)
    }

    // 720p 高清进度
    if (handlers.onProxy720pProgress) {
      subscribe(channel, 'proxy_720p_progress', handlers.onProxy720pProgress)
    }

    // 720p 高清完成
    if (handlers.onProxy720pComplete) {
      subscribe(channel, 'proxy_720p_complete', handlers.onProxy720pComplete)
    }

    // Proxy 视频进度（兼容旧接口）
    if (handlers.onProxyProgress) {
      subscribe(channel, 'proxy_progress', handlers.onProxyProgress)
    }

    // Proxy 视频完成（兼容旧接口）
    if (handlers.onProxyComplete) {
      subscribe(channel, 'proxy_complete', handlers.onProxyComplete)
    }

    console.log(`[SSE] 已订阅视频进度事件: ${jobId}`)
  }

  /**
   * 关闭连接
   */
  function close() {
    if (eventSource.value) {
      eventSource.value.close()
      eventSource.value = null
      isConnected.value = false
      currentChannel.value = null
      listeners.clear()
      console.log('[SSE] 连接已关闭')
    }
  }

  // 组件卸载时自动关闭
  onUnmounted(() => {
    close()
  })

  return {
    subscribe,
    unsubscribe,
    subscribeVideoProgress,
    close,
    isConnected,
    currentChannel
  }
}
