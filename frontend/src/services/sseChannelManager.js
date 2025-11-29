/**
 * SSE 频道管理器（增强版）
 *
 * 职责：
 * - 管理多个 SSE 连接（全局频道、单任务频道、模型频道）
 * - 自动重连机制
 * - 事件分发和处理
 * - 支持全局事件流和任务事件流
 */

import EventEmitter from 'eventemitter3'

class SSEChannelManager extends EventEmitter {
  constructor() {
    super()

    // 频道连接池 { channelId: EventSource }
    this.channels = new Map()

    // 频道配置 { channelId: config }
    this.channelConfigs = new Map()

    // 重连状态 { channelId: { timer, attempts } }
    this.reconnectState = new Map()

    // 基础 URL
    this.baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

    console.log('[SSEChannelManager] 频道管理器已初始化')
  }

  /**
   * 订阅全局事件流（所有任务状态变化）
   * @param {Object} handlers - 事件处理器
   *   {
   *     onInitialState: (state) => void,
   *     onQueueUpdate: (queue) => void,
   *     onJobStatus: (jobId, status, data) => void,
   *     onJobProgress: (jobId, progress, data) => void
   *   }
   * @returns {Function} 取消订阅函数
   */
  subscribeGlobal(handlers = {}) {
    const channelId = 'global'
    const url = `${this.baseURL}/api/events/global`

    return this._subscribe(channelId, url, {
      initial_state: (data) => {
        console.log('[SSE Global] 初始状态:', data)
        handlers.onInitialState?.(data)
      },
      queue_update: (data) => {
        console.log('[SSE Global] 队列更新:', data)
        handlers.onQueueUpdate?.(data.queue)
      },
      job_status: (data) => {
        // 兼容处理：优先使用percent，fallback到progress
        const percent = data.percent ?? data.progress ?? 0
        console.log('[SSE Global] 任务状态:', data)
        // 使用 data.id 而非 data.job_id，兼容全局频道的字段名
        handlers.onJobStatus?.(data.id || data.job_id, data.status, { ...data, percent })
      },
      job_progress: (data) => {
        // 兼容处理：优先使用percent，fallback到progress
        const percent = data.percent ?? data.progress ?? 0
        console.log('[SSE Global] 任务进度:', data.id || data.job_id, percent)
        // 使用 data.id 而非 data.job_id，兼容全局频道的字段名
        handlers.onJobProgress?.(data.id || data.job_id, percent, { ...data, percent })
      },
      connected: (data) => {
        console.log('[SSE Global] 连接成功:', data)
        handlers.onConnected?.(data)
      },
      ping: () => {
        // 心跳，不处理
      }
    })
  }

  /**
   * 订阅单个任务事件流
   * @param {string} jobId - 任务ID
   * @param {Object} handlers - 事件处理器
   *   {
   *     onProgress: (data) => void,
   *     onSignal: (signal, data) => void,
   *     onComplete: (data) => void,
   *     onFailed: (data) => void,
   *     onProxyProgress: (progress) => void
   *   }
   * @returns {Function} 取消订阅函数
   */
  subscribeJob(jobId, handlers = {}) {
    const channelId = `job:${jobId}`
    const url = `${this.baseURL}/api/stream/${jobId}`

    return this._subscribe(channelId, url, {
      initial_state: (data) => {
        console.log(`[SSE Job ${jobId}] 初始状态:`, data)
        handlers.onInitialState?.(data)
      },
      progress: (data) => {
        // 兼容处理：优先使用percent，fallback到progress
        const percent = data.percent ?? data.progress ?? 0
        console.log(`[SSE Job ${jobId}] 进度:`, percent)
        handlers.onProgress?.({ ...data, percent })
      },
      signal: (data) => {
        // 兼容处理：优先使用signal，fallback到code
        const signal = data.signal || data.code
        console.log(`[SSE Job ${jobId}] 信号:`, signal)

        // 分发特定信号事件
        if (signal === 'job_complete') {
          handlers.onComplete?.(data)
        } else if (signal === 'job_failed') {
          handlers.onFailed?.(data)
        } else if (signal === 'job_paused') {
          handlers.onPaused?.(data)
        } else if (signal === 'job_canceled') {
          handlers.onCanceled?.(data)
        } else if (signal === 'job_resumed') {
          // 新增：处理任务恢复信号
          handlers.onResumed?.(data)
        }

        handlers.onSignal?.(signal, data)
      },
      align_progress: (data) => {
        console.log(`[SSE Job ${jobId}] 对齐进度:`, data)
        handlers.onAlignProgress?.(data)
      },
      segment: (data) => {
        console.log(`[SSE Job ${jobId}] 片段:`, data)
        handlers.onSegment?.(data)
      },
      aligned: (data) => {
        console.log(`[SSE Job ${jobId}] 对齐完成:`, data)
        handlers.onAligned?.(data)
      },
      proxy_progress: (data) => {
        console.log(`[SSE Job ${jobId}] Proxy 进度:`, data.progress)
        handlers.onProxyProgress?.(data.progress)
      },
      proxy_complete: (data) => {
        console.log(`[SSE Job ${jobId}] Proxy 完成`)
        handlers.onProxyComplete?.()
      },
      connected: (data) => {
        console.log(`[SSE Job ${jobId}] 连接成功`)
        handlers.onConnected?.(data)
      },
      ping: () => {
        // 心跳，不处理
      }
    })
  }

  /**
   * 订阅模型下载进度
   * @param {Object} handlers - 事件处理器
   *   {
   *     onProgress: (modelId, progress, data) => void,
   *     onComplete: (modelId, data) => void
   *   }
   * @returns {Function} 取消订阅函数
   */
  subscribeModels(handlers = {}) {
    const channelId = 'models'
    const url = `${this.baseURL}/api/models/stream`

    return this._subscribe(channelId, url, {
      model_progress: (data) => {
        console.log('[SSE Models] 下载进度:', data.model_id, data.progress)
        handlers.onProgress?.(data.model_id, data.progress, data)
      },
      model_complete: (data) => {
        console.log('[SSE Models] 下载完成:', data.model_id)
        handlers.onComplete?.(data.model_id, data)
      },
      connected: (data) => {
        console.log('[SSE Models] 连接成功')
        handlers.onConnected?.(data)
      },
      ping: () => {
        // 心跳，不处理
      }
    })
  }

  /**
   * 通用订阅方法（内部使用）
   * @private
   */
  _subscribe(channelId, url, eventHandlers) {
    // 检查是否已存在连接
    const existingConnection = this.channels.get(channelId)

    // 关键修复：即使连接存在，也需要重建以确保新的事件处理器被绑定
    // 原因：EventSource 的事件监听器是在创建时绑定的，复用连接会导致新组件的回调无法被调用
    if (existingConnection) {
      console.log(`[SSEChannelManager] 频道 ${channelId} 存在旧连接，关闭后重建以更新事件处理器`)
      // 关闭旧连接
      existingConnection.close()
      this.channels.delete(channelId)
      // 清除重连定时器
      const reconnectInfo = this.reconnectState.get(channelId)
      if (reconnectInfo?.timer) {
        clearTimeout(reconnectInfo.timer)
      }
      this.reconnectState.delete(channelId)
    }

    // 保存配置
    this.channelConfigs.set(channelId, { url, eventHandlers })

    // 创建连接
    this._createConnection(channelId, url, eventHandlers)

    // 返回取消订阅函数
    return () => this._requestUnsubscribe(channelId)
  }

  /**
   * 请求取消订阅（延迟执行，避免页面切换时误关闭）
   * @private
   */
  _requestUnsubscribe(channelId) {
    // 不立即关闭，给一个短暂的缓冲期
    // 如果用户快速切换回来，可以复用连接
    console.log(`[SSEChannelManager] 延迟取消订阅: ${channelId}`)
  }

  /**
   * 创建 EventSource 连接
   * @private
   */
  _createConnection(channelId, url, eventHandlers) {
    try {
      console.log(`[SSEChannelManager] 创建连接: ${channelId} -> ${url}`)

      const eventSource = new EventSource(url, {
        withCredentials: false
      })

      this.channels.set(channelId, eventSource)

      // 监听所有事件类型
      for (const [eventType, handler] of Object.entries(eventHandlers)) {
        eventSource.addEventListener(eventType, (event) => {
          try {
            const data = JSON.parse(event.data)
            handler(data)
          } catch (error) {
            console.error(`[SSE ${channelId}] 解析事件失败:`, eventType, error, event.data)
          }
        })
      }

      // 错误处理
      eventSource.onerror = (error) => {
        console.error(`[SSE ${channelId}] 连接错误:`, error)

        // 检查连接状态
        if (eventSource.readyState === EventSource.CLOSED) {
          console.warn(`[SSE ${channelId}] 连接已关闭，尝试重连`)
          this._scheduleReconnect(channelId)
        }
      }

      // 清除重连状态
      this.reconnectState.delete(channelId)

    } catch (error) {
      console.error(`[SSE ${channelId}] 创建连接失败:`, error)
      this._scheduleReconnect(channelId)
    }
  }

  /**
   * 计划重连
   * @private
   */
  _scheduleReconnect(channelId) {
    const config = this.channelConfigs.get(channelId)
    if (!config) {
      console.warn(`[SSE ${channelId}] 无配置信息，无法重连`)
      return
    }

    // 获取重连状态
    let reconnectInfo = this.reconnectState.get(channelId)
    if (!reconnectInfo) {
      reconnectInfo = { attempts: 0, timer: null }
      this.reconnectState.set(channelId, reconnectInfo)
    }

    // 清除旧的定时器
    if (reconnectInfo.timer) {
      clearTimeout(reconnectInfo.timer)
    }

    reconnectInfo.attempts++

    // 最大重连次数限制（防止任务不存在时无限重连）
    const MAX_RECONNECT_ATTEMPTS = 5
    if (reconnectInfo.attempts > MAX_RECONNECT_ATTEMPTS) {
      console.warn(`[SSE ${channelId}] 已达到最大重连次数 (${MAX_RECONNECT_ATTEMPTS})，停止重连`)
      // 清理资源
      this.channels.delete(channelId)
      this.channelConfigs.delete(channelId)
      this.reconnectState.delete(channelId)
      return
    }

    // 计算重连延迟（指数退避，最大 30 秒）
    const delay = Math.min(1000 * Math.pow(2, reconnectInfo.attempts - 1), 30000)

    console.log(`[SSE ${channelId}] ${delay}ms 后尝试第 ${reconnectInfo.attempts}/${MAX_RECONNECT_ATTEMPTS} 次重连`)

    reconnectInfo.timer = setTimeout(() => {
      // 先关闭旧连接
      const oldEventSource = this.channels.get(channelId)
      if (oldEventSource) {
        oldEventSource.close()
        this.channels.delete(channelId)
      }

      // 创建新连接
      this._createConnection(channelId, config.url, config.eventHandlers)
    }, delay)
  }

  /**
   * 取消订阅指定频道
   * @param {string} channelId - 频道ID
   */
  unsubscribe(channelId) {
    console.log(`[SSEChannelManager] 取消订阅: ${channelId}`)

    // 关闭 EventSource
    const eventSource = this.channels.get(channelId)
    if (eventSource) {
      eventSource.close()
      this.channels.delete(channelId)
    }

    // 清除重连定时器
    const reconnectInfo = this.reconnectState.get(channelId)
    if (reconnectInfo?.timer) {
      clearTimeout(reconnectInfo.timer)
    }
    this.reconnectState.delete(channelId)

    // 删除配置
    this.channelConfigs.delete(channelId)
  }

  /**
   * 取消所有订阅
   */
  unsubscribeAll() {
    console.log('[SSEChannelManager] 取消所有订阅')

    for (const channelId of this.channels.keys()) {
      this.unsubscribe(channelId)
    }
  }

  /**
   * 手动重连指定频道
   * @param {string} channelId - 频道ID
   */
  reconnect(channelId) {
    console.log(`[SSEChannelManager] 手动重连: ${channelId}`)

    this.unsubscribe(channelId)

    const config = this.channelConfigs.get(channelId)
    if (config) {
      this._createConnection(channelId, config.url, config.eventHandlers)
    } else {
      console.error(`[SSE ${channelId}] 无配置信息，无法重连`)
    }
  }

  /**
   * 获取所有活跃频道
   * @returns {string[]}
   */
  getActiveChannels() {
    return Array.from(this.channels.keys())
  }

  /**
   * 检查频道是否已连接
   * @param {string} channelId - 频道ID
   * @returns {boolean}
   */
  isChannelActive(channelId) {
    const eventSource = this.channels.get(channelId)
    return eventSource && eventSource.readyState === EventSource.OPEN
  }

  /**
   * 销毁管理器
   */
  destroy() {
    console.log('[SSEChannelManager] 销毁管理器')
    this.unsubscribeAll()
    this.removeAllListeners()
  }
}

// 导出单例
export const sseChannelManager = new SSEChannelManager()
export default sseChannelManager
