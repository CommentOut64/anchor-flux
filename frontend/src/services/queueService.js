/**
 * 队列管理服务 - V3.0全局SSE
 *
 * 功能:
 * - 连接全局SSE流
 * - 监听队列变化、任务状态变化
 * - 提供事件订阅机制
 */

export class QueueService {
  constructor() {
    this.eventSource = null
    this.listeners = new Map()
    this.reconnectTimer = null
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
  }

  /**
   * 连接全局SSE
   */
  connect() {
    if (this.eventSource) {
      this.disconnect()
    }

    try {
      this.eventSource = new EventSource('/api/events/global')

      // 初始状态
      this.eventSource.addEventListener('initial_state', (e) => {
        try {
          const data = JSON.parse(e.data)
          console.log('[QueueService] 收到初始状态:', data)
          this._emit('initial_state', data)
          this.reconnectAttempts = 0 // 重置重连计数
        } catch (err) {
          console.error('[QueueService] 解析initial_state失败:', err)
        }
      })

      // 队列更新事件
      this.eventSource.addEventListener('queue_update', (e) => {
        try {
          const data = JSON.parse(e.data)
          console.log('[QueueService] 队列更新:', data)
          this._emit('queue_update', data)
        } catch (err) {
          console.error('[QueueService] 解析queue_update失败:', err)
        }
      })

      // 任务状态事件
      this.eventSource.addEventListener('job_status', (e) => {
        try {
          const data = JSON.parse(e.data)
          console.log('[QueueService] 任务状态:', data)
          this._emit('job_status', data)
        } catch (err) {
          console.error('[QueueService] 解析job_status失败:', err)
        }
      })

      // 任务进度事件
      this.eventSource.addEventListener('job_progress', (e) => {
        try {
          const data = JSON.parse(e.data)
          console.log('[QueueService] 任务进度:', data)
          this._emit('job_progress', data)
        } catch (err) {
          console.error('[QueueService] 解析job_progress失败:', err)
        }
      })

      // 连接打开
      this.eventSource.onopen = () => {
        console.log('[QueueService] 全局SSE已连接')
        this._emit('connected')
      }

      // 连接错误
      this.eventSource.onerror = (err) => {
        console.error('[QueueService] SSE连接错误:', err)
        this._emit('error', err)

        // 自动重连
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
          this.reconnectAttempts++
          const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000)
          console.log(`[QueueService] ${delay}ms后尝试第${this.reconnectAttempts}次重连...`)

          this.reconnectTimer = setTimeout(() => {
            console.log('[QueueService] 尝试重连...')
            this.connect()
          }, delay)
        } else {
          console.warn('[QueueService] 达到最大重连次数，停止重连')
          this._emit('max_reconnect_reached')
        }
      }

    } catch (err) {
      console.error('[QueueService] 创建EventSource失败:', err)
      this._emit('error', err)
    }
  }

  /**
   * 断开连接
   */
  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }

    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
      console.log('[QueueService] SSE连接已断开')
      this._emit('disconnected')
    }
  }

  /**
   * 订阅事件
   *
   * @param {string} event - 事件名称
   * @param {Function} callback - 回调函数
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, [])
    }
    this.listeners.get(event).push(callback)
  }

  /**
   * 取消订阅
   *
   * @param {string} event - 事件名称
   * @param {Function} callback - 回调函数
   */
  off(event, callback) {
    if (!this.listeners.has(event)) {
      return
    }

    const callbacks = this.listeners.get(event)
    const index = callbacks.indexOf(callback)
    if (index > -1) {
      callbacks.splice(index, 1)
    }
  }

  /**
   * 触发事件
   *
   * @param {string} event - 事件名称
   * @param {any} data - 事件数据
   */
  _emit(event, data) {
    const callbacks = this.listeners.get(event) || []
    callbacks.forEach(cb => {
      try {
        cb(data)
      } catch (err) {
        console.error(`[QueueService] 事件回调执行失败 (${event}):`, err)
      }
    })
  }

  /**
   * 检查连接状态
   *
   * @returns {boolean} 是否已连接
   */
  isConnected() {
    return this.eventSource && this.eventSource.readyState === EventSource.OPEN
  }

  /**
   * 重置重连计数
   */
  resetReconnectAttempts() {
    this.reconnectAttempts = 0
  }
}

/**
 * 创建单例
 */
let instance = null

export function getQueueService() {
  if (!instance) {
    instance = new QueueService()
  }
  return instance
}
