/**
 * 心跳服务
 * 用于维持前端与后端的连接，实现浏览器标签页重用
 */

class HeartbeatService {
  constructor() {
    this.clientId = this._getOrCreateClientId()
    this.intervalId = null
    this.HEARTBEAT_INTERVAL = 5000 // 5秒发送一次心跳
    this.isRunning = false
  }

  /**
   * 获取或创建客户端ID
   * @returns {string} 客户端ID
   */
  _getOrCreateClientId() {
    let clientId = localStorage.getItem('client_id')
    if (!clientId) {
      clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      localStorage.setItem('client_id', clientId)
    }
    return clientId
  }

  /**
   * 启动心跳服务
   */
  async start() {
    if (this.isRunning) {
      console.warn('[HeartbeatService] 心跳服务已在运行')
      return
    }

    console.log('[HeartbeatService] 启动心跳服务', { clientId: this.clientId })

    // 注册客户端
    await this._register()

    // 启动心跳定时器
    this.intervalId = setInterval(() => {
      this._sendHeartbeat()
    }, this.HEARTBEAT_INTERVAL)

    // 立即发送一次心跳
    this._sendHeartbeat()

    // 页面关闭时注销
    window.addEventListener('beforeunload', () => {
      this._unregister()
    })

    // 页面可见性变化时处理
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        this._sendHeartbeat() // 立即发送心跳
      }
    })

    this.isRunning = true
  }

  /**
   * 停止心跳服务
   */
  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
    this.isRunning = false
    console.log('[HeartbeatService] 心跳服务已停止')
  }

  /**
   * 注册客户端
   */
  async _register() {
    try {
      const response = await fetch('/api/system/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          client_id: this.clientId,
          user_agent: navigator.userAgent
        })
      })

      if (response.ok) {
        console.log('[HeartbeatService] 客户端注册成功')
      } else {
        console.warn('[HeartbeatService] 客户端注册失败', response.status)
      }
    } catch (e) {
      console.warn('[HeartbeatService] 客户端注册失败:', e)
    }
  }

  /**
   * 发送心跳
   */
  async _sendHeartbeat() {
    try {
      const response = await fetch('/api/system/heartbeat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: this.clientId })
      })

      if (response.ok) {
        const data = await response.json()
        // console.log('[HeartbeatService] 心跳成功', data)
      }
    } catch (e) {
      // 后端可能已关闭，忽略错误
      // console.warn('[HeartbeatService] 心跳失败:', e)
    }
  }

  /**
   * 注销客户端
   */
  _unregister() {
    // 使用 sendBeacon 确保页面关闭时请求能发出
    const data = JSON.stringify({ client_id: this.clientId })
    const blob = new Blob([data], { type: 'application/json' })

    if (navigator.sendBeacon) {
      navigator.sendBeacon('/api/system/unregister', blob)
      console.log('[HeartbeatService] 客户端已注销')
    }
  }

  /**
   * 获取客户端ID
   */
  getClientId() {
    return this.clientId
  }
}

// 导出单例
export const heartbeatService = new HeartbeatService()
