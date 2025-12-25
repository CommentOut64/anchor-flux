/**
 * 心跳服务
 * 用于维持前端与后端的连接
 * 
 * 策略：当检测到后端重启（连接断开后恢复），自动刷新页面
 * 这样可以确保只有最新打开的页面是有效的，避免多个前端互相干扰
 */

class HeartbeatService {
  constructor() {
    this.clientId = this._getOrCreateClientId()
    this.intervalId = null
    this.HEARTBEAT_INTERVAL = 3000 // 3秒发送一次心跳
    this.RETRY_INTERVAL = 1000 // 心跳失败后1秒重试
    this.isRunning = false
    this.consecutiveFailures = 0
    this.MAX_FAILURES_BEFORE_FAST_RETRY = 3 // 连续失败3次后开始快速重试
    this.MAX_FAILURES_BEFORE_STALE = 10 // 连续失败10次后认为页面过期
    this.isStale = false // 页面是否已过期（后端已关闭太久）
  }

  /**
   * 获取或创建客户端ID
   * 每次页面加载都生成新的 clientId
   * @returns {string} 客户端ID
   */
  _getOrCreateClientId() {
    // 每次页面加载都生成新的 clientId
    // 这样可以确保每个页面实例都是唯一的
    const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    localStorage.setItem('client_id', clientId)
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
    this._scheduleNextHeartbeat()

    // 立即发送一次心跳
    this._sendHeartbeat()

    // 页面关闭时注销
    window.addEventListener('beforeunload', () => {
      this._unregister()
    })

    // 页面可见性变化时处理
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && !this.isStale) {
        // 页面变为可见时，立即发送心跳
        this._sendHeartbeat()
      }
    })

    this.isRunning = true
  }

  /**
   * 调度下一次心跳
   */
  _scheduleNextHeartbeat() {
    if (this.intervalId) {
      clearTimeout(this.intervalId)
    }
    
    // 如果页面已过期，停止心跳
    if (this.isStale) {
      return
    }
    
    // 根据失败次数决定下一次心跳的间隔
    const interval = this.consecutiveFailures >= this.MAX_FAILURES_BEFORE_FAST_RETRY 
      ? this.RETRY_INTERVAL  // 失败多次后快速重试
      : this.HEARTBEAT_INTERVAL
    
    this.intervalId = setTimeout(() => {
      this._sendHeartbeat()
      this._scheduleNextHeartbeat()
    }, interval)
  }

  /**
   * 停止心跳服务
   */
  stop() {
    if (this.intervalId) {
      clearTimeout(this.intervalId)
      this.intervalId = null
    }
    this.isRunning = false
    this.consecutiveFailures = 0
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
        this.consecutiveFailures = 0
        return true
      } else {
        console.warn('[HeartbeatService] 客户端注册失败', response.status)
        return false
      }
    } catch (e) {
      console.warn('[HeartbeatService] 客户端注册失败:', e.message)
      return false
    }
  }

  /**
   * 发送心跳
   */
  async _sendHeartbeat() {
    if (this.isStale) {
      return // 页面已过期，不再发送心跳
    }
    
    try {
      const response = await fetch('/api/system/heartbeat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: this.clientId })
      })

      if (response.ok) {
        // 心跳成功
        if (this.consecutiveFailures >= this.MAX_FAILURES_BEFORE_FAST_RETRY) {
          // 之前失败多次，现在恢复了 - 说明后端重启过
          console.log('[HeartbeatService] 后端连接已恢复，检测到后端已重启，刷新页面...')
          this._showReloadNotification()
          return
        }
        this.consecutiveFailures = 0
      } else {
        this.consecutiveFailures++
      }
    } catch (e) {
      this.consecutiveFailures++
      
      if (this.consecutiveFailures === 1) {
        console.warn('[HeartbeatService] 心跳失败，后端可能正在重启...')
      }
      
      // 如果失败次数过多，标记页面为过期
      if (this.consecutiveFailures >= this.MAX_FAILURES_BEFORE_STALE) {
        this._markAsStale()
      }
    }
  }
  
  /**
   * 显示重新加载通知
   */
  _showReloadNotification() {
    // 停止心跳
    this.stop()
    
    // 创建通知遮罩
    const overlay = document.createElement('div')
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 99999;
    `
    
    const dialog = document.createElement('div')
    dialog.style.cssText = `
      background: white;
      padding: 30px 40px;
      border-radius: 12px;
      text-align: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      max-width: 400px;
    `
    dialog.innerHTML = `
      <h3 style="margin: 0 0 15px 0; color: #333;">后端服务已重启</h3>
      <p style="margin: 0 0 20px 0; color: #666;">检测到后端服务已重启，页面将自动刷新以同步状态。</p>
      <button id="reload-btn" style="
        background: #4CAF50;
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 16px;
      ">立即刷新</button>
      <p style="margin: 15px 0 0 0; color: #999; font-size: 12px;">3秒后自动刷新...</p>
    `
    
    overlay.appendChild(dialog)
    document.body.appendChild(overlay)
    
    // 点击按钮立即刷新
    document.getElementById('reload-btn').onclick = () => {
      window.location.reload()
    }
    
    // 3秒后自动刷新
    setTimeout(() => {
      window.location.reload()
    }, 3000)
  }
  
  /**
   * 标记页面为过期
   */
  _markAsStale() {
    if (this.isStale) return
    
    this.isStale = true
    this.stop()
    
    console.warn('[HeartbeatService] 页面已过期，后端长时间无响应')
    
    // 显示过期提示
    const overlay = document.createElement('div')
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 99999;
    `
    
    const dialog = document.createElement('div')
    dialog.style.cssText = `
      background: white;
      padding: 30px 40px;
      border-radius: 12px;
      text-align: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
      max-width: 400px;
    `
    dialog.innerHTML = `
      <h3 style="margin: 0 0 15px 0; color: #e74c3c;">连接已断开</h3>
      <p style="margin: 0 0 20px 0; color: #666;">无法连接到后端服务，请检查服务是否正常运行。</p>
      <button id="retry-btn" style="
        background: #3498db;
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 16px;
      ">重新连接</button>
    `
    
    overlay.appendChild(dialog)
    document.body.appendChild(overlay)
    
    document.getElementById('retry-btn').onclick = () => {
      window.location.reload()
    }
  }

  /**
   * 注销客户端
   */
  _unregister() {
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
