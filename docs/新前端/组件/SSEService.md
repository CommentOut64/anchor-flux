# SSEService - Server-Sent Events 服务

## 概述

SSEService 是基于 Server-Sent Events (SSE) 的实时通信服务，用于接收后端任务进度更新、状态变化等实时消息。它提供自动重连、心跳检测、消息队���等企业级功能，是 TaskMonitor 和转录任务实时更新的核心支撑。

## 功能特性

1. **稳定的连接管理**
   - 自动重连机制
   - 指数退避策略
   - 心跳检测
   - 连接状态监控

2. **消息处理**
   - 事件类型分发
   - 消息队列缓冲
   - 顺序保证
   - 去重处理

3. **错误处理**
   - 连接超时检测
   - 网络异常恢复
   - 错误回调通知
   - 降级到轮询

4. **性能优化**
   - 批量消息处理
   - 内存占用控制
   - 连接池管理
   - 资源自动清理

## 技术依赖

```json
{
  "eventemitter3": "^5.0.0"
}
```

## API 接口

### 初始化配置

```typescript
interface SSEConfig {
  // 连接配置
  url: string                    // SSE 端点 URL
  withCredentials?: boolean      // 携带凭证，默认true

  // 重连配置
  reconnect?: boolean            // 自动重连，默认true
  reconnectInterval?: number     // 初始重连间隔（毫秒），默认1000
  reconnectDecay?: number        // 退避系数，默认1.5
  maxReconnectInterval?: number  // 最大重连间隔（毫秒），默认30000
  maxReconnectAttempts?: number  // 最大重连次数，默认Infinity

  // 心跳配置
  heartbeatInterval?: number     // 心跳间隔（毫秒），默认30000
  heartbeatTimeout?: number      // 心跳超时（毫秒），默认10000

  // 消息配置
  messageQueueSize?: number      // 消息队列大小，默认100
  enableDedup?: boolean          // 启用去重，默认true

  // 降级配置
  fallbackToPolling?: boolean    // 降级到轮询，默认true
  pollingInterval?: number       // 轮询间隔（毫秒）��默认5000
}
```

### 核心方法

```typescript
class SSEService extends EventEmitter {
  /**
   * 连接到 SSE 端点
   * @param url SSE URL
   * @param config 配置选项
   */
  connect(url?: string, config?: Partial<SSEConfig>): void

  /**
   * 断开连接
   */
  disconnect(): void

  /**
   * 订阅特定事件类型
   * @param event 事件类型
   * @param handler 处理函数
   */
  subscribe(event: string, handler: (data: any) => void): () => void

  /**
   * 取消订阅
   * @param event 事件类型
   * @param handler 处理函数
   */
  unsubscribe(event: string, handler?: (data: any) => void): void

  /**
   * 获取连接状态
   */
  getStatus(): ConnectionStatus

  /**
   * 获取统计信息
   */
  getStats(): SSEStats

  /**
   * 手动触发重连
   */
  reconnect(): void

  /**
   * 清除消息队列
   */
  clearQueue(): void
}

enum ConnectionStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  RECONNECTING = 'reconnecting',
  FAILED = 'failed'
}

interface SSEStats {
  status: ConnectionStatus
  connected: boolean
  reconnectCount: number
  messagesReceived: number
  lastMessageTime: number | null
  uptime: number
}
```

## 核心实现

```javascript
// services/sseService.js
import EventEmitter from 'eventemitter3'

class SSEService extends EventEmitter {
  constructor(config = {}) {
    super()

    this.config = {
      withCredentials: true,
      reconnect: true,
      reconnectInterval: 1000,
      reconnectDecay: 1.5,
      maxReconnectInterval: 30000,
      maxReconnectAttempts: Infinity,
      heartbeatInterval: 30000,
      heartbeatTimeout: 10000,
      messageQueueSize: 100,
      enableDedup: true,
      fallbackToPolling: true,
      pollingInterval: 5000,
      ...config
    }

    this.eventSource = null
    this.status = 'disconnected'
    this.reconnectCount = 0
    this.currentReconnectInterval = this.config.reconnectInterval
    this.messageQueue = []
    this.lastMessageId = null
    this.heartbeatTimer = null
    this.pollingTimer = null

    // 统计信息
    this.stats = {
      messagesReceived: 0,
      lastMessageTime: null,
      connectedAt: null
    }

    // 去重集合
    this.processedMessageIds = new Set()
  }

  // 连接
  connect(url, config = {}) {
    if (this.status === 'connected' || this.status === 'connecting') {
      console.warn('[SSE] 已经连接或正在连接')
      return
    }

    // 更新配置
    if (url) this.config.url = url
    Object.assign(this.config, config)

    if (!this.config.url) {
      throw new Error('[SSE] 未指定 URL')
    }

    this.setStatus('connecting')
    this.createEventSource()
  }

  // 创建 EventSource
  createEventSource() {
    try {
      // 构建 URL（包含 Last-Event-ID）
      const url = new URL(this.config.url, window.location.origin)
      if (this.lastMessageId) {
        url.searchParams.set('lastEventId', this.lastMessageId)
      }

      this.eventSource = new EventSource(url.toString(), {
        withCredentials: this.config.withCredentials
      })

      // 连接打开
      this.eventSource.onopen = () => {
        console.log('[SSE] 连接成功')
        this.setStatus('connected')
        this.reconnectCount = 0
        this.currentReconnectInterval = this.config.reconnectInterval
        this.stats.connectedAt = Date.now()
        this.emit('connected')
        this.startHeartbeat()
        this.processMessageQueue()
      }

      // 接收消息
      this.eventSource.onmessage = (event) => {
        this.handleMessage('message', event)
      }

      // 连接错误
      this.eventSource.onerror = (error) => {
        console.error('[SSE] 连接错误:', error)
        this.handleError(error)
      }

      // 注册自定义事件监听器
      this.registerCustomEvents()
    } catch (error) {
      console.error('[SSE] 创建 EventSource 失败:', error)
      this.handleError(error)
    }
  }

  // 注册自定义事件
  registerCustomEvents() {
    // 任务进度事件
    this.eventSource.addEventListener('task_progress', (event) => {
      this.handleMessage('task_progress', event)
    })

    // 任务完成事件
    this.eventSource.addEventListener('task_complete', (event) => {
      this.handleMessage('task_complete', event)
    })

    // 任务失败事件
    this.eventSource.addEventListener('task_failed', (event) => {
      this.handleMessage('task_failed', event)
    })

    // 心跳事件
    this.eventSource.addEventListener('heartbeat', (event) => {
      this.handleHeartbeat(event)
    })
  }

  // 处理消息
  handleMessage(type, event) {
    try {
      const data = JSON.parse(event.data)

      // 更��统计
      this.stats.messagesReceived++
      this.stats.lastMessageTime = Date.now()

      // 记录最后的消息 ID
      if (event.lastEventId) {
        this.lastMessageId = event.lastEventId
      }

      // 去重检查
      if (this.config.enableDedup && data.id) {
        if (this.processedMessageIds.has(data.id)) {
          console.log(`[SSE] 忽略重复消息: ${data.id}`)
          return
        }
        this.processedMessageIds.add(data.id)

        // 限制集合大小
        if (this.processedMessageIds.size > 1000) {
          const firstItem = this.processedMessageIds.values().next().value
          this.processedMessageIds.delete(firstItem)
        }
      }

      // 触发事件
      this.emit(type, data)
      this.emit('message', { type, data })

      console.log(`[SSE] 收到消息 [${type}]:`, data)
    } catch (error) {
      console.error('[SSE] 消息解析失败:', error, event.data)
    }
  }

  // 处理心跳
  handleHeartbeat(event) {
    console.log('[SSE] 心跳响应')
    this.resetHeartbeat()
  }

  // 处理错误
  handleError(error) {
    this.emit('error', error)

    // 如果连接已关闭
    if (this.eventSource?.readyState === EventSource.CLOSED) {
      this.setStatus('disconnected')
      this.stopHeartbeat()

      // 尝试重连
      if (this.config.reconnect && this.reconnectCount < this.config.maxReconnectAttempts) {
        this.scheduleReconnect()
      } else {
        this.setStatus('failed')
        console.error('[SSE] 连接失败，不再重连')

        // 降级到轮询
        if (this.config.fallbackToPolling) {
          this.startPolling()
        }
      }
    }
  }

  // 计划重连
  scheduleReconnect() {
    this.setStatus('reconnecting')
    this.reconnectCount++

    console.log(`[SSE] ${this.currentReconnectInterval}ms 后尝试第 ${this.reconnectCount} 次重连`)

    setTimeout(() => {
      if (this.status === 'reconnecting') {
        this.createEventSource()

        // 指数退避
        this.currentReconnectInterval = Math.min(
          this.currentReconnectInterval * this.config.reconnectDecay,
          this.config.maxReconnectInterval
        )
      }
    }, this.currentReconnectInterval)
  }

  // 手动重连
  reconnect() {
    this.disconnect()
    this.reconnectCount = 0
    this.currentReconnectInterval = this.config.reconnectInterval
    this.connect()
  }

  // 断开��接
  disconnect() {
    console.log('[SSE] 断开连接')

    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
    }

    this.stopHeartbeat()
    this.stopPolling()
    this.setStatus('disconnected')
    this.emit('disconnected')
  }

  // 心跳管理
  startHeartbeat() {
    this.stopHeartbeat()

    this.heartbeatTimer = setTimeout(() => {
      console.warn('[SSE] 心跳超时，触发重连')
      this.handleError(new Error('心跳超时'))
    }, this.config.heartbeatInterval + this.config.heartbeatTimeout)
  }

  resetHeartbeat() {
    this.startHeartbeat()
  }

  stopHeartbeat() {
    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }

  // 降级轮询
  startPolling() {
    console.log('[SSE] 启动轮询模式')
    this.stopPolling()

    this.pollingTimer = setInterval(async () => {
      try {
        const response = await fetch(this.config.url.replace('/stream', '/poll'), {
          method: 'GET',
          credentials: this.config.withCredentials ? 'include' : 'omit',
          headers: {
            'Last-Event-ID': this.lastMessageId || ''
          }
        })

        if (!response.ok) throw new Error(`HTTP ${response.status}`)

        const data = await response.json()

        if (data.events && Array.isArray(data.events)) {
          data.events.forEach(event => {
            this.handleMessage(event.type || 'message', {
              data: JSON.stringify(event.data),
              lastEventId: event.id
            })
          })
        }
      } catch (error) {
        console.error('[SSE] 轮询失败:', error)
      }
    }, this.config.pollingInterval)
  }

  stopPolling() {
    if (this.pollingTimer) {
      clearInterval(this.pollingTimer)
      this.pollingTimer = null
    }
  }

  // 消息队列管理
  queueMessage(type, data) {
    this.messageQueue.push({ type, data, timestamp: Date.now() })

    // 限制队列大小
    if (this.messageQueue.length > this.config.messageQueueSize) {
      this.messageQueue.shift()
    }
  }

  processMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift()
      this.emit(message.type, message.data)
    }
  }

  clearQueue() {
    this.messageQueue = []
  }

  // 订阅
  subscribe(event, handler) {
    this.on(event, handler)

    // 返回取消订阅函数
    return () => {
      this.off(event, handler)
    }
  }

  // 取消订阅
  unsubscribe(event, handler) {
    if (handler) {
      this.off(event, handler)
    } else {
      this.removeAllListeners(event)
    }
  }

  // 状态管理
  setStatus(status) {
    if (this.status !== status) {
      const oldStatus = this.status
      this.status = status
      this.emit('status', { old: oldStatus, new: status })
    }
  }

  getStatus() {
    return {
      status: this.status,
      connected: this.status === 'connected',
      reconnecting: this.status === 'reconnecting'
    }
  }

  // 统计信息
  getStats() {
    return {
      status: this.status,
      connected: this.status === 'connected',
      reconnectCount: this.reconnectCount,
      messagesReceived: this.stats.messagesReceived,
      lastMessageTime: this.stats.lastMessageTime,
      uptime: this.stats.connectedAt ? Date.now() - this.stats.connectedAt : 0
    }
  }

  // 销毁
  destroy() {
    this.disconnect()
    this.removeAllListeners()
    this.messageQueue = []
    this.processedMessageIds.clear()
  }
}

// 导出单例
export const sseService = new SSEService()
export default sseService
```

## 使用示例

### 基础用法

```javascript
import sseService from '@/services/sseService'

// 连接
sseService.connect('/api/tasks/stream')

// 订阅任务进度
const unsubscribe = sseService.subscribe('task_progress', (data) => {
  console.log('任务进度:', data.progress)
})

// 取消订阅
unsubscribe()

// 断开连接
sseService.disconnect()
```

### 在 Store 中使用

```javascript
// stores/unifiedTaskStore.js
import sseService from '@/services/sseService'

export const useUnifiedTaskStore = defineStore('unifiedTask', {
  actions: {
    connectSSE() {
      sseService.connect('/api/tasks/stream')

      // 订阅任务进度
      sseService.subscribe('task_progress', (data) => {
        this.updateTaskProgress(data.jobId, data.progress)
      })

      // 订阅任务完成
      sseService.subscribe('task_complete', (data) => {
        this.completeTask(data.jobId, data.result)
      })

      // 订阅任务失败
      sseService.subscribe('task_failed', (data) => {
        this.failTask(data.jobId, data.error)
      })

      // 监听连接状态
      sseService.on('connected', () => {
        console.log('SSE ���连接')
      })

      sseService.on('disconnected', () => {
        console.log('SSE 已断开')
      })

      sseService.on('error', (error) => {
        console.error('SSE 错误:', error)
      })
    },

    disconnectSSE() {
      sseService.disconnect()
    }
  }
})
```

### 在组件中使用

```vue
<!-- components/TaskMonitor/index.vue -->
<script setup>
import { onMounted, onUnmounted } from 'vue'
import sseService from '@/services/sseService'

let unsubscribeProgress
let unsubscribeComplete

onMounted(() => {
  // 订阅任务事件
  unsubscribeProgress = sseService.subscribe('task_progress', handleProgress)
  unsubscribeComplete = sseService.subscribe('task_complete', handleComplete)
})

onUnmounted(() => {
  // 清理订阅
  unsubscribeProgress?.()
  unsubscribeComplete?.()
})

function handleProgress(data) {
  console.log('进度更新:', data)
}

function handleComplete(data) {
  console.log('任务完成:', data)
}
</script>
```

## 后端集成

### FastAPI 示例

```python
# backend/api/tasks.py
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

router = APIRouter()

@router.get("/stream")
async def task_stream(request: Request, lastEventId: str = None):
    async def event_generator():
        last_id = 0
        if lastEventId:
            try:
                last_id = int(lastEventId)
            except:
                pass

        while True:
            # 检查客户端是否断开
            if await request.is_disconnected():
                break

            # 获取任务更新（从队列或数据库）
            updates = await get_task_updates(last_id)

            for update in updates:
                last_id = update['id']

                # 发送事件
                yield f"id: {update['id']}\n"
                yield f"event: {update['type']}\n"
                yield f"data: {json.dumps(update['data'])}\n\n"

            # 发送心跳
            yield f"event: heartbeat\n"
            yield f"data: {json.dumps({'timestamp': time.time()})}\n\n"

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
```

## 性能优化

1. **连接复用**
   - 多个组件共享同一个 SSE 连接
   - 避免重复连接

2. **消息批处理**
   - 短时间内的多个消息合并处理

3. **内存管理**
   - 限制消息队列大小
   - 定期清理已处理消息 ID

## 最佳实践

1. **合理设置重连参数**
   - 初始间隔不宜过短
   - 指数退避避免服务器压力

2. **心跳检测**
   - 及时发现死连接
   - 触发重连机制

3. **降级方案**
   - SSE 不可用时降级到轮询
   - 保证基本功能可用

4. **资源清理**
   - 组件销毁时取消订阅
   - 页面卸载时断开连接

## 测试要点

1. 连接建立和断开
2. 自动重连机制
3. 心跳超时检���
4. 消息去重功能
5. 降级轮询模式

## 未来扩展

1. WebSocket 支持
2. 二进制消息传输
3. 消息压缩
4. 离线消息队列
5. 多端点负载均衡