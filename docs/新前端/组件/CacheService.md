# CacheService - 缓存服务

## 概述

CacheService 是多级缓存管理服务，提供内存缓存（LRU）+ IndexedDB 持久化的两级缓存机制，用于优化大数据加载性能和用户体验。它为项目数据、音频波形、视频缩略图等资源提供统一的缓存管理。

## 功能特性

1. **两级缓存架构**
   - L1: 内存 LRU 缓存（快速访问）
   - L2: IndexedDB 持久化（容量大）
   - 自动降级和提升

2. **智能缓存策略**
   - LRU 淘汰算法
   - 容量自动管理
   - 过期时间控制
   - 优先级管理

3. **数据类型支持**
   - 字符串（SRT内容）
   - 对象（项目���态）
   - 二进制（音频波形数据）
   - Blob（视频帧图片）

4. **性能优化**
   - 批量操作支持
   - 预加载机制
   - 懒清理策略
   - 压缩存储

## 技术依赖

```json
{
  "localforage": "^1.10.0",
  "lru-cache": "^10.0.0"
}
```

## API 接口

### 初始化配置

```typescript
interface CacheConfig {
  // L1 内存缓存配置
  memory: {
    max: number              // 最大条目数，默认100
    maxSize: number          // 最大大小（字节），默认50MB
    ttl: number             // 过期时间（毫秒），默认1小时
  }

  // L2 IndexedDB 配置
  indexedDB: {
    name: string            // 数据库名，默认'subtitle-editor-cache'
    storeName: string       // 存储名，默认'cache'
    version: number         // 版本号，默认1
  }

  // 全局配置
  enableCompression: boolean   // 启用压缩，默认true
  autoCleanup: boolean        // 自动清理，默认true
  cleanupInterval: number     // 清理间隔（毫秒），默认5分钟
}
```

### 核心方法

```typescript
class CacheService {
  /**
   * 获取缓存值
   * @param key 缓存键
   * @param options 选项
   * @returns 缓存值或null
   */
  async get<T>(key: string, options?: GetOptions): Promise<T | null>

  /**
   * 设置缓存值
   * @param key 缓存键
   * @param value 缓存值
   * @param options 选项
   */
  async set<T>(key: string, value: T, options?: SetOptions): Promise<void>

  /**
   * 删除缓存值
   * @param key 缓存键
   */
  async delete(key: string): Promise<void>

  /**
   * 检查键是否存在
   * @param key 缓存键
   */
  async has(key: string): Promise<boolean>

  /**
   * 清空所有缓存
   */
  async clear(): Promise<void>

  /**
   * 获取缓存统计信息
   */
  async stats(): Promise<CacheStats>

  /**
   * 批量获取
   * @param keys 键数组
   */
  async getMany<T>(keys: string[]): Promise<Map<string, T>>

  /**
   * 批量设置
   * @param entries 键值对数组
   */
  async setMany<T>(entries: Array<[string, T]>, options?: SetOptions): Promise<void>

  /**
   * 预加载数据
   * @param keys 需要预加载的键
   */
  async preload(keys: string[]): Promise<void>

  /**
   * 手动触发清理
   */
  async cleanup(): Promise<void>
}

interface GetOptions {
  preferMemory?: boolean     // 优先使用内存缓存
  skipIndexedDB?: boolean    // 跳过IndexedDB查询
}

interface SetOptions {
  ttl?: number              // 自定义过期时间
  priority?: number         // 优先级（0-10），默认5
  compress?: boolean        // 是否压缩，默认根据大小自动判断
  persistToDisk?: boolean   // 是否持久化到磁盘，默认true
}

interface CacheStats {
  memory: {
    size: number           // 当前条目数
    bytes: number          // 内存占用（字节）
    hitRate: number        // 命中率
  }
  indexedDB: {
    size: number           // 存储条目数
    bytes: number          // 磁盘占用（估计）
  }
  total: {
    hits: number           // 总命中次数
    misses: number         // 总未命中次数
  }
}
```

## 核心实现

```javascript
// services/cacheService.js
import localforage from 'localforage'
import { LRUCache } from 'lru-cache'

class CacheService {
  constructor(config = {}) {
    this.config = {
      memory: {
        max: 100,
        maxSize: 50 * 1024 * 1024, // 50MB
        ttl: 60 * 60 * 1000, // 1 hour
        ...config.memory
      },
      indexedDB: {
        name: 'subtitle-editor-cache',
        storeName: 'cache',
        version: 1,
        ...config.indexedDB
      },
      enableCompression: config.enableCompression ?? true,
      autoCleanup: config.autoCleanup ?? true,
      cleanupInterval: config.cleanupInterval ?? 5 * 60 * 1000 // 5 minutes
    }

    // 初始化 L1 内存缓存
    this.memoryCache = new LRUCache({
      max: this.config.memory.max,
      maxSize: this.config.memory.maxSize,
      ttl: this.config.memory.ttl,
      sizeCalculation: (value) => {
        return JSON.stringify(value).length
      },
      dispose: (value, key) => {
        console.log(`[Cache] 内存淘汰: ${key}`)
      }
    })

    // 初始化 L2 IndexedDB
    this.diskCache = localforage.createInstance({
      name: this.config.indexedDB.name,
      storeName: this.config.indexedDB.storeName,
      version: this.config.indexedDB.version
    })

    // 统计信息
    this.stats = {
      hits: 0,
      misses: 0
    }

    // 自动清理
    if (this.config.autoCleanup) {
      this.startAutoCleanup()
    }
  }

  // 获取缓存
  async get(key, options = {}) {
    const { preferMemory = true, skipIndexedDB = false } = options

    // L1: 内存缓存查询
    if (preferMemory && this.memoryCache.has(key)) {
      const cached = this.memoryCache.get(key)
      if (this.isValid(cached)) {
        this.stats.hits++
        console.log(`[Cache] 内存命中: ${key}`)
        return cached.value
      }
    }

    // L2: IndexedDB 查询
    if (!skipIndexedDB) {
      try {
        const cached = await this.diskCache.getItem(key)
        if (cached && this.isValid(cached)) {
          this.stats.hits++
          console.log(`[Cache] 磁盘命中: ${key}`)

          // 提升到内存缓存
          this.memoryCache.set(key, cached)

          return this.decompress(cached.value)
        }
      } catch (error) {
        console.error('[Cache] IndexedDB 读取失败:', error)
      }
    }

    this.stats.misses++
    return null
  }

  // 设置缓存
  async set(key, value, options = {}) {
    const {
      ttl = this.config.memory.ttl,
      priority = 5,
      compress = this.shouldCompress(value),
      persistToDisk = true
    } = options

    const cached = {
      value: compress ? this.compress(value) : value,
      timestamp: Date.now(),
      ttl,
      priority,
      compressed: compress
    }

    // L1: 写入内存缓存
    this.memoryCache.set(key, cached)

    // L2: 写入 IndexedDB
    if (persistToDisk) {
      try {
        await this.diskCache.setItem(key, cached)
        console.log(`[Cache] 已持久化: ${key}`)
      } catch (error) {
        console.error('[Cache] IndexedDB 写入失败:', error)
      }
    }
  }

  // 删除缓存
  async delete(key) {
    this.memoryCache.delete(key)
    await this.diskCache.removeItem(key)
  }

  // 检查是否存在
  async has(key) {
    if (this.memoryCache.has(key)) return true

    try {
      const value = await this.diskCache.getItem(key)
      return value !== null
    } catch {
      return false
    }
  }

  // 清空缓存
  async clear() {
    this.memoryCache.clear()
    await this.diskCache.clear()
    this.stats.hits = 0
    this.stats.misses = 0
    console.log('[Cache] 已清空所有缓存')
  }

  // 批量获取
  async getMany(keys) {
    const results = new Map()

    for (const key of keys) {
      const value = await this.get(key)
      if (value !== null) {
        results.set(key, value)
      }
    }

    return results
  }

  // 批量设置
  async setMany(entries, options = {}) {
    const promises = entries.map(([key, value]) =>
      this.set(key, value, options)
    )

    await Promise.all(promises)
  }

  // 预加载
  async preload(keys) {
    console.log(`[Cache] 预加载 ${keys.length} 个键`)
    await this.getMany(keys)
  }

  // 获取统计信息
  async getStats() {
    const memorySize = this.memoryCache.size
    const memoryBytes = this.memoryCache.calculatedSize

    // 估算 IndexedDB 大小
    let diskSize = 0
    try {
      const keys = await this.diskCache.keys()
      diskSize = keys.length
    } catch (error) {
      console.error('[Cache] 获取磁盘统计失败:', error)
    }

    const totalRequests = this.stats.hits + this.stats.misses
    const hitRate = totalRequests > 0 ? this.stats.hits / totalRequests : 0

    return {
      memory: {
        size: memorySize,
        bytes: memoryBytes,
        hitRate
      },
      indexedDB: {
        size: diskSize,
        bytes: 0 // 无法精确计算
      },
      total: {
        hits: this.stats.hits,
        misses: this.stats.misses
      }
    }
  }

  // 手动清理
  async cleanup() {
    console.log('[Cache] 执行清理')

    // 清理过期的 IndexedDB 条目
    try {
      const keys = await this.diskCache.keys()

      for (const key of keys) {
        const cached = await this.diskCache.getItem(key)
        if (!this.isValid(cached)) {
          await this.diskCache.removeItem(key)
          console.log(`[Cache] 清理过期: ${key}`)
        }
      }
    } catch (error) {
      console.error('[Cache] 清理失败:', error)
    }
  }

  // 自动清理
  startAutoCleanup() {
    this.cleanupTimer = setInterval(() => {
      this.cleanup()
    }, this.config.cleanupInterval)
  }

  stopAutoCleanup() {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }
  }

  // 辅助方法

  isValid(cached) {
    if (!cached || !cached.timestamp) return false

    const age = Date.now() - cached.timestamp
    return age < cached.ttl
  }

  shouldCompress(value) {
    if (!this.config.enableCompression) return false

    // 小于 1KB 不压缩
    const size = JSON.stringify(value).length
    return size > 1024
  }

  compress(value) {
    // 简单的压缩实现（实际项目可使用 pako 等库）
    if (typeof value === 'string') {
      return { __compressed: true, data: value }
    }
    return { __compressed: true, data: JSON.stringify(value) }
  }

  decompress(value) {
    if (value?.__compressed) {
      try {
        return JSON.parse(value.data)
      } catch {
        return value.data
      }
    }
    return value
  }

  // 销毁
  destroy() {
    this.stopAutoCleanup()
    this.memoryCache.clear()
  }
}

// 导出单例
export const cacheService = new CacheService()
export default cacheService
```

## 使用示例

### 基础用法

```javascript
import cacheService from '@/services/cacheService'

// 设置缓存
await cacheService.set('project-123', projectData, {
  ttl: 60 * 60 * 1000, // 1小时
  priority: 8
})

// 获取缓存
const data = await cacheService.get('project-123')

// 删除缓存
await cacheService.delete('project-123')
```

### 在 Store 中使用

```javascript
// stores/projectStore.js
import cacheService from '@/services/cacheService'

export const useProjectStore = defineStore('project', {
  actions: {
    async saveProject() {
      const cacheKey = `project-${this.meta.jobId}`

      await cacheService.set(cacheKey, {
        subtitles: this.subtitles,
        meta: this.meta,
        history: this.history
      }, {
        priority: 10, // 高优先级
        ttl: 24 * 60 * 60 * 1000 // 24小时
      })
    },

    async restoreProject(jobId) {
      const cacheKey = `project-${jobId}`
      const cached = await cacheService.get(cacheKey)

      if (cached) {
        this.subtitles = cached.subtitles
        this.meta = cached.meta
        this.history = cached.history
        return true
      }

      return false
    }
  }
})
```

### 波形数据缓存

```javascript
// components/WaveformTimeline/index.vue
async function loadPeaks() {
  const cacheKey = `peaks-${props.jobId}`

  // 尝试从缓存获取
  let peaks = await cacheService.get(cacheKey)

  if (!peaks) {
    // 从后端加载
    const response = await fetch(`/api/audio/peaks/${props.jobId}`)
    peaks = await response.json()

    // 缓存数据
    await cacheService.set(cacheKey, peaks, {
      ttl: 7 * 24 * 60 * 60 * 1000, // 7天
      priority: 7
    })
  }

  return peaks
}
```

## 性能指标

### 命中率监控

```javascript
// 定期输出统计信息
setInterval(async () => {
  const stats = await cacheService.getStats()
  console.log('缓存统计:', {
    内存条目数: stats.memory.size,
    内存占用: (stats.memory.bytes / 1024 / 1024).toFixed(2) + 'MB',
    命中率: (stats.memory.hitRate * 100).toFixed(1) + '%',
    磁盘条目数: stats.indexedDB.size
  })
}, 60000) // 每分钟
```

## 最佳实践

1. **合理设置 TTL**
   - 频繁变化的数据：短 TTL（几分钟）
   - 稳定数据：长 TTL（几天）
   - 不变数据：永久（需手动清理）

2. **优先级管理**
   - 核心数据：高优先级（8-10）
   - 常用数据：中优先级（5-7）
   - 临时数据：低优先级（1-4）

3. **容量控制**
   - 监控内存占用
   - 定期清理过期数据
   - 限制单个缓存大小

4. **错误处理**
   - IndexedDB 可能失败（隐私模式）
   - 降级到仅内存缓存
   - 提供回退机制

## 测试要点

1. LRU 淘汰算法正确性
2. 过期时间准确性
3. IndexedDB 读写性能
4. 批量操作效率
5. 清理机制可靠性

## 未来扩展

1. 支持 Service Worker 缓存
2. 网络请求缓存拦截
3. 智能预测性预加载
4. 跨标签页缓存共享
5. 缓存版本管理