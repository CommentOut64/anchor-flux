# ProjectStore - 项目数据管理

## 概述

ProjectStore 是字幕编辑器的核心数据管理器，负责管理所有与字幕项目相关的状态，包括字幕数据、播放器状态、视图配置等。它实现了完整的撤销/重做功能、自动保存、智能问题检测等高级特性。

## 功能特性

1. **字幕数据管理**
   - 字幕的增删改查
   - 时间轴调整
   - 文本编辑
   - 批量操作

2. **撤销/重做系统**
   - 基于 @vueuse/core 的 useRefHistory
   - 50步历史记录限制
   - 深拷贝保证数据独立性

3. **多级缓存策略**
   - 内存缓存（热数据）
   - IndexedDB 持久化（冷数据）
   - LRU 缓存淘汰策略

4. **智能问题检测**
   - 时间重叠检测
   - 字数过长警告
   - 实时验证反馈

5. **播放器状态同步**
   - 当前时间同步
   - 播放状态控制
   - 倍速控制
   - 音量管理

## 技术依赖

```json
{
  "pinia": "^3.0.0",
  "vue": "^3.4.0",
  "@vueuse/core": "^10.0.0",
  "localforage": "^1.10.0"
}
```

## 数据结构

```javascript
// 字幕数据结构
interface Subtitle {
  id: string           // 唯一标识
  start: number        // 开始时间（秒）
  end: number          // 结束时间（秒）
  text: string         // 字幕文本
  isDirty: boolean     // 是否已修改
}

// 项目元数据
interface ProjectMeta {
  jobId: string        // 任务ID
  videoPath: string    // 视频路径
  audioPath: string    // 音频路径
  peaksPath: string    // 波形数据路径
  duration: number     // 总时长
  filename: string     // 文件名
  lastSaved: number    // 最后保存时间
  isDirty: boolean     // 是否有未保存修改
}

// 播放器状态
interface PlayerState {
  currentTime: number  // 当前播放时间
  isPlaying: boolean   // 是否正在播放
  playbackRate: number // 播放速度
  volume: number       // 音量
}

// 视图状态
interface ViewState {
  theme: 'dark' | 'light'
  zoomLevel: number
  autoScroll: boolean
  selectedSubtitleId: string | null
}

// 验证错误
interface ValidationError {
  type: 'overlap' | 'too_long'
  index: number
  message: string
  severity: 'error' | 'warning'
}
```

## 核心实现

```javascript
// stores/projectStore.js
import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { useRefHistory } from '@vueuse/core'
import localforage from 'localforage'

export const useProjectStore = defineStore('project', () => {
  // ========== 状态定义 ==========
  const meta = ref({
    jobId: null,
    videoPath: null,
    audioPath: null,
    peaksPath: null,
    duration: 0,
    filename: '',
    lastSaved: Date.now(),
    isDirty: false
  })

  const subtitles = ref([])

  const player = ref({
    currentTime: 0,
    isPlaying: false,
    playbackRate: 1.0,
    volume: 1.0
  })

  const view = ref({
    theme: 'dark',
    zoomLevel: 100,
    autoScroll: true,
    selectedSubtitleId: null
  })

  // ========== 撤销/重做 ==========
  const {
    history,
    undo,
    redo,
    canUndo,
    canRedo,
    clear: clearHistory
  } = useRefHistory(subtitles, {
    deep: true,
    capacity: 50,
    clone: true
  })

  // ========== 计算属性 ==========
  const totalSubtitles = computed(() => subtitles.value.length)

  const currentSubtitle = computed(() => {
    return subtitles.value.find(s =>
      player.value.currentTime >= s.start &&
      player.value.currentTime < s.end
    )
  })

  const isDirty = computed(() => {
    return meta.value.isDirty || subtitles.value.some(s => s.isDirty)
  })

  // 智能问题检测
  const validationErrors = computed(() => {
    const errors = []

    // 检测时间重叠
    for (let i = 0; i < subtitles.value.length - 1; i++) {
      if (subtitles.value[i].end > subtitles.value[i + 1].start) {
        errors.push({
          type: 'overlap',
          index: i,
          message: `字幕 #${i+1} 与 #${i+2} 时间重叠`,
          severity: 'error'
        })
      }
    }

    // 检测字数过长
    subtitles.value.forEach((sub, i) => {
      if (sub.text.length > 30) {
        errors.push({
          type: 'too_long',
          index: i,
          message: `字幕 #${i+1} 超过30字`,
          severity: 'warning'
        })
      }
    })

    return errors
  })

  // ========== 多级缓存 ==========
  const memoryCache = new Map()
  const MAX_MEMORY_CACHE = 10

  // 自动保存
  watch([subtitles, meta], async () => {
    if (!meta.value.jobId) return

    // 内存缓存
    memoryCache.set(meta.value.jobId, {
      subtitles: subtitles.value,
      meta: meta.value
    })

    // LRU 淘汰
    if (memoryCache.size > MAX_MEMORY_CACHE) {
      const firstKey = memoryCache.keys().next().value
      memoryCache.delete(firstKey)
    }

    // IndexedDB 持久化
    try {
      await localforage.setItem(`project-${meta.value.jobId}`, {
        subtitles: subtitles.value,
        meta: meta.value,
        savedAt: Date.now()
      })
    } catch (error) {
      console.error('自动保存失败:', error)
    }
  }, {
    deep: true,
    throttle: 3000  // 3秒节流
  })

  // ========== Actions ==========
  function importSRT(srtContent, metadata) {
    const parsed = parseSRT(srtContent)
    subtitles.value = parsed.map((item, idx) => ({
      id: `subtitle-${Date.now()}-${idx}`,
      start: item.start,
      end: item.end,
      text: item.text,
      isDirty: false
    }))

    meta.value = {
      ...meta.value,
      ...metadata,
      lastSaved: Date.now(),
      isDirty: false
    }

    clearHistory()
  }

  async function restoreProject(jobId) {
    try {
      // 优先从内存缓存获取
      if (memoryCache.has(jobId)) {
        const cached = memoryCache.get(jobId)
        subtitles.value = cached.subtitles
        meta.value = cached.meta
        return true
      }

      // 从 IndexedDB 恢复
      const saved = await localforage.getItem(`project-${jobId}`)
      if (saved) {
        subtitles.value = saved.subtitles
        meta.value = saved.meta
        return true
      }
      return false
    } catch (error) {
      console.error('恢复项目失败:', error)
      return false
    }
  }

  function updateSubtitle(id, payload) {
    const index = subtitles.value.findIndex(s => s.id === id)
    if (index === -1) return

    subtitles.value[index] = {
      ...subtitles.value[index],
      ...payload,
      isDirty: true
    }
    meta.value.isDirty = true
  }

  function addSubtitle(insertIndex, payload) {
    const newSubtitle = {
      id: `subtitle-${Date.now()}`,
      start: payload.start || 0,
      end: payload.end || 0,
      text: payload.text || '',
      isDirty: true
    }
    subtitles.value.splice(insertIndex, 0, newSubtitle)
    meta.value.isDirty = true
  }

  function removeSubtitle(id) {
    const index = subtitles.value.findIndex(s => s.id === id)
    if (index !== -1) {
      subtitles.value.splice(index, 1)
      meta.value.isDirty = true
    }
  }

  function generateSRT() {
    let srtContent = ''
    subtitles.value.forEach((sub, index) => {
      srtContent += `${index + 1}\n`
      srtContent += `${formatTimestamp(sub.start)} --> ${formatTimestamp(sub.end)}\n`
      srtContent += `${sub.text}\n\n`
    })
    return srtContent
  }

  function seekTo(time) {
    player.value.currentTime = time
  }

  async function saveProject() {
    const srtContent = generateSRT()
    // TODO: 调用后端API保存
    meta.value.lastSaved = Date.now()
    meta.value.isDirty = false
    subtitles.value.forEach(s => s.isDirty = false)
  }

  return {
    // 状态
    meta,
    subtitles,
    player,
    view,

    // 计算属性
    totalSubtitles,
    currentSubtitle,
    isDirty,
    validationErrors,

    // 历史记录
    canUndo,
    canRedo,
    undo,
    redo,

    // 操作方法
    importSRT,
    restoreProject,
    updateSubtitle,
    addSubtitle,
    removeSubtitle,
    generateSRT,
    seekTo,
    saveProject
  }
})
```

## 与其他组件的关系

### 依赖关系
- 无直接依赖其他 Store

### 被依赖关系
- `VideoStage` 使用播放器状态
- `WaveformTimeline` 使用字幕数据和播放状态
- `SubtitleList` 使用字幕数据
- `ValidationPanel` 使用验证错误
- `EditorView` 协调所有状态

## 使用示例

```vue
<script setup>
import { useProjectStore } from '@/stores/projectStore'

const projectStore = useProjectStore()

// 导入SRT文件
async function handleSRTImport(srtContent) {
  projectStore.importSRT(srtContent, {
    jobId: 'xxx',
    filename: 'video.mp4',
    videoPath: '/api/media/xxx/video'
  })
}

// 编辑字幕
function handleSubtitleEdit(id, newText) {
  projectStore.updateSubtitle(id, { text: newText })
}

// 播放控制
function togglePlay() {
  projectStore.player.isPlaying = !projectStore.player.isPlaying
}

// 导出SRT
function exportSRT() {
  const content = projectStore.generateSRT()
  downloadFile(content, 'subtitle.srt')
}
</script>
```

## 性能优化

1. **虚拟滚动**
   - 配合 vue-virtual-scroller 处理大量字幕

2. **防抖更新**
   - 3秒防抖避免频繁写入 IndexedDB

3. **LRU 缓存**
   - 内存缓存最近10个项目

4. **计算属性缓存**
   - Vue 自动缓存计算结果

## 测试要点

1. 字幕CRUD操作
2. 撤销/重做功能
3. 自动保存和恢复
4. 问题检测准确性
5. 多项目切换

## 未来扩展

1. 字幕样式支持
2. 多语言字幕轨道
3. 字幕模板
4. 协作编辑
5. 版本控制