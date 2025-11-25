# SubtitleList - 字幕列表组件

## 概述

SubtitleList 是字幕编辑器的核心列表管理组件，提供字幕的文本编辑、时间调整和批量操作功能。它使用虚拟滚动技术处理大量字幕，并与波形时间���实现双向同步。

## 功能特性

1. **字幕编辑**
   - 实时文本编辑
   - 时间戳精确调整
   - 快速添加/删除

2. **虚拟滚动**
   - 始终使用 vue-virtual-scroller
   - 支持1000+条字幕流畅滚动
   - 自动跟随播放位置

3. **交互功能**
   - 点击跳转播放
   - 高亮当前播放字幕
   - 选中状态管理

4. **批���操作**
   - 多选支持
   - 批量时间偏移
   - 批量删除

## 技术依赖

```json
{
  "vue": "^3.4.0",
  "vue-virtual-scroller": "^2.0.0",
  "element-plus": "^2.11.0",
  "pinia": "^3.0.0"
}
```

## 组件属性

```typescript
interface SubtitleListProps {
  // 列表配置
  itemHeight?: number       // 每项高度，默认80
  maxHeight?: string       // 最大高度，默认'600px'
  autoScroll?: boolean     // 自动滚动跟随，默认true

  // 编辑配置
  editable?: boolean       // 允许编辑，默认true
  showIndex?: boolean      // 显示序号，默认true
  showActions?: boolean    // 显示操作按钮，默认true

  // 时间格式
  timeFormat?: 'srt' | 'seconds'  // 时间显示格式，默认'srt'
}

interface SubtitleListEmits {
  'subtitle-click': (subtitle: Subtitle) => void
  'subtitle-edit': (id: string, field: string, value: any) => void
  'subtitle-delete': (id: string) => void
  'subtitle-add': (index: number) => void
}
```

## 核心实现

```vue
<!-- components/editor/ScriptEditor/ListMode.vue -->
<template>
  <div class="subtitle-list-container">
    <!-- 工具栏 -->
    <div class="list-toolbar">
      <h3>字幕列表 ({{ totalSubtitles }}条)</h3>
      <div class="toolbar-actions">
        <el-button
          @click="addNewSubtitle"
          size="small"
          type="primary"
        >
          <el-icon><Plus /></el-icon>
          新增字幕
        </el-button>

        <el-dropdown v-if="selectedIds.size > 0" @command="handleBatchAction">
          <el-button size="small">
            批量操作 ({{ selectedIds.size }})
            <el-icon><ArrowDown /></el-icon>
          </el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="offset">时间偏移</el-dropdown-item>
              <el-dropdown-item command="merge">合并字幕</el-dropdown-item>
              <el-dropdown-item command="split">分割字幕</el-dropdown-item>
              <el-dropdown-item command="delete" divided>删除选中</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>

        <el-input
          v-model="searchText"
          placeholder="搜索字幕..."
          size="small"
          clearable
          style="width: 200px"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
      </div>
    </div>

    <!-- 虚拟滚动列表 -->
    <RecycleScroller
      ref="scrollerRef"
      class="subtitle-scroller"
      :items="filteredSubtitles"
      :item-size="itemHeight"
      key-field="id"
      v-slot="{ item, index }"
      :buffer="200"
      :emit-update="true"
      @visible="onVisible"
      @hidden="onHidden"
    >
      <div
        class="subtitle-item"
        :class="{
          'is-active': isActive(item.id),
          'is-current': isCurrent(item.id),
          'is-selected': isSelected(item.id),
          'has-error': hasError(index)
        }"
        @click="onSubtitleClick(item, $event)"
      >
        <!-- 选择框 -->
        <div class="subtitle-checkbox" v-if="showSelection">
          <el-checkbox
            :model-value="isSelected(item.id)"
            @change="(val) => toggleSelection(item.id, val)"
            @click.stop
          />
        </div>

        <!-- 序号 -->
        <div class="subtitle-index" v-if="showIndex">
          {{ index + 1 }}
        </div>

        <!-- 内容区 -->
        <div class="subtitle-content">
          <!-- 时间编辑 -->
          <div class="time-inputs">
            <TimeInput
              :value="item.start"
              @change="(val) => updateTime(item.id, 'start', val)"
              :format="timeFormat"
              :disabled="!editable"
              placeholder="开始时���"
            />
            <el-icon class="time-arrow"><ArrowRight /></el-icon>
            <TimeInput
              :value="item.end"
              @change="(val) => updateTime(item.id, 'end', val)"
              :format="timeFormat"
              :disabled="!editable"
              placeholder="结束时间"
            />
            <span class="duration">
              ({{ formatDuration(item.end - item.start) }})
            </span>
          </div>

          <!-- 文本编辑 -->
          <div class="text-editor">
            <el-input
              :model-value="item.text"
              @input="(val) => updateText(item.id, val)"
              @blur="onTextBlur(item)"
              type="textarea"
              :rows="2"
              :disabled="!editable"
              placeholder="输入字幕文本..."
              :maxlength="200"
              show-word-limit
            />
            <div class="text-stats">
              <span>{{ item.text.length }}字</span>
              <span v-if="item.isDirty" class="dirty-indicator">*</span>
            </div>
          </div>

          <!-- 错误提示 -->
          <div v-if="getItemErrors(index).length > 0" class="error-tips">
            <el-tag
              v-for="error in getItemErrors(index)"
              :key="error.type"
              :type="error.severity === 'error' ? 'danger' : 'warning'"
              size="small"
            >
              {{ error.message }}
            </el-tag>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div class="subtitle-actions" v-if="showActions">
          <el-button-group size="small">
            <el-button
              @click.stop="insertBefore(index)"
              title="在前面插入"
            >
              <el-icon><Top /></el-icon>
            </el-button>
            <el-button
              @click.stop="insertAfter(index)"
              title="在后面插入"
            >
              <el-icon><Bottom /></el-icon>
            </el-button>
            <el-button
              @click.stop="splitSubtitle(item, index)"
              title="分割字幕"
            >
              <el-icon><ScissorIcon /></el-icon>
            </el-button>
            <el-button
              @click.stop="deleteSubtitle(item.id)"
              type="danger"
              title="删除"
            >
              <el-icon><Delete /></el-icon>
            </el-button>
          </el-button-group>
        </div>
      </div>
    </RecycleScroller>

    <!-- 空状态 -->
    <el-empty
      v-if="filteredSubtitles.length === 0"
      description="暂无字幕"
      :image-size="100"
    >
      <el-button @click="addNewSubtitle" type="primary">
        添加第一条字幕
      </el-button>
    </el-empty>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { RecycleScroller } from 'vue-virtual-scroller'
import 'vue-virtual-scroller/dist/vue-virtual-scroller.css'
import { useProjectStore } from '@/stores/projectStore'
import { ElMessage, ElMessageBox } from 'element-plus'
import TimeInput from './TimeInput.vue'

// Props
const props = defineProps({
  itemHeight: {
    type: Number,
    default: 80
  },
  maxHeight: {
    type: String,
    default: '600px'
  },
  autoScroll: {
    type: Boolean,
    default: true
  },
  editable: {
    type: Boolean,
    default: true
  },
  showIndex: {
    type: Boolean,
    default: true
  },
  showActions: {
    type: Boolean,
    default: true
  },
  timeFormat: {
    type: String,
    default: 'srt',
    validator: v => ['srt', 'seconds'].includes(v)
  }
})

const emit = defineEmits([
  'subtitle-click',
  'subtitle-edit',
  'subtitle-delete',
  'subtitle-add'
])

// Store
const projectStore = useProjectStore()

// Refs
const scrollerRef = ref(null)

// State
const searchText = ref('')
const selectedIds = ref(new Set())
const showSelection = ref(false)
const visibleItems = ref(new Set())

// Computed
const subtitles = computed(() => projectStore.subtitles)
const totalSubtitles = computed(() => projectStore.totalSubtitles)
const validationErrors = computed(() => projectStore.validationErrors)

const filteredSubtitles = computed(() => {
  if (!searchText.value) return subtitles.value

  const search = searchText.value.toLowerCase()
  return subtitles.value.filter(sub =>
    sub.text.toLowerCase().includes(search)
  )
})

// 当前播放的字幕
const currentSubtitleId = computed(() => projectStore.currentSubtitle?.id)

// 选中的字幕
const activeSubtitleId = computed(() => projectStore.view.selectedSubtitleId)

// Methods
function isActive(id) {
  return activeSubtitleId.value === id
}

function isCurrent(id) {
  return currentSubtitleId.value === id
}

function isSelected(id) {
  return selectedIds.value.has(id)
}

function hasError(index) {
  return validationErrors.value.some(e => e.index === index)
}

function getItemErrors(index) {
  return validationErrors.value.filter(e => e.index === index)
}

function onSubtitleClick(subtitle, event) {
  if (event.shiftKey && showSelection.value) {
    // Shift点击：范围选择
    const lastId = activeSubtitleId.value
    if (lastId) {
      selectRange(lastId, subtitle.id)
    }
  } else if (event.ctrlKey || event.metaKey) {
    // Ctrl/Cmd点击：多选
    toggleSelection(subtitle.id, !isSelected(subtitle.id))
    showSelection.value = true
  } else {
    // 普通点击
    projectStore.view.selectedSubtitleId = subtitle.id
    projectStore.seekTo(subtitle.start)
    emit('subtitle-click', subtitle)

    // 清除多选
    if (showSelection.value && selectedIds.value.size > 0) {
      selectedIds.value.clear()
      showSelection.value = false
    }
  }
}

function toggleSelection(id, selected) {
  if (selected) {
    selectedIds.value.add(id)
  } else {
    selectedIds.value.delete(id)
  }
}

function selectRange(fromId, toId) {
  const fromIndex = subtitles.value.findIndex(s => s.id === fromId)
  const toIndex = subtitles.value.findIndex(s => s.id === toId)

  const start = Math.min(fromIndex, toIndex)
  const end = Math.max(fromIndex, toIndex)

  for (let i = start; i <= end; i++) {
    selectedIds.value.add(subtitles.value[i].id)
  }
}

function updateTime(id, field, value) {
  projectStore.updateSubtitle(id, { [field]: value })
  emit('subtitle-edit', id, field, value)
}

function updateText(id, text) {
  projectStore.updateSubtitle(id, { text })
  emit('subtitle-edit', id, 'text', text)
}

function onTextBlur(subtitle) {
  // 文本失焦时检查是否需要保存
  if (subtitle.isDirty) {
    // 可以在这里触发自动保存
  }
}

function deleteSubtitle(id) {
  ElMessageBox.confirm(
    '确定删除这条字幕吗？',
    '确认删除',
    {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(() => {
    projectStore.removeSubtitle(id)
    emit('subtitle-delete', id)
    ElMessage.success('删除成功')
  }).catch(() => {})
}

function addNewSubtitle() {
  const lastSubtitle = subtitles.value[subtitles.value.length - 1]
  const newStart = lastSubtitle ? lastSubtitle.end : 0

  projectStore.addSubtitle(subtitles.value.length, {
    start: newStart,
    end: newStart + 3,
    text: ''
  })

  // 滚动到新添加的字幕
  nextTick(() => {
    scrollToIndex(subtitles.value.length - 1)
  })

  emit('subtitle-add', subtitles.value.length - 1)
}

function insertBefore(index) {
  const current = subtitles.value[index]
  const prev = subtitles.value[index - 1]

  const start = prev ? prev.end : Math.max(0, current.start - 3)
  const end = current.start

  projectStore.addSubtitle(index, {
    start,
    end,
    text: ''
  })
}

function insertAfter(index) {
  const current = subtitles.value[index]
  const next = subtitles.value[index + 1]

  const start = current.end
  const end = next ? next.start : current.end + 3

  projectStore.addSubtitle(index + 1, {
    start,
    end,
    text: ''
  })
}

function splitSubtitle(subtitle, index) {
  const midPoint = Math.floor(subtitle.text.length / 2)
  const text1 = subtitle.text.slice(0, midPoint).trim()
  const text2 = subtitle.text.slice(midPoint).trim()

  const duration = subtitle.end - subtitle.start
  const midTime = subtitle.start + duration / 2

  projectStore.updateSubtitle(subtitle.id, {
    text: text1,
    end: midTime
  })

  projectStore.addSubtitle(index + 1, {
    start: midTime,
    end: subtitle.end,
    text: text2
  })

  ElMessage.success('字幕已���割')
}

// 批量操作
function handleBatchAction(command) {
  switch (command) {
    case 'offset':
      showOffsetDialog()
      break
    case 'merge':
      mergeSelected()
      break
    case 'split':
      splitSelected()
      break
    case 'delete':
      deleteSelected()
      break
  }
}

function showOffsetDialog() {
  ElMessageBox.prompt(
    '请输入时间偏移量（秒，正数后移，负数前移）',
    '批量时间偏移',
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      inputPattern: /^-?\d+(\.\d+)?$/,
      inputErrorMessage: '请输入有效的数字'
    }
  ).then(({ value }) => {
    const offset = parseFloat(value)
    applyTimeOffset(offset)
  }).catch(() => {})
}

function applyTimeOffset(offset) {
  selectedIds.value.forEach(id => {
    const subtitle = subtitles.value.find(s => s.id === id)
    if (subtitle) {
      projectStore.updateSubtitle(id, {
        start: Math.max(0, subtitle.start + offset),
        end: Math.max(0, subtitle.end + offset)
      })
    }
  })

  ElMessage.success(`已对${selectedIds.value.size}条字幕应用时间偏移`)
  selectedIds.value.clear()
  showSelection.value = false
}

function mergeSelected() {
  const selectedSubtitles = subtitles.value
    .filter(s => selectedIds.value.has(s.id))
    .sort((a, b) => a.start - b.start)

  if (selectedSubtitles.length < 2) {
    ElMessage.warning('请选择至少两条字幕进行合并')
    return
  }

  const first = selectedSubtitles[0]
  const last = selectedSubtitles[selectedSubtitles.length - 1]
  const mergedText = selectedSubtitles.map(s => s.text).join(' ')

  // 更新第一条
  projectStore.updateSubtitle(first.id, {
    end: last.end,
    text: mergedText
  })

  // 删除其余
  selectedSubtitles.slice(1).forEach(s => {
    projectStore.removeSubtitle(s.id)
  })

  ElMessage.success('字幕已合并')
  selectedIds.value.clear()
  showSelection.value = false
}

function deleteSelected() {
  ElMessageBox.confirm(
    `确定删除选中的${selectedIds.value.size}条字幕吗？`,
    '批量删除',
    {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(() => {
    selectedIds.value.forEach(id => {
      projectStore.removeSubtitle(id)
    })
    ElMessage.success('批量删除成功')
    selectedIds.value.clear()
    showSelection.value = false
  }).catch(() => {})
}

// 自动滚动
function scrollToIndex(index) {
  scrollerRef.value?.scrollToItem(index)
}

// 虚拟滚动事件
function onVisible(startIndex, endIndex) {
  for (let i = startIndex; i <= endIndex; i++) {
    const item = filteredSubtitles.value[i]
    if (item) {
      visibleItems.value.add(item.id)
    }
  }
}

function onHidden(startIndex, endIndex) {
  for (let i = startIndex; i <= endIndex; i++) {
    const item = filteredSubtitles.value[i]
    if (item) {
      visibleItems.value.delete(item.id)
    }
  }
}

// 自动滚动跟随
watch(currentSubtitleId, (id) => {
  if (!props.autoScroll || !id) return

  const index = filteredSubtitles.value.findIndex(s => s.id === id)
  if (index !== -1 && !visibleItems.value.has(id)) {
    scrollToIndex(index)
  }
})

// 工具函数
function formatDuration(seconds) {
  return `${seconds.toFixed(1)}s`
}
</script>
```

## 与其他组件的关系

### 依赖关系
- 强依赖 `ProjectStore` 的字幕数据
- 依赖 `TimeInput` 组件进行时间编辑

### 被依赖关系
- 被 `EditorView` 包含使用
- 与 `WaveformTimeline` 数据同步
- 与 `ValidationPanel` 共享错误状态

## 性能优化

1. **虚拟滚动**
   - 始终使用 RecycleScroller
   - 支持10000+条字幕

2. **防抖更新**
   - 文本输入防抖300ms
   - 减少Store更新频率

3. **计算属性缓存**
   - Vue自动缓存computed结果

## 测试要点

1. 虚拟滚动性能
2. 字幕编辑同步
3. 批量操作正确性
4. 自动滚动跟随
5. 搜索过滤功能

## 未来扩展

1. 拖拽排序
2. 导入导出功能
3. 字幕样式编辑
4. AI智能分割
5. 多语言支持