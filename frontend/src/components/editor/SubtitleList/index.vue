<template>
  <div class="subtitle-list">
    <!-- 工具栏 -->
    <div class="list-toolbar">
      <div class="toolbar-left">
        <span class="subtitle-count">{{ totalSubtitles }} 条字幕</span>
      </div>

      <div class="toolbar-center">
        <div class="search-box">
          <svg class="search-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
          </svg>
          <input
            v-model="searchText"
            type="text"
            placeholder="搜索字幕..."
            class="search-input"
          />
          <button v-if="searchText" class="search-clear" @click="searchText = ''">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
          </button>
        </div>
      </div>

      <div class="toolbar-right">
        <button class="toolbar-btn" @click="addNewSubtitle" title="添加字幕">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
          </svg>
        </button>
      </div>
    </div>

    <!-- 字幕列表 -->
    <div class="list-container" ref="listRef">
      <div v-if="filteredSubtitles.length === 0" class="empty-state">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zM4 18V6h16v12H4zm2-1h2v-2H6v2zm0-3h8v-2H6v2zm10 3h2v-5h-2v5zm-4 0h2v-2h-2v2zm0-3h4v-2h-4v2z"/>
        </svg>
        <p>暂无字幕</p>
        <button class="add-first-btn" @click="addNewSubtitle">添加第一条字幕</button>
      </div>

      <div
        v-for="(subtitle, index) in filteredSubtitles"
        :key="subtitle.id"
        class="subtitle-item"
        :class="{
          'is-active': activeSubtitleId === subtitle.id,
          'is-current': currentSubtitleId === subtitle.id,
          'has-error': hasError(index)
        }"
        @click="onSubtitleClick(subtitle)"
      >
        <!-- 序号 -->
        <div class="item-index">{{ index + 1 }}</div>

        <!-- 主内容 -->
        <div class="item-content">
          <!-- 时间行 -->
          <div class="time-row">
            <input
              type="text"
              class="time-input"
              :value="formatTime(subtitle.start)"
              @change="e => updateTime(subtitle.id, 'start', parseTime(e.target.value))"
              @focus="e => e.target.select()"
            />
            <span class="time-arrow">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M16.01 11H4v2h12.01v3L20 12l-3.99-4z"/>
              </svg>
            </span>
            <input
              type="text"
              class="time-input"
              :value="formatTime(subtitle.end)"
              @change="e => updateTime(subtitle.id, 'end', parseTime(e.target.value))"
              @focus="e => e.target.select()"
            />
            <span class="duration-tag">{{ formatDuration(subtitle.end - subtitle.start) }}</span>
          </div>

          <!-- 文本行 -->
          <div class="text-row">
            <textarea
              class="text-input"
              :value="subtitle.text"
              @input="e => updateText(subtitle.id, e.target.value)"
              placeholder="输入字幕文本..."
              rows="2"
            ></textarea>
            <span class="char-count" :class="{ warning: subtitle.text.length > 30 }">
              {{ subtitle.text.length }}
            </span>
          </div>

          <!-- 错误提示 -->
          <div v-if="getItemErrors(index).length > 0" class="error-tags">
            <span
              v-for="error in getItemErrors(index)"
              :key="error.type"
              class="error-tag"
              :class="error.severity"
            >
              {{ error.message }}
            </span>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div class="item-actions">
          <button class="action-btn" @click.stop="insertBefore(index)" title="在前面插入">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 14l5-5 5 5z"/>
            </svg>
          </button>
          <button class="action-btn" @click.stop="insertAfter(index)" title="在后面插入">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 10l5 5 5-5z"/>
            </svg>
          </button>
          <button class="action-btn action-btn--danger" @click.stop="deleteSubtitle(subtitle.id)" title="删除">
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { useProjectStore } from '@/stores/projectStore'

// Props
const props = defineProps({
  autoScroll: { type: Boolean, default: true },
  editable: { type: Boolean, default: true }
})

const emit = defineEmits(['subtitle-click', 'subtitle-edit', 'subtitle-delete', 'subtitle-add'])

// Store
const projectStore = useProjectStore()

// Refs
const listRef = ref(null)

// State
const searchText = ref('')

// Computed
const subtitles = computed(() => projectStore.subtitles)
const totalSubtitles = computed(() => projectStore.totalSubtitles)
const validationErrors = computed(() => projectStore.validationErrors)
const currentSubtitleId = computed(() => projectStore.currentSubtitle?.id)
const activeSubtitleId = computed(() => projectStore.view.selectedSubtitleId)

const filteredSubtitles = computed(() => {
  if (!searchText.value) return subtitles.value
  const search = searchText.value.toLowerCase()
  return subtitles.value.filter(sub => sub.text.toLowerCase().includes(search))
})

// Methods
function hasError(index) {
  return validationErrors.value.some(e => e.index === index)
}

function getItemErrors(index) {
  return validationErrors.value.filter(e => e.index === index)
}

function onSubtitleClick(subtitle) {
  projectStore.view.selectedSubtitleId = subtitle.id
  projectStore.seekTo(subtitle.start)
  emit('subtitle-click', subtitle)
}

function updateTime(id, field, value) {
  if (isNaN(value)) return
  projectStore.updateSubtitle(id, { [field]: value })
  emit('subtitle-edit', id, field, value)
}

function updateText(id, text) {
  projectStore.updateSubtitle(id, { text })
  emit('subtitle-edit', id, 'text', text)
}

function deleteSubtitle(id) {
  if (confirm('确定删除这条字幕吗?')) {
    projectStore.removeSubtitle(id)
    emit('subtitle-delete', id)
  }
}

function addNewSubtitle() {
  const lastSubtitle = subtitles.value[subtitles.value.length - 1]
  const newStart = lastSubtitle ? lastSubtitle.end : 0
  projectStore.addSubtitle(subtitles.value.length, {
    start: newStart,
    end: newStart + 3,
    text: ''
  })
  nextTick(() => {
    scrollToBottom()
  })
  emit('subtitle-add', subtitles.value.length - 1)
}

function insertBefore(index) {
  const current = subtitles.value[index]
  const prev = subtitles.value[index - 1]
  const start = prev ? prev.end : Math.max(0, current.start - 3)
  const end = current.start
  projectStore.addSubtitle(index, { start, end, text: '' })
}

function insertAfter(index) {
  const current = subtitles.value[index]
  const next = subtitles.value[index + 1]
  const start = current.end
  const end = next ? next.start : current.end + 3
  projectStore.addSubtitle(index + 1, { start, end, text: '' })
}

function scrollToBottom() {
  if (listRef.value) {
    listRef.value.scrollTop = listRef.value.scrollHeight
  }
}

function scrollToItem(index) {
  const items = listRef.value?.querySelectorAll('.subtitle-item')
  if (items && items[index]) {
    items[index].scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

// 自动滚动跟随当前播放
watch(currentSubtitleId, (id) => {
  if (!props.autoScroll || !id) return
  const index = filteredSubtitles.value.findIndex(s => s.id === id)
  if (index !== -1) {
    nextTick(() => scrollToItem(index))
  }
})

// 时间格式化
function formatTime(seconds) {
  if (isNaN(seconds)) return '00:00.000'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  const ms = Math.round((seconds % 1) * 1000)
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`
}

function parseTime(str) {
  const match = str.match(/(\d+):(\d+)\.?(\d*)/)
  if (!match) return NaN
  const m = parseInt(match[1])
  const s = parseInt(match[2])
  const ms = match[3] ? parseInt(match[3].padEnd(3, '0')) : 0
  return m * 60 + s + ms / 1000
}

function formatDuration(seconds) {
  if (isNaN(seconds) || seconds < 0) return '0.0s'
  return seconds.toFixed(1) + 's'
}
</script>

<style lang="scss" scoped>
.subtitle-list {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-primary);
}

// 工具栏 - 针对 350px 宽度优化
.list-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;  // 减少内边距
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-default);
  gap: 12px;

  .toolbar-left {
    .subtitle-count {
      font-size: 12px;
      color: var(--text-secondary);
      white-space: nowrap;
    }
  }

  .toolbar-center {
    flex: 1;
    max-width: 180px;  // 缩小搜索框
    min-width: 100px;
  }

  .search-box {
    display: flex;
    align-items: center;
    background: var(--bg-tertiary);
    border-radius: var(--radius-md);
    padding: 5px 10px;
    gap: 6px;

    .search-icon {
      width: 14px;
      height: 14px;
      color: var(--text-muted);
      flex-shrink: 0;
    }

    .search-input {
      flex: 1;
      min-width: 0;
      background: transparent;
      border: none;
      color: var(--text-normal);
      font-size: 12px;

      &::placeholder { color: var(--text-muted); }
    }

    .search-clear {
      width: 16px;
      height: 16px;
      color: var(--text-muted);
      flex-shrink: 0;
      svg { width: 100%; height: 100%; }
      &:hover { color: var(--text-normal); }
    }
  }

  .toolbar-btn {
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    transition: all var(--transition-fast);
    flex-shrink: 0;

    svg { width: 18px; height: 18px; }

    &:hover {
      background: var(--bg-tertiary);
      color: var(--primary);
    }
  }
}

// 列表容器
.list-container {
  flex: 1;
  overflow-y: auto;
  padding: 6px;

  &::-webkit-scrollbar {
    width: 6px;
  }
  &::-webkit-scrollbar-track {
    background: transparent;
  }
  &::-webkit-scrollbar-thumb {
    background: var(--border-default);
    border-radius: 3px;
    &:hover { background: var(--text-muted); }
  }
}

// 空状态
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px 16px;
  color: var(--text-muted);

  svg {
    width: 48px;
    height: 48px;
    margin-bottom: 12px;
    opacity: 0.5;
  }

  p {
    font-size: 13px;
    margin-bottom: 12px;
  }

  .add-first-btn {
    padding: 6px 16px;
    background: var(--primary);
    color: white;
    border-radius: var(--radius-md);
    font-size: 13px;
    transition: background var(--transition-fast);
    &:hover { background: var(--primary-hover); }
  }
}

// 字幕项 - 紧凑布局
.subtitle-item {
  display: flex;
  gap: 10px;
  padding: 10px;  // 减少内边距
  margin-bottom: 6px;
  background: var(--bg-secondary);
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  transition: all var(--transition-fast);
  cursor: pointer;

  &:hover {
    background: var(--bg-tertiary);

    .item-actions { opacity: 1; }
  }

  &.is-active {
    border-color: var(--primary);
    background: rgba(88, 166, 255, 0.08);
  }

  &.is-current {
    border-color: var(--success);
    background: rgba(63, 185, 80, 0.08);

    .item-index { background: var(--success); color: white; }
  }

  &.has-error {
    border-color: var(--danger);
  }
}

// 序号 - 缩小尺寸
.item-index {
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-tertiary);
  border-radius: var(--radius-sm);
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  flex-shrink: 0;
}

// 内容区
.item-content {
  flex: 1;
  min-width: 0;
}

// 时间行 - 优化间距
.time-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  flex-wrap: wrap;

  .time-input {
    width: 75px;  // 缩小宽度
    padding: 3px 6px;
    background: var(--bg-tertiary);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-normal);
    text-align: center;

    &:focus {
      border-color: var(--primary);
      outline: none;
    }
  }

  .time-arrow {
    color: var(--text-muted);
    svg { width: 14px; height: 14px; }
  }

  .duration-tag {
    padding: 2px 6px;
    background: var(--bg-tertiary);
    border-radius: var(--radius-full);
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text-muted);
  }
}

// 文本行 - 优化尺寸
.text-row {
  position: relative;

  .text-input {
    width: 100%;
    padding: 6px 8px;
    padding-right: 35px;
    background: var(--bg-tertiary);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--text-normal);
    resize: none;
    line-height: 1.4;

    &:focus {
      border-color: var(--primary);
      outline: none;
    }

    &::placeholder { color: var(--text-muted); }
  }

  .char-count {
    position: absolute;
    right: 6px;
    bottom: 6px;
    font-size: 10px;
    font-family: var(--font-mono);
    color: var(--text-muted);

    &.warning { color: var(--warning); }
  }
}

// 错误标签
.error-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 6px;

  .error-tag {
    padding: 2px 6px;
    font-size: 10px;
    border-radius: var(--radius-full);

    &.error {
      background: rgba(248, 81, 73, 0.15);
      color: var(--danger);
    }

    &.warning {
      background: rgba(210, 153, 34, 0.15);
      color: var(--warning);
    }
  }
}

// 操作按钮 - 始终可见，更小尺寸
.item-actions {
  display: flex;
  flex-direction: column;
  gap: 2px;
  opacity: 1;  // 始终显示

  .action-btn {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    color: var(--text-muted);
    transition: all var(--transition-fast);

    svg { width: 14px; height: 14px; }

    &:hover {
      background: var(--bg-tertiary);
      color: var(--text-normal);
    }

    &--danger:hover {
      background: rgba(248, 81, 73, 0.15);
      color: var(--danger);
    }
  }
}
</style>
