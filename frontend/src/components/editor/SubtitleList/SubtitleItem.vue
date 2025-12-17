<template>
  <div
    class="subtitle-item"
    :class="itemClasses"
    @click="handleClick"
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
          :readonly="subtitle.isDraft"
          @change="e => updateTime('start', parseTime(e.target.value))"
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
          :readonly="subtitle.isDraft"
          @change="e => updateTime('end', parseTime(e.target.value))"
          @focus="e => e.target.select()"
        />
        <span class="duration-tag">{{ formatDuration(subtitle.end - subtitle.start) }}</span>

        <!-- 草稿状态指示器 -->
        <span v-if="subtitle.isDraft" class="draft-indicator">
          <span class="spinner"></span>
          <span class="draft-text">生成中</span>
        </span>

        <!-- 置信度徽章 -->
        <span
          v-else-if="showConfidenceBadge"
          class="confidence-badge"
          :class="confidenceBadgeClass"
          :title="`置信度: ${(subtitle.confidence * 100).toFixed(0)}%`"
        >
          {{ (subtitle.confidence * 100).toFixed(0) }}%
        </span>
      </div>

      <!-- 文本行（三态视图） -->
      <div class="text-row">
        <!-- 状态1: 草稿锁定模式 (isDraft=true) -->
        <div
          v-if="subtitle.isDraft"
          class="text-display text-draft"
        >
          {{ subtitle.text || '...' }}
        </div>

        <!-- 状态2: 只读高亮预览模式 (isDraft=false & 非编辑) -->
        <div
          v-else-if="!isEditing"
          class="text-display text-preview"
          :class="{ 'can-edit': editable }"
          @click.stop="startEditing"
          v-html="renderTextWithHighlight()"
        ></div>

        <!-- 状态3: 编辑模式 (isDraft=false & 编辑中) -->
        <textarea
          v-else
          ref="editTextarea"
          class="text-input"
          :value="subtitle.text"
          @input="e => handleTextInput(e.target.value)"
          @blur="stopEditing"
          @keydown.enter.ctrl="stopEditing"
          @keydown.escape="cancelEditing"
          placeholder="输入字幕文本..."
          rows="2"
        ></textarea>

        <span class="char-count" :class="{ warning: subtitle.text.length > 30 }">
          {{ subtitle.text.length }}
        </span>
      </div>

      <!-- 警告提示 -->
      <div v-if="showWarning" class="warning-banner">
        <span class="warning-text">{{ warningMessage }}</span>
      </div>

      <!-- 错误标签 -->
      <div v-if="errors.length > 0" class="error-tags">
        <span
          v-for="error in errors"
          :key="error.type"
          class="error-tag"
          :class="error.severity"
        >
          {{ error.message }}
        </span>
      </div>
    </div>

    <!-- 操作按钮 -->
    <div v-if="!subtitle.isDraft" class="item-actions" @click.stop>
      <button class="action-btn" @click="$emit('insert-before')" title="在前面插入">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M7 14l5-5 5 5z"/>
        </svg>
      </button>
      <button class="action-btn" @click="$emit('insert-after')" title="在后面插入">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M7 10l5 5 5-5z"/>
        </svg>
      </button>
      <button class="action-btn action-btn--danger" @click="handleDelete" title="删除">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup>
/**
 * SubtitleItem - 字幕列表单项组件
 *
 * Phase 5 双模态架构: 实现三态视图
 * 1. 草稿锁定模式: isDraft=true, 灰色斜体, 只读
 * 2. 高亮预览模式: isDraft=false & 非编辑, 显示置信度高亮
 * 3. 编辑模式: isDraft=false & 编辑中, 可编辑文本
 */
import { ref, computed, nextTick, watch } from 'vue'

const props = defineProps({
  subtitle: { type: Object, required: true },
  index: { type: Number, required: true },
  isActive: { type: Boolean, default: false },
  isCurrent: { type: Boolean, default: false },
  editable: { type: Boolean, default: true },
  errors: { type: Array, default: () => [] }
})

const emit = defineEmits([
  'click',
  'update-time',
  'update-text',
  'delete',
  'insert-before',
  'insert-after'
])

// 编辑状态
const isEditing = ref(false)
const editTextarea = ref(null)
const originalText = ref('')

// 计算属性
const itemClasses = computed(() => ({
  'is-active': props.isActive,
  'is-current': props.isCurrent,
  'is-draft': props.subtitle.isDraft,
  'has-error': props.errors.length > 0,
  'warning-low-confidence': props.subtitle.warning_type === 'low_confidence',
  'warning-high-perplexity': props.subtitle.warning_type === 'high_perplexity',
  'warning-both': props.subtitle.warning_type === 'both'
}))

// 置信度徽章
const showConfidenceBadge = computed(() => {
  return props.subtitle.confidence !== undefined && props.subtitle.confidence < 0.9
})

const confidenceBadgeClass = computed(() => {
  const conf = props.subtitle.confidence
  if (conf >= 0.8) return 'badge-good'
  if (conf >= 0.6) return 'badge-warning'
  return 'badge-danger'
})

// 警告信息
const showWarning = computed(() => {
  return props.subtitle.warning_type && props.subtitle.warning_type !== 'none'
})

const warningMessage = computed(() => {
  const type = props.subtitle.warning_type
  const messages = {
    'low_confidence': '低置信度，建议人工审核',
    'high_perplexity': 'LLM 困惑度较高，可能有语法问题',
    'both': '低置信度 + 高困惑度，强烈建议审核'
  }
  return messages[type] || ''
})

// 点击处理
function handleClick() {
  emit('click', props.subtitle)
}

// 时间更新
function updateTime(field, value) {
  if (isNaN(value) || props.subtitle.isDraft) return
  emit('update-time', props.subtitle.id, field, value)
}

// 开始编辑
function startEditing() {
  if (!props.editable || props.subtitle.isDraft) return
  originalText.value = props.subtitle.text
  isEditing.value = true
  nextTick(() => {
    if (editTextarea.value) {
      editTextarea.value.focus()
      editTextarea.value.select()
    }
  })
}

// 停止编辑
function stopEditing() {
  isEditing.value = false
}

// 取消编辑（恢复原文）
function cancelEditing() {
  if (originalText.value !== props.subtitle.text) {
    emit('update-text', props.subtitle.id, originalText.value)
  }
  isEditing.value = false
}

// 文本输入处理
function handleTextInput(text) {
  emit('update-text', props.subtitle.id, text)
}

// 删除处理
function handleDelete() {
  emit('delete', props.subtitle.id)
}

// 渲染带置信度高亮的文本
function renderTextWithHighlight() {
  const words = props.subtitle.words
  const text = props.subtitle.text

  // 如果没有字级数据，直接返回文本
  if (!words || words.length === 0) {
    return escapeHtml(text)
  }

  const WARN_THRESHOLD = 0.5
  const CRITICAL_THRESHOLD = 0.3

  let html = ''
  for (let i = 0; i < words.length; i++) {
    const word = words[i]
    const conf = word.confidence !== undefined ? word.confidence : 1.0
    const wordText = escapeHtml(word.word)

    if (conf < CRITICAL_THRESHOLD) {
      html += `<span class="word-critical">${wordText}</span>`
    } else if (conf < WARN_THRESHOLD) {
      html += `<span class="word-warning">${wordText}</span>`
    } else {
      html += wordText
    }

    // 智能添加空格：英文单词之间加空格，中文字符之间不加
    if (i < words.length - 1) {
      const nextWord = words[i + 1].word
      // 如果当前词或下一词是中文字符，不加空格
      // 如果下一词是标点符号，不加空格
      const isChinese = (char) => char && /[\u4e00-\u9fff]/.test(char)
      const isPunctuation = (char) => char && /[,.!?;:'"()[\]{}，。！？；：""''（）【】《》、]/.test(char)

      if (!isChinese(wordText[wordText.length - 1]) &&
          !isChinese(nextWord[0]) &&
          !isPunctuation(nextWord[0])) {
        html += ' '
      }
    }
  }
  return html
}

// HTML 转义
function escapeHtml(text) {
  if (!text) return ''
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

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
// 字幕项
.subtitle-item {
  display: flex;
  gap: 10px;
  padding: 10px;
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

  // 草稿状态样式
  &.is-draft {
    background: rgba(128, 128, 128, 0.05);
    border-color: rgba(128, 128, 128, 0.2);
    cursor: wait;

    .item-index {
      background: #6b7280;
      color: white;
    }

    .text-display {
      color: #9ca3af;
      font-style: italic;
    }
  }

  // 置信度警告高亮样式
  &.warning-low-confidence {
    border-color: var(--warning);
    background: rgba(210, 153, 34, 0.06);
    .item-index { background: var(--warning); color: white; }
  }

  &.warning-high-perplexity {
    border-color: #e67700;
    background: rgba(230, 119, 0, 0.06);
    .item-index { background: #e67700; color: white; }
  }

  &.warning-both {
    border-color: var(--danger);
    background: rgba(248, 81, 73, 0.08);
    border-width: 2px;
    .item-index { background: var(--danger); color: white; }
  }
}

// 序号
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

// 时间行
.time-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
  flex-wrap: wrap;

  .time-input {
    width: 75px;
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

    &[readonly] {
      cursor: wait;
      opacity: 0.6;
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

// 草稿状态指示器
.draft-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  background: rgba(107, 114, 128, 0.15);
  border-radius: var(--radius-full);
  font-size: 10px;
  color: #9ca3af;

  .spinner {
    width: 10px;
    height: 10px;
    border: 2px solid rgba(156, 163, 175, 0.3);
    border-top-color: #9ca3af;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .draft-text {
    font-style: italic;
  }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

// 置信度徽章
.confidence-badge {
  padding: 2px 6px;
  border-radius: var(--radius-full);
  font-size: 10px;
  font-family: var(--font-mono);
  font-weight: 600;

  &.badge-good {
    background: rgba(63, 185, 80, 0.15);
    color: var(--success);
  }

  &.badge-warning {
    background: rgba(210, 153, 34, 0.15);
    color: var(--warning);
  }

  &.badge-danger {
    background: rgba(248, 81, 73, 0.15);
    color: var(--danger);
  }
}

// 文本行
.text-row {
  position: relative;

  .text-display {
    width: 100%;
    min-height: 45px;
    padding: 6px 35px 6px 8px;
    background: var(--bg-tertiary);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--text-normal);
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
  }

  // 草稿文本样式
  .text-draft {
    color: #9ca3af;
    font-style: italic;
    cursor: wait;
    background: rgba(107, 114, 128, 0.08);
  }

  // 预览文本样式
  .text-preview {
    cursor: default;

    &.can-edit {
      cursor: text;
      &:hover {
        border-color: var(--primary);
        background: var(--bg-secondary);
      }
    }

    // 字级警告高亮样式
    :deep(.word-warning) {
      background-color: rgba(255, 193, 7, 0.25);
      border-bottom: 2px solid var(--warning, #ffc107);
      padding: 0 2px;
      border-radius: 2px;
    }

    :deep(.word-critical) {
      background-color: rgba(244, 67, 54, 0.25);
      border-bottom: 2px solid var(--error, #f44336);
      padding: 0 2px;
      border-radius: 2px;
      font-weight: 500;
    }
  }

  .text-input {
    width: 100%;
    padding: 6px 8px;
    padding-right: 35px;
    background: var(--bg-tertiary);
    border: 1px solid var(--primary);
    border-radius: var(--radius-sm);
    font-size: 12px;
    color: var(--text-normal);
    resize: none;
    line-height: 1.4;
    outline: none;

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

// 警告横幅
.warning-banner {
  margin-top: 6px;
  padding: 4px 8px;
  background: rgba(210, 153, 34, 0.1);
  border-left: 3px solid var(--warning);
  border-radius: var(--radius-sm);

  .warning-text {
    font-size: 11px;
    color: var(--warning);
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

// 操作按钮
.item-actions {
  display: flex;
  flex-direction: column;
  gap: 2px;
  opacity: 0.5;
  transition: opacity var(--transition-fast);

  .action-btn {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    color: var(--text-muted);
    background: transparent;
    border: none;
    cursor: pointer;
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
