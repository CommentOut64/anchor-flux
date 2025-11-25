# AIAssistant - AI 助手组件

## 概述

AIAssistant 是智能字幕辅助编辑组件，提供基于 AI 的字幕优化建议、翻译、润色等功能。它与 ProjectStore 集成，为用户提供智能化的字幕编辑辅助。

## 功能特性

1. **智能建议**
   - 文本润色优化
   - 标点符号修正
   - 断句优化建议
   - 时长调整建议

2. **翻译功能**
   - 多语言翻译
   - 批量翻译
   - 术语库支持
   - 翻译质量评估

3. **问题检测**
   - 语法错误检测
   - 逻辑连贯性检查
   - 专业术语建议
   - 文化适配建议

4. **交互体验**
   - 流式响应显示
   - 建议对比预览
   - 一键应用/拒绝
   - 历史记录追溯

## 技术依赖

```json
{
  "vue": "^3.4.0",
  "element-plus": "^2.11.0",
  "pinia": "^3.0.0",
  "markdown-it": "^14.0.0"
}
```

## 组件属性

```typescript
interface AIAssistantProps {
  // 上下文配置
  context?: Subtitle          // 当前字幕上下文
  contextWindow?: number      // 上下文窗口大小，默认3

  // 功能开关
  enableTranslation?: boolean    // 启用翻译，默认true
  enablePolish?: boolean         // 启用润色，默认true
  enableSuggestion?: boolean     // 启用建议，默认true

  // API配置
  apiEndpoint?: string          // API端点
  apiKey?: string              // API密钥
  model?: string               // 模型名称，默认'gpt-4'

  // UI配置
  showHistory?: boolean         // 显示历史，默认true
  maxHistoryItems?: number      // 最大历史数，默认20
}

interface AIAssistantEmits {
  'apply-suggestion': (suggestion: Suggestion) => void
  'translate': (text: string, targetLang: string) => void
  'polish': (text: string) => void
  'error': (error: Error) => void
}
```

## 核心实现

```vue
<!-- components/AIAssistant/index.vue -->
<template>
  <div class="ai-assistant">
    <!-- 功能选择 -->
    <div class="assistant-toolbar">
      <el-segmented v-model="activeMode" :options="modeOptions" />

      <el-dropdown v-if="activeMode === 'translate'" @command="handleLangChange">
        <el-button size="small">
          {{ targetLanguage }}<el-icon><ArrowDown /></el-icon>
        </el-button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="en">English</el-dropdown-item>
            <el-dropdown-item command="ja">日本語</el-dropdown-item>
            <el-dropdown-item command="ko">한국어</el-dropdown-item>
            <el-dropdown-item command="es">Español</el-dropdown-item>
            <el-dropdown-item command="fr">Français</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>

    <!-- 当前字幕显示 -->
    <div class="current-context" v-if="context">
      <div class="context-header">
        <span>当前字幕 #{{ currentIndex + 1 }}</span>
        <el-tag size="small">{{ formatTime(context.start) }} - {{ formatTime(context.end) }}</el-tag>
      </div>
      <div class="context-text">{{ context.text }}</div>
    </div>

    <!-- 输入区域 -->
    <div class="input-area">
      <el-input
        v-model="inputText"
        type="textarea"
        :rows="3"
        :placeholder="inputPlaceholder"
        @keydown.ctrl.enter="handleSubmit"
      />
      <div class="input-actions">
        <el-button
          @click="handleSubmit"
          type="primary"
          :loading="loading"
          :disabled="!inputText.trim()"
        >
          {{ submitButtonText }}
        </el-button>
        <el-button @click="clearInput">清空</el-button>
      </div>
    </div>

    <!-- AI 响应区域 -->
    <div class="response-area" v-if="currentResponse || loading">
      <div class="response-header">
        <el-icon class="ai-icon"><ChatDotRound /></el-icon>
        <span>AI 建议</span>
        <el-tag v-if="currentResponse?.confidence" :type="confidenceType">
          置信度: {{ (currentResponse.confidence * 100).toFixed(0) }}%
        </el-tag>
      </div>

      <!-- 加载状态 -->
      <div v-if="loading" class="response-loading">
        <el-skeleton :rows="3" animated />
      </div>

      <!-- 响应内容 -->
      <div v-else-if="currentResponse" class="response-content">
        <!-- 润色/翻译结果 -->
        <div v-if="activeMode === 'polish' || activeMode === 'translate'" class="result-text">
          {{ currentResponse.text }}
        </div>

        <!-- 建议列表 -->
        <div v-else-if="activeMode === 'suggest'" class="suggestions-list">
          <div
            v-for="(suggestion, index) in currentResponse.suggestions"
            :key="index"
            class="suggestion-item"
          >
            <div class="suggestion-content">
              <el-icon><InfoFilled /></el-icon>
              <span>{{ suggestion.description }}</span>
            </div>
            <el-button
              @click="applySuggestion(suggestion)"
              size="small"
              type="primary"
              link
            >
              应用
            </el-button>
          </div>
        </div>

        <!-- 对比视图 -->
        <div v-if="showComparison" class="comparison-view">
          <div class="comparison-item original">
            <div class="comparison-label">原文</div>
            <div class="comparison-text">{{ context?.text }}</div>
          </div>
          <el-icon class="comparison-arrow"><Right /></el-icon>
          <div class="comparison-item modified">
            <div class="comparison-label">修改后</div>
            <div class="comparison-text">{{ currentResponse.text }}</div>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div class="response-actions">
          <el-button
            v-if="canApply"
            @click="applyResponse"
            type="primary"
          >
            应用到字幕
          </el-button>
          <el-button @click="copyResponse">复制</el-button>
          <el-button @click="regenerate" :loading="loading">重新生成</el-button>
        </div>

        <!-- 解释说明 -->
        <div v-if="currentResponse.explanation" class="response-explanation">
          <el-collapse>
            <el-collapse-item title="查看详细解释" name="1">
              <div v-html="renderMarkdown(currentResponse.explanation)"></div>
            </el-collapse-item>
          </el-collapse>
        </div>
      </div>
    </div>

    <!-- 历史记录 -->
    <div v-if="showHistory && history.length > 0" class="history-section">
      <div class="history-header">
        <span>历史记录</span>
        <el-button @click="clearHistory" size="small" link>清空</el-button>
      </div>
      <el-scrollbar max-height="200px">
        <div class="history-list">
          <div
            v-for="item in history"
            :key="item.id"
            class="history-item"
            @click="loadHistory(item)"
          >
            <div class="history-mode">{{ getModeLabel(item.mode) }}</div>
            <div class="history-preview">{{ item.preview }}</div>
            <div class="history-time">{{ formatHistoryTime(item.timestamp) }}</div>
          </div>
        </div>
      </el-scrollbar>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { ElMessage } from 'element-plus'
import MarkdownIt from 'markdown-it'

const md = new MarkdownIt()

// Props & Emits
const props = defineProps({
  context: {
    type: Object,
    default: null
  },
  contextWindow: {
    type: Number,
    default: 3
  },
  enableTranslation: {
    type: Boolean,
    default: true
  },
  enablePolish: {
    type: Boolean,
    default: true
  },
  enableSuggestion: {
    type: Boolean,
    default: true
  },
  apiEndpoint: {
    type: String,
    default: '/api/ai/assist'
  },
  apiKey: String,
  model: {
    type: String,
    default: 'gpt-4'
  },
  showHistory: {
    type: Boolean,
    default: true
  },
  maxHistoryItems: {
    type: Number,
    default: 20
  }
})

const emit = defineEmits(['apply-suggestion', 'translate', 'polish', 'error'])

// Store
const projectStore = useProjectStore()

// State
const activeMode = ref('polish')
const targetLanguage = ref('en')
const inputText = ref('')
const loading = ref(false)
const currentResponse = ref(null)
const history = ref([])

// Mode options
const modeOptions = computed(() => {
  const options = []
  if (props.enablePolish) {
    options.push({ label: '润色', value: 'polish' })
  }
  if (props.enableTranslation) {
    options.push({ label: '翻译', value: 'translate' })
  }
  if (props.enableSuggestion) {
    options.push({ label: '建议', value: 'suggest' })
  }
  return options
})

// Computed
const currentIndex = computed(() => {
  if (!props.context) return -1
  return projectStore.subtitles.findIndex(s => s.id === props.context.id)
})

const inputPlaceholder = computed(() => {
  switch (activeMode.value) {
    case 'polish':
      return '输入需要润色的文本，或使用当前字幕...'
    case 'translate':
      return '输入需要翻译的文本，或使用当前字幕...'
    case 'suggest':
      return '描述您想要的改进方向...'
    default:
      return '输入文本...'
  }
})

const submitButtonText = computed(() => {
  switch (activeMode.value) {
    case 'polish': return '润色'
    case 'translate': return '翻译'
    case 'suggest': return '获取建议'
    default: return '提交'
  }
})

const showComparison = computed(() => {
  return (activeMode.value === 'polish' || activeMode.value === 'translate') &&
         currentResponse.value &&
         props.context
})

const canApply = computed(() => {
  return currentResponse.value &&
         (activeMode.value === 'polish' || activeMode.value === 'translate') &&
         props.context
})

const confidenceType = computed(() => {
  if (!currentResponse.value?.confidence) return 'info'
  const conf = currentResponse.value.confidence
  if (conf >= 0.8) return 'success'
  if (conf >= 0.6) return 'warning'
  return 'danger'
})

// Methods
async function handleSubmit() {
  if (!inputText.value.trim() && !props.context) {
    ElMessage.warning('请输入文本或选择一条字幕')
    return
  }

  const text = inputText.value.trim() || props.context?.text

  loading.value = true
  currentResponse.value = null

  try {
    const response = await fetch(props.apiEndpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(props.apiKey && { 'Authorization': `Bearer ${props.apiKey}` })
      },
      body: JSON.stringify({
        mode: activeMode.value,
        text,
        targetLanguage: targetLanguage.value,
        context: getContext(),
        model: props.model
      })
    })

    if (!response.ok) throw new Error('API请求失败')

    const data = await response.json()
    currentResponse.value = data

    // 添加到历史
    addToHistory({
      mode: activeMode.value,
      request: text,
      response: data,
      preview: text.substring(0, 50) + (text.length > 50 ? '...' : '')
    })

    // 触发相应事件
    switch (activeMode.value) {
      case 'polish':
        emit('polish', data.text)
        break
      case 'translate':
        emit('translate', data.text, targetLanguage.value)
        break
    }
  } catch (error) {
    console.error('AI请求失败:', error)
    ElMessage.error('AI请求失败，请稍后重试')
    emit('error', error)
  } finally {
    loading.value = false
  }
}

function getContext() {
  if (!props.context || !props.contextWindow) return null

  const currentIdx = currentIndex.value
  const subtitles = projectStore.subtitles

  const start = Math.max(0, currentIdx - props.contextWindow)
  const end = Math.min(subtitles.length, currentIdx + props.contextWindow + 1)

  return {
    before: subtitles.slice(start, currentIdx).map(s => s.text),
    current: props.context.text,
    after: subtitles.slice(currentIdx + 1, end).map(s => s.text)
  }
}

function applyResponse() {
  if (!canApply.value) return

  projectStore.updateSubtitle(props.context.id, {
    text: currentResponse.value.text
  })

  ElMessage.success('已应用到字幕')
  emit('apply-suggestion', {
    subtitleId: props.context.id,
    oldText: props.context.text,
    newText: currentResponse.value.text
  })

  clearInput()
}

function applySuggestion(suggestion) {
  if (suggestion.action) {
    suggestion.action()
  }

  ElMessage.success('已应用建议')
  emit('apply-suggestion', suggestion)
}

function copyResponse() {
  if (!currentResponse.value) return

  const text = currentResponse.value.text ||
               currentResponse.value.suggestions?.map(s => s.description).join('\n') ||
               ''

  navigator.clipboard.writeText(text)
  ElMessage.success('已复制到剪贴板')
}

async function regenerate() {
  await handleSubmit()
}

function clearInput() {
  inputText.value = ''
  currentResponse.value = null
}

function handleLangChange(lang) {
  targetLanguage.value = lang
}

function getModeLabel(mode) {
  const labels = {
    polish: '润色',
    translate: '翻译',
    suggest: '建议'
  }
  return labels[mode] || mode
}

function addToHistory(item) {
  history.value.unshift({
    id: Date.now(),
    ...item,
    timestamp: Date.now()
  })

  // 限制历史数量
  if (history.value.length > props.maxHistoryItems) {
    history.value = history.value.slice(0, props.maxHistoryItems)
  }

  // 持久化到 localStorage
  localStorage.setItem('ai-assistant-history', JSON.stringify(history.value))
}

function loadHistory(item) {
  inputText.value = item.request
  currentResponse.value = item.response
}

function clearHistory() {
  history.value = []
  localStorage.removeItem('ai-assistant-history')
  ElMessage.success('历史记录已清空')
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = (seconds % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

function formatHistoryTime(timestamp) {
  const now = Date.now()
  const diff = now - timestamp

  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`
  return new Date(timestamp).toLocaleDateString()
}

function renderMarkdown(text) {
  return md.render(text)
}

// 监听上下文变化
watch(() => props.context, (newContext) => {
  if (newContext && !inputText.value) {
    // 可以选择自动填充当前字幕文本
    // inputText.value = newContext.text
  }
})

// 从 localStorage 恢复历史
if (props.showHistory) {
  try {
    const saved = localStorage.getItem('ai-assistant-history')
    if (saved) {
      history.value = JSON.parse(saved)
    }
  } catch (error) {
    console.error('恢复历史记录失败:', error)
  }
}
</script>
```

## 样式定义

```scss
.ai-assistant {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  height: 100%;
  overflow: auto;

  // 工具栏
  .assistant-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-light);
  }

  // 当前上下文
  .current-context {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    border-left: 3px solid var(--primary-color);

    .context-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-size: 13px;
      color: var(--text-muted);
    }

    .context-text {
      font-size: 14px;
      line-height: 1.6;
    }
  }

  // 输入区域
  .input-area {
    .input-actions {
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }
  }

  // 响应区域
  .response-area {
    padding: 16px;
    background: var(--bg-secondary);
    border-radius: 8px;

    .response-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
      font-weight: 500;

      .ai-icon {
        color: var(--primary-color);
      }
    }

    .response-loading {
      padding: 12px;
    }

    .response-content {
      .result-text {
        padding: 12px;
        background: var(--bg-primary);
        border-radius: 6px;
        line-height: 1.8;
        margin-bottom: 12px;
      }

      .suggestions-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 12px;

        .suggestion-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background: var(--bg-primary);
          border-radius: 6px;

          .suggestion-content {
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 1;
          }
        }
      }

      .comparison-view {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding: 12px;
        background: var(--bg-primary);
        border-radius: 6px;

        .comparison-item {
          flex: 1;
          padding: 12px;
          border-radius: 6px;

          &.original {
            background: var(--bg-tertiary);
          }

          &.modified {
            background: var(--color-success-light);
          }

          .comparison-label {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 8px;
          }

          .comparison-text {
            line-height: 1.6;
          }
        }

        .comparison-arrow {
          color: var(--text-muted);
        }
      }

      .response-actions {
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
      }

      .response-explanation {
        margin-top: 12px;
        font-size: 14px;
        color: var(--text-secondary);
      }
    }
  }

  // 历史记录
  .history-section {
    .history-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-weight: 500;
    }

    .history-list {
      display: flex;
      flex-direction: column;
      gap: 8px;

      .history-item {
        padding: 8px 12px;
        background: var(--bg-secondary);
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s;

        &:hover {
          background: var(--bg-hover);
          transform: translateX(4px);
        }

        .history-mode {
          font-size: 12px;
          color: var(--primary-color);
          margin-bottom: 4px;
        }

        .history-preview {
          font-size: 14px;
          color: var(--text-primary);
          margin-bottom: 4px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .history-time {
          font-size: 12px;
          color: var(--text-muted);
        }
      }
    }
  }
}
```

## API 接口

### 请求格式

```typescript
interface AIAssistRequest {
  mode: 'polish' | 'translate' | 'suggest'
  text: string
  targetLanguage?: string    // 仅翻译模式需要
  context?: {                // 可选的上下文
    before: string[]
    current: string
    after: string[]
  }
  model?: string
}
```

### 响应格式

```typescript
interface AIAssistResponse {
  // 润色/翻译模式
  text?: string
  confidence?: number         // 0-1 置信度

  // 建议模式
  suggestions?: Array<{
    type: string
    description: string
    action?: Function
  }>

  // 通用
  explanation?: string       // Markdown 格式的详细解释
  alternatives?: string[]    // 备选方案
}
```

## 与其他组件的关系

### 依赖关系
- 依赖 `ProjectStore` 获取字幕数据
- 使用后端 AI API 服务

### 被依赖关系
- 被 `EditorView` 作为标签页包含
- 可独立使用

## 性能优化

1. **请求防抖**
   - 避免频繁 API 调用

2. **响应缓存**
   - 相同请求复用结果

3. **流式响应**
   - 长文本逐步显示

## 测试要点

1. AI API 集成
2. 建议应用准确性
3. 历史记录持久化
4. 错误处理
5. 多语言翻译

## 未来扩展

1. 支持更多 AI 模型
2. 自定义提示词
3. 批量处理
4. 术语库管理
5. 学习用户偏好