# ValidationPanel - 验证面板组件

## 概述

ValidationPanel 是字幕验证和问题检查面板组件，实时显示字幕中的问题（如时间重叠、字数过长等），并提供快速定位和修复功能。它与 ProjectStore 的验证系统深度集成，帮助用户快速发现并解决字幕问题。

## 功能特性

1. **实时问题检测**
   - 时间重叠检查
   - 字数限制警告
   - 格式错误提示
   - 自定义验证规则

2. **问题分类显示**
   - 错误（Error）- 必须修复
   - 警告（Warning）- 建议修复
   - 信息（Info）- 提示建议

3. **快速定位修复**
   - 点击问题跳转到对应字幕
   - 批量修复建议
   - 一键修复功能

4. **统计和报告**
   - 问题统计概览
   - 导出验证报告
   - 历史问题追踪

## 技术依赖

```json
{
  "vue": "^3.4.0",
  "element-plus": "^2.11.0",
  "pinia": "^3.0.0"
}
```

## 组件属性

```typescript
interface ValidationPanelProps {
  // 显示配置
  collapsible?: boolean      // 可折叠，默认true
  defaultExpanded?: boolean   // 默认展开，默认false
  maxHeight?: string         // 最大高度，默认'300px'

  // 验证规则配置
  enableOverlapCheck?: boolean    // 时间重叠检查，默认true
  enableLengthCheck?: boolean     // 字数检查，默认true
  maxTextLength?: number          // 最大字数，默认30
  minGapTime?: number            // 最小间隔时间（秒），默认0.1

  // 自动修复配置
  enableAutoFix?: boolean        // 启用自动修复，默认true
  autoFixOnSave?: boolean        // 保存时自动修复，默认false
}

interface ValidationPanelEmits {
  'issue-click': (issue: ValidationIssue) => void
  'fix-issue': (issue: ValidationIssue) => void
  'fix-all': (type: string) => void
}
```

## 核心实现

```vue
<!-- components/ValidationPanel/index.vue -->
<template>
  <div class="validation-panel">
    <!-- 折叠头部 -->
    <div
      class="panel-header"
      @click="togglePanel"
      :class="{ collapsible }"
    >
      <div class="header-title">
        <el-icon><WarningFilled /></el-icon>
        <span>问题检查</span>
        <el-badge
          :value="totalIssues"
          :type="badgeType"
          v-if="totalIssues > 0"
        />
      </div>
      <el-icon v-if="collapsible" class="collapse-icon" :class="{ expanded }">
        <ArrowDown />
      </el-icon>
    </div>

    <!-- 面板内容 -->
    <el-collapse-transition>
      <div v-show="expanded" class="panel-content">
        <!-- 统计概览 -->
        <div class="statistics">
          <div class="stat-item error" v-if="errorCount > 0">
            <el-icon><CircleCloseFilled /></el-icon>
            <span>{{ errorCount }} 个错误</span>
          </div>
          <div class="stat-item warning" v-if="warningCount > 0">
            <el-icon><WarningFilled /></el-icon>
            <span>{{ warningCount }} 个警告</span>
          </div>
          <div class="stat-item info" v-if="infoCount > 0">
            <el-icon><InfoFilled /></el-icon>
            <span>{{ infoCount }} 个提示</span>
          </div>
          <div class="stat-item success" v-if="totalIssues === 0">
            <el-icon><CircleCheckFilled /></el-icon>
            <span>没有发现问题</span>
          </div>
        </div>

        <!-- 工具栏 -->
        <div class="toolbar" v-if="totalIssues > 0">
          <el-button
            v-if="enableAutoFix && hasFixableIssues"
            @click="fixAllIssues"
            size="small"
            type="primary"
          >
            修复所有可修复问题
          </el-button>
          <el-button
            @click="exportReport"
            size="small"
          >
            导出报告
          </el-button>
          <el-input
            v-model="filterText"
            placeholder="搜索问题..."
            size="small"
            clearable
            style="width: 200px"
          >
            <template #prefix>
              <el-icon><Search /></el-icon>
            </template>
          </el-input>
        </div>

        <!-- 问题列表 -->
        <el-scrollbar :max-height="maxHeight">
          <div class="issues-list">
            <div
              v-for="issue in filteredIssues"
              :key="issue.id"
              class="issue-item"
              :class="issue.severity"
              @click="handleIssueClick(issue)"
            >
              <!-- 问题图标 -->
              <el-icon class="issue-icon">
                <CircleCloseFilled v-if="issue.severity === 'error'" />
                <WarningFilled v-else-if="issue.severity === 'warning'" />
                <InfoFilled v-else />
              </el-icon>

              <!-- 问题信息 -->
              <div class="issue-content">
                <div class="issue-message">
                  {{ issue.message }}
                </div>
                <div class="issue-location">
                  字幕 #{{ issue.index + 1 }}
                  <span v-if="issue.time">
                    ({{ formatTime(issue.time) }})
                  </span>
                </div>
              </div>

              <!-- 操作按钮 -->
              <div class="issue-actions" @click.stop>
                <el-button
                  v-if="issue.fixable && enableAutoFix"
                  @click="fixIssue(issue)"
                  size="small"
                  type="primary"
                  link
                >
                  修复
                </el-button>
                <el-button
                  @click="jumpToSubtitle(issue)"
                  size="small"
                  link
                >
                  跳转
                </el-button>
              </div>
            </div>
          </div>

          <!-- 空状态 -->
          <el-empty
            v-if="filteredIssues.length === 0 && filterText"
            description="没有匹配的问题"
            :image-size="80"
          />
        </el-scrollbar>
      </div>
    </el-collapse-transition>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { ElMessage, ElMessageBox } from 'element-plus'

// Props & Emits
const props = defineProps({
  collapsible: {
    type: Boolean,
    default: true
  },
  defaultExpanded: {
    type: Boolean,
    default: false
  },
  maxHeight: {
    type: String,
    default: '300px'
  },
  enableOverlapCheck: {
    type: Boolean,
    default: true
  },
  enableLengthCheck: {
    type: Boolean,
    default: true
  },
  maxTextLength: {
    type: Number,
    default: 30
  },
  minGapTime: {
    type: Number,
    default: 0.1
  },
  enableAutoFix: {
    type: Boolean,
    default: true
  },
  autoFixOnSave: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['issue-click', 'fix-issue', 'fix-all'])

// Store
const projectStore = useProjectStore()

// State
const expanded = ref(props.defaultExpanded)
const filterText = ref('')

// Computed
const validationErrors = computed(() => {
  const errors = []
  const subtitles = projectStore.subtitles

  // 时间重叠检查
  if (props.enableOverlapCheck) {
    for (let i = 0; i < subtitles.length - 1; i++) {
      if (subtitles[i].end > subtitles[i + 1].start) {
        errors.push({
          id: `overlap-${i}`,
          type: 'overlap',
          severity: 'error',
          index: i,
          message: `与下一条字幕时间重叠 (${formatOverlap(
            subtitles[i].end - subtitles[i + 1].start
          )})`,
          time: subtitles[i].start,
          fixable: true,
          fix: () => fixOverlap(i)
        })
      }

      // 间隔时间检查
      const gap = subtitles[i + 1].start - subtitles[i].end
      if (gap > 0 && gap < props.minGapTime) {
        errors.push({
          id: `gap-${i}`,
          type: 'gap',
          severity: 'warning',
          index: i,
          message: `与下一条字幕间隔过小 (${formatTime(gap)})`,
          time: subtitles[i].start,
          fixable: true,
          fix: () => fixGap(i)
        })
      }
    }
  }

  // 字数检查
  if (props.enableLengthCheck) {
    subtitles.forEach((sub, i) => {
      if (sub.text.length > props.maxTextLength) {
        errors.push({
          id: `length-${i}`,
          type: 'length',
          severity: 'warning',
          index: i,
          message: `文本过长 (${sub.text.length}字，建议不超过${props.maxTextLength}字)`,
          time: sub.start,
          fixable: false
        })
      }

      // 空字幕检查
      if (!sub.text.trim()) {
        errors.push({
          id: `empty-${i}`,
          type: 'empty',
          severity: 'error',
          index: i,
          message: '字幕文本为空',
          time: sub.start,
          fixable: false
        })
      }
    })
  }

  // 时长检查
  subtitles.forEach((sub, i) => {
    const duration = sub.end - sub.start
    if (duration < 0.5) {
      errors.push({
        id: `duration-short-${i}`,
        type: 'duration',
        severity: 'warning',
        index: i,
        message: `显示时间过短 (${duration.toFixed(1)}秒)`,
        time: sub.start,
        fixable: false
      })
    } else if (duration > 7) {
      errors.push({
        id: `duration-long-${i}`,
        type: 'duration',
        severity: 'warning',
        index: i,
        message: `显示时间过长 (${duration.toFixed(1)}秒)`,
        time: sub.start,
        fixable: false
      })
    }
  })

  return errors
})

const filteredIssues = computed(() => {
  if (!filterText.value) return validationErrors.value

  const search = filterText.value.toLowerCase()
  return validationErrors.value.filter(issue =>
    issue.message.toLowerCase().includes(search) ||
    `#${issue.index + 1}`.includes(search)
  )
})

const totalIssues = computed(() => validationErrors.value.length)
const errorCount = computed(() =>
  validationErrors.value.filter(i => i.severity === 'error').length
)
const warningCount = computed(() =>
  validationErrors.value.filter(i => i.severity === 'warning').length
)
const infoCount = computed(() =>
  validationErrors.value.filter(i => i.severity === 'info').length
)

const badgeType = computed(() => {
  if (errorCount.value > 0) return 'danger'
  if (warningCount.value > 0) return 'warning'
  return 'info'
})

const hasFixableIssues = computed(() =>
  validationErrors.value.some(i => i.fixable)
)

// Methods
function togglePanel() {
  if (props.collapsible) {
    expanded.value = !expanded.value
  }
}

function handleIssueClick(issue) {
  jumpToSubtitle(issue)
  emit('issue-click', issue)
}

function jumpToSubtitle(issue) {
  const subtitle = projectStore.subtitles[issue.index]
  if (subtitle) {
    projectStore.view.selectedSubtitleId = subtitle.id
    projectStore.seekTo(subtitle.start)
  }
}

function fixIssue(issue) {
  if (issue.fix) {
    issue.fix()
    ElMessage.success('问题已修复')
    emit('fix-issue', issue)
  }
}

async function fixAllIssues() {
  const fixableIssues = validationErrors.value.filter(i => i.fixable)

  if (fixableIssues.length === 0) {
    ElMessage.warning('没有可自动修复的问题')
    return
  }

  try {
    await ElMessageBox.confirm(
      `将自动修复 ${fixableIssues.length} 个问题，是否继续？`,
      '批量修复',
      {
        confirmButtonText: '修复',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    fixableIssues.forEach(issue => {
      if (issue.fix) issue.fix()
    })

    ElMessage.success(`已修复 ${fixableIssues.length} 个问题`)
    emit('fix-all', 'all')
  } catch {
    // User cancelled
  }
}

function fixOverlap(index) {
  const current = projectStore.subtitles[index]
  const next = projectStore.subtitles[index + 1]

  // 调整当前字幕结束时间
  projectStore.updateSubtitle(current.id, {
    end: next.start - 0.1
  })
}

function fixGap(index) {
  const current = projectStore.subtitles[index]
  const next = projectStore.subtitles[index + 1]

  // 扩展当前字幕结束时间
  projectStore.updateSubtitle(current.id, {
    end: next.start - props.minGapTime
  })
}

function exportReport() {
  const report = {
    generated: new Date().toISOString(),
    totalSubtitles: projectStore.subtitles.length,
    issues: validationErrors.value.map(issue => ({
      type: issue.type,
      severity: issue.severity,
      subtitle: issue.index + 1,
      message: issue.message
    }))
  }

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `validation-report-${Date.now()}.json`
  a.click()
  URL.revokeObjectURL(url)

  ElMessage.success('报告已导出')
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = (seconds % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

function formatOverlap(seconds) {
  return `${Math.abs(seconds).toFixed(1)}秒`
}

// Auto-expand on errors
watch(errorCount, (count) => {
  if (count > 0 && !expanded.value) {
    expanded.value = true
  }
})
</script>
```

## 样式定义

```scss
.validation-panel {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    user-select: none;

    &.collapsible {
      cursor: pointer;

      &:hover {
        background: var(--bg-hover);
      }
    }

    .header-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 500;
    }

    .collapse-icon {
      transition: transform 0.3s;

      &.expanded {
        transform: rotate(180deg);
      }
    }
  }

  .panel-content {
    padding: 16px;

    .statistics {
      display: flex;
      gap: 16px;
      margin-bottom: 16px;

      .stat-item {
        display: flex;
        align-items: center;
        gap: 4px;
        font-size: 14px;

        &.error { color: var(--color-danger); }
        &.warning { color: var(--color-warning); }
        &.info { color: var(--color-info); }
        &.success { color: var(--color-success); }
      }
    }

    .toolbar {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-light);
    }

    .issues-list {
      display: flex;
      flex-direction: column;
      gap: 8px;

      .issue-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        background: var(--bg-primary);
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s;

        &:hover {
          transform: translateX(4px);
        }

        &.error {
          border-left: 3px solid var(--color-danger);
          .issue-icon { color: var(--color-danger); }
        }

        &.warning {
          border-left: 3px solid var(--color-warning);
          .issue-icon { color: var(--color-warning); }
        }

        &.info {
          border-left: 3px solid var(--color-info);
          .issue-icon { color: var(--color-info); }
        }

        .issue-content {
          flex: 1;

          .issue-message {
            font-size: 14px;
            color: var(--text-primary);
          }

          .issue-location {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
          }
        }

        .issue-actions {
          display: flex;
          gap: 8px;
        }
      }
    }
  }
}
```

## 与其他组件的关系

### 依赖关系
- 强依赖 `ProjectStore` 的字幕数据和验证系统
- 使用 Element Plus 的 UI 组件

### 被依赖关系
- 被 `EditorView` 包含使用
- 与 `SubtitleList` 共享错误状态显示

## 性能优化

1. **计算属性缓存**
   - Vue 自动缓存验证结果
   - 只在字幕变化时重新计算

2. **虚拟滚动**
   - 问题列表超过50条时考虑虚拟滚动

3. **防抖搜索**
   - 搜索输入使用防抖处理

## 测试要点

1. 各种问题检测准确性
2. 自动修复功能正确性
3. 跳转定位功能
4. 批量修复安全性
5. 导出功能完整性

## 未来扩展

1. 自定义验证规则
2. AI 辅助修复建议
3. 问题优先级排序
4. 实时协作冲突检测
5. 批量智能调整