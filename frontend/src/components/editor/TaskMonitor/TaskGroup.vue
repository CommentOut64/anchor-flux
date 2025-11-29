<template>
  <div class="task-group" :class="`variant-${variant}`">
    <!-- 标题栏 -->
    <div class="group-header" @click="toggleCollapse">
      <div class="header-left">
        <span class="status-dot"></span>
        <span class="group-title">{{ title }}</span>
        <span class="group-count">({{ count }})</span>
      </div>
      <div class="header-right">
        <svg
          class="collapse-icon"
          :class="{ collapsed: isCollapsed }"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M7 10l5 5 5-5z"/>
        </svg>
      </div>
    </div>

    <!-- 内容区（使用 CSS Grid 折叠） -->
    <div
      class="group-content-wrapper"
      :class="{ collapsed: isCollapsed }"
    >
      <div class="group-content">
        <slot></slot>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  title: { type: String, required: true },
  count: { type: Number, default: 0 },
  variant: {
    type: String,
    default: 'default',
    validator: (v) => ['default', 'primary', 'success', 'warning', 'danger'].includes(v)
  },
  defaultCollapsed: { type: Boolean, default: false }
})

const isCollapsed = ref(props.defaultCollapsed)

function toggleCollapse() {
  isCollapsed.value = !isCollapsed.value
}
</script>

<style lang="scss" scoped>
.task-group {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-default);
  border-radius: 8px;
  margin-bottom: 12px;
  overflow: hidden;
  width: 100%;  // 强制宽度为 100%，防止动画导致宽度变化
  box-sizing: border-box;  // 确保 border 和 padding 不影响总宽度
}

.group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  cursor: pointer;
  user-select: none;
  transition: background 0.2s;
  width: 100%;  // 强制宽度
  box-sizing: border-box;

  &:hover {
    background: var(--bg-elevated);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;  // 允许内容压缩
    flex: 1;  // 占据剩余空间
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;  // 不压缩
  }

  .group-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .group-count {
    font-size: 11px;
    color: var(--text-muted);
  }

  .header-right {
    flex-shrink: 0;  // 右侧图标不压缩
  }

  .collapse-icon {
    width: 16px;
    height: 16px;
    color: var(--text-muted);
    transition: transform 0.3s ease;

    &.collapsed {
      transform: rotate(-90deg);
    }
  }
}

// CSS Grid 折叠动画
.group-content-wrapper {
  display: grid;
  grid-template-rows: 1fr;
  transition: grid-template-rows 300ms ease-out;
  width: 100%;  // 强制宽度
  box-sizing: border-box;
  // 防止折叠时内容溢出导致宽度变化
  min-width: 0;
  overflow: hidden;  // 确保内容不会溢出

  &.collapsed {
    grid-template-rows: 0fr;
  }
}

.group-content {
  overflow: hidden;  // 隐藏溢出内容
  // 折叠时移除 padding，防止露出内容
  transition: padding 300ms ease-out;
  padding: 0 12px 12px;
  width: 100%;  // 强制宽度
  box-sizing: border-box;
  // 防止内容溢出导致宽度变化
  min-width: 0;
}

.group-content-wrapper.collapsed .group-content {
  padding: 0;
}

// 变体样式
.variant-primary .status-dot { background: var(--primary); }
.variant-success .status-dot { background: var(--success); }
.variant-warning .status-dot { background: var(--warning); }
.variant-danger .status-dot { background: var(--danger); }
</style>
