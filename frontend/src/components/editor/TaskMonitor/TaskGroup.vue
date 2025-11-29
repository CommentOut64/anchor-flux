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
}

.group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
  transition: background 0.2s;

  &:hover {
    background: var(--bg-elevated);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-muted);
  }

  .group-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .group-count {
    font-size: 12px;
    color: var(--text-muted);
  }

  .collapse-icon {
    width: 20px;
    height: 20px;
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

  &.collapsed {
    grid-template-rows: 0fr;
  }
}

.group-content {
  overflow: hidden;
  padding: 0 12px 12px;
}

// 变体样式
.variant-primary .status-dot { background: var(--primary); }
.variant-success .status-dot { background: var(--success); }
.variant-warning .status-dot { background: var(--warning); }
.variant-danger .status-dot { background: var(--danger); }
</style>
