# 任务监控器 UI 系统设计文档 (Task Monitor System Design)

## 1\. 设计理念 (Design Philosophy)

  * **渐进式披露 (Progressive Disclosure)：** 默认仅展示核心信息（正在处理、排队中），次要信息（失败、暂停、完成）折叠收纳，防止信息过载。
  * **关注点分离 (Separation of Concerns)：** 活跃任务（Processing/Queued）强调实时性与操控；历史任务（Finished/Failed）强调归档与批量处理。
  * **即时反馈 (Immediate Feedback)：** 所有的状态变更（完成、拖动、暂停）都必须伴随明确的视觉反馈与平滑动画，建立操作的因果联系。
  * **容错与弹性 (Flexibility)：** 界面需适应任意数量的任务（从0到100+），在有限的窗口高度内通过滚动与折叠保持可用性。

-----

## 2\. 界面布局与视觉架构

### 2.1 整体结构 (Flexbox Layout)

采用“定海神针 + 滚动列表”的结构，确保核心任务永远可见。

  * **Header (固定区)：**
      * 内容：**Processing (正在处理)** 任务卡片。
      * 行为：常驻顶部，不可折叠，不随下方列表滚动。
  * **Body (滚动区)：**
      * 内容：包含 Failed, Queued, Paused, Finished 四个折叠分组。
      * 行为：占据剩余空间 (`flex-1`)，允许 Y 轴滚动 (`overflow-y-auto`)。
  * **Footer (视觉遮罩)：**
      * 内容：**底部渐变遮罩 (Bottom Gradient Mask)**。
      * 样式：`height: 48px`, `pointer-events: none`。
      * 效果：从背景色渐变到透明 (`bg-gradient-to-t from-white to-transparent`)。
      * 作用：代替“查看更多”的文字提示，优雅地暗示下方还有内容，解决“标题栏恰好卡在视口边缘”的视觉尴尬。

### 2.2 分组状态与默认行为

| 分组名称 | 默认状态 | 标题栏设计 | 排序规则 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Processing** | **常驻** | 无标题栏 | N/A | 核心区域，单一任务。 |
| **Failed** | **折叠** 🔴 | 🔴 失败 (n) | **时间倒序** (最新失败在前) | 用户通常在队列结束后批量处理，无需强制展开。 |
| **Queued** | **展开** 🔵 | 🔵 排队中 (n) | **自定义顺序** (拖动) | 用户需频繁调整优先级，默认展开。 |
| **Paused** | **折叠** 🟡 | 🟡 已暂停 (n) | **时间倒序** (最新暂停在前) | 待决策区。最近暂停的恢复概率最高。 |
| **Finished** | **折叠** 🟢 | 🟢 已完成 (n) | **时间倒序** (最新完成在前) | 仅作查阅，避免占据屏幕。 |
| **Canceled** | **移除** ⚪ | 不显示 | N/A | 垃圾回收原则，直接从列表中移除。 |

-----

## 3\. 核心动画与交互细节 (Animation & Interaction)

### 3.1 完美折叠/展开动画 (The Accordion)

解决 `height: auto` 无法做 CSS Transition 的问题，确保在狭小窗口内展开丝滑。

  * **技术方案：** CSS Grid `grid-template-rows`
  * **状态切换：**
      * 折叠态：`grid-rows-[0fr]`
      * 展开态：`grid-rows-[1fr]`
  * **过渡属性：** `transition-[grid-template-rows] duration-300 ease-out`
  * **注意：** 内部容器必须设置 `overflow-hidden`。

### 3.2 任务完成过渡 (Completion Transition)

采用 **“庆祝 - 离场 - 替补”** 三阶段动画，建立任务流转的连贯感。

1.  **Phase 1: 庆祝 (Success State) - 1.0s**
      * 触发：后端推送 `status: finished`。
      * 表现：**原地不动**。进度条变绿 (100%)，Loading 图标变为 ✅ Checkmark，文本变为“已完成”。
2.  **Phase 2: 离场 (Exit) - 0.5s**
      * 表现：Processing 卡片向上滑动并淡出 (`opacity: 0`, `translateY: -20px`)。
      * DOM 操作：DOM 元素在此阶段结束时才真正移动到 Finished 列表。
      * 同时反馈：Finished 分组标题栏的计数器数字跳动 (+1)。
3.  **Phase 3: 替补 (Enter) - 0.5s**
      * 表现：Queued 列表的第一位任务向上滑动 (`translateY: 20px` -\> `0`) 填补 Processing 的空缺，并开始 Loading 动画。

### 3.3 拖动排序与自动滚动 (Auto-Scrolling)

确保长列表中拖动任务时的手感。

  * **交互库推荐：** `Sortable.js` / `Vue.Draggable`
  * **动态热区 (Dynamic Hot Zones)：**
      * **上滚区：** 视口顶部 15% 区域。
      * **下滚区：** 视口底部 15% 区域。
  * **速度控制 (Velocity)：**
      * 基于鼠标距离边缘的距离计算速度。越靠近边缘，滚动越快。
      * **参数建议：** `scrollSensitivity: 80` (像素), `scrollSpeed: 20` (像素/帧)。防止一进入边缘就“光速”滚到底。
  * **视觉反馈：**
      * 被拖拽项：半透明 (`opacity: 0.8`) + 投影 (`shadow-lg`)。
      * 占位项 (Ghost)：虚线边框 (`border-dashed`) + 浅灰背景。

-----

## 4\. 技术实现指南 (Technical Implementation)

### 4.1 数据结构 (Vue Store 示例)

建议使用 `computed` 属性根据状态对主任务列表进行切片。

```javascript
// store/tasks.js

// 核心：所有任务数据
const allTasks = ref([...]); 

// 1. Processing (单项)
const processingTask = computed(() => allTasks.value.find(t => t.status === 'processing'));

// 2. Failed (倒序)
const failedTasks = computed(() => 
  allTasks.value.filter(t => t.status === 'failed')
  .sort((a, b) => new Date(b.failedAt) - new Date(a.failedAt))
);

// 3. Queued (严格遵循后端 Sequence)
const queuedTasks = computed(() => {
  // 假设 backendQueueOrder 是一个 ID 数组 ['id_1', 'id_2']
  return backendQueueOrder.value
    .map(id => allTasks.value.find(t => t.id === id))
    .filter(Boolean);
});

// 4. Paused (倒序)
const pausedTasks = computed(() => 
  allTasks.value.filter(t => t.status === 'paused')
  .sort((a, b) => new Date(b.pausedAt) - new Date(a.pausedAt))
);

// 5. Finished (倒序)
const finishedTasks = computed(() => 
  allTasks.value.filter(t => t.status === 'finished')
  .sort((a, b) => new Date(b.finishedAt) - new Date(a.finishedAt))
);
```

### 4.2 拖动逻辑代码片段

```javascript
// handleDragChange.js
function onDragEnd(event) {
  const { newIndex, oldIndex } = event;
  if (newIndex === oldIndex) return;

  // 1. 乐观更新 (Optimistic UI): 立即在前端重排 queueOrder
  const newOrder = [...currentQueueOrder];
  const [movedItem] = newOrder.splice(oldIndex, 1);
  newOrder.splice(newIndex, 0, movedItem);
  updateLocalQueue(newOrder);

  // 2. 发送请求
  api.reorderQueue(newOrder).catch(() => {
    // 3. 失败回滚
    showToast('排序失败，正在还原...');
    updateLocalQueue(currentQueueOrder);
  });
}
```

### 4.3 CSS Grid 动画组件 (Tailwind)

```html
<template>
  <div class="group border rounded-lg bg-white">
    <button @click="toggle" class="w-full flex justify-between p-3">
      <span>{{ title }} ({{ count }})</span>
      <span :class="isOpen ? 'rotate-180' : ''" class="transition-transform duration-300">▼</span>
    </button>

    <div 
      class="grid transition-[grid-template-rows] duration-300 ease-out"
      :class="isOpen ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'"
    >
      <div class="overflow-hidden">
        <div class="p-3 border-t">
          <slot />
        </div>
      </div>
    </div>
  </div>
</template>
```

-----

## 5\. 总结 (Summary)

本设计通过**分区折叠**解决了空间拥挤问题，通过**智能排序**解决了信息查找效率问题，通过**细腻的过渡动画**解决了状态变化的感知问题。

**关键待办清单 (Checklist):**

  * [ ] 实现 Header 固定 + Body 滚动的 Flex 布局。
  * [ ] 添加底部渐变遮罩层。
  * [ ] 封装基于 Grid 的折叠组件。
  * [ ] 配置 Sortable.js 的滚动热区与速度。
  * [ ] 实现任务完成的 1s 延迟庆祝 + 离场动画逻辑。
  * [ ] 确保“已取消”任务被静默移除。