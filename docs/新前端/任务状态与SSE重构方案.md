# 任务状态与 SSE 通信架构重构方案

> 目标：彻底修复动态进度区卡住、暂停/取消无响应、恢复任务状态错误等问题，并实现任务队列拖动排序
> 原则：在保证彻底修复的前提下，最小化代码改动

---

## 一、问题根源分析

### 1.1 SSE 频道错位（最严重）

| 操作 | 后端推送频道 | 前端监听频道 | 结果 |
|------|-------------|-------------|------|
| `pause_job()` | `global` | `job:{jobId}` | 收不到 |
| `cancel_job()` | `global` | `job:{jobId}` | 收不到 |
| Worker 任务结束 | `job:{jobId}` | `job:{jobId}` | 正常 |

**核心矛盾**：
- `pause_job()` / `cancel_job()` 在 HTTP 请求处理中调用 `_notify_job_status()`，只推送到 `global` 频道
- 但前端 EditorView 监听的是单任务频道 `job:{jobId}`，所以收不到暂停/取消信号
- 只有 Worker 线程的 `finally` 块会推送到单任务频道，但此时任务已经结束

### 1.2 状态更新不同步

**后端 `pause_job()` 的问题**（`job_queue_service.py:108-141`）：

```python
if job_id == self.running_job_id:
    job.paused = True          # ✓ 设置标志
    job.message = "暂停中..."   # ✓ 设置消息
    # ✗ 没有更新 job.status！仍然是 "processing"
```

**后果**：

- `_notify_job_status(job_id, job.status)` 发送的是 `"processing"` 而非 `"paused"`
- 前端即使收到消息，状态也是错的

### 1.3 恢复任务 API 语义混乱

**当前实现**：

| 场景 | 前端调用 | 后端行为 | 问题 |
|------|---------|---------|------|
| 恢复暂停任务 | `restoreJob()` | 从 checkpoint 加载 | 语义错误，应重新入队 |
| 断点续传 | `restoreJob()` | 从 checkpoint 加载 | 正确 |

**前端硬编码状态**（`EditorView.vue:610-623`）：
```javascript
async function resumeTranscription() {
  await transcriptionApi.restoreJob(props.jobId)
  taskStatus.value = 'processing'  // 错误：应该是 'queued'
}
```

### 1.4 问题影响链路

```
用户点击暂停
    ↓
前端调用 pauseJob API
    ↓
后端设置 job.paused = true（但 status 仍是 processing）
    ↓
后端推送到 global 频道
    ↓
前端乐观更新为 paused
    ↓
SSE 单任务频道无消息
    ↓
如果 HTTP 响应慢或失败，前端状态永远不一致
    ↓
进度条卡住，按钮无响应
```

---

## 二、设计目标

1. **SSE 信号可靠送达**：暂停/取消等关键操作必须推送到对应的单任务频道
2. **状态机一致性**：后端 `job.status` 必须及时、准确地反映真实状态
3. **API 语义清晰**：区分"恢复暂停任务"和"断点续传"两种场景
4. **前端响应式更新**：状态由后端 SSE 事件驱动，避免硬编码
5. **最小改动原则**：只修改必要的代码路径

---

## 三、架构设计

### 3.1 统一 SSE 推送策略

**新规则**：所有状态变更必须同时推送到两个频道

```
状态变更 ──┬──→ global 频道（队列监控、TaskMonitor）
           └──→ job:{jobId} 频道（EditorView）
```

### 3.2 任务状态机

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
created ──→ queued ──→ processing ──→ finished                │
              │            │                                  │
              │            │ (pause)                          │
              │            ▼                                  │
              │         paused ────────────────────────────→ (resume)
              │            │
              │            │ (cancel)
              ▼            ▼
           canceled ◄────────
```

**状态转换触发点**：

| 触发操作 | 状态变化 | SSE 事件 |
|---------|---------|----------|
| 加入队列 | → `queued` | `job_status` |
| 开始执行 | → `processing` | `job_status` |
| 暂停（正在执行） | → `paused` | `signal: job_paused` |
| 暂停（排队中） | → `paused` | `job_status` |
| 恢复暂停 | → `queued` | `job_status` |
| 取消 | → `canceled` | `signal: job_canceled` |
| 完成 | → `finished` | `signal: job_complete` |
| 失败 | → `failed` | `signal: job_failed` |

### 3.3 新增 API：恢复暂停任务

```
POST /api/resume/{job_id}
```

**行为**：
1. 检查任务状态是否为 `paused`
2. 将任务状态改为 `queued`
3. 将任务加入队列尾部（或指定位置）
4. 推送 SSE 通知

**与 `restore-job` 的区别**：

| API | 用途 | 状态变化 |
|-----|------|---------|
| `POST /api/resume/{job_id}` | 恢复暂停的任务 | `paused` → `queued` |
| `POST /api/restore-job/{job_id}` | 从 checkpoint 断点续传 | 根据 checkpoint |

---

## 四、具体修改方案

### 4.1 后端修改

#### 4.1.1 `job_queue_service.py` - 修复 `pause_job()`

**位置**：第 108-141 行

**修改内容**：
```python
def pause_job(self, job_id: str) -> bool:
    job = self.jobs.get(job_id)
    if not job:
        return False

    with self.lock:
        if job_id == self.running_job_id:
            # 正在执行的任务：设置暂停标志
            job.paused = True
            job.status = "paused"  # [新增] 立即更新状态
            job.message = "暂停中..."
            logger.info(f"设置暂停标志: {job_id}")
        elif job_id in self.queue:
            # 排队中的任务：直接移除并标记
            self.queue.remove(job_id)
            job.status = "paused"
            job.message = "已暂停（未开始）"
            logger.info(f"从队列移除: {job_id}")

    self._save_state()
    self._notify_queue_change()
    self._notify_job_status(job_id, job.status)

    # [新增] 同时推送到单任务频道
    self._notify_job_signal(job_id, "job_paused")

    return True
```

#### 4.1.2 `job_queue_service.py` - 修复 `cancel_job()`

**位置**：第 143-193 行

**修改内容**：在函数末尾添加单任务频道推送：

```python
def cancel_job(self, job_id: str, delete_data: bool = False) -> bool:
    # ... 现有逻辑 ...

    self._save_state()
    self._notify_queue_change()
    self._notify_job_status(job_id, job.status)

    # [新增] 同时推送到单任务频道
    self._notify_job_signal(job_id, "job_canceled")

    return result
```

#### 4.1.3 `job_queue_service.py` - 新增 `_notify_job_signal()` 方法

**位置**：在 `_notify_job_progress()` 后面添加（约第 447 行后）

```python
def _notify_job_signal(self, job_id: str, signal: str):
    """推送关键信号到单任务SSE频道"""
    job = self.jobs.get(job_id)
    if not job:
        return

    data = {
        "signal": signal,
        "job_id": job_id,
        "status": job.status,
        "message": job.message,
        "percent": round(job.progress, 1)
    }

    self.sse_manager.broadcast_sync(f"job:{job_id}", "signal", data)
    logger.debug(f"[单任务SSE] 推送信号: {job_id[:8]}... -> {signal}")
```

#### 4.1.4 `job_queue_service.py` - 新增 `resume_job()` 方法

**位置**：在 `pause_job()` 后面添加（约第 142 行后）

```python
def resume_job(self, job_id: str) -> bool:
    """
    恢复暂停的任务（重新加入队列）

    Args:
        job_id: 任务ID

    Returns:
        bool: 是否成功
    """
    job = self.jobs.get(job_id)
    if not job:
        return False

    if job.status != "paused":
        logger.warning(f"任务未暂停，无法恢复: {job_id}, status={job.status}")
        return False

    with self.lock:
        # 重新加入队列
        if job_id not in self.queue:
            self.queue.append(job_id)

        job.status = "queued"
        job.paused = False
        job.message = f"已恢复，排队中 (位置: {len(self.queue)})"
        logger.info(f"恢复暂停任务: {job_id}")

    self._save_state()
    self._notify_queue_change()
    self._notify_job_status(job_id, job.status)

    # 推送到单任务频道
    self._notify_job_signal(job_id, "job_resumed")

    return True
```

#### 4.1.5 `transcription_routes.py` - 新增 `/resume` 路由

**位置**：在 `/pause` 路由后面添加（约第 333 行后）

```python
@router.post("/resume/{job_id}")
async def resume_job(job_id: str):
    """恢复暂停的任务（重新加入队列）"""
    queue_service = get_queue_service()
    ok = queue_service.resume_job(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="无法恢复任务（任务未暂停或不存在）")

    job = queue_service.get_job(job_id)
    return {
        "job_id": job_id,
        "resumed": True,
        "status": job.status if job else "queued",
        "queue_position": list(queue_service.queue).index(job_id) + 1 if job_id in queue_service.queue else 0
    }
```

---

### 4.2 前端修改

#### 4.2.1 `api/transcription.js` - 新增 `resumeJob()` API

**修改内容**：添加新的 API 方法

```javascript
// 恢复暂停的任务（重新加入队列）
export async function resumeJob(jobId) {
  const response = await fetch(`${API_BASE}/api/resume/${jobId}`, {
    method: 'POST'
  })
  if (!response.ok) throw new Error('恢复任务失败')
  return response.json()
}
```

#### 4.2.2 `EditorView.vue` - 修复 `resumeTranscription()`

**位置**：第 610-623 行

**修改内容**：

```javascript
async function resumeTranscription() {
  try {
    // [修改] 使用新的 resumeJob API
    const result = await transcriptionApi.resumeJob(props.jobId)

    // [修改] 根据后端返回值设置状态
    taskStatus.value = result.status || 'queued'
    isTranscribing.value = result.status === 'processing'

    taskStore.updateTaskStatus(props.jobId, result.status || 'queued')
    console.log('[EditorView] 任务已恢复，状态:', result.status)
  } catch (error) {
    console.error('恢复任务失败:', error)
  }
}
```

#### 4.2.3 `sseChannelManager.js` - 新增 `job_resumed` 信号处理

**位置**：第 108-124 行的 `signal` 处理中

**修改内容**：

```javascript
signal: (data) => {
  const signal = data.signal || data.code
  console.log(`[SSE Job ${jobId}] 信号:`, signal)

  if (signal === 'job_complete') {
    handlers.onComplete?.(data)
  } else if (signal === 'job_failed') {
    handlers.onFailed?.(data)
  } else if (signal === 'job_paused') {
    handlers.onPaused?.(data)
  } else if (signal === 'job_canceled') {
    handlers.onCanceled?.(data)
  } else if (signal === 'job_resumed') {  // [新增]
    handlers.onResumed?.(data)
  }

  handlers.onSignal?.(signal, data)
}
```

#### 4.2.4 `EditorView.vue` - SSE 订阅增加信号处理

**位置**：SSE 订阅回调处（需找到 `subscribeJob` 调用处）

**修改内容**：添加 `onPaused`、`onCanceled`、`onResumed` 回调

```javascript
const unsubscribe = sseChannelManager.subscribeJob(props.jobId, {
  onProgress: (data) => {
    taskProgress.value = data.percent
    taskPhase.value = data.phase
    // ...
  },
  onComplete: (data) => {
    taskStatus.value = 'finished'
    isTranscribing.value = false
    // ...
  },
  // [新增] 暂停信号处理
  onPaused: (data) => {
    console.log('[EditorView] 收到暂停信号:', data)
    taskStatus.value = 'paused'
    isTranscribing.value = false
    taskStore.updateTaskStatus(props.jobId, 'paused')
  },
  // [新增] 取消信号处理
  onCanceled: (data) => {
    console.log('[EditorView] 收到取消信号:', data)
    taskStatus.value = 'canceled'
    isTranscribing.value = false
    taskStore.updateTaskStatus(props.jobId, 'canceled')
  },
  // [新增] 恢复信号处理
  onResumed: (data) => {
    console.log('[EditorView] 收到恢复信号:', data)
    taskStatus.value = data.status || 'queued'
    isTranscribing.value = data.status === 'processing'
    taskStore.updateTaskStatus(props.jobId, data.status || 'queued')
  }
})
```

#### 4.2.5 `TaskMonitor/index.vue` - 修复 `resumeTask()`

**位置**：第 236-244 行

**修改内容**：

```javascript
async function resumeTask(jobId) {
  try {
    // [修改] 使用新的 resumeJob API
    const result = await transcriptionApi.resumeJob(jobId)

    // [修改] 根据后端返回值设置状态
    taskStore.updateTaskStatus(jobId, result.status || 'queued')
  } catch (error) {
    console.error('恢复任务失败:', error)
  }
}
```

---

## 五、修改清单汇总

### 后端（2 个文件 / 5 处修改）

| 文件 | 位置 | 修改类型 | 说明 |
|------|------|---------|------|
| `job_queue_service.py` | 108-141 | 修改 | `pause_job()` 增加状态更新和信号推送 |
| `job_queue_service.py` | 143-193 | 修改 | `cancel_job()` 增加信号推送 |
| `job_queue_service.py` | ~142 | 新增 | `resume_job()` 方法 |
| `job_queue_service.py` | ~447 | 新增 | `_notify_job_signal()` 方法 |
| `transcription_routes.py` | ~333 | 新增 | `/resume/{job_id}` 路由 |

> 注：拖动排序后端已完整支持，无需修改

### 前端（4 个文件 / 6 处修改 + 1 依赖）

| 文件 | 位置 | 修改类型 | 说明 |
|------|------|---------|------|
| `transcriptionApi.js` | - | 新增 | `resumeJob()` API 方法 |
| `sseChannelManager.js` | 108-124 | 修改 | 增加 `job_resumed` 信号处理 |
| `EditorView.vue` | 610-623 | 修改 | `resumeTranscription()` 改用新 API |
| `EditorView.vue` | SSE订阅 | 修改 | 增加 `onPaused/onCanceled/onResumed` 回调 |
| `TaskMonitor/index.vue` | 236-244 | 修改 | `resumeTask()` 改用新 API |
| `TaskMonitor/index.vue` | 全文 | 新增 | 拖动排序功能（vuedraggable） |
| `package.json` | dependencies | 新增 | `vuedraggable@next` 依赖 |

---

## 六、验证测试用例

### 6.1 暂停功能

1. **正在转录时暂停**
   - 点击暂停按钮
   - 预期：进度条显示"已暂停"，按钮变为恢复按钮
   - 验证：SSE 收到 `signal: job_paused`

2. **排队中暂停**
   - 在 TaskMonitor 中暂停排队任务
   - 预期：任务从队列移除，状态变为"已暂停"

### 6.2 恢复功能

1. **恢复暂停任务**
   - 点击恢复按钮
   - 预期：状态变为"排队中"（非"转录中"）
   - 验证：任务加入队列尾部

2. **恢复后有其他任务在执行**
   - 任务 A 在执行，任务 B 暂停后恢复
   - 预期：任务 B 状态为"排队中"，等待 A 完成后才开始

### 6.3 取消功能

1. **取消正在执行的任务**
   - 点击取消按钮
   - 预期：状态变为"已取消"，SSE 收到 `signal: job_canceled`

### 6.4 进度显示

1. **进度实时更新**
   - 转录过程中观察进度条
   - 预期：进度条平滑更新，不卡顿

2. **暂停后进度保持**
   - 暂停任务后
   - 预期：进度保持在暂停时的值，不归零

### 6.5 拖动排序

1. **基本排序**
   - 有 2 个以上排队任务时，拖动任务 B 到任务 A 前面
   - 预期：队列顺序更新为 [B, A]，后端返回成功

2. **拖动反馈**
   - 拖动过程中
   - 预期：被拖动元素有阴影和缩放效果，目标位置显示占位符

3. **SSE 同步**
   - 拖动排序后
   - 预期：所有监听 global 频道的客户端都能收到 queue_update 事件

4. **排序失败回滚**
   - 模拟网络错误
   - 预期：列表恢复到拖动前的顺序

5. **单任务禁用**
   - 只有 1 个排队任务
   - 预期：拖动功能禁用，无法拖动

---

## 七、风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| SSE 连接断开 | 状态同步丢失 | 已有重连机制 + 初始状态回调 |
| 并发暂停/恢复 | 状态竞争 | 后端使用锁保护 |
| checkpoint 损坏 | 无法断点续传 | 与恢复暂停任务解耦，互不影响 |

---

## 八、实施顺序

### 阶段一：SSE 信号修复（核心问题）

1. **第一步**：后端新增 `_notify_job_signal()` 方法
2. **第二步**：后端修改 `pause_job()` 和 `cancel_job()`
3. **第三步**：后端新增 `resume_job()` 方法和路由
4. **第四步**：前端新增 `resumeJob()` API
5. **第五步**：前端修改 SSE 信号处理
6. **第六步**：前端修改 `resumeTranscription()` 和 `resumeTask()`
7. **第七步**：测试验证

### 阶段二：拖动排序功能

8. **第八步**：安装 `vuedraggable@next` 依赖
9. **第九步**：修改 TaskMonitor 组件，实现拖动排序
10. **第十步**：测试拖动排序和 SSE 同步

---

## 九、任务队列拖动排序功能

### 9.1 功能需求

用户可以在 TaskMonitor 组件中通过拖动任务项来改变队列顺序，实现自定义优先级调整。

**交互设计**：
- 仅 `queued`（排队中）状态的任务可以拖动排序
- `processing`（处理中）、`paused`（已暂停）、`finished`（已完成）的任务不参与拖动
- 拖动时显示视觉反馈（阴影、占位符）
- 释放后立即同步到后端

### 9.2 现有基础设施

**后端已支持**（无需修改）：

| 组件 | 位置 | 功能 |
|------|------|------|
| `reorder_queue()` | `job_queue_service.py:691-729` | 重排队列顺序 |
| `POST /api/reorder-queue` | `transcription_routes.py` | HTTP 路由 |
| `reorderQueue()` | `transcriptionApi.js:154-158` | 前端 API |

后端 `reorder_queue()` 逻辑：
```python
def reorder_queue(self, job_ids: list) -> bool:
    with self.lock:
        # 验证所有job_id都在队列中
        current_queue_set = set(self.queue)
        new_queue_set = set(job_ids)
        if current_queue_set != new_queue_set:
            return False  # 任务ID不匹配

        # 更新队列顺序
        self.queue.clear()
        for job_id in job_ids:
            self.queue.append(job_id)

    self._save_state()
    self._notify_queue_change()  # 推送SSE通知
    return True
```

### 9.3 前端实现方案

#### 方案选择：vuedraggable（推荐）

使用 `vuedraggable` 库（基于 Sortable.js）实现拖动排序：

```bash
npm install vuedraggable@next
```

#### 9.3.1 TaskMonitor 组件修改

**位置**：`frontend/src/components/editor/TaskMonitor/index.vue`

**模板修改**：

```vue
<template>
  <div class="task-monitor">
    <!-- 头部保持不变 -->
    <div class="monitor-header">
      <h3>后台任务</h3>
      <span class="task-count">{{ filteredTasks.length }}</span>
    </div>

    <div class="task-list" v-if="filteredTasks.length > 0">
      <!-- [新增] 可拖动的排队任务区域 -->
      <draggable
        v-model="queuedTasks"
        :disabled="queuedTasks.length < 2"
        item-key="job_id"
        handle=".drag-handle"
        ghost-class="task-item--ghost"
        drag-class="task-item--dragging"
        @end="onDragEnd"
        class="draggable-list"
      >
        <template #item="{ element: task }">
          <div
            class="task-item is-queued"
            :class="{ 'is-current': task.job_id === currentJobId }"
          >
            <!-- [新增] 拖动手柄 -->
            <div class="drag-handle" title="拖动排序">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M11 18c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm-2-8c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0-6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm6 4c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
              </svg>
            </div>

            <!-- 任务内容（复用现有结构） -->
            <div class="task-info">
              <!-- ... 现有任务信息内容 ... -->
            </div>

            <!-- 操作按钮 -->
            <div class="task-actions">
              <!-- ... 现有操作按钮 ... -->
            </div>
          </div>
        </template>
      </draggable>

      <!-- 非排队状态的任务（不可拖动） -->
      <div
        v-for="task in nonQueuedTasks"
        :key="task.job_id"
        class="task-item"
        :class="{
          'is-current': task.job_id === currentJobId,
          'is-finished': task.status === 'finished',
          'is-processing': task.status === 'processing',
          'is-paused': task.status === 'paused'
        }"
      >
        <!-- 任务内容（复用现有结构） -->
      </div>
    </div>

    <!-- 其他保持不变 -->
  </div>
</template>
```

**脚本修改**：

```javascript
import { ref, computed, watch } from 'vue'
import draggable from 'vuedraggable'
import { transcriptionApi } from '@/services/api'

// 组件注册
const components = { draggable }

// 分离排队任务和非排队任务
const queuedTasks = ref([])

const nonQueuedTasks = computed(() => {
  return filteredTasks.value.filter(t => t.status !== 'queued')
})

// 监听 store 变化，更新可拖动列表
watch(
  () => taskStore.tasks,
  (tasks) => {
    queuedTasks.value = tasks.filter(t => t.status === 'queued')
  },
  { immediate: true, deep: true }
)

// 拖动结束处理
async function onDragEnd(event) {
  if (event.oldIndex === event.newIndex) return

  // 获取新的任务ID顺序
  const newOrder = queuedTasks.value.map(t => t.job_id)

  console.log('[TaskMonitor] 队列重排:', newOrder)

  try {
    const result = await transcriptionApi.reorderQueue(newOrder)
    if (!result.reordered) {
      console.error('[TaskMonitor] 队列重排失败')
      // 恢复原顺序
      queuedTasks.value = taskStore.tasks.filter(t => t.status === 'queued')
    }
  } catch (error) {
    console.error('[TaskMonitor] 队列重排请求失败:', error)
    // 恢复原顺序
    queuedTasks.value = taskStore.tasks.filter(t => t.status === 'queued')
  }
}
```

**样式修改**：

```scss
// 拖动手柄
.drag-handle {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  cursor: grab;
  flex-shrink: 0;
  opacity: 0.5;
  transition: opacity 0.2s;

  svg {
    width: 16px;
    height: 16px;
  }

  &:hover {
    opacity: 1;
    color: var(--text-secondary);
  }

  &:active {
    cursor: grabbing;
  }
}

// 拖动中的元素样式
.task-item--dragging {
  opacity: 0.9;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  transform: scale(1.02);
  z-index: 100;
}

// 占位符样式
.task-item--ghost {
  opacity: 0.3;
  background: var(--bg-tertiary);
  border: 2px dashed var(--border-default);
}

// 排队中的任务显示拖动手柄
.task-item.is-queued {
  display: flex;
  gap: 8px;

  .drag-handle {
    display: flex;
  }
}

// 非排队任务隐藏拖动手柄（如果有的话）
.task-item:not(.is-queued) {
  .drag-handle {
    display: none;
  }
}

// 可拖动列表容器
.draggable-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
```

### 9.4 任务分组显示

为了更好的用户体验，建议将任务按状态分组显示：

```
┌──────────────────────────────┐
│ 后台任务                    3 │
├──────────────────────────────┤
│ ▼ 正在处理 (1)               │
│   [任务A] 转录中 45%         │
├──────────────────────────────┤
│ ▼ 排队中 (2) 可拖动排序      │
│   ⋮⋮ [任务B] 排队中 #1      │
│   ⋮⋮ [任务C] 排队中 #2      │
├──────────────────────────────┤
│ ▼ 已暂停 (1)                 │
│   [任务D] 已暂停             │
├──────────────────────────────┤
│ ▼ 已完成 (2)                 │
│   [任务E] 完成               │
│   [任务F] 完成               │
└──────────────────────────────┘
```

### 9.5 SSE 同步机制

拖动排序后，后端会通过 `_notify_queue_change()` 推送更新：

```
用户拖动任务B到任务C前面
    ↓
前端调用 reorderQueue([B, C])
    ↓
后端 reorder_queue() 更新队列顺序
    ↓
后端推送 queue_update 事件到 global 频道
    ↓
前端 SSE 收到队列更新
    ↓
taskStore 自动同步最新队列
```

**前端已有的 SSE 订阅**（`subscribeGlobal`）会自动处理队列更新：

```javascript
// sseChannelManager.js
queue_update: (data) => {
  console.log('[SSE Global] 队列更新:', data)
  handlers.onQueueUpdate?.(data.queue)
}
```

### 9.6 修改清单（拖动排序）

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `TaskMonitor/index.vue` | 修改 | 引入 vuedraggable，实现拖动逻辑 |
| `package.json` | 依赖 | 添加 `vuedraggable@next` |
| 后端 | 无 | 已有完整支持 |

### 9.7 交互细节

1. **视觉反馈**
   - 拖动时任务卡片轻微放大 + 阴影
   - 目标位置显示虚线占位符
   - 鼠标变为抓取手势

2. **防误操作**
   - 只有"排队中"的任务可拖动
   - 少于 2 个排队任务时禁用拖动
   - 拖动失败时自动回滚

3. **队列位置显示**
   - 每个排队任务显示队列位置 `#1`、`#2`
   - 拖动后位置编号自动更新

---

## 十、附录：关键代码位置索引

```
backend/
├── app/
│   ├── services/
│   │   └── job_queue_service.py
│   │       ├── pause_job()           # 108-141
│   │       ├── cancel_job()          # 143-193
│   │       ├── reorder_queue()       # 691-729 (拖动排序，已实现)
│   │       ├── _notify_job_status()  # 410-427
│   │       └── _notify_job_progress()# 429-446
│   └── api/routes/
│       └── transcription_routes.py
│           ├── /pause/{job_id}       # 325-332
│           ├── /reorder-queue        # (已实现)
│           └── /restore-job/{job_id} # 702-709

frontend/
├── src/
│   ├── services/
│   │   ├── api/
│   │   │   └── transcriptionApi.js
│   │   │       └── reorderQueue()    # 154-158 (已实现)
│   │   └── sseChannelManager.js
│   │       └── subscribeJob()        # 93-154
│   ├── views/
│   │   └── EditorView.vue
│   │       ├── pauseTranscription()  # 596-608
│   │       ├── resumeTranscription() # 610-623
│   │       └── cancelTranscription() # 625-639
│   └── components/editor/
│       ├── EditorHeader.vue          # 暂停/恢复/取消按钮
│       └── TaskMonitor/index.vue
│           ├── resumeTask()          # 236-244
│           └── onDragEnd()           # [待实现] 拖动排序
```
