<template>
  <div class="task-list-view">
    <!-- 顶部导航栏 -->
    <header class="task-header">
      <div class="header-left">
        <h1 class="app-title">
          <svg class="app-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM5 10h9v2H5zm0-3h9v2H5zm0 6h6v2H5z"/>
          </svg>
          VideoSRT
        </h1>
      </div>
      <div class="header-right">
        <el-button type="primary" @click="showUploadDialog = true">
          <el-icon><Upload /></el-icon>
          上传视频
        </el-button>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="task-main">
      <!-- 空状态 -->
      <div v-if="tasks.length === 0" class="empty-state">
        <svg class="empty-icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14z"/>
        </svg>
        <h2 class="empty-title">还没有任务</h2>
        <p class="empty-desc">点击上方"上传视频"按钮开始创建字幕任务</p>
        <el-button type="primary" size="large" @click="showUploadDialog = true">
          <el-icon><Upload /></el-icon>
          上传视频
        </el-button>
      </div>

      <!-- 任务列表 -->
      <div v-else class="task-grid">
        <div
          v-for="task in tasks"
          :key="task.job_id"
          class="task-card"
          :class="`status-${task.status}`"
          @click="handleTaskClick(task)"
        >
          <!-- 视频缩略图 -->
          <div class="task-thumbnail">
            <img
              v-if="thumbnailCache[task.job_id] && thumbnailCache[task.job_id] !== null"
              :src="thumbnailCache[task.job_id]"
              class="thumbnail-image"
              alt="Video thumbnail"
            />
            <div
              v-else
              class="thumbnail-placeholder"
              :class="{ 'clickable': thumbnailCache[task.job_id] === null }"
              @click.stop="thumbnailCache[task.job_id] === null && getThumbnailUrl(task.job_id, true)"
              :title="thumbnailCache[task.job_id] === null ? '点击重试' : ''"
            >
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
              </svg>
            </div>
            <!-- 缩略图加载中 -->
            <div
              v-if="thumbnailCache[task.job_id] === undefined"
              class="thumbnail-loading"
            >
              <svg class="loading-spinner" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" fill="none"/>
              </svg>
            </div>
            <div v-if="task.status !== 'finished'" class="status-overlay">
              <span class="status-text">{{ getStatusText(task.status) }}</span>
            </div>
          </div>

          <!-- 任务信息 -->
          <div class="task-info">
            <h3 class="task-title" :title="task.filename">{{ getTaskDisplayName(task.filename) }}</h3>
            <div class="task-meta">
              <span class="meta-item">
                <el-icon><Clock /></el-icon>
                {{ formatDate(task.createdAt) }}
              </span>
              <span v-if="task.status === 'processing' || task.status === 'queued'" class="meta-item">
                <el-icon><Loading /></el-icon>
                {{ task.progress }}%
              </span>
            </div>

            <!-- 进度条 -->
            <el-progress
              v-if="task.status === 'processing' || task.status === 'queued'"
              :percentage="task.progress"
              :show-text="false"
              :stroke-width="4"
            />
          </div>

          <!-- 操作按钮 -->
          <div class="task-actions" @click.stop>
            <el-button
              type="primary"
              size="small"
              @click="openEditor(task.job_id)"
            >
              <el-icon><Edit /></el-icon>
              {{ task.status === 'finished' ? '编辑' : '查看' }}
            </el-button>
            <el-button
              size="small"
              @click="deleteTask(task.job_id)"
            >
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
          </div>
        </div>
      </div>
    </main>

    <!-- 上传对话框 -->
    <el-dialog
      v-model="showUploadDialog"
      title="上传视频"
      width="500px"
      :close-on-click-modal="false"
    >
      <el-upload
        ref="uploadRef"
        drag
        :auto-upload="false"
        :limit="1"
        accept="video/*"
        :on-change="handleFileChange"
      >
        <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
        <div class="el-upload__text">
          拖拽视频文件到此处，或 <em>点击选择</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持 MP4, AVI, MOV 等常见视频格式
          </div>
        </template>
      </el-upload>

      <template #footer>
        <el-button @click="showUploadDialog = false">取消</el-button>
        <el-button type="primary" :loading="uploading" @click="handleUpload">
          {{ uploading ? '上传中...' : '开始上传' }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Upload, UploadFilled, Edit, Delete, Clock, Loading } from '@element-plus/icons-vue'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { transcriptionApi } from '@/services/api'

const router = useRouter()
const taskStore = useUnifiedTaskStore()

// 响应式数据
const showUploadDialog = ref(false)
const uploading = ref(false)
const uploadRef = ref(null)
const selectedFile = ref(null)
const thumbnailCache = ref({})  // 缩略图缓存，避免重复加载

// 计算属性 - 使用 computed 包装确保响应式
const tasks = computed(() => taskStore.tasks)

// 监听任务列表变化，自动加载新增任务的缩略图
watch(
  () => tasks.value?.length,
  (newLength) => {
    if (newLength && tasks.value) {
      tasks.value.forEach(task => {
        // 只加载还没缓存的任务的缩略图
        if (!(task.job_id in thumbnailCache.value)) {
          setTimeout(() => {
            getThumbnailUrl(task.job_id)
          }, 100)
        }
      })
    }
  }
)

// 处理文件选择
function handleFileChange(file) {
  selectedFile.value = file.raw
}

// 处理上传
async function handleUpload() {
  if (!selectedFile.value) {
    ElMessage.warning('请先选择视频文件')
    return
  }

  uploading.value = true
  try {
    // 上传文件到后端
    const { job_id, filename, queue_position } = await transcriptionApi.uploadFile(
      selectedFile.value,
      (percent) => {
        // 上传进度回调
        console.log(`上传进度: ${percent}%`)
      }
    )

    // 添加任务到 store
    taskStore.addTask({
      job_id,
      filename,
      file_path: null,
      status: 'queued',
      phase: 'uploading',
      progress: 0,
      message: `已加入队列 (位置: ${queue_position})`,
      settings: {}
    })

    // 启动转录任务（使用默认设置）
    const defaultSettings = {
      model: 'medium',
      compute_type: 'float16',
      device: 'cuda',
      batch_size: 16,
      word_timestamps: false
    }

    await transcriptionApi.startJob(job_id, defaultSettings)

    // 更新任务状态
    taskStore.updateTask(job_id, {
      status: 'queued',
      phase: 'transcribing',
      message: '等待转录...'
    })

    // 修复：上传完成后立即从后端同步任务列表（确保 UI 及时更新）
    await taskStore.syncTasksFromBackend()

    // 新增：延迟3秒后尝试加载新任务的缩略图（给后端时间处理视频）
    setTimeout(() => {
      console.log(`[TaskListView] 尝试为新任务 ${job_id} 加载缩略图`)
      getThumbnailUrl(job_id, true)
    }, 3000)

    ElMessage.success('上传成功，已加入转录队列')
    showUploadDialog.value = false
    selectedFile.value = null
    uploadRef.value?.clearFiles()
  } catch (error) {
    console.error('上传失败:', error)
    ElMessage.error(`上传失败: ${error.message || '未知错误'}`)
  } finally {
    uploading.value = false
  }
}

// 打开编辑器
function openEditor(jobId) {
  router.push(`/editor/${jobId}`)
}

// 处理任务卡片点击 - 所有状态都可以打开编辑器查看
function handleTaskClick(task) {
  openEditor(task.job_id)
}

// 删除任务
async function deleteTask(jobId) {
  try {
    await ElMessageBox.confirm(
      '确定要删除这个任务吗？此操作无法撤销。',
      '确认删除',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    // 调用后端 API 删除任务数据
    try {
      await transcriptionApi.cancelJob(jobId, true)
    } catch (error) {
      console.warn('调用后端删除失败，继续清理本地记录:', error)
      // 即使后端删除失败，也继续清理本地记录
      // 这有助于修复幽灵任务问题
    }

    // 从本地 store 中删除
    await taskStore.deleteTask(jobId)

    // 修复：删除完成后立即从后端同步任务列表（确保 UI 及时更新）
    await taskStore.syncTasksFromBackend()

    ElMessage.success('任务已删除')
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除任务失败:', error)
      ElMessage.error(`删除失败: ${error.message}`)
    }
  }
}

// 格式化日期 - 显示为 YYYY-MM-DD HH:mm 格式（第二阶段修复：实时更新）
function formatDate(timestamp) {
  if (!timestamp) return ''
  const date = new Date(timestamp)

  // 格式化为 YYYY-MM-DD HH:mm
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  const hours = String(date.getHours()).padStart(2, '0')
  const minutes = String(date.getMinutes()).padStart(2, '0')

  return `${year}-${month}-${day} ${hours}:${minutes}`
}

// 获取状态文本（与后端状态枚举保持一致）
function getStatusText(status) {
  const statusMap = {
    created: '已创建',
    queued: '排队中',
    processing: '转录中',
    paused: '已暂停',
    finished: '已完成',
    failed: '失败',
    canceled: '已取消'
  }
  return statusMap[status] || status
}

// 去除文件扩展名（第五阶段修复：任务名显示）
function getTaskDisplayName(filename) {
  if (!filename) return ''
  // 去除文件扩展名
  const lastDotIndex = filename.lastIndexOf('.')
  if (lastDotIndex > 0) {
    return filename.substring(0, lastDotIndex)
  }
  return filename
}

// 组件挂载
onMounted(() => {
  // 任务列表在 store 初始化时已自动加载 (restoreTasks)
  // 无需手动调用

  // 异步���载所有任务的缩略图（不阻塞UI）
  if (tasks.value && tasks.value.length > 0) {
    tasks.value.forEach(task => {
      // 延迟加载缩略图，避免过多并发请求
      setTimeout(() => {
        getThumbnailUrl(task.job_id)
      }, 100)
    })
  }
})

// 获取任务缩略图（带缓存和重试机制）
async function getThumbnailUrl(jobId, forceReload = false) {
  // 强制重新加载时清除缓存
  if (forceReload && thumbnailCache.value[jobId]) {
    delete thumbnailCache.value[jobId]
  }

  // 检查缓存（避免重复请求）
  if (thumbnailCache.value[jobId] !== undefined && !forceReload) {
    return thumbnailCache.value[jobId]
  }

  // 标记为加载中
  thumbnailCache.value[jobId] = undefined

  try {
    const result = await transcriptionApi.getThumbnail(jobId)
    const thumbnail = result.thumbnail || null

    // 如果获取失败但视频可能还在处理中，标记为"待重试"而非永久失败
    if (!thumbnail) {
      const task = taskStore.getTask(jobId)
      // 如果任务正在处理中，保持undefined状态以便后续重试
      if (task && (task.status === 'processing' || task.status === 'queued')) {
        console.log(`[TaskListView] 任务 ${jobId} 正在处理中，稍后重试加载缩略图`)
        // 不缓存null，保持为undefined，允许后续重试
        return null
      }
    }

    thumbnailCache.value[jobId] = thumbnail
    return thumbnail
  } catch (error) {
    console.warn(`获取缩略图失败 [${jobId}]:`, error)
    // 失败时也设置为null（而非undefined），这样至少显示占位符
    thumbnailCache.value[jobId] = null
    return null
  }
}
</script>

<style lang="scss" scoped>
@use '@/styles/variables' as *;

// 加载动画
@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.task-list-view {
  min-height: 100vh;
  background: var(--bg-primary);
  display: flex;
  flex-direction: column;
}

// 顶部导航栏
.task-header {
  height: 64px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-default);
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: $z-sticky;
  box-shadow: var(--shadow-sm);

  .header-left {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .app-title {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
  }

  .app-icon {
    width: 28px;
    height: 28px;
    color: var(--primary);
  }
}

// 主内容区
.task-main {
  flex: 1;
  padding: 32px 24px;
  max-width: 1400px;
  width: 100%;
  margin: 0 auto;
}

// 空状态
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 24px;
  text-align: center;

  .empty-icon {
    width: 120px;
    height: 120px;
    color: var(--text-disabled);
    opacity: 0.5;
    margin-bottom: 24px;
  }

  .empty-title {
    font-size: 24px;
    color: var(--text-primary);
    margin: 0 0 12px;
  }

  .empty-desc {
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0 0 32px;
  }
}

// 任务网格
.task-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
}

// 任务卡片
.task-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  overflow: hidden;
  transition: all var(--transition-normal);
  cursor: pointer;  // 所有状态都可点击

  &:hover {
    border-color: var(--primary);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
  }

  .task-thumbnail {
    position: relative;
    width: 100%;
    padding-top: 56.25%; // 16:9
    background: var(--bg-tertiary);

    .thumbnail-image {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
      background: var(--bg-tertiary);
    }

    .thumbnail-placeholder {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;

      svg {
        width: 64px;
        height: 64px;
        color: var(--text-disabled);
        opacity: 0.3;
      }

      &.clickable {
        cursor: pointer;
        transition: background var(--transition-fast);

        &:hover {
          background: rgba(0, 0, 0, 0.05);

          svg {
            opacity: 0.5;
          }
        }
      }
    }

    .thumbnail-loading {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.1);
      cursor: pointer;

      .loading-spinner {
        width: 40px;
        height: 40px;
        color: var(--primary);
        opacity: 0.6;
        animation: spin 2s linear infinite;
      }
    }

    .status-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.6);
      display: flex;
      align-items: center;
      justify-content: center;

      .status-text {
        color: white;
        font-size: 14px;
        font-weight: 500;
      }
    }
  }

  .task-info {
    padding: 16px;

    .task-title {
      font-size: 15px;
      font-weight: 500;
      color: var(--text-primary);
      margin: 0 0 8px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .task-meta {
      display: flex;
      gap: 16px;
      margin-bottom: 12px;

      .meta-item {
        font-size: 12px;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 4px;

        .el-icon {
          font-size: 14px;
        }
      }
    }
  }

  .task-actions {
    padding: 0 16px 16px;
    display: flex;
    gap: 8px;

    .el-button {
      flex: 1;
    }
  }
}
</style>
