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
            <div class="thumbnail-placeholder">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
              </svg>
            </div>
            <div v-if="task.status !== 'finished'" class="status-overlay">
              <span class="status-text">{{ getStatusText(task.status) }}</span>
            </div>
          </div>

          <!-- 任务信息 -->
          <div class="task-info">
            <h3 class="task-title" :title="task.filename">{{ task.filename }}</h3>
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
import { ref, onMounted } from 'vue'
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

// 计算属性 - 直接使用 store 中的 tasks (已是 computed)
const tasks = taskStore.tasks

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
    await transcriptionApi.cancelJob(jobId, true)

    // 从本地 store 中删除
    await taskStore.deleteTask(jobId)

    ElMessage.success('任务已删除')
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除任务失败:', error)
      ElMessage.error(`删除失败: ${error.message}`)
    }
  }
}

// 格式化日期
function formatDate(timestamp) {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date
  
  // 1小时内
  if (diff < 3600000) {
    const minutes = Math.floor(diff / 60000)
    return `${minutes} 分钟前`
  }
  
  // 今天
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  
  // 其他
  return date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
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

// 组件挂载
onMounted(() => {
  // 任务列表在 store 初始化时已自动加载 (restoreTasks)
  // 无需手动调用
})
</script>

<style lang="scss" scoped>
@use '@/styles/variables' as *;

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
