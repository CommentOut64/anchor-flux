<!--
  任务队列测试组件 - 用于测试V2.2队列管理功能
  注意：这是临时测试组件，不考虑美观性
-->
<template>
  <el-card class="task-queue-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <el-icon><List /></el-icon>
        <span>任务队列测试面板</span>
        <div style="margin-left: auto;">
          <el-button
            type="primary"
            size="small"
            @click="refreshQueue"
            :loading="loading"
          >
            <el-icon><Refresh /></el-icon>
            刷新队列
          </el-button>
        </div>
      </div>
    </template>

    <!-- 批量上传区域 -->
    <div class="batch-upload-section">
      <el-divider content-position="left">批量上传测试</el-divider>
      <el-upload
        class="upload-area"
        drag
        multiple
        :auto-upload="false"
        :on-change="handleFileChange"
        :file-list="uploadFileList"
        accept=".mp4,.avi,.mkv,.mov,.wmv,.flv,.webm,.m4v,.mp3,.wav,.flac,.aac,.ogg,.m4a,.wma"
      >
        <div class="upload-content">
          <el-icon class="upload-icon"><UploadFilled /></el-icon>
          <div class="upload-text">
            拖拽多个文件到此处，或<em>点击选择多个文件</em>
          </div>
          <div class="upload-hint">
            支持批量上传多个视频/音频文件
          </div>
        </div>
      </el-upload>

      <div v-if="uploadFileList.length > 0" class="file-list-preview">
        <el-tag
          v-for="file in uploadFileList"
          :key="file.uid"
          closable
          @close="removeFile(file)"
          style="margin: 4px;"
        >
          {{ file.name }}
        </el-tag>
      </div>

      <el-button
        v-if="uploadFileList.length > 0"
        type="primary"
        @click="batchUpload"
        :loading="uploading"
        style="margin-top: 12px; width: 100%;"
      >
        <el-icon><Upload /></el-icon>
        上传所有文件并创建任务 ({{ uploadFileList.length }} 个)
      </el-button>
    </div>

    <!-- 任务队列显示 -->
    <div class="task-queue-section">
      <el-divider content-position="left">任务队列状态</el-divider>

      <div v-if="tasks.length === 0" class="no-tasks">
        <el-empty description="队列中没有任务" />
      </div>

      <el-table
        v-else
        :data="tasks"
        style="width: 100%"
        :highlight-current-row="true"
        stripe
      >
        <el-table-column prop="job_id" label="任务ID" width="100">
          <template #default="scope">
            {{ scope.row.job_id.substring(0, 8) }}...
          </template>
        </el-table-column>

        <el-table-column prop="filename" label="文件名" min-width="150" />

        <el-table-column prop="status" label="状态" width="100">
          <template #default="scope">
            <el-tag
              :type="getStatusType(scope.row.status)"
              size="small"
            >
              {{ getStatusText(scope.row.status) }}
            </el-tag>
          </template>
        </el-table-column>

        <el-table-column prop="queue_position" label="队列位置" width="80">
          <template #default="scope">
            <span v-if="scope.row.queue_position === 0">
              <el-tag type="warning" size="small">执行中</el-tag>
            </span>
            <span v-else-if="scope.row.queue_position > 0">
              第 {{ scope.row.queue_position }} 位
            </span>
            <span v-else>-</span>
          </template>
        </el-table-column>

        <el-table-column prop="progress" label="进度" width="100">
          <template #default="scope">
            <el-progress
              :percentage="scope.row.progress || 0"
              :stroke-width="6"
              :format="() => `${scope.row.progress || 0}%`"
            />
          </template>
        </el-table-column>

        <el-table-column prop="message" label="消息" min-width="150" />

        <el-table-column label="操作" width="240" fixed="right">
          <template #default="scope">
            <!-- 开始/继续按钮 -->
            <el-button
              v-if="canStart(scope.row)"
              type="success"
              size="small"
              @click="startTask(scope.row)"
              :loading="scope.row.starting"
            >
              {{ scope.row.status === 'paused' ? '继续' : '开始' }}
            </el-button>

            <!-- 暂停按钮 -->
            <el-button
              v-if="canPause(scope.row)"
              type="warning"
              size="small"
              @click="pauseTask(scope.row)"
              :loading="scope.row.pausing"
            >
              暂停
            </el-button>

            <!-- 取消按钮 -->
            <el-button
              v-if="canCancel(scope.row)"
              type="danger"
              size="small"
              @click="cancelTask(scope.row)"
              :loading="scope.row.canceling"
            >
              取消
            </el-button>

            <!-- 插队按钮（测试V2.4功能） -->
            <el-button
              v-if="canPrioritize(scope.row)"
              type="primary"
              size="small"
              @click="prioritizeTask(scope.row)"
              :loading="scope.row.prioritizing"
            >
              插队
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- 队列统计 -->
      <div class="queue-stats" v-if="tasks.length > 0">
        <el-row :gutter="20" style="margin-top: 20px;">
          <el-col :span="6">
            <el-statistic title="总任务数" :value="tasks.length" />
          </el-col>
          <el-col :span="6">
            <el-statistic title="排队中" :value="queuedCount" />
          </el-col>
          <el-col :span="6">
            <el-statistic title="执行中" :value="runningCount" />
          </el-col>
          <el-col :span="6">
            <el-statistic title="已完成" :value="finishedCount" />
          </el-col>
        </el-row>
      </div>
    </div>

    <!-- 测试说明 -->
    <div class="test-instructions">
      <el-divider content-position="left">测试说明</el-divider>
      <el-alert type="info" :closable="false" show-icon>
        <template #title>V2.2 任务队列测试要点</template>
        <template #default>
          <ol style="margin: 8px 0; padding-left: 20px;">
            <li>批量上传3个视频文件</li>
            <li>快速点击所有"开始"按钮</li>
            <li>观察：应该只有1个任务显示"执行中"，其他显示"排队中"</li>
            <li>测试暂停：点击执行中任务的"暂停"</li>
            <li>测试取消：点击排队中任务的"取消"</li>
            <li>测试插队：点击排队任务的"插队"按钮（如果实现了V2.4）</li>
          </ol>
        </template>
      </el-alert>
    </div>
  </el-card>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

// 文件上传相关
const uploadFileList = ref([])
const uploading = ref(false)

// 任务队列相关
const tasks = ref([])
const loading = ref(false)
const pollTimer = ref(null)

// 计算属性
const queuedCount = computed(() => tasks.value.filter(t => t.status === 'queued').length)
const runningCount = computed(() => tasks.value.filter(t => t.status === 'processing').length)
const finishedCount = computed(() => tasks.value.filter(t => t.status === 'finished').length)

// 文件处理
function handleFileChange(file, fileList) {
  uploadFileList.value = fileList
}

function removeFile(file) {
  const index = uploadFileList.value.findIndex(f => f.uid === file.uid)
  if (index > -1) {
    uploadFileList.value.splice(index, 1)
  }
}

// 批量上传
async function batchUpload() {
  if (uploadFileList.value.length === 0) {
    ElMessage.warning('请先选择文件')
    return
  }

  uploading.value = true
  let successCount = 0
  let failCount = 0

  for (const file of uploadFileList.value) {
    try {
      // 创建FormData
      const formData = new FormData()
      formData.append('file', file.raw)

      // 上传文件
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      successCount++
      ElMessage.success(`文件 ${file.name} 上传成功，已加入队列`)

      // 添加到任务列表
      const jobData = response.data
      tasks.value.push({
        job_id: jobData.job_id,
        filename: jobData.filename,
        status: 'uploaded',
        queue_position: jobData.queue_position || -1,
        progress: 0,
        message: '已上传，等待开始',
        starting: false,
        pausing: false,
        canceling: false,
        prioritizing: false
      })

    } catch (error) {
      failCount++
      console.error('上传失败:', error)
      ElMessage.error(`文件 ${file.name} 上传失败`)
    }
  }

  // 清空文件列表
  uploadFileList.value = []
  uploading.value = false

  ElMessage.info(`上传完成：成功 ${successCount} 个，失败 ${failCount} 个`)

  // 刷新队列状态
  await refreshQueue()
}

// 刷新队列状态
async function refreshQueue() {
  loading.value = true

  try {
    // 获取所有任务状态
    const updatedTasks = []

    for (const task of tasks.value) {
      try {
        const response = await axios.get(`/api/status/${task.job_id}`)
        const data = response.data

        updatedTasks.push({
          ...task,
          status: data.status,
          queue_position: data.queue_position,
          progress: data.progress || 0,
          message: data.message || '',
        })
      } catch (error) {
        console.error(`获取任务 ${task.job_id} 状态失败:`, error)
        updatedTasks.push(task)
      }
    }

    tasks.value = updatedTasks
  } catch (error) {
    console.error('刷新队列失败:', error)
    ElMessage.error('刷新队列失败')
  } finally {
    loading.value = false
  }
}

// 任务控制函数
async function startTask(task) {
  task.starting = true

  try {
    // 准备默认设置
    const settings = {
      model: 'medium',
      compute_type: 'float16',
      device: 'cuda',
      batch_size: 16,
      word_timestamps: false
    }

    // 发送开始请求
    const formData = new FormData()
    formData.append('job_id', task.job_id)
    formData.append('settings', JSON.stringify(settings))

    await axios.post('/api/start', formData)

    ElMessage.success(`任务 ${task.filename} 已加入队列`)
    task.status = 'queued'

    // 刷新状态
    setTimeout(() => refreshQueue(), 1000)
  } catch (error) {
    console.error('启动任务失败:', error)
    ElMessage.error('启动任务失败')
  } finally {
    task.starting = false
  }
}

async function pauseTask(task) {
  task.pausing = true

  try {
    await axios.post(`/api/pause/${task.job_id}`)
    ElMessage.success('暂停请求已发送')

    // 刷新状态
    setTimeout(() => refreshQueue(), 2000)
  } catch (error) {
    console.error('暂停任务失败:', error)
    ElMessage.error('暂停任务失败')
  } finally {
    task.pausing = false
  }
}

async function cancelTask(task) {
  try {
    await ElMessageBox.confirm(
      '确定要取消此任务吗？',
      '确认取消',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    task.canceling = true

    await axios.post(`/api/cancel/${task.job_id}`)
    ElMessage.success('任务已取消')

    // 从列表中移除
    const index = tasks.value.findIndex(t => t.job_id === task.job_id)
    if (index > -1) {
      tasks.value.splice(index, 1)
    }
  } catch (error) {
    if (error !== 'cancel') {
      console.error('取消任务失败:', error)
      ElMessage.error('取消任务失败')
    }
  } finally {
    task.canceling = false
  }
}

async function prioritizeTask(task) {
  task.prioritizing = true

  try {
    await axios.post(`/api/prioritize/${task.job_id}`)
    ElMessage.success('任务已插队到队首')

    // 刷新状态
    setTimeout(() => refreshQueue(), 1000)
  } catch (error) {
    console.error('插队失败:', error)
    ElMessage.warning('插队功能未实现（需要V2.4）')
  } finally {
    task.prioritizing = false
  }
}

// 状态判断函数
function canStart(task) {
  return ['uploaded', 'paused', 'failed'].includes(task.status)
}

function canPause(task) {
  return ['processing', 'queued'].includes(task.status)
}

function canCancel(task) {
  return !['finished', 'canceled'].includes(task.status)
}

function canPrioritize(task) {
  return task.status === 'queued' && task.queue_position > 1
}

// 状态显示函数
function getStatusType(status) {
  const statusMap = {
    'uploaded': 'info',
    'queued': 'warning',
    'processing': 'primary',
    'finished': 'success',
    'failed': 'danger',
    'canceled': 'info',
    'paused': 'warning'
  }
  return statusMap[status] || 'info'
}

function getStatusText(status) {
  const statusMap = {
    'uploaded': '已上传',
    'queued': '排队中',
    'processing': '处理中',
    'finished': '已完成',
    'failed': '失败',
    'canceled': '已取消',
    'paused': '已暂停'
  }
  return statusMap[status] || status
}

// 定时刷新
function startPolling() {
  pollTimer.value = setInterval(() => {
    // 只有有进行中的任务才刷新
    if (tasks.value.some(t => ['processing', 'queued'].includes(t.status))) {
      refreshQueue()
    }
  }, 3000)
}

function stopPolling() {
  if (pollTimer.value) {
    clearInterval(pollTimer.value)
    pollTimer.value = null
  }
}

// 生命周期
onMounted(() => {
  startPolling()
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped>
.task-queue-card {
  margin-bottom: 20px;
  border-radius: 16px;
  overflow: hidden;
  border: 2px solid #e4e7ed;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  color: #2c3e50;
}

.batch-upload-section {
  margin-bottom: 30px;
}

.upload-content {
  text-align: center;
  padding: 40px 20px;
}

.upload-icon {
  font-size: 3rem;
  color: #409eff;
  margin-bottom: 12px;
}

.upload-text {
  font-size: 1rem;
  color: #606266;
  margin-bottom: 8px;
}

.upload-hint {
  color: #909399;
  font-size: 0.9rem;
}

.file-list-preview {
  margin-top: 12px;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 8px;
}

.task-queue-section {
  margin-bottom: 30px;
}

.no-tasks {
  padding: 40px 0;
}

.queue-stats {
  padding: 20px;
  background: #f5f7fa;
  border-radius: 8px;
}

.test-instructions {
  margin-top: 20px;
}

/* 自定义 Element Plus 样式 */
:deep(.el-upload-dragger) {
  border: 2px dashed #d9d9d9;
  border-radius: 12px;
  background-color: #fafafa;
  transition: all 0.3s ease;
}

:deep(.el-upload-dragger:hover) {
  border-color: #409eff;
  background-color: #f0f9ff;
}

:deep(.el-divider__text) {
  background: white;
  font-weight: 600;
  color: #303133;
}
</style>