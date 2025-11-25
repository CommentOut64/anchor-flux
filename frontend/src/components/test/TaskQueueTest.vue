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

    <!-- 从input目录选择文件 -->
    <div class="input-files-section">
      <el-divider content-position="left">从input目录选择文件</el-divider>

      <el-alert v-if="loadingInputFiles" type="info" :closable="false">
        正在加载input目录文件...
      </el-alert>

      <div v-else-if="inputFiles.length > 0" class="input-files-grid">
        <el-checkbox-group v-model="selectedInputFiles">
          <el-checkbox
            v-for="file in inputFiles"
            :key="file.name"
            :label="file.name"
            class="file-checkbox"
          >
            <span class="file-name">{{ file.name }}</span>
            <span class="file-size">({{ formatFileSize(file.size) }})</span>
          </el-checkbox>
        </el-checkbox-group>

        <el-button
          v-if="selectedInputFiles.length > 0"
          type="primary"
          @click="createJobsFromInput"
          :loading="creatingJobs"
          style="margin-top: 12px; width: 100%;"
        >
          <el-icon><VideoPlay /></el-icon>
          创建任务并启动 ({{ selectedInputFiles.length }} 个文件)
        </el-button>
      </div>

      <el-empty v-else description="input目录中没有媒体文件" />
    </div>

    <!-- 批量上传区域 -->
    <div class="batch-upload-section">
      <el-divider content-position="left">手动上传文件</el-divider>
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

        <el-table-column label="操作" width="300" fixed="right">
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

            <!-- 插队按钮组（V2.4功能） -->
            <el-dropdown
              v-if="canPrioritize(scope.row)"
              split-button
              type="primary"
              size="small"
              @click="prioritizeTask(scope.row, 'gentle')"
              @command="(cmd) => prioritizeTask(scope.row, cmd)"
              :loading="scope.row.prioritizing"
            >
              插队
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="gentle">
                    <el-icon><CaretRight /></el-icon>
                    温和插队
                    <span style="color: #909399; font-size: 12px; margin-left: 8px;">
                      (等当前任务完成)
                    </span>
                  </el-dropdown-item>
                  <el-dropdown-item command="force">
                    <el-icon><Warning /></el-icon>
                    强制插队
                    <span style="color: #909399; font-size: 12px; margin-left: 8px;">
                      (暂停当前任务)
                    </span>
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
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

    <!-- 插队设置 -->
    <div class="prioritize-settings-section">
      <el-divider content-position="left">插队设置 (V2.4)</el-divider>

      <el-form :inline="true" label-width="120px">
        <el-form-item label="默认插队模式">
          <el-select
            v-model="defaultPrioritizeMode"
            placeholder="选择默认模式"
            @change="updatePrioritizeSettings"
            :loading="loadingSettings"
          >
            <el-option value="gentle" label="温和插队">
              <span>温和插队</span>
              <span style="color: #909399; font-size: 12px; margin-left: 8px;">
                放到队列头部，等当前任务完成
              </span>
            </el-option>
            <el-option value="force" label="强制插队">
              <span>强制插队</span>
              <span style="color: #909399; font-size: 12px; margin-left: 8px;">
                暂停当前任务，完成后自动恢复
              </span>
            </el-option>
          </el-select>
        </el-form-item>
      </el-form>

      <el-alert type="info" :closable="false" style="margin-top: 12px;">
        <template #title>插队模式说明</template>
        <template #default>
          <ul style="margin: 8px 0; padding-left: 20px;">
            <li><strong>温和插队</strong>: 将任务放到等待队列的头部，当前正在执行的任务继续完成</li>
            <li><strong>强制插队</strong>: 暂停当前任务A -> 执行插队任务B -> B完成后自动恢复A</li>
          </ul>
        </template>
      </el-alert>
    </div>

    <!-- 测试说明 -->
    <div class="test-instructions">
      <el-divider content-position="left">测试说明</el-divider>
      <el-alert type="info" :closable="false" show-icon>
        <template #title>V3.0 任务队列测试要点</template>
        <template #default>
          <ol style="margin: 8px 0; padding-left: 20px;">
            <li><strong>[V3.0新增]</strong> 页面加载时自动连接全局SSE，实时接收队列状态更新</li>
            <li>从input目录选择3个视频文件，点击"创建任务并启动"</li>
            <li>观察：任务状态和队列位置<strong>实时更新</strong>，无需手动刷新</li>
            <li>测试<strong>温和插队</strong>：点击排队任务的"插队"按钮</li>
            <li>观察：队列顺序立即更新，当前任务继续执行</li>
            <li>测试<strong>强制插队</strong>：点击插队按钮右侧下拉菜单选择"强制插队"</li>
            <li>观察：当前任务状态实时变为"暂停中"，插队任务开始执行</li>
            <li>观察：插队任务完成后，被中断的任务自动恢复到队列头部</li>
            <li>修改"默认插队模式"设置，测试设置是否生效</li>
            <li><strong>[降级测试]</strong> 断开后端连接，观察是否自动切换到轮询模式</li>
          </ol>
        </template>
      </el-alert>
    </div>
  </el-card>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { CaretRight, Warning } from '@element-plus/icons-vue'
import axios from 'axios'
import { getQueueService } from '@/services/queueService'

// input目录文件相关
const inputFiles = ref([])
const selectedInputFiles = ref([])
const loadingInputFiles = ref(false)
const creatingJobs = ref(false)

// 文件上传相关
const uploadFileList = ref([])
const uploading = ref(false)

// 任务队列相关
const tasks = ref([])
const loading = ref(false)
const pollTimer = ref(null)

// 插队设置相关
const defaultPrioritizeMode = ref('gentle')
const loadingSettings = ref(false)

// 全局SSE服务 (V3.0)
const queueService = getQueueService()
const useGlobalSSE = ref(false) // 是否使用全局SSE

// 计算属性
const queuedCount = computed(() => tasks.value.filter(t => t.status === 'queued').length)
const runningCount = computed(() => tasks.value.filter(t => t.status === 'processing').length)
const finishedCount = computed(() => tasks.value.filter(t => t.status === 'finished').length)

// 加载input目录文件
async function loadInputFiles() {
  loadingInputFiles.value = true
  try {
    const response = await axios.get('/api/files')
    // 修复：后端返回 {files: [...], input_dir: "..."} 格式
    const files = response.data.files || response.data || []
    inputFiles.value = files
    ElMessage.success(`从input目录加载了 ${files.length} 个文件`)
  } catch (error) {
    console.error('加载input文件失败:', error)
    ElMessage.error('加载input目录文件失败')
  } finally {
    loadingInputFiles.value = false
  }
}

// 格式化文件大小
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
}

// 从input目录创建任务
async function createJobsFromInput() {
  if (selectedInputFiles.value.length === 0) {
    ElMessage.warning('请选择至少一个文件')
    return
  }

  creatingJobs.value = true
  let successCount = 0
  let failCount = 0
  const totalFiles = selectedInputFiles.value.length

  // 逐个创建任务，但添加超时和更好的错误处理
  for (let i = 0; i < selectedInputFiles.value.length; i++) {
    const filename = selectedInputFiles.value[i]
    try {
      ElMessage.info(`正在创建任务 (${i + 1}/${totalFiles}): ${filename}`)

      // 创建任务
      const formData = new FormData()
      formData.append('filename', filename)

      const response = await axios.post('/api/create-job', formData, {
        timeout: 30000  // 30秒超时
      })
      const jobData = response.data

      // 添加到任务列表
      const newTask = {
        job_id: jobData.job_id,
        filename: jobData.filename,
        status: 'created',
        queue_position: -1,
        progress: 0,
        message: '任务已创建',
        starting: false,
        pausing: false,
        canceling: false,
        prioritizing: false
      }
      tasks.value.push(newTask)

      // 自动启动任务（添加超时）
      try {
        await startTaskWithTimeout(newTask, 30000)
        successCount++
      } catch (startError) {
        console.error(`启动任务失败 ${filename}:`, startError)
        newTask.status = 'failed'
        newTask.message = '启动失败: ' + (startError.message || '超时')
        failCount++
      }

    } catch (error) {
      failCount++
      console.error('创建任务失败:', error)
      ElMessage.error(`文件 ${filename} 创建任务失败: ${error.message || '未知错误'}`)
    }
  }

  // 清空选择
  selectedInputFiles.value = []
  creatingJobs.value = false

  ElMessage.info(`创建完成：成功 ${successCount} 个，失败 ${failCount} 个`)

  // 刷新队列状态
  setTimeout(() => refreshQueue(), 1000)
}

// 带超时的启动任务
async function startTaskWithTimeout(task, timeoutMs = 30000) {
  return new Promise(async (resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error('请求超时'))
    }, timeoutMs)

    try {
      await startTask(task)
      clearTimeout(timeoutId)
      resolve()
    } catch (error) {
      clearTimeout(timeoutId)
      reject(error)
    }
  })
}

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
        // 忽略404错误（任务可能已被清理）
        if (error.response && error.response.status === 404) {
          console.log(`任务 ${task.job_id} 未找到，可能已被清理`)
          // 如果任务状态是已完成，保留它；否则移除
          if (task.status === 'finished') {
            updatedTasks.push(task)
          }
        } else {
          console.error(`获取任务 ${task.job_id} 状态失败:`, error)
          updatedTasks.push(task)
        }
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

    // 发送开始请求（添加超时）
    const formData = new FormData()
    formData.append('job_id', task.job_id)
    formData.append('settings', JSON.stringify(settings))

    await axios.post('/api/start', formData, {
      timeout: 30000  // 30秒超时
    })

    ElMessage.success(`任务 ${task.filename} 已加入队列`)
    task.status = 'queued'

    // 刷新状态
    setTimeout(() => refreshQueue(), 1000)
  } catch (error) {
    console.error('启动任务失败:', error)
    ElMessage.error('启动任务失败: ' + (error.message || '未知错误'))
    throw error  // 重新抛出错误，让调用者知道失败了
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

async function prioritizeTask(task, mode = null) {
  task.prioritizing = true

  // 如果没有指定模式，使用默认模式
  const actualMode = mode || defaultPrioritizeMode.value

  try {
    const response = await axios.post(`/api/prioritize/${task.job_id}?mode=${actualMode}`)
    const result = response.data

    if (result.mode === 'gentle') {
      ElMessage.success('温和插队成功，任务已移到队列头部')
    } else if (result.mode === 'force') {
      if (result.interrupted_job_id) {
        ElMessage.success('强制插队成功，当前任务已暂停，完成后将自动恢复')
      } else {
        ElMessage.success('强制插队成功，任务已移到队列头部')
      }
    }

    // 刷新状态
    setTimeout(() => refreshQueue(), 1000)
  } catch (error) {
    console.error('插队失败:', error)
    const errorMsg = error.response?.data?.detail || '插队失败'
    ElMessage.error(errorMsg)
  } finally {
    task.prioritizing = false
  }
}

// 加载插队设置
async function loadPrioritizeSettings() {
  loadingSettings.value = true
  try {
    const response = await axios.get('/api/queue-settings')
    defaultPrioritizeMode.value = response.data.default_prioritize_mode || 'gentle'
  } catch (error) {
    console.error('加载设置失败:', error)
  } finally {
    loadingSettings.value = false
  }
}

// 更新插队设置
async function updatePrioritizeSettings() {
  loadingSettings.value = true
  try {
    await axios.post('/api/queue-settings', {
      default_prioritize_mode: defaultPrioritizeMode.value
    })
    ElMessage.success(`默认插队模式已设置为: ${defaultPrioritizeMode.value === 'gentle' ? '温和插队' : '强制插队'}`)
  } catch (error) {
    console.error('更新设置失败:', error)
    ElMessage.error('更新设置失败')
  } finally {
    loadingSettings.value = false
  }
}

// 状态判断函数
function canStart(task) {
  return ['uploaded', 'created', 'paused', 'failed'].includes(task.status)
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
    'created': 'info',
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
    'created': '已创建',
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

// ========== 全局SSE相关 (V3.0) ==========

// 初始化全局SSE
function initGlobalSSE() {
  console.log('[TaskQueueTest] 尝试连接全局SSE...')

  // 订阅初始状态
  queueService.on('initial_state', (data) => {
    console.log('[TaskQueueTest] 收到初始状态:', data)
    useGlobalSSE.value = true

    // 更新任务列表
    updateTasksFromSSE(data)

    // 停止轮询，改用SSE
    stopPolling()
    ElMessage.success('已连接全局SSE，实时更新已启用')
  })

  // 订阅队列更新
  queueService.on('queue_update', (data) => {
    console.log('[TaskQueueTest] 队列更新:', data)
    // 更新队列位置
    updateQueuePositions(data)
  })

  // 订阅任务状态
  queueService.on('job_status', (data) => {
    console.log('[TaskQueueTest] 任务状态:', data)
    // 更新单个任务状态
    updateTaskStatus(data)
  })

  // 订阅任务进度
  queueService.on('job_progress', (data) => {
    console.log('[TaskQueueTest] 任务进度:', data)
    // 更新单个任务进度
    updateTaskProgress(data)
  })

  // 错误处理
  queueService.on('error', () => {
    if (useGlobalSSE.value) {
      ElMessage.warning('SSE连接中断，切换到轮询模式')
      useGlobalSSE.value = false
      startPolling()
    }
  })

  // 连接SSE
  queueService.connect()
}

// 从SSE数据更新任务列表
function updateTasksFromSSE(data) {
  const { jobs, queue, running } = data

  // 创建任务映射
  const tasksMap = new Map()

  jobs.forEach(job => {
    tasksMap.set(job.id, {
      job_id: job.id,
      filename: job.filename,
      status: job.status,
      progress: job.progress,
      message: job.message,
      queue_position: -1,
      starting: false,
      pausing: false,
      canceling: false,
      prioritizing: false
    })
  })

  // 计算队列位置
  queue.forEach((jobId, index) => {
    const task = tasksMap.get(jobId)
    if (task) {
      task.queue_position = index + 1
    }
  })

  // 设置正在执行的任务
  if (running && tasksMap.has(running)) {
    tasksMap.get(running).queue_position = 0
  }

  tasks.value = Array.from(tasksMap.values())
}

// 更新队列位置
function updateQueuePositions(data) {
  const { queue, running } = data

  // 重置所有队列位置
  tasks.value.forEach(task => {
    task.queue_position = -1
  })

  // 设置队列位置
  queue.forEach((jobId, index) => {
    const task = tasks.value.find(t => t.job_id === jobId)
    if (task) {
      task.queue_position = index + 1
    }
  })

  // 设置正在执行的任务
  if (running) {
    const task = tasks.value.find(t => t.job_id === running)
    if (task) {
      task.queue_position = 0
    }
  }
}

// 更新任务状态
function updateTaskStatus(data) {
  const { id, status, message, filename } = data

  const task = tasks.value.find(t => t.job_id === id)
  if (task) {
    task.status = status
    task.message = message
    if (filename) {
      task.filename = filename
    }
  } else {
    // 新任务，添加到列表
    tasks.value.push({
      job_id: id,
      filename: filename || id,
      status: status,
      message: message,
      progress: 0,
      queue_position: -1,
      starting: false,
      pausing: false,
      canceling: false,
      prioritizing: false
    })
  }
}

// 更新任务进度
function updateTaskProgress(data) {
  const { id, progress, message, phase } = data

  const task = tasks.value.find(t => t.job_id === id)
  if (task) {
    task.progress = progress
    if (message) {
      task.message = message
    }
    if (phase) {
      task.phase = phase
    }
  }
}

// 生命周期
onMounted(() => {
  loadInputFiles() // 加载input目录文件
  loadPrioritizeSettings() // 加载插队设置

  // 尝试连接全局SSE (V3.0)
  initGlobalSSE()

  // 如果SSE连接失败，使用轮询模式
  setTimeout(() => {
    if (!useGlobalSSE.value) {
      console.log('[TaskQueueTest] SSE未连接，使用轮询模式')
      startPolling()
    }
  }, 2000)
})

onUnmounted(() => {
  stopPolling()
  queueService.disconnect()
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

.input-files-section {
  margin-bottom: 30px;
}

.input-files-grid {
  background: #f5f7fa;
  padding: 16px;
  border-radius: 8px;
  max-height: 300px;
  overflow-y: auto;
}

.file-checkbox {
  display: block;
  margin: 8px 0;
  width: 100%;
}

.file-name {
  font-weight: 500;
  color: #303133;
}

.file-size {
  color: #909399;
  font-size: 0.9rem;
  margin-left: 8px;
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

.prioritize-settings-section {
  margin-bottom: 30px;
  padding: 16px;
  background: #fafafa;
  border-radius: 8px;
  border: 1px solid #e4e7ed;
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