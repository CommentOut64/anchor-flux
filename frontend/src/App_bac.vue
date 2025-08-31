<template>
  <el-container class="app-container">
    <!-- 顶部标题栏 -->
    <el-header class="header">
      <el-row justify="space-between" align="middle">
        <el-col :span="18">
          <h1 class="title">
            <el-icon><VideoPlay /></el-icon>
            Video To SRT 转录工具
          </h1>
        </el-col>
        <el-col :span="6" class="header-actions">
          <!-- 模型状态按钮 -->
          <ModelStatusButton />
          
          <!-- 硬件信息按钮 -->
          <el-button 
            type="info" 
            size="small" 
            @click="showHardwareDialog = true" 
            plain
          >
            <el-icon><Monitor /></el-icon>
            硬件信息
          </el-button>
        </el-col>
      </el-row>
    </el-header>

    <!-- 主要内容区域 -->
    <el-main class="main-content">
      <el-row :gutter="20" justify="center">
        <el-col :xs="24" :sm="20" :md="16" :lg="14" :xl="12">
          
          <!-- 1. 文件选择区域 -->
          <FileSelector
            :show-upload="showUpload"
            :available-files="availableFiles"
            :selected-file="selectedFile"
            :loading-files="loadingFiles"
            :creating="creating"
            :uploading="uploading"
            :upload-progress="uploadProgress"
            :input-dir-path="inputDirPath"
            :job-id="jobId"
            @toggle-mode="toggleUploadMode"
            @refresh-files="loadFiles"
            @select-file="selectFile"
            @clear-selection="clearSelection"
            @create-job="createJob"
            @upload-file="handleUpload"
          />

          <!-- 2. 参数设置区域 -->
          <TranscriptionSettings
            :job-id="jobId"
            :settings="settings"
            :starting="starting"
            :processing="processing"
            :canceling="canceling"
            :can-restart="canRestart"
            @start-job="startJob"
            @cancel-job="cancelJob"
            @restart-job="restartJob"
            @reset-selection="resetSelection"
            @show-hardware="showHardwareDialog = true"
          />

          <!-- 3. 进度显示区域 -->
          <ProgressDisplay
            :job-id="jobId"
            :progress="progress"
            :status="status"
            :status-text="statusText"
            :download-url="downloadUrl"
            :last-error="lastError"
            :phase="phase"
            :language="language"
            @download="downloadFile"
            @copy-to-source="copyResultToSource"
          />

        </el-col>
      </el-row>
    </el-main>

    <!-- 硬件信息对话框 -->
    <HardwareDialog v-model="showHardwareDialog" />
  </el-container>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from "vue"
import { ElMessage, ElMessageBox } from 'element-plus'
import { VideoPlay, Monitor } from '@element-plus/icons-vue'

// 导入组件
import FileSelector from './components/file-management/FileSelector.vue'
import TranscriptionSettings from './components/transcription/TranscriptionSettings.vue'
import ProgressDisplay from './components/transcription/ProgressDisplay.vue'
import HardwareDialog from './components/hardware/HardwareDialog.vue'
import ModelStatusButton from './components/models/ModelStatusButton.vue'

// 导入服务
import { FileService } from './services/fileService.js'
import { TranscriptionService } from './services/transcriptionService.js'

// 文件选择相关
const availableFiles = ref([])
const selectedFile = ref(null)
const loadingFiles = ref(false)
const creating = ref(false)
const inputDirPath = ref('input/')
const uploading = ref(false)
const uploadProgress = ref(0)
const showUpload = ref(false) // 默认使用本地input模式

// 硬件信息对话框
const showHardwareDialog = ref(false)

// 任务相关
const jobId = ref("")
const starting = ref(false)
const processing = ref(false)
const canceling = ref(false)
const canRestart = ref(false)

// 进度相关
const progress = ref(0)
const status = ref("")
const statusText = ref("请先选择文件")
const downloadUrl = ref("")
const lastError = ref("")
const phase = ref("")
const language = ref("")

// 设置对象
const settings = reactive({
  model: 'medium',
  compute_type: 'float16',
  device: 'auto',
  batch_size: 16,
  word_timestamps: false,
  // CPU亲和性设置
  cpu_affinity_enabled: false,
  cpu_affinity_strategy: 'auto',
  cpu_affinity_custom_cores: [],
  cpu_affinity_exclude_cores: []
})

// 文件管理方法
async function loadFiles() {
  try {
    loadingFiles.value = true
    const files = await FileService.getFiles()
    availableFiles.value = files.files || []
    inputDirPath.value = files.input_dir
    
    if (availableFiles.value.length === 0) {
      ElMessage.info('input 目录中没有找到支持的媒体文件')
    }
  } catch (error) {
    console.error('获取文件列表失败:', error)
    ElMessage.error('获取文件列表失败：' + (error.response?.data?.detail || error.message))
  } finally {
    loadingFiles.value = false
  }
}

function toggleUploadMode() {
  showUpload.value = !showUpload.value
  if (!showUpload.value) {
    loadFiles()
  }
}

function selectFile(filename) {
  selectedFile.value = filename
  resetJobState()
}

function clearSelection() {
  selectedFile.value = null
  resetJobState()
}

async function createJob(filename) {
  try {
    creating.value = true
    const result = await FileService.createJob(filename)
    jobId.value = result.job_id
    selectedFile.value = result.filename
    resetJobState()
    ElMessage.success('任务创建成功')
  } catch (error) {
    console.error('创建任务失败:', error)
    ElMessage.error('创建任务失败：' + (error.response?.data?.detail || error.message))
  } finally {
    creating.value = false
  }
}

async function handleUpload(file, onProgress) {
  try {
    uploading.value = true
    const result = await FileService.uploadFile(file, onProgress)
    
    jobId.value = result.job_id
    selectedFile.value = result.filename
    resetJobState()
    
    ElMessage.success('文件上传成功，任务已创建')
    
    // 刷新文件列表
    if (!showUpload.value) {
      await loadFiles()
    }
  } catch (error) {
    console.error('上传文件失败:', error)
    ElMessage.error('上传文件失败：' + (error.response?.data?.detail || error.message))
  } finally {
    uploading.value = false
    uploadProgress.value = 0
  }
}

// 转录控制方法
async function startJob() {
  if (!jobId.value) {
    ElMessage.error('请先选择文件')
    return
  }

  try {
    starting.value = true
    await TranscriptionService.startJob(jobId.value, settings)
    
    processing.value = true
    starting.value = false
    canRestart.value = false
    
    // 开始轮询状态
    startStatusPolling()
    
    ElMessage.success('转录任务已启动')
  } catch (error) {
    console.error('启动任务失败:', error)
    ElMessage.error('启动任务失败：' + (error.response?.data?.detail || error.message))
    starting.value = false
  }
}

async function cancelJob() {
  if (!jobId.value) return

  try {
    canceling.value = true
    await TranscriptionService.cancelJob(jobId.value)
    
    processing.value = false
    canceling.value = false
    canRestart.value = true
    
    stopStatusPolling()
    
    ElMessage.success('任务已取消')
  } catch (error) {
    console.error('取消任务失败:', error)
    ElMessage.error('取消任务失败：' + (error.response?.data?.detail || error.message))
    canceling.value = false
  }
}

async function restartJob() {
  if (!jobId.value) return

  try {
    await ElMessageBox.confirm('确定要重新开始转录吗？', '确认操作', {
      type: 'warning'
    })
    
    resetJobState()
    await startJob()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('重新启动失败:', error)
      ElMessage.error('重新启动失败')
    }
  }
}

function resetSelection() {
  selectedFile.value = null
  jobId.value = ""
  resetJobState()
}

function resetJobState() {
  processing.value = false
  starting.value = false
  canceling.value = false
  canRestart.value = false
  progress.value = 0
  status.value = ""
  statusText.value = selectedFile.value ? "已选择文件，可开始转录" : "请先选择文件"
  downloadUrl.value = ""
  lastError.value = ""
  phase.value = ""
  language.value = ""
  
  stopStatusPolling()
}

// 状态轮询
let statusTimer = null

function startStatusPolling() {
  if (statusTimer) return
  
  statusTimer = setInterval(async () => {
    if (!jobId.value || !processing.value) {
      stopStatusPolling()
      return
    }
    
    try {
      const jobStatus = await TranscriptionService.getJobStatus(jobId.value)
      updateJobStatus(jobStatus)
    } catch (error) {
      console.error('获取任务状态失败:', error)
    }
  }, 1000)
}

function stopStatusPolling() {
  if (statusTimer) {
    clearInterval(statusTimer)
    statusTimer = null
  }
}

function updateJobStatus(jobStatus) {
  status.value = jobStatus.status
  progress.value = jobStatus.progress || 0
  statusText.value = jobStatus.status_text || ""
  lastError.value = jobStatus.last_error || ""
  phase.value = jobStatus.phase || ""
  language.value = jobStatus.language || ""
  
  if (jobStatus.download_url) {
    downloadUrl.value = jobStatus.download_url
  }
  
  // 任务完成或失败时停止轮询
  if (jobStatus.status === 'finished' || jobStatus.status === 'failed') {
    processing.value = false
    canRestart.value = true
    stopStatusPolling()
    
    if (jobStatus.status === 'finished') {
      ElMessage.success('转录完成！')
    } else if (jobStatus.status === 'failed') {
      ElMessage.error('转录失败：' + (lastError.value || '未知错误'))
    }
  }
}

// 下载和复制
async function downloadFile() {
  if (!downloadUrl.value) return
  
  try {
    const link = document.createElement('a')
    link.href = downloadUrl.value
    link.download = ''
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    ElMessage.success('文件下载已开始')
  } catch (error) {
    console.error('下载失败:', error)
    ElMessage.error('下载失败')
  }
}

async function copyResultToSource() {
  if (!jobId.value) return
  
  try {
    await TranscriptionService.copyResultToSource(jobId.value)
    ElMessage.success('转录结果已复制到源文件目录')
  } catch (error) {
    console.error('复制失败:', error)
    ElMessage.error('复制失败：' + (error.response?.data?.detail || error.message))
  }
}

// 自动预加载模型
async function startInitialPreload() {
  try {
    console.log('[App] 🚀 系统启动，准备自动预加载模型...')
    
    // 延迟10秒确保前后端完全就绪
    setTimeout(async () => {
      try {
        console.log('[App] 📡 开始检查后端连接状态...')
        
        // 检查后端连接
        const pingResponse = await fetch('/api/ping', { timeout: 5000 })
        if (!pingResponse.ok) {
          console.log('[App] ❌ 后端连接失败，跳过自动预加载')
          return
        }
        console.log('[App] ✅ 后端连接正常')
        
        // 检查当前预加载状态
        const statusResponse = await fetch('/api/models/preload/status', { timeout: 5000 })
        if (statusResponse.ok) {
          const statusResult = await statusResponse.json()
          if (statusResult.success) {
            const status = statusResult.data
            console.log('[App] 📊 当前预加载状态:', status)
            
            // 如果已经在预加载或已有模型，跳过
            if (status.is_preloading) {
              console.log('[App] ⚠️ 预加载已在进行中，跳过自动启动')
              return
            }
            if (status.loaded_models > 0) {
              console.log('[App] ✅ 模型已预加载完成，跳过自动启动')
              return
            }
          }
        }
        
        console.log('[App] 🎯 启动自动预加载...')
        const preloadResponse = await fetch('/api/models/preload/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
        
        if (preloadResponse.ok) {
          const result = await preloadResponse.json()
          if (result.success) {
            console.log('[App] ✅ 模型预加载已启动')
            ElMessage.success('模型预加载已启动，可在右上角查看进度', { duration: 3000 })
          } else {
            console.log('[App] ⚠️ 预加载启动失败:', result.message)
            ElMessage.info('模型将在首次使用时自动加载', { duration: 2000 })
          }
        } else {
          console.log('[App] ❌ 预加载请求失败，状态码:', preloadResponse.status)
        }
      } catch (error) {
        console.log('[App] ❌ 自动预加载异常:', error.message)
        ElMessage.info('模型将在首次使用时自动加载', { duration: 2000 })
      }
    }, 10000) // 延迟10秒
  } catch (error) {
    console.log('[App] ❌ 预加载初始化失败:', error)
  }
}

// 生命周期
onMounted(async () => {
  console.log('[App] 🎬 应用程序启动')
  
  // 加载文件列表
  await loadFiles()
  
  // 启动自动预加载
  startInitialPreload()
})

onUnmounted(() => {
  stopStatusPolling()
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
}

.title {
  margin: 0;
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 28px;
  font-weight: 600;
  color: #2c3e50;
  background: linear-gradient(135deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  align-items: center;
}

.main-content {
  padding: 40px 20px;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .title {
    font-size: 20px;
  }
  
  .header-actions {
    gap: 8px;
  }
  
  .main-content {
    padding: 20px 10px;
  }
}

@media (max-width: 480px) {
  .title {
    font-size: 18px;
  }
  
  .header {
    padding: 0 10px;
  }
}
</style>
