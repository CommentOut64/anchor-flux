<template>
  <el-container class="app-container">
    <!-- 顶部标题 -->
    <el-header class="header">
      <el-row justify="center">
        <el-col :span="24">
          <h1 class="title">
            <el-icon><VideoPlay /></el-icon>
            Video To SRT 转录工具
          </h1>
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
const status = ref("")
const progress = ref(0)
const statusText = ref("请先选择文件")
const downloadUrl = ref("")
const processing = ref(false)
const starting = ref(false)
const canceling = ref(false)
const lastError = ref("")
const phase = ref("")
const language = ref("")
const canRestart = ref(false)
const pollTimer = ref(null)

const settings = reactive({
  model: "medium",
  compute_type: "float16",
  device: "cuda",
  batch_size: 16,
  word_timestamps: false,
})

// 文件上传处理
async function handleUpload(uploadFile) {
  if (!uploadFile) {
    ElMessage.warning('请选择文件')
    return
  }
  
  // 检查文件类型
  if (!FileService.isSupportedFile(uploadFile.name)) {
    ElMessage.error('不支持的文件格式，请上传视频或音频文件')
    return
  }
  
  uploading.value = true
  uploadProgress.value = 0
  
  try {
    const data = await TranscriptionService.uploadFile(uploadFile.raw, (progressEvent) => {
      uploadProgress.value = Math.round(
        (progressEvent.loaded / progressEvent.total) * 100
      )
    })
    
    jobId.value = data.job_id
    selectedFile.value = {
      name: data.filename,
      originalName: data.original_name,
      size: uploadFile.size
    }
    status.value = "ready"
    statusText.value = "文件已上传，可开始转录"
    canRestart.value = false
    showUpload.value = false
    
    ElMessage.success('文件上传成功！转录任务已创建。')
    
    // 刷新文件列表
    loadFiles()
  } catch (error) {
    console.error('文件上传失败:', error)
    ElMessage.error('文件上传失败：' + (error.response?.data?.detail || error.message))
  } finally {
    uploading.value = false
    uploadProgress.value = 0
  }
}

// 切换上传模式
function toggleUploadMode() {
  showUpload.value = !showUpload.value
  if (showUpload.value) {
    // 清除已选择的文件
    selectedFile.value = null
  }
}

// 加载可用文件列表
async function loadFiles() {
  loadingFiles.value = true
  try {
    const data = await FileService.getFiles()
    availableFiles.value = data.files || []
    inputDirPath.value = data.input_dir || 'input/'
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

// 选择文件
function selectFile(file) {
  selectedFile.value = file
  ElMessage.success(`已选择文件：${file.name}`)
}

// 清除选择
function clearSelection() {
  selectedFile.value = null
}

// 创建转录任务
async function createJob() {
  if (!selectedFile.value) {
    ElMessage.warning('请先选择文件')
    return
  }
  
  creating.value = true
  try {
    const data = await TranscriptionService.createJob(selectedFile.value.name)
    jobId.value = data.job_id
    status.value = "ready"
    statusText.value = "文件已准备就绪，可开始转录"
    canRestart.value = false
    
    ElMessage.success('转录任务创建成功！')
  } catch (error) {
    console.error('创建任务失败:', error)
    ElMessage.error('创建任务失败：' + (error.response?.data?.detail || error.message))
  } finally {
    creating.value = false
  }
}

// 重置选择
function resetSelection() {
  ElMessageBox.confirm('确定要重新选择文件吗？这将清除当前的转录进度。', '确认操作', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning',
  }).then(() => {
    // 重置所有状态
    selectedFile.value = null
    jobId.value = ""
    status.value = ""
    progress.value = 0
    statusText.value = "请先选择文件"
    downloadUrl.value = ""
    processing.value = false
    canRestart.value = false
    lastError.value = ""
    phase.value = ""
    language.value = ""
    
    // 清除轮询定时器
    if (pollTimer.value) {
      clearTimeout(pollTimer.value)
      pollTimer.value = null
    }
    
    // 刷新文件列表
    loadFiles()
    ElMessage.success('已重置，可以重新选择文件')
  }).catch(() => {
    // 用户取消操作
  })
}

async function startJob() {
  if (!jobId.value) {
    ElMessage.warning('请先选择文件并创建任务')
    return
  }
  
  starting.value = true
  processing.value = true
  lastError.value = ""
  
  try {
    await TranscriptionService.startJob(jobId.value, settings)
    ElMessage.success('转录任务已启动！')
    poll() // 开始轮询状态
  } catch (e) {
    const errorMessage = "启动失败: " + (e?.message || e)
    statusText.value = errorMessage
    processing.value = false
    ElMessage.error(errorMessage)
  } finally {
    starting.value = false
  }
}

async function cancelJob() {
  if (!jobId.value) return
  
  try {
    await ElMessageBox.confirm('确定要取消当前转录任务吗？', '确认操作', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
    })
    
    canceling.value = true
    await TranscriptionService.cancelJob(jobId.value)
    ElMessage.success('任务已取消')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('取消任务失败')
    }
  } finally {
    canceling.value = false
  }
}

async function restartJob() {
  if (!jobId.value) return
  
  try {
    await ElMessageBox.confirm('确定要重新转录当前文件吗？', '确认操作', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
    })
    
    // 重置转录相关状态
    status.value = ""
    progress.value = 0
    phase.value = ""
    statusText.value = "重新开始转录"
    downloadUrl.value = ""
    lastError.value = ""
    language.value = ""
    canRestart.value = false
    
    // 清除之前的轮询
    if (pollTimer.value) {
      clearTimeout(pollTimer.value)
      pollTimer.value = null
    }
    
    await startJob()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('重新转录失败')
    }
  }
}

async function poll() {
  clearTimeout(pollTimer.value)
  if (!jobId.value) return
  
  try {
    const data = await TranscriptionService.getJobStatus(jobId.value)
    if (data.error) {
      statusText.value = data.error
      processing.value = false
      return
    }
    
    status.value = data.status
    progress.value = data.progress || 0
    statusText.value = data.message || data.status
    phase.value = data.phase || ""
    language.value = data.language || ""
    lastError.value = data.error || ""
    
    if (status.value === "finished") {
      processing.value = false
      downloadUrl.value = TranscriptionService.getDownloadUrl(jobId.value)
      canRestart.value = true
      ElMessage.success('转录完成！可以下载字幕文件了。')
    } else if (status.value === "failed" || status.value === "canceled") {
      processing.value = false
      canRestart.value = true
      if (status.value === "failed") {
        ElMessage.error('转录失败：' + (lastError.value || '未知错误'))
      }
    } else {
      // 继续轮询
      pollTimer.value = setTimeout(poll, 1500)
    }
  } catch (e) {
    // 网络错误：稍后重试
    pollTimer.value = setTimeout(poll, 2500)
  }
}

// 下载文件
function downloadFile() {
  if (downloadUrl.value) {
    window.open(downloadUrl.value, '_blank')
    ElMessage.success('开始下载字幕文件')
  }
}

// 复制结果到源目录
async function copyResultToSource() {
  if (!jobId.value) {
    ElMessage.warning('没有可复制的结果')
    return
  }
  
  try {
    await TranscriptionService.copyResultToSource(jobId.value)
    ElMessage.success('字幕文件已复制到源文件目录！')
  } catch (error) {
    console.error('复制结果失败:', error)
    ElMessage.error('复制结果失败：' + (error.response?.data?.detail || error.message))
  }
}

// 组件卸载时清理定时器
onUnmounted(() => {
  if (pollTimer.value) {
    clearTimeout(pollTimer.value)
  }
})

onMounted(() => {
  // 页面加载时自动获取文件列表
  loadFiles()
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
}

.header {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  text-align: center;
  padding: 30px 0;
}

.title {
  color: #409eff;
  font-size: 2.5rem;
  font-weight: 700;
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
}

.title .el-icon {
  font-size: 2.5rem;
}

.main-content {
  padding: 40px 20px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .title {
    font-size: 2rem;
  }
  
  .title .el-icon {
    font-size: 2rem;
  }
  
  .main-content {
    padding: 20px 10px;
  }
}

/* 自定义 Element Plus 样式 */
:deep(.el-card__header) {
  background: linear-gradient(90deg, #f8f9fa, #e9ecef);
  border-bottom: 2px solid #dee2e6;
}

:deep(.el-button) {
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.3s ease;
}

:deep(.el-tag) {
  border-radius: 6px;
  font-weight: 500;
}
</style>