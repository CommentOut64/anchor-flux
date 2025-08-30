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
          <!-- <p class="subtitle">基于 WhisperX 的高性能视频字幕生成工具</p> -->
        </el-col>
      </el-row>
    </el-header>

    <!-- 主要内容区域 -->
    <el-main class="main-content">
      <el-row :gutter="20" justify="center">
        <el-col :xs="24" :sm="20" :md="16" :lg="14" :xl="12">
          
          <!-- 1. 文件选择区域 -->
          <el-card class="upload-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><FolderOpened /></el-icon>
                <span>1. 选择媒体文件</span>
                <div style="margin-left: auto; display: flex; gap: 8px;">
                  <el-button 
                    type="info" 
                    size="small"
                    @click="showHardwareDialog = true"
                    plain
                  >
                    <el-icon><Monitor /></el-icon>
                    硬件信息
                  </el-button>
                  <el-button 
                    type="success" 
                    size="small"
                    @click="toggleUploadMode"
                    :disabled="creating || uploading"
                  >
                    <el-icon><Upload /></el-icon>
                    {{ showUpload ? '切换到本地模式' : '切换到上传模式' }}
                  </el-button>
                  <el-button 
                    type="primary" 
                    size="small"
                    @click="loadFiles"
                    :loading="loadingFiles"
                    v-if="!showUpload"
                  >
                    <el-icon><Refresh /></el-icon>
                    刷新
                  </el-button>
                </div>
              </div>
            </template>
            
            <div class="file-selection-area">
              <!-- 上传模式 -->
              <div v-if="showUpload" class="upload-mode">
                <el-upload
                  class="upload-area"
                  drag
                  :auto-upload="false"
                  :on-change="handleUpload"
                  :show-file-list="false"
                  :disabled="uploading"
                  accept=".mp4,.avi,.mkv,.mov,.wmv,.flv,.webm,.m4v,.mp3,.wav,.flac,.aac,.ogg,.m4a,.wma"
                >
                  <div class="upload-content">
                    <el-icon class="upload-icon"><UploadFilled /></el-icon>
                    <div class="upload-text" v-if="!uploading">
                      拖拽文件到此处，或<em>点击选择</em>
                    </div>
                    <div class="upload-text" v-else>
                      正在上传... {{ uploadProgress }}%
                    </div>
                    <div class="upload-hint">
                      支持格式：MP4, AVI, MKV, MOV, WMV, MP3, WAV, FLAC 等
                    </div>
                  </div>
                </el-upload>
                
                <!-- 上传进度条 -->
                <el-progress
                  v-if="uploading"
                  :percentage="uploadProgress"
                  :stroke-width="8"
                  status="active"
                  style="margin-top: 16px;"
                />
              </div>
              
              <!-- 浏览模式 -->
              <div v-else class="browse-mode">
                <!-- 目录提示 -->
                <el-alert
                  title="请将视频/音频文件放入 input 目录中"
                  type="info"
                  :closable="false"
                  show-icon
                >
                  <template #default>
                    <p>支持格式：MP4, AVI, MKV, MOV, WMV, MP3, WAV, FLAC 等</p>
                    <p>目录路径：<code>{{ inputDirPath }}</code></p>
                  </template>
                </el-alert>

              <!-- 文件列表 -->
              <div v-if="availableFiles.length > 0" class="file-list">
                <el-table 
                  :data="availableFiles" 
                  style="width: 100%"
                  @row-click="selectFile"
                  :highlight-current-row="true"
                  :current-row-key="selectedFile?.name"
                  row-key="name"
                >
                  <el-table-column type="selection" width="40" />
                  <el-table-column prop="name" label="文件名" min-width="200">
                    <template #default="scope">
                      <div class="file-name">
                        <el-icon><VideoPlay v-if="isVideoFile(scope.row.name)" /><Headphone v-else /></el-icon>
                        {{ scope.row.name }}
                      </div>
                    </template>
                  </el-table-column>
                  <el-table-column prop="size" label="大小" width="100">
                    <template #default="scope">
                      {{ formatFileSize(scope.row.size) }}
                    </template>
                  </el-table-column>
                  <el-table-column prop="modified" label="修改时间" width="160" />
                  <el-table-column label="操作" width="120">
                    <template #default="scope">
                      <el-button
                        type="primary"
                        size="small"
                        @click.stop="selectFile(scope.row)"
                        :disabled="creating"
                      >
                        选择
                      </el-button>
                    </template>
                  </el-table-column>
                </el-table>
              </div>

              <!-- 无文件提示 -->
              <el-empty v-else-if="!loadingFiles" description="input 目录中没有找到支持的媒体文件">
                <el-button type="primary" @click="loadFiles">刷新文件列表</el-button>
              </el-empty>

              <!-- 加载中 -->
              <div v-if="loadingFiles" class="loading-files">
                <el-icon class="is-loading"><Loading /></el-icon>
                <span>正在加载文件列表...</span>
              </div>
              </div>

              <!-- 选中的文件 -->
              <div v-if="selectedFile && !showUpload" class="selected-file">
                <el-tag type="success" size="large" closable @close="clearSelection">
                  <el-icon><Document /></el-icon>
                  已选择：{{ selectedFile.name }} 
                  <span v-if="selectedFile.size">({{ formatFileSize(selectedFile.size) }})</span>
                  <span v-if="selectedFile.originalName && selectedFile.originalName !== selectedFile.name">
                    <br><small>原文件名: {{ selectedFile.originalName }}</small>
                  </span>
                </el-tag>
                
                <el-button 
                  v-if="!jobId"
                  type="primary" 
                  size="large"
                  :loading="creating"
                  @click="createJob"
                  style="margin-top: 12px;"
                >
                  <el-icon v-if="!creating"><FolderAdd /></el-icon>
                  {{ creating ? "创建中..." : "创建转录任务" }}
                </el-button>
              </div>
            </div>
          </el-card>

          <!-- 2. 参数设置区域 -->
          <el-card v-if="jobId" class="settings-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><Setting /></el-icon>
                <span>2. 设置参数并开始转录</span>
              </div>
            </template>
            
            <el-form 
              :model="settings" 
              label-width="100px" 
              label-position="left"
              @submit.prevent="startJob"
            >
              <el-row :gutter="20">
                <el-col :span="12">
                  <el-form-item label="模型选择">
                    <el-select v-model="settings.model" placeholder="选择模型">
                      <el-option label="tiny (最快)" value="tiny" />
                      <el-option label="base (较快)" value="base" />
                      <el-option label="small (平衡)" value="small" />
                      <el-option label="medium (推荐)" value="medium" />
                      <el-option label="large-v2 (高精度)" value="large-v2" />
                      <el-option label="large-v3 (最高精度)" value="large-v3" />
                    </el-select>
                  </el-form-item>
                </el-col>
                <el-col :span="12">
                  <el-form-item label="计算类型">
                    <el-select v-model="settings.compute_type">
                      <el-option label="float16 (推荐)" value="float16" />
                      <el-option label="float32 (高精度)" value="float32" />
                      <el-option label="int8 (省内存)" value="int8" />
                    </el-select>
                  </el-form-item>
                </el-col>
                <el-col :span="12">
                  <el-form-item label="运行设备">
                    <el-select v-model="settings.device">
                      <el-option label="CUDA GPU" value="cuda" />
                      <el-option label="CPU" value="cpu" />
                    </el-select>
                  </el-form-item>
                </el-col>
                <el-col :span="12">
                  <el-form-item label="批大小">
                    <el-input-number 
                      v-model="settings.batch_size" 
                      :min="1" 
                      :max="32"
                      controls-position="right"
                    />
                  </el-form-item>
                </el-col>
                <el-col :span="24">
                  <el-form-item>
                    <el-checkbox v-model="settings.word_timestamps">
                      启用词级时间戳（更精确但较慢）
                    </el-checkbox>
                  </el-form-item>
                </el-col>
              </el-row>

              <!-- 操作按钮 -->
              <div class="action-buttons">
                <el-button 
                  type="primary" 
                  size="large"
                  :loading="starting"
                  :disabled="processing || starting"
                  @click="startJob"
                >
                  <el-icon v-if="!starting"><VideoPlay /></el-icon>
                  {{ starting ? "启动中..." : "开始转录" }}
                </el-button>
                
                <el-button 
                  type="danger"
                  size="large"
                  :loading="canceling"
                  :disabled="!processing || canceling"
                  @click="cancelJob"
                >
                  <el-icon v-if="!canceling"><VideoPause /></el-icon>
                  {{ canceling ? "取消中..." : "取消任务" }}
                </el-button>
                
                <el-button 
                  type="warning"
                  size="large"
                  :disabled="processing || !canRestart"
                  @click="restartJob"
                >
                  <el-icon><RefreshRight /></el-icon>
                  重新转录
                </el-button>
                
                <el-button 
                  type="success"
                  size="large"
                  @click="resetSelection"
                >
                  <el-icon><FolderOpened /></el-icon>
                  重新选择文件
                </el-button>
              </div>
            </el-form>
          </el-card>

          <!-- 3. 进度显示区域 -->
          <el-card v-if="jobId" class="progress-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <el-icon><Timer /></el-icon>
                <span>3. 转录进度</span>
              </div>
            </template>
            
            <div class="progress-content">
              <!-- 进度条 -->
              <el-progress
                :percentage="progress"
                :stroke-width="12"
                :status="getProgressStatus()"
              />
              
              <!-- 状态信息 -->
              <div class="status-info">
                <el-tag :type="getStatusType()" size="large">
                  <el-icon><Clock /></el-icon>
                  {{ phaseLabel }} ({{ progress }}%)
                </el-tag>
                <p class="status-text">{{ statusText }}</p>
                <p v-if="language" class="language-info">
                  <el-icon><Flag /></el-icon>
                  检测语言: <strong>{{ language }}</strong>
                </p>
              </div>

              <!-- 错误信息 -->
              <el-alert
                v-if="status === 'failed'"
                :title="'任务失败：' + lastError"
                type="error"
                show-icon
                :closable="false"
              />

              <!-- 下载按钮 -->
              <div v-if="status === 'finished' && downloadUrl" class="download-section">
                <el-button 
                  type="success" 
                  size="large"
                  @click="downloadFile"
                >
                  <el-icon><Download /></el-icon>
                  下载 SRT 字幕文件
                </el-button>
                
                <el-button 
                  type="warning" 
                  size="large"
                  @click="copyResultToSource"
                  style="margin-left: 12px;"
                >
                  <el-icon><FolderAdd /></el-icon>
                  复制到源文件目录
                </el-button>
              </div>
            </div>
          </el-card>

        </el-col>
      </el-row>
    </el-main>
    
    <!-- 硬件信息对话框 -->
    <HardwareDialog v-model="showHardwareDialog" />
  </el-container>
</template>

<script setup>
import { ref, reactive, onMounted, computed, onUnmounted } from "vue";
import axios from "axios";
import { ElMessage, ElMessageBox } from 'element-plus';
import HardwareDialog from './components/hardware/HardwareDialog.vue';

// 文件选择相关
const availableFiles = ref([]);
const selectedFile = ref(null);
const loadingFiles = ref(false);
const creating = ref(false);
const inputDirPath = ref('input/');
const uploading = ref(false);
const uploadProgress = ref(0);
const showUpload = ref(false); // 默认使用本地input模式

// 硬件信息对话框
const showHardwareDialog = ref(false);

// 任务相关 
const jobId = ref("");
const status = ref("");
const progress = ref(0);
const statusText = ref("请先选择文件");
const downloadUrl = ref("");
const processing = ref(false);
const starting = ref(false);
const canceling = ref(false);
const lastError = ref("");
const phase = ref("");
const language = ref("");
const canRestart = ref(false);
const pollTimer = ref(null);

const settings = reactive({
  model: "medium",
  compute_type: "float16",
  device: "cuda",
  batch_size: 16,
  word_timestamps: false,
});

const phaseMap = {
  extract: "提取音频",
  split: "分段处理",
  transcribe: "语音转录",
  srt: "生成字幕",
  pending: "等待处理",
  "": "等待处理",
};

const phaseLabel = computed(() => phaseMap[phase.value] || phase.value || "等待处理");

// 获取进度条状态
function getProgressStatus() {
  if (status.value === 'finished') return 'success';
  if (status.value === 'failed') return 'exception';
  return 'active';
}

// 获取状态标签类型
function getStatusType() {
  switch (status.value) {
    case 'finished': return 'success';
    case 'failed': return 'danger';
    case 'canceled': return 'warning';
    case 'processing': return 'primary';
    default: return 'info';
  }
}

// 检查是否为视频文件
function isVideoFile(filename) {
  const videoExtensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'];
  const ext = filename.toLowerCase().substring(filename.lastIndexOf('.'));
  return videoExtensions.includes(ext);
}

// 文件上传处理
async function handleUpload(uploadFile) {
  if (!uploadFile) {
    ElMessage.warning('请选择文件');
    return;
  }
  
  // 检查文件类型
  const videoExtensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'];
  const audioExtensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'];
  const ext = uploadFile.name.toLowerCase().substring(uploadFile.name.lastIndexOf('.'));
  
  if (![...videoExtensions, ...audioExtensions].includes(ext)) {
    ElMessage.error('不支持的文件格式，请上传视频或音频文件');
    return;
  }
  
  uploading.value = true;
  uploadProgress.value = 0;
  
  try {
    const formData = new FormData();
    formData.append('file', uploadFile.raw);
    
    const { data } = await axios.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: (progressEvent) => {
        uploadProgress.value = Math.round(
          (progressEvent.loaded / progressEvent.total) * 100
        );
      }
    });
    
    jobId.value = data.job_id;
    selectedFile.value = {
      name: data.filename,
      originalName: data.original_name,
      size: uploadFile.size
    };
    status.value = "ready";
    statusText.value = "文件已上传，可开始转录";
    canRestart.value = false;
    showUpload.value = false;
    
    ElMessage.success('文件上传成功！转录任务已创建。');
    
    // 刷新文件列表
    loadFiles();
  } catch (error) {
    console.error('文件上传失败:', error);
    ElMessage.error('文件上传失败：' + (error.response?.data?.detail || error.message));
  } finally {
    uploading.value = false;
    uploadProgress.value = 0;
  }
}

// 切换上传模式
function toggleUploadMode() {
  showUpload.value = !showUpload.value;
  if (showUpload.value) {
    // 清除已选择的文件
    selectedFile.value = null;
  }
}

// 加载可用文件列表
async function loadFiles() {
  loadingFiles.value = true;
  try {
    const { data } = await axios.get('/api/files');
    availableFiles.value = data.files || [];
    if (availableFiles.value.length === 0) {
      ElMessage.info('input 目录中没有找到支持的媒体文件');
    }
  } catch (error) {
    console.error('获取文件列表失败:', error);
    ElMessage.error('获取文件列表失败：' + (error.response?.data?.detail || error.message));
  } finally {
    loadingFiles.value = false;
  }
}

// 选择文件
function selectFile(file) {
  selectedFile.value = file;
  ElMessage.success(`已选择文件：${file.name}`);
}

// 清除选择
function clearSelection() {
  selectedFile.value = null;
}

// 创建转录任务
async function createJob() {
  if (!selectedFile.value) {
    ElMessage.warning('请先选择文件');
    return;
  }
  
  creating.value = true;
  try {
    const fd = new FormData();
    fd.append('filename', selectedFile.value.name);
    
    const { data } = await axios.post('/api/create-job', fd);
    jobId.value = data.job_id;
    status.value = "ready";
    statusText.value = "文件已准备就绪，可开始转录";
    canRestart.value = false;
    
    ElMessage.success('转录任务创建成功！');
  } catch (error) {
    console.error('创建任务失败:', error);
    ElMessage.error('创建任务失败：' + (error.response?.data?.detail || error.message));
  } finally {
    creating.value = false;
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
    selectedFile.value = null;
    jobId.value = "";
    status.value = "";
    progress.value = 0;
    statusText.value = "请先选择文件";
    downloadUrl.value = "";
    processing.value = false;
    canRestart.value = false;
    lastError.value = "";
    phase.value = "";
    language.value = "";
    
    // 清除轮询定时器
    if (pollTimer.value) {
      clearTimeout(pollTimer.value);
      pollTimer.value = null;
    }
    
    // 刷新文件列表
    loadFiles();
    ElMessage.success('已重置，可以重新选择文件');
  }).catch(() => {
    // 用户取消操作
  });
}

// 格式化文件大小
function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

async function startJob() {
  if (!jobId.value) {
    ElMessage.warning('请先选择文件并创建任务');
    return;
  }
  
  starting.value = true;
  processing.value = true;
  lastError.value = "";
  
  try {
    const fd = new FormData();
    fd.append("job_id", jobId.value);
    fd.append("settings", JSON.stringify(settings));
    await axios.post("/api/start", fd);
    
    ElMessage.success('转录任务已启动！');
    poll(); // 开始轮询状态
  } catch (e) {
    const errorMessage = "启动失败: " + (e?.message || e);
    statusText.value = errorMessage;
    processing.value = false;
    ElMessage.error(errorMessage);
  } finally {
    starting.value = false;
  }
}

async function cancelJob() {
  if (!jobId.value) return;
  
  try {
    await ElMessageBox.confirm('确定要取消当前转录任务吗？', '确认操作', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
    });
    
    canceling.value = true;
    await axios.post(`/api/cancel/${jobId.value}`);
    ElMessage.success('任务已取消');
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('取消任务失败');
    }
  } finally {
    canceling.value = false;
  }
}

async function restartJob() {
  if (!jobId.value) return;
  
  try {
    await ElMessageBox.confirm('确定要重新转录当前文件吗？', '确认操作', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning',
    });
    
    // 重置转录相关状态
    status.value = "";
    progress.value = 0;
    phase.value = "";
    statusText.value = "重新开始转录";
    downloadUrl.value = "";
    lastError.value = "";
    language.value = "";
    canRestart.value = false;
    
    // 清除之前的轮询
    if (pollTimer.value) {
      clearTimeout(pollTimer.value);
      pollTimer.value = null;
    }
    
    await startJob();
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('重新转录失败');
    }
  }
}

async function poll() {
  clearTimeout(pollTimer.value);
  if (!jobId.value) return;
  
  try {
    const { data } = await axios.get(`/api/status/${jobId.value}`);
    if (data.error) {
      statusText.value = data.error;
      processing.value = false;
      return;
    }
    
    status.value = data.status;
    progress.value = data.progress || 0;
    statusText.value = data.message || data.status;
    phase.value = data.phase || "";
    language.value = data.language || "";
    lastError.value = data.error || "";
    
    if (status.value === "finished") {
      processing.value = false;
      downloadUrl.value = `/api/download/${jobId.value}`;
      canRestart.value = true;
      ElMessage.success('转录完成！可以下载字幕文件了。');
    } else if (status.value === "failed" || status.value === "canceled") {
      processing.value = false;
      canRestart.value = true;
      if (status.value === "failed") {
        ElMessage.error('转录失败：' + (lastError.value || '未知错误'));
      }
    } else {
      // 继续轮询
      pollTimer.value = setTimeout(poll, 1500);
    }
  } catch (e) {
    // 网络错误：稍后重试
    pollTimer.value = setTimeout(poll, 2500);
  }
}

// 下载文件
function downloadFile() {
  if (downloadUrl.value) {
    window.open(downloadUrl.value, '_blank');
    ElMessage.success('开始下载字幕文件');
  }
}

// 复制结果到源目录
async function copyResultToSource() {
  if (!jobId.value) {
    ElMessage.warning('没有可复制的结果');
    return;
  }
  
  try {
    const { data } = await axios.post(`/api/copy-result/${jobId.value}`);
    ElMessage.success('字幕文件已复制到源文件目录！');
  } catch (error) {
    console.error('复制结果失败:', error);
    ElMessage.error('复制结果失败：' + (error.response?.data?.detail || error.message));
  }
}

// 组件卸载时清理定时器
onUnmounted(() => {
  if (pollTimer.value) {
    clearTimeout(pollTimer.value);
  }
});

onMounted(() => {
  // 页面加载时自动获取文件列表
  loadFiles();
});
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  /* background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); */
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
  /* text-shadow: 0 2px 4px rgba(91, 90, 90, 0.3); */
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
}

.title .el-icon {
  font-size: 2.5rem;
}

.subtitle {
  color: rgba(255, 255, 255, 0.9);
  font-size: 1.1rem;
  margin: 10px 0 0;
  font-weight: 300;
}

.main-content {
  padding: 40px 20px;
}

.upload-card,
.settings-card,
.progress-card {
  margin-bottom: 20px;
  border-radius: 16px;
  overflow: hidden;
  border: none;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 1.1rem;
  font-weight: 600;
  color: #2c3e50;
}

.file-selection-area {
  text-align: left;
}

.file-list {
  margin: 20px 0;
}

.file-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.selected-file {
  text-align: center;
  margin: 20px 10px;
}

.loading-files {
  text-align: center;
  padding: 40px 0;
  color: #606266;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.upload-mode {
  margin-bottom: 20px;
}

.upload-content {
  text-align: center;
  padding: 60px 20px;
}

.browse-mode {
  /* 保持现有布局 */
}

.upload-area {
  text-align: center;
}

.upload-icon {
  font-size: 4rem;
  color: #409eff;
  margin-bottom: 16px;
}

.upload-text {
  font-size: 1.2rem;
  color: #606266;
  margin-bottom: 8px;
}

.upload-hint {
  color: #909399;
  font-size: 0.9rem;
}

.file-info {
  margin: 20px 0;
}

.upload-actions {
  margin-top: 20px;
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
}

.upload-progress {
  margin-top: 20px;
}

.progress-text {
  margin-top: 8px;
  color: #606266;
  font-size: 0.9rem;
}

.action-buttons {
  text-align: center;
  margin-top: 20px;
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
}

.progress-content {
  text-align: center;
}

.status-info {
  margin: 20px 0;
}

.status-text {
  margin: 12px 0;
  font-size: 1rem;
  color: #606266;
}

.language-info {
  color: #67c23a;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

.download-section {
  margin-top: 20px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .title {
    font-size: 2rem;
  }
  
  .title .el-icon {
    font-size: 2rem;
  }
  
  .subtitle {
    font-size: 1rem;
  }
  
  .main-content {
    padding: 20px 10px;
  }
  
  .action-buttons,
  .upload-actions {
    flex-direction: column;
    align-items: center;
  }
  
  .action-buttons .el-button,
  .upload-actions .el-button {
    width: 200px;
  }
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

:deep(.el-card__header) {
  background: linear-gradient(90deg, #f8f9fa, #e9ecef);
  border-bottom: 2px solid #dee2e6;
}

:deep(.el-progress-bar__outer) {
  border-radius: 8px;
}

:deep(.el-progress-bar__inner) {
  border-radius: 8px;
  background: linear-gradient(90deg, #67c23a, #85ce61);
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

:deep(.el-select),
:deep(.el-input-number) {
  width: 100%;
}
</style>
