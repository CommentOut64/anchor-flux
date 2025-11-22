<template>
  <el-container class="app-container">
    <!-- 顶部标题 -->
    <el-header class="header">
      <el-row justify="space-between" align="middle">
        <el-col :span="18">
          <h1 class="title">
            <el-icon><VideoPlay /></el-icon>
            Video To SRT 转录工具
          </h1>
        </el-col>
        <el-col :span="6" style="text-align: center; display: flex; gap: 0px; justify-content: flex-end;">
          <el-button type="success" @click="showModelDownload = true">
            <el-icon><Download /></el-icon>
            模型下载
          </el-button>
        </el-col>
      </el-row>
    </el-header>

    <!-- 主要内容区域 -->
    <el-main class="main-content">
      <el-row :gutter="20" justify="center">
        <el-col :xs="24" :sm="20" :md="16" :lg="14" :xl="12">
          <!-- 0. 未完成任务列表 -->
          <IncompleteJobs
            ref="incompleteJobsRef"
            :current-job-id="jobId"
            @restore-job="handleRestoreJob"
            @delete-job="handleDeleteJob"
          />

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
            :pausing="pausing"
            :can-restart="canRestart"
            :status="status"
            @start-job="startJob"
            @cancel-job="cancelJobConfirm"
            @pause-job="pauseJob"
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

    <!-- 模型预加载对话框 -->
    <ModelManager v-model="showModelPreload" />

    <!-- 模型下载管理对话框 -->
    <ModelDownloadManager v-model="showModelDownload" />
  </el-container>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from "vue";
import { ElMessage, ElMessageBox } from "element-plus";

// 导入组件
import FileSelector from "./components/file-management/FileSelector.vue";
import TranscriptionSettings from "./components/transcription/TranscriptionSettings.vue";
import ProgressDisplay from "./components/transcription/ProgressDisplay.vue";
import IncompleteJobs from "./components/transcription/IncompleteJobs.vue";
import HardwareDialog from "./components/hardware/HardwareDialog.vue";
import ModelStatusButton from "./components/models/ModelStatusButton.vue";
import ModelManager from "./components/models/ModelManager.vue";
import ModelDownloadManager from "./components/models/ModelDownloadManager.vue";

// 导入服务
import { FileService } from "./services/fileService.js";
import { TranscriptionService } from "./services/transcriptionService.js";

// 安全获取错误消息的辅助函数
const getErrorMessage = (error) => {
  if (!error) return "未知错误";
  return (
    error.response?.data?.detail ||
    error.message ||
    error.toString() ||
    "未知错误"
  );
};

// 文件选择相关
const availableFiles = ref([]);
const selectedFile = ref(null);
const loadingFiles = ref(false);
const creating = ref(false);
const inputDirPath = ref("input/");
const uploading = ref(false);
const uploadProgress = ref(0);
const showUpload = ref(false); // 默认使用本地input模式

// 硬件信息对话框
const showHardwareDialog = ref(false);

// 模型管理对话框
const showModelPreload = ref(false);  // 模型预加载管理
const showModelDownload = ref(false); // 模型下载管理

// 任务相关
const jobId = ref("");
const status = ref("");
const progress = ref(0);
const statusText = ref("请先选择文件");
const downloadUrl = ref("");
const processing = ref(false);
const starting = ref(false);
const canceling = ref(false);
const pausing = ref(false);  // 新增：暂停中状态
const lastError = ref("");
const phase = ref("");
const language = ref("");
const canRestart = ref(false);
const pollTimer = ref(null);

// 断点续传相关
const resumeInfo = ref(null); // 存储恢复信息
const incompleteJobsRef = ref(null); // 未完成任务组件引用

const settings = reactive({
  model: "medium",
  compute_type: "float16",
  device: "cuda",
  batch_size: 16,
  word_timestamps: false,
  // CPU亲和性配置
  cpu_affinity_enabled: true,
  cpu_affinity_strategy: "auto",
  cpu_affinity_custom_cores: null,
  cpu_affinity_exclude_cores: null,
});

// 文件上传处理
async function handleUpload(uploadFile) {
  if (!uploadFile) {
    ElMessage.warning("请选择文件");
    return;
  }

  // 检查文件类型
  if (!FileService.isSupportedFile(uploadFile.name)) {
    ElMessage.error("不支持的文件格式，请上传视频或音频文件");
    return;
  }

  uploading.value = true;
  uploadProgress.value = 0;

  try {
    const data = await TranscriptionService.uploadFile(
      uploadFile.raw,
      (progressEvent) => {
        uploadProgress.value = Math.round(
          (progressEvent.loaded / progressEvent.total) * 100
        );
      }
    );

    jobId.value = data.job_id;
    selectedFile.value = {
      name: data.filename,
      originalName: data.original_name,
      size: uploadFile.size,
    };
    status.value = "ready";
    statusText.value = "文件已上传，可开始转录";
    canRestart.value = false;
    showUpload.value = false;

    ElMessage.success("文件上传成功！转录任务已创建。");

    // 检查是否有断点可以恢复
    await checkResumeStatus();

    // 刷新文件列表
    loadFiles();
  } catch (error) {
    console.error("文件上传失败:", error);
    ElMessage.error("文件上传失败：" + getErrorMessage(error));
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
    const data = await FileService.getFiles();
    availableFiles.value = data.files || [];
    inputDirPath.value = data.input_dir || "input/";
    if (availableFiles.value.length === 0) {
      // 延迟一点时间确保组件完全初始化
      setTimeout(() => {
        try {
          ElMessage.info("input 目录中没有找到支持的媒体文件");
        } catch (error) {
          console.log("ElMessage 调用失败:", error);
        }
      }, 100);
    }
  } catch (error) {
    console.error("获取文件列表失败:", error);
    // 延迟一点时间确保组件完全初始化
    setTimeout(() => {
      try {
        ElMessage.error("获取文件列表失败：" + getErrorMessage(error));
      } catch (msgError) {
        console.log("ElMessage 调用失败:", msgError);
      }
    }, 100);
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
    ElMessage.warning("请先选择文件");
    return;
  }

  creating.value = true;
  try {
    const data = await TranscriptionService.createJob(selectedFile.value.name);
    jobId.value = data.job_id;
    status.value = "ready";
    statusText.value = "文件已准备就绪，可开始转录";
    canRestart.value = false;

    // 检查是否有断点可以恢复
    await checkResumeStatus();

    ElMessage.success("转录任务创建成功！");
  } catch (error) {
    console.error("创建任务失败:", error);
    ElMessage.error("创建任务失败：" + getErrorMessage(error));
  } finally {
    creating.value = false;
  }
}

// 检查断点续传状态
async function checkResumeStatus() {
  if (!jobId.value) return;

  try {
    const data = await TranscriptionService.checkResume(jobId.value);
    if (data.can_resume) {
      resumeInfo.value = data;
      ElMessage.info({
        message: data.message || `检测到上次进度 (${data.progress}%)，将从断点继续`,
        duration: 5000,
      });
    } else {
      resumeInfo.value = null;
    }
  } catch (error) {
    console.error("检查恢复状态失败:", error);
    resumeInfo.value = null;
  }
}

// 重置选择
function resetSelection() {
  ElMessageBox.confirm(
    "确定要重新选择文件吗？这将清除当前的转录进度。",
    "确认操作",
    {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    }
  )
    .then(() => {
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
      ElMessage.success("已重置，可以重新选择文件");
    })
    .catch(() => {
      // 用户取消操作
    });
}

async function startJob() {
  if (!jobId.value) {
    ElMessage.warning("请先选择文件并创建任务");
    return;
  }

  starting.value = true;
  processing.value = true;
  lastError.value = "";

  try {
    // 如果有恢复信息，显示提示
    if (resumeInfo.value && resumeInfo.value.can_resume) {
      ElMessage.info({
        message: `🔄 从断点继续（已完成 ${resumeInfo.value.processed_segments}/${resumeInfo.value.total_segments} 段）`,
        duration: 3000,
      });
    }

    await TranscriptionService.startJob(jobId.value, settings);
    ElMessage.success(resumeInfo.value?.can_resume ? "继续转录任务！" : "转录任务已启动！");

    // 清除恢复信息
    resumeInfo.value = null;

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
    await ElMessageBox.confirm("确定要取消当前转录任务吗？", "确认操作", {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: "warning",
    });

    canceling.value = true;
    await TranscriptionService.cancelJob(jobId.value);
    ElMessage.success("任务已取消");
  } catch (error) {
    if (error !== "cancel") {
      ElMessage.error("取消任务失败");
    }
  } finally {
    canceling.value = false;
  }
}

// 暂停任务
async function pauseJob() {
  if (!jobId.value) return;

  try {
    pausing.value = true;
    await TranscriptionService.pauseJob(jobId.value);
    ElMessage.success("任务暂停请求已发送，正在保存断点...");
  } catch (error) {
    ElMessage.error("暂停任务失败");
    console.error("暂停任务失败:", error);
  } finally {
    pausing.value = false;
  }
}

// 取消任务确认（带删除选项）
async function cancelJobConfirm() {
  if (!jobId.value) return;

  try {
    await ElMessageBox.confirm(
      "确定要取消当前任务吗？",
      "取消任务",
      {
        confirmButtonText: "仅取消",
        cancelButtonText: "返回",
        distinguishCancelAndClose: true,
        type: "warning"
      }
    );

    // 询问是否删除数据
    const deleteData = await ElMessageBox.confirm(
      "是否同时删除任务数据？\n删除后无法恢复！",
      "删除数据",
      {
        confirmButtonText: "删除数据",
        cancelButtonText: "保留数据",
        distinguishCancelAndClose: true,
        type: "warning"
      }
    ).then(() => true).catch(() => false);

    canceling.value = true;
    await TranscriptionService.cancelJob(jobId.value, deleteData);

    if (deleteData) {
      ElMessage.success("任务已取消并删除数据");
      // 重置状态
      resetSelection();
    } else {
      ElMessage.success("任务已取消，数据已保留");
    }

    // 刷新未完成任务列表
    if (incompleteJobsRef.value) {
      incompleteJobsRef.value.refreshJobs();
    }
  } catch (error) {
    if (error !== "cancel") {
      ElMessage.error("取消任务失败");
      console.error("取消任务失败:", error);
    }
  } finally {
    canceling.value = false;
  }
}

async function restartJob() {
  if (!jobId.value) return;

  // 先检查是否有断点
  await checkResumeStatus();

  const confirmMessage = resumeInfo.value?.can_resume
    ? `检测到上次进度 (${resumeInfo.value.progress}%)，确定要继续转录吗？`
    : "确定要重新转录当前文件吗？";

  try {
    await ElMessageBox.confirm(confirmMessage, "确认操作", {
      confirmButtonText: "确定",
      cancelButtonText: "取消",
      type: resumeInfo.value?.can_resume ? "info" : "warning",
    });

    // 重置转录相关状态
    status.value = "";
    progress.value = 0;
    phase.value = "";
    statusText.value = resumeInfo.value?.can_resume ? "继续转录" : "重新开始转录";
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
    if (error !== "cancel") {
      ElMessage.error("重新转录失败");
    }
  }
}

async function poll() {
  clearTimeout(pollTimer.value);
  if (!jobId.value) return;

  try {
    const data = await TranscriptionService.getJobStatus(jobId.value);
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
      downloadUrl.value = TranscriptionService.getDownloadUrl(jobId.value);
      canRestart.value = true;
      ElMessage.success("转录完成！可以下载字幕文件了。");

      // 刷新未完成任务列表
      if (incompleteJobsRef.value) {
        incompleteJobsRef.value.refreshJobs();
      }
    } else if (status.value === "failed" || status.value === "canceled") {
      processing.value = false;
      canRestart.value = true;
      if (status.value === "failed") {
        ElMessage.error("转录失败：" + (lastError.value || "未知错误"));
      }

      // 刷新未完成任务列表
      if (incompleteJobsRef.value) {
        incompleteJobsRef.value.refreshJobs();
      }
    } else if (status.value === "paused") {
      processing.value = false;
      canRestart.value = true;
      ElMessage.info("任务已暂停，可随时恢复");

      // 刷新未完成任务列表
      if (incompleteJobsRef.value) {
        incompleteJobsRef.value.refreshJobs();
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
    window.open(downloadUrl.value, "_blank");
    ElMessage.success("开始下载字幕文件");
  }
}

// 复制结果到源目录
async function copyResultToSource() {
  if (!jobId.value) {
    ElMessage.warning("没有可复制的结果");
    return;
  }

  try {
    await TranscriptionService.copyResultToSource(jobId.value);
    ElMessage.success("字幕文件已复制到源文件目录！");
  } catch (error) {
    console.error("复制结果失败:", error);
    ElMessage.error("复制结果失败：" + getErrorMessage(error));
  }
}

// 恢复任务
async function handleRestoreJob(job) {
  try {
    // 调用恢复接口
    const data = await TranscriptionService.restoreJob(job.job_id);

    // 更新当前任务信息
    jobId.value = data.job_id;
    selectedFile.value = {
      name: data.filename,
      size: 0
    };
    status.value = data.status;
    progress.value = data.progress || 0;
    statusText.value = data.message || "已恢复";
    phase.value = data.phase || "";
    processing.value = false;
    canRestart.value = true;

    ElMessage.success(`任务已恢复：${data.filename} (${data.progress}%)`);

    // 刷新未完成任务列表
    if (incompleteJobsRef.value) {
      incompleteJobsRef.value.refreshJobs();
    }
  } catch (error) {
    console.error("恢复任务失败:", error);
    ElMessage.error("恢复任务失败：" + getErrorMessage(error));
  }
}

// 删除任务处理
function handleDeleteJob(jobId) {
  // 如果删除的是当前任务，重置状态
  if (jobId === jobId.value) {
    resetSelection();
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

  // 页面加载完成后，启动模型预加载
  startInitialPreload();
});

// 初始模型预加载
async function startInitialPreload() {
  try {
    console.log("[App] 系统启动，准备自动预加载模型...");

    // 延迟10秒确保前后端完全就绪
    setTimeout(async () => {
      try {
        console.log("[App] 开始检查后端连接状态...");

        // 检查后端连接
        const pingResponse = await fetch("/api/ping", { timeout: 5000 });
        if (!pingResponse.ok) {
          console.log("[App] 后端连接失败，跳过自动预加载");
          return;
        }
        console.log("[App] 后端连接正常");

        // 检查当前预加载状态
        const statusResponse = await fetch("/api/models/preload/status", {
          timeout: 5000,
        });
        if (statusResponse.ok) {
          const statusResult = await statusResponse.json();
          if (statusResult.success) {
            const status = statusResult.data;
            console.log("[App] 当前预加载状态:", status);

            // 如果已经在预加载或已有模型，跳过
            if (status.is_preloading) {
              console.log("[App] 预加载已在进行中，跳过自动启动");
              return;
            }
            if (status.loaded_models > 0) {
              console.log("[App] 模型已预加载完成，跳过自动启动");
              return;
            }
          }
        }

        console.log("[App] 启动自动预加载...");
        const preloadResponse = await fetch("/api/models/preload/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        });

        if (preloadResponse.ok) {
          const result = await preloadResponse.json();
          if (result.success) {
            console.log("[App] ✅ 模型预加载已启动");
            // 使用 nextTick 确保组件完全初始化后再调用 ElMessage
            await new Promise((resolve) => setTimeout(resolve, 100));
            ElMessage.success("模型预加载已启动，可在右上角查看进度", {
              duration: 3000,
            });
          } else {
            console.log("[App] ⚠️ 预加载启动失败:", result.message);
            // 使用 nextTick 确保组件完全初始化后再调用 ElMessage
            await new Promise((resolve) => setTimeout(resolve, 100));
            ElMessage.info("模型将在首次使用时自动加载", { duration: 2000 });
          }
        } else {
          console.log(
            "[App] ❌ 预加载请求失败，状态码:",
            preloadResponse.status
          );
        }
      } catch (error) {
        const errorMsg = error?.message || error?.toString() || "未知错误";
        console.log("[App] ❌ 自动预加载异常:", errorMsg);
        // 使用 nextTick 确保组件完全初始化后再调用 ElMessage
        try {
          await new Promise((resolve) => setTimeout(resolve, 100));
          // 使用可选链操作符安全调用 ElMessage
          if (typeof ElMessage?.info === "function") {
            ElMessage.info("模型将在首次使用时自动加载", { duration: 2000 });
          }
        } catch (msgError) {
          const msgErrorMsg =
            msgError?.message || msgError?.toString() || "未知错误";
          console.log("[App] ElMessage 调用失败:", msgErrorMsg);
        }
      }
    }, 10000); // 延迟10秒
  } catch (error) {
    const errorMsg = error?.message || error?.toString() || "未知错误";
    console.log("[App] ❌ 预加载初始化失败:", errorMsg);
  }
}
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
  padding: 0px 0px;
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
