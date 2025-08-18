<template>
  <div class="layout">
    <header>
      <h1>Video To SRT 转录工具</h1>
    </header>
    <main>
      <!-- 上传 -->
      <section class="card">
        <h2>1. 上传媒体文件</h2>
        <input type="file" @change="onFile" accept="video/*,audio/*" />
        <p v-if="fileName">
          已选择: {{ fileName }}
          <span v-if="fileSize">({{ formatFileSize(fileSize) }})</span>
        </p>
        <button :disabled="!file || uploading" @click="doUpload">
          {{ uploading ? "上传中..." : "上传" }}
        </button>

        <!-- 上传进度条 -->
        <div v-if="uploading" class="upload-progress">
          <div class="progress-wrapper">
            <div
              class="progress-bar"
              :style="{ width: uploadProgress + '%' }"
            ></div>
          </div>
          <p>
            上传进度: {{ uploadProgress }}% ({{ formatSpeed(uploadSpeed) }})
          </p>
        </div>

        <p v-if="uploadError" class="error">{{ uploadError }}</p>
      </section>

      <!-- 参数设置 -->
      <section class="card" v-if="jobId">
        <h2>2. 设置参数并开始</h2>
        <form @submit.prevent="startJob">
          <div class="grid">
            <label
              >模型
              <select v-model="settings.model">
                <option value="tiny">tiny</option>
                <option value="base">base</option>
                <option value="small">small</option>
                <option value="medium">medium</option>
                <option value="large-v2">large-v2</option>
                <option value="large-v3">large-v3</option>
              </select>
            </label>
            <label
              >计算类型
              <select v-model="settings.compute_type">
                <option value="float16">float16</option>
                <option value="float32">float32</option>
                <option value="int8">int8</option>
              </select>
            </label>
            <label
              >设备
              <select v-model="settings.device">
                <option value="cuda">cuda</option>
                <option value="cpu">cpu</option>
              </select>
            </label>
            <label
              >批大小<input
                type="number"
                v-model.number="settings.batch_size"
                min="1"
            /></label>
            <label
              >词级时间戳
              <input type="checkbox" v-model="settings.word_timestamps" />
            </label>
          </div>
          <div class="actions">
            <button :disabled="processing || starting" type="submit">
              {{ starting ? "启动中..." : "开始转录" }}
            </button>
            <button
              type="button"
              class="secondary"
              :disabled="!processing || canceling"
              @click="cancelJob"
            >
              {{ canceling ? "取消中..." : "取消任务" }}
            </button>
            <button
              type="button"
              class="secondary"
              :disabled="processing || !canRestart"
              @click="restartJob"
            >
              重新开始
            </button>
          </div>
        </form>
      </section>

      <!-- 进度显示 -->
      <section class="card" v-if="jobId">
        <h2>3. 进度</h2>
        <div class="progress-wrapper">
          <div class="progress-bar" :style="{ width: progress + '%' }"></div>
        </div>
        <p class="phase">阶段: {{ phaseLabel }} ({{ progress }}%)</p>
        <p>{{ statusText }}</p>
        <p v-if="language">
          检测语言: <strong>{{ language }}</strong>
        </p>
        <p v-if="status === 'failed'" class="error">
          任务失败：{{ lastError }}
        </p>
        <a v-if="status === 'finished' && downloadUrl" :href="downloadUrl"
          >下载 SRT</a
        >
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from "vue";
import axios from "axios";

const file = ref(null);
const fileName = ref("");
const fileSize = ref(0);
const jobId = ref("");
const status = ref("");
const progress = ref(0);
const statusText = ref("等待上传");
const downloadUrl = ref("");
const processing = ref(false);
const uploading = ref(false);
const uploadProgress = ref(0);
const uploadSpeed = ref(0);
const uploadError = ref("");
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
  split: "分段",
  transcribe: "转录",
  srt: "生成字幕",
  pending: "等待",
  "": "等待",
};
const phaseLabel = computed(() => phaseMap[phase.value] || phase.value || "—");

function onFile(e) {
  file.value = e.target.files[0];
  fileName.value = file.value?.name || "";
  fileSize.value = file.value?.size || 0;
  uploadProgress.value = 0;
  uploadError.value = "";
}

// 格式化文件大小
function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// 格式化上传速度
function formatSpeed(bytesPerSecond) {
  if (bytesPerSecond === 0) return "0 B/s";
  const k = 1024;
  const sizes = ["B/s", "KB/s", "MB/s", "GB/s"];
  const i = Math.floor(Math.log(bytesPerSecond) / Math.log(k));
  return (
    parseFloat((bytesPerSecond / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  );
}

async function doUpload() {
  if (!file.value) return;
  uploading.value = true;
  uploadError.value = "";
  uploadProgress.value = 0;
  uploadSpeed.value = 0;

  try {
    const fd = new FormData();
    fd.append("file", file.value);

    const startTime = Date.now();
    let lastLoaded = 0;
    let lastTime = startTime;

    const { data } = await axios.post("/api/upload", fd, {
      timeout: 0, // 不设置超时
      onUploadProgress: (progressEvent) => {
        const now = Date.now();
        const loaded = progressEvent.loaded;
        const total = progressEvent.total;

        if (total) {
          uploadProgress.value = Math.round((loaded / total) * 100);
        }

        // 计算上传速度
        if (now - lastTime > 1000) {
          // 每秒更新一次速度
          const timeDiff = (now - lastTime) / 1000;
          const bytesDiff = loaded - lastLoaded;
          uploadSpeed.value = bytesDiff / timeDiff;
          lastLoaded = loaded;
          lastTime = now;
        }
      },
    });

    jobId.value = data.job_id;
    status.value = "uploaded";
    statusText.value = "文件已上传, 可开始转录";
    canRestart.value = false;
    uploadProgress.value = 100;
    console.log("上传成功:", data);
  } catch (e) {
    console.error("上传失败:", e);
    if (e.code === "ECONNABORTED") {
      uploadError.value = "上传超时，请检查网络连接或尝试上传较小的文件";
    } else if (e.response) {
      uploadError.value = `上传失败: ${e.response.status} ${e.response.statusText}`;
    } else if (e.request) {
      uploadError.value = "网络连接失败，请检查后端服务是否正常运行";
    } else {
      uploadError.value = "上传失败: " + (e.message || e);
    }
  } finally {
    uploading.value = false;
  }
}

async function startJob() {
  if (!jobId.value) return;
  starting.value = true;
  processing.value = true;
  lastError.value = "";
  try {
    const fd = new FormData();
    fd.append("job_id", jobId.value);
    fd.append("settings", JSON.stringify(settings));
    await axios.post("/api/start", fd);
    poll();
  } catch (e) {
    statusText.value = "启动失败: " + (e?.message || e);
    processing.value = false;
  } finally {
    starting.value = false;
  }
}

async function cancelJob() {
  if (!jobId.value) return;
  canceling.value = true;
  try {
    await axios.post(`/api/cancel/${jobId.value}`);
  } catch (e) {
    // ignore
  } finally {
    canceling.value = false;
  }
}

async function restartJob() {
  if (!jobId.value) return;
  status.value = "";
  progress.value = 0;
  phase.value = "";
  statusText.value = "重新开始";
  await startJob();
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
    } else if (status.value === "failed" || status.value === "canceled") {
      processing.value = false;
      canRestart.value = true;
    } else {
      pollTimer.value = setTimeout(poll, 1500);
    }
  } catch (e) {
    // 网络错误：稍后重试
    pollTimer.value = setTimeout(poll, 2500);
  }
}

onMounted(() => {});
</script>

<style>
body,
html {
  margin: 0;
  padding: 0;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, "Noto Sans", sans-serif;
  background: #f5f7fa;
  color: #222;
}
.layout {
  max-width: 960px;
  margin: 0 auto;
  padding: 24px;
}
header {
  text-align: center;
  margin-bottom: 24px;
}
h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: 1px;
}
main {
  display: flex;
  flex-direction: column;
  gap: 24px;
}
.card {
  background: #fff;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 16px -4px rgba(0, 0, 0, 0.08);
  border: 1px solid #e5e9ef;
}
.card h2 {
  margin-top: 0;
  font-size: 18px;
  display: flex;
  align-items: center;
  gap: 8px;
}
button {
  background: #4f46e5;
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
  transition: 0.2s;
}
button:hover:not(:disabled) {
  background: #4338ca;
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px;
  margin: 12px 0;
}
label {
  font-size: 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  color: #555;
}
select,
input[type="number"] {
  padding: 6px 8px;
  border: 1px solid #ccc;
  border-radius: 6px;
  font-size: 14px;
  background: #fff;
}
.progress-wrapper {
  height: 16px;
  background: #eef1f5;
  border-radius: 8px;
  overflow: hidden;
  margin: 8px 0 4px;
  position: relative;
}
.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #6366f1, #818cf8);
  transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
a {
  color: #2563eb;
  text-decoration: none;
  font-weight: 500;
}
a:hover {
  text-decoration: underline;
}
.actions {
  display: flex;
  gap: 12px;
  margin-top: 12px;
  flex-wrap: wrap;
}
button.secondary {
  background: #64748b;
}
button.secondary:hover:not(:disabled) {
  background: #475569;
}
.error {
  color: #dc2626;
  font-size: 13px;
  margin: 4px 0;
}
.phase {
  font-size: 13px;
  color: #555;
  margin: 4px 0;
}
.upload-progress {
  margin-top: 12px;
  padding: 12px;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}
.upload-progress .progress-wrapper {
  margin: 8px 0;
}
.upload-progress p {
  margin: 4px 0;
  font-size: 13px;
  color: #475569;
}
</style>
