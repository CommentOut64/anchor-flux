<template>
  <el-dialog
    v-model="visible"
    title="硬件信息详情"
    width="800px"
    :before-close="handleClose"
    class="hardware-dialog"
    destroy-on-close
  >
    <div v-if="loading" class="loading-content">
      <el-skeleton :rows="8" animated />
      <div class="loading-text">正在检测硬件信息...</div>
    </div>

    <div v-else-if="error" class="error-content">
      <el-alert
        title="硬件检测失败"
        type="error"
        :description="error"
        show-icon
        :closable="false"
      />
      <div class="retry-section">
        <el-button type="primary" @click="refreshHardware" :loading="loading">
          <el-icon><Refresh /></el-icon>
          重新检测
        </el-button>
      </div>
    </div>

    <div v-else-if="hardwareStatus" class="hardware-detail-content">
      <!-- GPU 详细信息 -->
      <el-card class="info-card" shadow="hover">
        <template #header>
          <div class="card-header">
            <el-icon><Monitor /></el-icon>
            <span>GPU 详细信息</span>
          </div>
        </template>

        <div class="detail-grid">
          <div class="detail-item">
            <span class="label">CUDA 支持:</span>
            <el-tag
              :type="
                hardwareStatus.hardware.gpu.cuda_available
                  ? 'success'
                  : 'danger'
              "
              size="small"
            >
              {{
                hardwareStatus.hardware.gpu.cuda_available
                  ? "✓ 可用"
                  : "✗ 不可用"
              }}
            </el-tag>
          </div>

          <div class="detail-item">
            <span class="label">GPU 数量:</span>
            <span class="value"
              >{{ hardwareStatus.hardware.gpu.count }} 个</span
            >
          </div>

          <div class="detail-item">
            <span class="label">总显存:</span>
            <span class="value">{{
              formatMemory(hardwareStatus.hardware.gpu.total_memory_mb)
            }}</span>
          </div>

          <div class="detail-item">
            <span class="label">可用显存:</span>
            <span class="value">{{
              formatMemory(hardwareStatus.hardware.gpu.available_memory_mb)
            }}</span>
          </div>

          <div class="detail-item">
            <span class="label">显存使用率:</span>
            <div class="progress-container">
              <el-progress
                :percentage="hardwareStatus.hardware.gpu.memory_usage_percent"
                :stroke-width="8"
                :color="
                  getMemoryColor(
                    hardwareStatus.hardware.gpu.memory_usage_percent
                  )
                "
                :show-text="false"
                style="width: 120px"
              />
              <span class="percentage"
                >{{ hardwareStatus.hardware.gpu.memory_usage_percent }}%</span
              >
            </div>
          </div>

          <div
            class="detail-item"
            v-if="hardwareStatus.hardware.gpu.cuda_version"
          >
            <span class="label">CUDA 版本:</span>
            <span class="value">{{
              hardwareStatus.hardware.gpu.cuda_version
            }}</span>
          </div>
        </div>
      </el-card>

      <!-- CPU 详细信息 -->
      <el-card class="info-card" shadow="hover">
        <template #header>
          <div class="card-header">
            <el-icon><Cpu /></el-icon>
            <span>CPU 详细信息</span>
          </div>
        </template>

        <div class="detail-grid">
          <div class="detail-item">
            <span class="label">物理核心:</span>
            <span class="value"
              >{{ hardwareStatus.hardware.cpu.cores }} 核</span
            >
          </div>

          <div class="detail-item">
            <span class="label">逻辑线程:</span>
            <span class="value"
              >{{ hardwareStatus.hardware.cpu.threads }} 线程</span
            >
          </div>

          <div class="detail-item">
            <span class="label">CPU 使用率:</span>
            <div class="progress-container">
              <el-progress
                :percentage="hardwareStatus.hardware.cpu.usage_percent || 0"
                :stroke-width="8"
                :color="
                  getCpuColor(hardwareStatus.hardware.cpu.usage_percent || 0)
                "
                :show-text="false"
                style="width: 120px"
              />
              <span class="percentage"
                >{{ hardwareStatus.hardware.cpu.usage_percent || 0 }}%</span
              >
            </div>
          </div>

          <div class="detail-item" v-if="hardwareStatus.hardware.cpu.name">
            <span class="label">处理器型号:</span>
            <span class="value">{{ hardwareStatus.hardware.cpu.name }}</span>
          </div>

          <div class="detail-item" v-if="hardwareStatus.hardware.cpu.frequency">
            <span class="label">基础频率:</span>
            <span class="value"
              >{{ hardwareStatus.hardware.cpu.frequency }} MHz</span
            >
          </div>
        </div>
      </el-card>

      <!-- 内存详细信息 -->
      <el-card class="info-card" shadow="hover">
        <template #header>
          <div class="card-header">
            <el-icon><Monitor /></el-icon>
            <span>内存详细信息</span>
          </div>
        </template>

        <div class="detail-grid">
          <div class="detail-item">
            <span class="label">总内存:</span>
            <span class="value">{{
              formatMemory(hardwareStatus.hardware.memory.total_mb)
            }}</span>
          </div>

          <div class="detail-item">
            <span class="label">已使用:</span>
            <span class="value">{{
              formatMemory(hardwareStatus.hardware.memory.used_mb)
            }}</span>
          </div>

          <div class="detail-item">
            <span class="label">可用内存:</span>
            <span class="value">{{
              formatMemory(hardwareStatus.hardware.memory.available_mb)
            }}</span>
          </div>

          <div class="detail-item">
            <span class="label">使用率:</span>
            <div class="progress-container">
              <el-progress
                :percentage="hardwareStatus.hardware.memory.usage_percent"
                :stroke-width="8"
                :color="
                  getMemoryColor(hardwareStatus.hardware.memory.usage_percent)
                "
                :show-text="false"
                style="width: 120px"
              />
              <span class="percentage"
                >{{ hardwareStatus.hardware.memory.usage_percent }}%</span
              >
            </div>
          </div>

          <div
            class="detail-item"
            v-if="hardwareStatus.hardware.memory.swap_total_mb"
          >
            <span class="label">交换文件总量:</span>
            <span class="value">{{
              formatMemory(hardwareStatus.hardware.memory.swap_total_mb)
            }}</span>
          </div>

          <div
            class="detail-item"
            v-if="hardwareStatus.hardware.memory.swap_used_mb"
          >
            <span class="label">交换文件已用:</span>
            <span class="value">{{
              formatMemory(hardwareStatus.hardware.memory.swap_used_mb)
            }}</span>
          </div>
        </div>
      </el-card>

      <!-- 系统优化建议 -->
      <el-card class="info-card" shadow="hover">
        <template #header>
          <div class="card-header">
            <el-icon><Setting /></el-icon>
            <span>系统优化建议</span>
          </div>
        </template>

        <div class="optimization-content">
          <div class="optimization-section">
            <h4 class="subsection-title">转录优化配置</h4>
            <div class="detail-grid">
              <div class="detail-item">
                <span class="label">推荐设备:</span>
                <el-tag
                  :type="
                    hardwareStatus.optimization.transcription.device === 'cuda'
                      ? 'success'
                      : 'warning'
                  "
                  size="small"
                >
                  {{
                    hardwareStatus.optimization.transcription.device.toUpperCase()
                  }}
                </el-tag>
              </div>

              <div class="detail-item">
                <span class="label">批处理大小:</span>
                <span class="value">{{
                  hardwareStatus.optimization.transcription.batch_size
                }}</span>
              </div>

              <div class="detail-item">
                <span class="label">并发数:</span>
                <span class="value">{{
                  hardwareStatus.optimization.transcription.concurrency
                }}</span>
              </div>

              <div class="detail-item">
                <span class="label">推荐模型:</span>
                <span class="value">{{
                  hardwareStatus.optimization.transcription.recommended_model ||
                  "medium"
                }}</span>
              </div>
            </div>
          </div>

          <div class="optimization-section">
            <h4 class="subsection-title">系统优化配置</h4>
            <div class="detail-grid">
              <div class="detail-item">
                <span class="label">内存映射:</span>
                <el-tag
                  :type="
                    hardwareStatus.optimization.system.use_memory_mapping
                      ? 'success'
                      : 'info'
                  "
                  size="small"
                >
                  {{
                    hardwareStatus.optimization.system.use_memory_mapping
                      ? "启用"
                      : "禁用"
                  }}
                </el-tag>
              </div>

              <div class="detail-item">
                <span class="label">CPU 绑定核心:</span>
                <span class="value"
                  >{{
                    hardwareStatus.optimization.system.cpu_affinity_cores.length
                  }}
                  核</span
                >
              </div>

              <div
                class="detail-item"
                v-if="
                  hardwareStatus.optimization.system.cpu_affinity_cores.length >
                  0
                "
              >
                <span class="label">绑定核心编号:</span>
                <span class="value">{{
                  hardwareStatus.optimization.system.cpu_affinity_cores.join(
                    ", "
                  )
                }}</span>
              </div>

              <div class="detail-item">
                <span class="label">进程优先级:</span>
                <span class="value">{{
                  hardwareStatus.optimization.system.process_priority || "正常"
                }}</span>
              </div>
            </div>
          </div>
        </div>
      </el-card>
    </div>

    <template #footer>
      <div class="dialog-footer">
        <el-button @click="refreshHardware" :loading="loading" type="primary">
          <el-icon><Refresh /></el-icon>
          刷新硬件信息
        </el-button>
        <el-button @click="handleClose">关闭</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script>
import { hardwareService } from "@/services/hardwareService";
import { ElMessage } from "element-plus";
import { Monitor, Cpu, Refresh, Setting } from "@element-plus/icons-vue";

export default {
  name: "HardwareDialog",
  components: {
    Monitor,
    Cpu,
    Refresh,
    Setting,
  },
  props: {
    modelValue: {
      type: Boolean,
      default: false,
    },
  },
  emits: ["update:modelValue"],
  data() {
    return {
      loading: false,
      error: null,
      hardwareStatus: null,
    };
  },
  computed: {
    visible: {
      get() {
        return this.modelValue;
      },
      set(value) {
        this.$emit("update:modelValue", value);
      },
    },
  },
  watch: {
    visible(newVal) {
      if (newVal && !this.hardwareStatus) {
        this.loadHardwareStatus();
      }
    },
  },
  methods: {
    async loadHardwareStatus() {
      this.loading = true;
      this.error = null;
      try {
        const response = await hardwareService.getStatus();
        if (response.success) {
          this.hardwareStatus = response;
        } else {
          this.error = response.message || "硬件检测失败";
        }
      } catch (error) {
        this.error = error.message || "无法连接到服务器";
        ElMessage.error("硬件检测失败");
      } finally {
        this.loading = false;
      }
    },
    async refreshHardware() {
      await this.loadHardwareStatus();
      if (!this.error) {
        ElMessage.success("硬件信息已刷新");
      }
    },
    handleClose() {
      this.visible = false;
    },
    formatMemory(mb) {
      if (mb >= 1024) {
        return `${(mb / 1024).toFixed(1)} GB`;
      }
      return `${mb} MB`;
    },
    getMemoryColor(percentage) {
      if (percentage < 50) return "#67c23a";
      if (percentage < 80) return "#e6a23c";
      return "#f56c6c";
    },
    getCpuColor(percentage) {
      if (percentage < 30) return "#67c23a";
      if (percentage < 70) return "#e6a23c";
      return "#f56c6c";
    },
  },
};
</script>

<style scoped>
.hardware-dialog {
  border-radius: 8px;
}

.loading-content {
  text-align: center;
  padding: 40px 20px;
}

.loading-text {
  margin-top: 16px;
  color: #909399;
  font-size: 14px;
}

.error-content {
  padding: 20px 0;
}

.retry-section {
  margin-top: 20px;
  text-align: center;
}

.hardware-detail-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
  max-height: 600px;
  overflow-y: auto;
  padding-right: 4px;
}

.info-card {
  border-radius: 8px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: bold;
  color: #303133;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 16px;
}

.detail-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 8px;
  border: 1px solid #dee2e6;
  transition: all 0.3s ease;
}

.detail-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.label {
  font-size: 14px;
  color: #606266;
  font-weight: 500;
  min-width: 120px;
}

.value {
  font-weight: bold;
  color: #303133;
  font-size: 14px;
}

.progress-container {
  display: flex;
  align-items: center;
  gap: 8px;
}

.percentage {
  font-size: 12px;
  color: #909399;
  font-weight: 500;
  min-width: 35px;
}

.optimization-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.optimization-section {
  border-left: 3px solid #409eff;
  padding-left: 16px;
}

.subsection-title {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #303133;
  font-weight: 600;
}

.dialog-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 10px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .detail-grid {
    grid-template-columns: 1fr;
  }

  .detail-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .label {
    min-width: auto;
  }

  .progress-container {
    align-self: stretch;
    justify-content: space-between;
  }
}

/* 自定义滚动条 */
.hardware-detail-content::-webkit-scrollbar {
  width: 6px;
}

.hardware-detail-content::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.hardware-detail-content::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.hardware-detail-content::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}
</style>
