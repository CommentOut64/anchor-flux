<template>
  <el-card class="hardware-status-bar" shadow="hover">
    <div class="status-container">
      <div class="status-left">
        <el-icon><Cpu /></el-icon>
        <span class="status-title">硬件状态</span>

        <div v-if="loading" class="status-loading">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>检测中...</span>
        </div>

        <div v-else-if="error" class="status-error">
          <el-icon><Warning /></el-icon>
          <span>检测失败</span>
        </div>

        <div v-else-if="hardwareStatus" class="status-info">
          <!-- GPU 状态 -->
          <div class="status-item">
            <el-tag
              :type="
                hardwareStatus.hardware.gpu.cuda_available
                  ? 'success'
                  : 'warning'
              "
              size="small"
              effect="plain"
            >
              GPU:
              {{
                hardwareStatus.hardware.gpu.cuda_available
                  ? "CUDA可用"
                  : "CPU模式"
              }}
            </el-tag>
          </div>

          <!-- 内存状态 -->
          <div class="status-item">
            <el-tag
              :type="
                getMemoryStatusType(
                  hardwareStatus.hardware.memory.usage_percent
                )
              "
              size="small"
              effect="plain"
            >
              内存: {{ hardwareStatus.hardware.memory.usage_percent }}%
            </el-tag>
          </div>

          <!-- 推荐设备 -->
          <div class="status-item">
            <el-tag
              :type="
                hardwareStatus.optimization.transcription.device === 'cuda'
                  ? 'success'
                  : 'info'
              "
              size="small"
              effect="plain"
            >
              推荐:
              {{
                hardwareStatus.optimization.transcription.device.toUpperCase()
              }}
            </el-tag>
          </div>
        </div>
      </div>

      <div class="status-right">
        <el-button
          type="primary"
          size="small"
          @click="openHardwareDialog"
          :loading="loading"
        >
          <el-icon><Monitor /></el-icon>
          查看详情
        </el-button>

        <el-button
          type="info"
          size="small"
          @click="refreshHardware"
          :loading="loading"
          plain
        >
          <el-icon><Refresh /></el-icon>
        </el-button>
      </div>
    </div>

    <!-- 硬件详情对话框 -->
    <HardwareDialog v-model="showHardwareDialog" />
  </el-card>
</template>

<script>
import { hardwareService } from "@/services/hardwareService";
import { ElMessage } from "element-plus";
import {
  Cpu,
  Monitor,
  Refresh,
  Loading,
  Warning,
} from "@element-plus/icons-vue";
import HardwareDialog from "./HardwareDialog.vue";

export default {
  name: "HardwareStatusBar",
  components: {
    Cpu,
    Monitor,
    Refresh,
    Loading,
    Warning,
    HardwareDialog,
  },
  data() {
    return {
      loading: true,
      error: null,
      hardwareStatus: null,
      showHardwareDialog: false,
    };
  },
  mounted() {
    this.loadHardwareStatus();
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
      } finally {
        this.loading = false;
      }
    },
    async refreshHardware() {
      await this.loadHardwareStatus();
      if (!this.error) {
        ElMessage.success("硬件状态已刷新");
      }
    },
    openHardwareDialog() {
      this.showHardwareDialog = true;
    },
    getMemoryStatusType(percentage) {
      if (percentage < 50) return "success";
      if (percentage < 80) return "warning";
      return "danger";
    },
  },
};
</script>

<style scoped>
.hardware-status-bar {
  margin-bottom: 20px;
  border-radius: 8px;
}

.status-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  padding: 8px 0;
}

.status-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
}

.status-title {
  font-weight: 600;
  color: #303133;
  font-size: 14px;
}

.status-loading,
.status-error {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}

.status-loading {
  color: #409eff;
}

.status-error {
  color: #f56c6c;
}

.status-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-item {
  display: flex;
  align-items: center;
}

.status-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .status-container {
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
  }

  .status-left {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .status-info {
    flex-wrap: wrap;
    gap: 6px;
  }

  .status-right {
    justify-content: center;
  }
}

@media (max-width: 480px) {
  .status-info {
    flex-direction: column;
    align-items: flex-start;
    width: 100%;
  }

  .status-item {
    width: 100%;
  }
}
</style>
