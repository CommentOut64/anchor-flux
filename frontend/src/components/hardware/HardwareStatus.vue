<template>
  <el-card class="hardware-status-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <el-icon><Cpu /></el-icon>
        <span>硬件检测状态</span>
        <el-button 
          type="primary" 
          size="small"
          @click="refreshHardware"
          :loading="loading"
          style="margin-left: auto;"
        >
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
      </div>
    </template>

    <div v-if="loading" class="loading-content">
      <el-skeleton :rows="3" animated />
      <div class="loading-text">正在检测硬件...</div>
    </div>

    <div v-else-if="error" class="error-content">
      <el-alert
        title="硬件检测失败"
        type="error"
        :description="error"
        show-icon
        :closable="false"
      />
    </div>

    <div v-else-if="hardwareStatus" class="hardware-content">
      <!-- GPU 信息 -->
      <div class="hardware-section">
        <h4 class="section-title">
          <el-icon><Monitor /></el-icon>
          GPU 信息
        </h4>
        <div class="hardware-grid">
          <div class="hardware-item">
            <span class="label">CUDA 支持:</span>
            <el-tag :type="hardwareStatus.hardware.gpu.cuda_available ? 'success' : 'warning'" size="small">
              {{ hardwareStatus.hardware.gpu.cuda_available ? '✓ 可用' : '✗ 不可用' }}
            </el-tag>
          </div>
          <div class="hardware-item">
            <span class="label">GPU 数量:</span>
            <span class="value">{{ hardwareStatus.hardware.gpu.count }} 个</span>
          </div>
          <div class="hardware-item">
            <span class="label">总显存:</span>
            <span class="value">{{ formatMemory(hardwareStatus.hardware.gpu.total_memory_mb) }}</span>
          </div>
        </div>
      </div>

      <!-- CPU 信息 -->
      <div class="hardware-section">
        <h4 class="section-title">
          <el-icon><Cpu /></el-icon>
          CPU 信息
        </h4>
        <div class="hardware-grid">
          <div class="hardware-item">
            <span class="label">物理核心:</span>
            <span class="value">{{ hardwareStatus.hardware.cpu.cores }} 核</span>
          </div>
          <div class="hardware-item">
            <span class="label">逻辑线程:</span>
            <span class="value">{{ hardwareStatus.hardware.cpu.threads }} 线程</span>
          </div>
        </div>
      </div>

      <!-- 内存信息 -->
      <div class="hardware-section">
        <h4 class="section-title">
          <el-icon><MemoryStick /></el-icon>
          内存信息
        </h4>
        <div class="hardware-grid">
          <div class="hardware-item">
            <span class="label">总内存:</span>
            <span class="value">{{ formatMemory(hardwareStatus.hardware.memory.total_mb) }}</span>
          </div>
          <div class="hardware-item">
            <span class="label">可用内存:</span>
            <span class="value">{{ formatMemory(hardwareStatus.hardware.memory.available_mb) }}</span>
          </div>
          <div class="hardware-item">
            <span class="label">使用率:</span>
            <el-progress 
              :percentage="hardwareStatus.hardware.memory.usage_percent"
              :stroke-width="6"
              :color="getMemoryColor(hardwareStatus.hardware.memory.usage_percent)"
              :show-text="false"
              style="width: 80px;"
            />
            <span class="percentage">{{ hardwareStatus.hardware.memory.usage_percent }}%</span>
          </div>
        </div>
      </div>

      <!-- 优化配置 -->
      <div class="hardware-section">
        <h4 class="section-title">
          <el-icon><Tools /></el-icon>
          推荐配置
        </h4>
        <div class="optimization-grid">
          <div class="optimization-item">
            <span class="label">推荐设备:</span>
            <el-tag :type="hardwareStatus.optimization.transcription.device === 'cuda' ? 'success' : 'info'" size="small">
              {{ hardwareStatus.optimization.transcription.device.toUpperCase() }}
            </el-tag>
          </div>
          <div class="optimization-item">
            <span class="label">批处理大小:</span>
            <span class="value">{{ hardwareStatus.optimization.transcription.batch_size }}</span>
          </div>
          <div class="optimization-item">
            <span class="label">并发数:</span>
            <span class="value">{{ hardwareStatus.optimization.transcription.concurrency }}</span>
          </div>
          <div class="optimization-item">
            <span class="label">内存映射:</span>
            <el-tag :type="hardwareStatus.optimization.system.use_memory_mapping ? 'success' : 'info'" size="small">
              {{ hardwareStatus.optimization.system.use_memory_mapping ? '启用' : '禁用' }}
            </el-tag>
          </div>
          <div class="optimization-item">
            <span class="label">CPU 绑定核心:</span>
            <span class="value">{{ hardwareStatus.optimization.system.cpu_affinity_cores.length }} 核</span>
          </div>
        </div>
      </div>
    </div>
  </el-card>
</template>

<script>
import { hardwareService } from '@/services/hardwareService'
import { ElMessage } from 'element-plus'
import { Monitor, Cpu, Refresh, Tools } from '@element-plus/icons-vue'

export default {
  name: 'HardwareStatus',
  components: {
    Monitor,
    Cpu,
    Refresh,
    Tools
  },
  data() {
    return {
      loading: true,
      error: null,
      hardwareStatus: null
    }
  },
  mounted() {
    this.loadHardwareStatus()
  },
  methods: {
    async loadHardwareStatus() {
      this.loading = true
      this.error = null
      try {
        const response = await hardwareService.getStatus()
        if (response.success) {
          this.hardwareStatus = response
          ElMessage.success('硬件检测完成')
        } else {
          this.error = response.message || '硬件检测失败'
          ElMessage.warning('硬件检测部分失败')
        }
      } catch (error) {
        this.error = error.message || '无法连接到服务器'
        ElMessage.error('硬件检测失败')
      } finally {
        this.loading = false
      }
    },
    async refreshHardware() {
      await this.loadHardwareStatus()
    },
    formatMemory(mb) {
      if (mb >= 1024) {
        return `${(mb / 1024).toFixed(1)} GB`
      }
      return `${mb} MB`
    },
    getMemoryColor(percentage) {
      if (percentage < 50) return '#67c23a'
      if (percentage < 80) return '#e6a23c'
      return '#f56c6c'
    }
  }
}
</script>

<style scoped>
.hardware-status-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: bold;
}

.loading-content {
  text-align: center;
  padding: 20px 0;
}

.loading-text {
  margin-top: 16px;
  color: #909399;
}

.error-content {
  padding: 10px 0;
}

.hardware-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.hardware-section {
  border-left: 3px solid #409eff;
  padding-left: 12px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #303133;
}

.hardware-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}

.optimization-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}

.hardware-item,
.optimization-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: #f8f9fa;
  border-radius: 4px;
  border: 1px solid #e9ecef;
}

.label {
  font-size: 13px;
  color: #606266;
  font-weight: 500;
}

.value {
  font-weight: bold;
  color: #303133;
}

.percentage {
  margin-left: 8px;
  font-size: 12px;
  color: #909399;
}

@media (max-width: 768px) {
  .hardware-grid,
  .optimization-grid {
    grid-template-columns: 1fr;
  }
}
</style>