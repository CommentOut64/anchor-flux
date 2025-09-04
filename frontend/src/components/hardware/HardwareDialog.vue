<template>
  <el-dialog
    v-model="visible"
    title="硬件信息总览"
    width="700px"
    :before-close="handleClose"
    class="modern-hardware-dialog"
    destroy-on-close
  >
    <div v-if="loading" class="loading-content">
      <div class="loading-text">正在检测硬件信息...</div>
    </div>

    <div v-else-if="error" class="error-content">
      <h3>硬件检测失败</h3>
      <p class="error-message">{{ error }}</p>
      <el-button type="primary" @click="refreshHardware" :loading="loading" class="retry-btn">
        重新检测
      </el-button>
    </div>

    <div v-else-if="hardwareStatus" class="hardware-overview">
      <!-- 硬件概览卡片网格 -->
      <div class="hardware-grid">
        <!-- GPU 卡片 -->
        <div class="hardware-card gpu-card">
          <div class="card-header">
            <div class="header-info">
              <h3>GPU 显卡</h3>
            </div>
          </div>
          <div class="card-content">
            <div class="primary-info">
              <div class="info-item">
                <span class="label">型号</span>
                <span class="value gpu-name" :class="{ 'long-name': getGpuName().length > 25 }">{{ getGpuName() || 'N/A' }}</span>
              </div>
              <div class="info-item">
                <span class="label">显存大小</span>
                <span class="value">{{ formatMemory(hardwareStatus.hardware.gpu.total_memory_mb) }}</span>
              </div>
              <div class="info-item">
                <span class="label">CUDA支持</span>
                <el-tag 
                  :type="hardwareStatus.hardware.gpu.cuda_available ? 'success' : 'danger'"
                  size="small"
                >
                  {{ hardwareStatus.hardware.gpu.cuda_available ? '可用' : '不可用' }}
                </el-tag>
              </div>
            </div>
          </div>
        </div>

        <!-- CPU 卡片 -->
        <div class="hardware-card cpu-card">
          <div class="card-header">
            <div class="header-info">
              <h3>CPU 处理器</h3>
            </div>
          </div>
          <div class="card-content">
            <div class="primary-info">
              <div class="info-item">
                <span class="label">型号</span>
                <span class="value cpu-name" :class="{ 'long-name': getCpuName().length > 25 }">{{ getCpuName() || 'N/A' }}</span>
              </div>
              <div class="info-item">
                <span class="label">核心/线程</span>
                <span class="value">{{ hardwareStatus.hardware.cpu.cores }}C/{{ hardwareStatus.hardware.cpu.threads }}T</span>
              </div>
              <div class="info-item" v-if="hardwareStatus.hardware.cpu.max_frequency">
                <span class="label">基准频率</span>
                <span class="value">{{ formatFrequency(hardwareStatus.hardware.cpu.max_frequency) }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 内存卡片 -->
        <div class="hardware-card memory-card">
          <div class="card-header">
            <div class="header-info">
              <h3>系统内存</h3>
            </div>
          </div>
          <div class="card-content">
            <div class="primary-info">
              <div class="info-item">
                <span class="label">总大小</span>
                <span class="value">{{ formatMemory(hardwareStatus.hardware.memory.total_mb) }}</span>
              </div>
              <div class="info-item">
                <span class="label">使用率</span>
                <div class="usage-display">
                  <el-progress
                    :percentage="hardwareStatus.hardware.memory.usage_percent"
                    :stroke-width="4"
                    :color="getMemoryColor(hardwareStatus.hardware.memory.usage_percent)"
                    :show-text="false"
                    class="usage-bar"
                  />
                  <span class="usage-text">{{ hardwareStatus.hardware.memory.usage_percent }}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 转录优化配置 -->
      <div class="optimization-section">
        <div class="section-header">
          <h2>转录优化配置</h2>
        </div>
        
        <div class="optimization-grid">
          <div class="optimization-item">
            <div class="opt-content">
              <h4>推荐设备</h4>
              <el-tag 
                :type="hardwareStatus.optimization.transcription.device === 'cuda' ? 'success' : 'warning'"
                size="small"
              >
                {{ hardwareStatus.optimization.transcription.device.toUpperCase() }}
              </el-tag>
            </div>
          </div>
          
          <div class="optimization-item">
            <div class="opt-content">
              <h4>批处理大小</h4>
              <span class="opt-value">{{ hardwareStatus.optimization.transcription.batch_size }}</span>
            </div>
          </div>
          
          <div class="optimization-item">
            <div class="opt-content">
              <h4>并发处理数</h4>
              <span class="opt-value">{{ hardwareStatus.optimization.transcription.concurrency }}</span>
            </div>
          </div>
          
          <div class="optimization-item">
            <div class="opt-content">
              <h4>推荐模型</h4>
              <span class="opt-value">{{ hardwareStatus.optimization.transcription.recommended_model || "medium" }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <template #footer>
      <div class="dialog-footer">
        <el-button @click="refreshHardware" :loading="loading" type="primary" class="refresh-btn">
          刷新信息
        </el-button>
        <el-button @click="handleClose" class="close-btn">关闭</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<script>
import { hardwareService } from "@/services/hardwareService";
import { ElMessage } from "element-plus";

export default {
  name: "HardwareDialog",
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
    formatFrequency(mhz) {
      if (mhz >= 1000) {
        return `${(mhz / 1000).toFixed(1)} GHz`;
      }
      return `${mhz} MHz`;
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
    getGpuName() {
      // 尝试从不同的字段获取GPU名称
      let name = '';
      if (this.hardwareStatus?.hardware?.gpu?.name) {
        name = this.hardwareStatus.hardware.gpu.name;
      } else if (this.hardwareStatus?.hardware?.gpu?.device_name) {
        name = this.hardwareStatus.hardware.gpu.device_name;
      } else if (this.hardwareStatus?.hardware?.gpu?.cuda_available) {
        return "NVIDIA GPU";
      } else {
        return "未知显卡";
      }

      // 优化GPU名称显示 - 简化长名称以便更好地显示
      // 处理NVIDIA GeForce系列 (例: NVIDIA GeForce RTX 4060 Laptop GPU -> RTX 4060 Laptop)
      if (name.includes('NVIDIA GeForce')) {
        // 移除NVIDIA GeForce前缀和GPU后缀
        name = name.replace(/NVIDIA GeForce\s*/i, '').replace(/\s*GPU\s*$/i, '');
        
        // 处理RTX系列
        if (name.includes('RTX')) {
          // 提取RTX和后续的数字部分，保留重要后缀如Ti、SUPER、Laptop等
          const rtxMatch = name.match(/RTX\s*(\d+(?:\s*Ti)?(?:\s*SUPER)?)\s*(.*)/i);
          if (rtxMatch) {
            const model = rtxMatch[1];
            const suffix = rtxMatch[2].trim();
            // 保留重要的后缀如Laptop、Mobile等
            const importantSuffixes = ['Laptop', 'Mobile', 'Max-Q'];
            const filteredSuffix = suffix.split(' ').filter(word => 
              importantSuffixes.some(suffix => word.toLowerCase().includes(suffix.toLowerCase()))
            ).join(' ');
            name = `RTX ${model}${filteredSuffix ? ' ' + filteredSuffix : ''}`;
          }
        }
        // 处理GTX系列
        else if (name.includes('GTX')) {
          const gtxMatch = name.match(/GTX\s*(\d+(?:\s*Ti)?(?:\s*SUPER)?)\s*(.*)/i);
          if (gtxMatch) {
            const model = gtxMatch[1];
            const suffix = gtxMatch[2].trim();
            const importantSuffixes = ['Laptop', 'Mobile', 'Max-Q'];
            const filteredSuffix = suffix.split(' ').filter(word => 
              importantSuffixes.some(suffix => word.toLowerCase().includes(suffix.toLowerCase()))
            ).join(' ');
            name = `GTX ${model}${filteredSuffix ? ' ' + filteredSuffix : ''}`;
          }
        }
      }
      // 处理AMD显卡
      else if (name.includes('AMD') || name.includes('Radeon')) {
        name = name.replace(/AMD\s*/i, '').replace(/Radeon\s*/i, 'RX ');
        name = name.replace(/\s+/g, ' ').trim();
      }
      // 处理Intel显卡
      else if (name.includes('Intel')) {
        name = name.replace(/Intel\(R\)\s*/i, 'Intel ').replace(/Graphics/i, '');
        name = name.replace(/\s+/g, ' ').trim();
      }
      
      // 清理多余的空格和符号
      name = name.replace(/\s+/g, ' ').replace(/[()]/g, '').trim();
      
      return name;
    },
    getCpuName() {
      // 获取CPU名称并优化显示
      if (this.hardwareStatus?.hardware?.cpu?.name) {
        let name = this.hardwareStatus.hardware.cpu.name;
        // 移除商标符号和常见后缀
        name = name.replace(/\(R\)/g, '').replace(/\(TM\)/g, '').replace(/®/g, '').replace(/™/g, '');
        // 移除频率信息和CPU标识
        name = name.replace(/CPU\s*@.*/, '').replace(/\s*CPU\s*$/, '').trim();
        // 简化Intel和AMD的型号名称
        name = name.replace(/Intel\s+Core\s+/i, 'Intel ');
        name = name.replace(/AMD\s+Ryzen\s+/i, 'AMD Ryzen ');
        // 压缩多个空格
        name = name.replace(/\s+/g, ' ').trim();
        return name;
      }
      return "未知处理器";
    },
  },
};
</script>

<style scoped>
/* 对话框整体样式 */
.modern-hardware-dialog {
  border-radius: 16px;
  overflow: hidden;
}

/* 加载状态 */
.loading-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  min-height: 120px;
}

.loading-text {
  color: #606266;
  font-size: 16px;
  font-weight: 500;
}

/* 错误状态 */
.error-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  text-align: center;
}

.error-content h3 {
  color: #303133;
  margin: 0 0 10px 0;
  font-size: 1.25rem;
  font-weight: 600;
}

.error-message {
  color: #606266;
  margin: 0 0 20px 0;
  font-size: 14px;
  line-height: 1.5;
}

.retry-btn {
  border-radius: 6px;
  padding: 8px 16px;
  font-weight: 500;
}

/* 硬件概览 */
.hardware-overview {
  padding: 0;
}

.hardware-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

/* 硬件卡片样式 */
.hardware-card {
  background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
  border: 1px solid #e4e7ed;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}

.hardware-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border-color: #c6e2ff;
}

/* 卡片头部 */
.card-header {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
}

.header-info h3 {
  margin: 0;
  color: #303133;
  font-size: 1rem;
  font-weight: 600;
}

/* 卡片内容 */
.card-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.primary-info {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.7);
  border-radius: 6px;
  border: 1px solid rgba(0, 0, 0, 0.05);
  min-height: 32px;
}

.label {
  color: #606266;
  font-size: 0.8rem;
  font-weight: 500;
  margin-top: 2px;
  flex-shrink: 0;
}

.value {
  color: #303133;
  font-weight: 600;
  font-size: 0.8rem;
  text-align: right;
  margin-left: 8px;
}

.gpu-name, .cpu-name {
  max-width: 140px;
  word-wrap: break-word;
  word-break: break-word;
  hyphens: auto;
  line-height: 1.3;
  text-align: right;
}

/* 针对超长型号名称的特殊处理 */
.long-name {
  font-size: 0.75rem;
  line-height: 1.2;
}

.usage-display {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 80px;
}

.usage-bar {
  flex: 1;
}

.usage-text {
  font-size: 0.7rem;
  color: #909399;
  font-weight: 500;
  min-width: 30px;
}

/* 优化配置区域 */
.optimization-section {
  background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
  border: 1px solid #e4e7ed;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.section-header {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid #f0f0f0;
}

.section-header h2 {
  margin: 0;
  color: #303133;
  font-size: 1rem;
  font-weight: 600;
}

.optimization-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px;
}

.optimization-item {
  display: flex;
  align-items: center;
  padding: 12px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 8px;
  border: 1px solid rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease;
}

.optimization-item:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.opt-content {
  flex: 1;
}

.opt-content h4 {
  margin: 0 0 4px 0;
  color: #303133;
  font-size: 0.8rem;
  font-weight: 600;
}

.opt-value {
  color: #409eff;
  font-weight: 600;
  font-size: 0.8rem;
}

/* 底部按钮 */
.dialog-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 16px;
}

.refresh-btn {
  background: linear-gradient(135deg, #409eff, #67c23a);
  border: none;
  border-radius: 6px;
  padding: 8px 16px;
  font-weight: 600;
  color: white;
  transition: all 0.3s ease;
}

.refresh-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.3);
}

.close-btn {
  border-radius: 6px;
  padding: 8px 16px;
  font-weight: 500;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .hardware-grid {
    grid-template-columns: 1fr;
    gap: 12px;
  }
  
  .optimization-grid {
    grid-template-columns: 1fr;
    gap: 8px;
  }
  
  .hardware-card, .optimization-section {
    padding: 12px;
  }
  
  .dialog-footer {
    flex-direction: column;
    gap: 8px;
  }
  
  .refresh-btn, .close-btn {
    width: 100%;
  }
}

@media (max-width: 480px) {
  .gpu-name, .cpu-name {
    max-width: 120px;
    font-size: 0.75rem;
  }
  
  .info-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
  
  .value {
    text-align: left;
    margin-left: 0;
  }
}

/* Element Plus 组件自定义样式 */
:deep(.el-dialog__header) {
  background: linear-gradient(135deg, #409eff 0%, #67c23a 100%);
  color: white;
  padding: 16px 20px;
  margin: 0;
}

:deep(.el-dialog__title) {
  color: white;
  font-weight: 600;
  font-size: 1.125rem;
}

:deep(.el-dialog__headerbtn .el-dialog__close) {
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.125rem;
}

:deep(.el-dialog__headerbtn .el-dialog__close:hover) {
  color: white;
}

:deep(.el-dialog__body) {
  padding: 20px;
}

:deep(.el-dialog__footer) {
  padding: 0 20px 20px 20px;
}

:deep(.el-progress-bar__outer) {
  border-radius: 4px;
}

:deep(.el-progress-bar__inner) {
  border-radius: 4px;
}

:deep(.el-tag--small) {
  padding: 4px 8px;
  font-size: 0.75rem;
  border-radius: 4px;
  font-weight: 500;
}
</style>
