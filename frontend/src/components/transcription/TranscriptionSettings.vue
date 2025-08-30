<!--
  转录设置和控制组件
-->
<template>
  <el-card v-if="jobId" class="settings-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <el-icon><Setting /></el-icon>
        <span>2. 设置参数并开始转录</span>
        <div style="margin-left:auto; display:flex; gap:8px;">
          <el-button 
            type="info" 
            size="small"
            @click="$emit('show-hardware')"
            plain>
            <el-icon><Monitor /></el-icon>
            硬件信息
          </el-button>
        </div>
      </div>
    </template>
    
    <el-form 
      :model="settings" 
      label-width="100px" 
      label-position="left"
      @submit.prevent="$emit('start-job')"
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
          @click="$emit('start-job')"
        >
          <el-icon v-if="!starting"><VideoPlay /></el-icon>
          {{ starting ? "启动中..." : "开始转录" }}
        </el-button>
        
        <el-button 
          type="danger"
          size="large"
          :loading="canceling"
          :disabled="!processing || canceling"
          @click="$emit('cancel-job')"
        >
          <el-icon v-if="!canceling"><VideoPause /></el-icon>
          {{ canceling ? "取消中..." : "取消任务" }}
        </el-button>
        
        <el-button 
          type="warning"
          size="large"
          :disabled="processing || !canRestart"
          @click="$emit('restart-job')"
        >
          <el-icon><RefreshRight /></el-icon>
          重新转录
        </el-button>
        
        <el-button 
          type="success"
          size="large"
          @click="$emit('reset-selection')"
        >
          <el-icon><FolderOpened /></el-icon>
          重新选择文件
        </el-button>
      </div>
    </el-form>
  </el-card>
</template>

<script setup>
import { defineProps, defineEmits } from 'vue'

// 定义props
const props = defineProps({
  jobId: String,
  settings: Object,
  starting: Boolean,
  processing: Boolean,
  canceling: Boolean,
  canRestart: Boolean
})

// 定义emits
const emits = defineEmits([
  'start-job',
  'cancel-job', 
  'restart-job',
  'reset-selection',
  'show-hardware'
])
</script>

<style scoped>
.settings-card {
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

.action-buttons {
  text-align: center;
  margin-top: 20px;
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .action-buttons {
    flex-direction: column;
    align-items: center;
  }
  
  .action-buttons .el-button {
    width: 200px;
  }
}

:deep(.el-select),
:deep(.el-input-number) {
  width: 100%;
}
</style>