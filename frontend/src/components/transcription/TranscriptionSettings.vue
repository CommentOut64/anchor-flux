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
          <!-- 模型状态按钮 -->
          <ModelStatusButton />
          
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
        <!-- 开始/恢复转录按钮 -->
        <el-button
          v-if="status !== 'paused'"
          type="primary"
          size="large"
          :loading="starting"
          :disabled="processing || starting"
          @click="$emit('start-job')"
        >
          <el-icon v-if="!starting"><VideoPlay /></el-icon>
          {{ starting ? "启动中..." : "开始转录" }}
        </el-button>

        <!-- 恢复任务按钮（仅在暂停时显示） -->
        <el-button
          v-if="status === 'paused'"
          type="success"
          size="large"
          :loading="starting"
          :disabled="starting"
          @click="$emit('start-job')"
        >
          <el-icon v-if="!starting"><VideoPlay /></el-icon>
          {{ starting ? "恢复中..." : "恢复任务" }}
        </el-button>

        <!-- 暂停任务按钮（处理中时显示） -->
        <el-button
          v-if="processing"
          type="warning"
          size="large"
          :loading="pausing"
          :disabled="pausing"
          @click="$emit('pause-job')"
        >
          <el-icon v-if="!pausing"><VideoPause /></el-icon>
          {{ pausing ? "暂停中..." : "暂停任务" }}
        </el-button>

        <!-- 取消任务按钮（有任务且未完成时显示） -->
        <el-button
          v-if="canRestart || processing || status === 'paused'"
          type="danger"
          size="large"
          :loading="canceling"
          :disabled="canceling"
          @click="$emit('cancel-job')"
        >
          <el-icon v-if="!canceling"><Delete /></el-icon>
          {{ canceling ? "取消中..." : "取消任务" }}
        </el-button>

        <!-- 重新转录按钮（任务失败或取消后显示） -->
        <el-button
          v-if="!processing && (status === 'failed' || status === 'canceled')"
          type="warning"
          size="large"
          @click="$emit('restart-job')"
        >
          <el-icon><RefreshRight /></el-icon>
          重新转录
        </el-button>

        <!-- 重新选择文件按钮 -->
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
import { defineProps, defineEmits, ref, watch, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import ModelStatusButton from '../models/ModelStatusButton.vue'
import { modelAPI } from '../../services/api.js'

// 定义props
const props = defineProps({
  jobId: String,
  settings: Object,
  starting: Boolean,
  processing: Boolean,
  canceling: Boolean,
  pausing: Boolean,  // 新增：暂停中
  canRestart: Boolean,
  status: String  // 新增：任务状态
})

// 定义emits
const emits = defineEmits([
  'start-job',
  'cancel-job',
  'pause-job',  // 新增：暂停任务
  'restart-job',
  'reset-selection',
  'show-hardware'
])

// 追踪原始设置用于检测变更
const originalSettings = ref({})
const hasModelSettingsChanged = ref(false)

// 监听设置变化
watch(() => props.settings, (newSettings, oldSettings) => {
  if (!oldSettings) {
    // 初始化时保存原始设置
    originalSettings.value = { ...newSettings }
    return
  }
  
  // 检查模型相关参数是否变更
  const modelRelatedKeys = ['model', 'compute_type', 'device']
  const changed = modelRelatedKeys.some(key => 
    newSettings[key] !== originalSettings.value[key]
  )
  
  if (changed && !hasModelSettingsChanged.value) {
    hasModelSettingsChanged.value = true
    showModelReloadConfirm()
  }
}, { deep: true })

// 显示模型重新加载确认对话框
async function showModelReloadConfirm() {
  try {
    await ElMessageBox.confirm(
      '检测到模型相关参数已修改，是否立即重新预加载模型以获得最佳性能？',
      '参数变更提示',
      {
        confirmButtonText: '立即预加载',
        cancelButtonText: '稍后手动',
        type: 'info',
        distinguishCancelAndClose: true
      }
    )
    
    // 用户确认，开始预加载
    await startModelPreload()
    
  } catch (action) {
    if (action === 'cancel') {
      ElMessage.info('您可以稍后通过模型状态按钮手动预加载')
    }
  } finally {
    // 更新原始设置和重置标志
    originalSettings.value = { ...props.settings }
    hasModelSettingsChanged.value = false
  }
}

// 启动模型预加载
async function startModelPreload() {
  try {
    const result = await modelAPI.startPreload()
    if (result.success) {
      ElMessage.success('模型预加载已启动，将在后台进行')
    } else {
      ElMessage.warning(result.message || '预加载启动失败，转录时将自动加载模型')
    }
  } catch (error) {
    console.error('启动预加载失败:', error)
    ElMessage.warning('预加载启动失败，转录时将自动加载模型')
  }
}

// 初始化时保存设置
nextTick(() => {
  if (props.settings) {
    originalSettings.value = { ...props.settings }
  }
})
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