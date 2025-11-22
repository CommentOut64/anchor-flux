<template>
  <el-card v-if="incompleteJobs.length > 0" class="incomplete-jobs-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <span>
          <el-icon><Clock /></el-icon>
          未完成任务 ({{ incompleteJobs.length }})
        </span>
        <el-button
          type="primary"
          size="small"
          @click="refreshJobs"
          :loading="loading"
        >
          刷新
        </el-button>
      </div>
    </template>

    <el-scrollbar max-height="300px">
      <div class="job-list">
        <div
          v-for="job in incompleteJobs"
          :key="job.job_id"
          class="job-item"
          :class="{ 'job-item-active': currentJobId === job.job_id }"
        >
          <div class="job-info">
            <div class="job-name">
              <el-icon><VideoPlay /></el-icon>
              {{ job.filename }}
            </div>
            <div class="job-progress">
              <el-progress
                :percentage="job.progress"
                :stroke-width="8"
                :show-text="false"
              />
              <span class="progress-text">
                {{ job.processed_segments }}/{{ job.total_segments }} 段 ({{ job.progress }}%)
              </span>
            </div>
          </div>
          <div class="job-actions">
            <el-button
              type="success"
              size="small"
              @click="continueJob(job)"
              :disabled="currentJobId === job.job_id"
            >
              <el-icon><VideoPlay /></el-icon>
              继续
            </el-button>
            <el-button
              type="danger"
              size="small"
              @click="deleteJob(job)"
            >
              <el-icon><Delete /></el-icon>
              删除
            </el-button>
          </div>
        </div>
      </div>
    </el-scrollbar>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Clock, VideoPlay, Delete } from '@element-plus/icons-vue'
import { TranscriptionService } from '../../services/transcriptionService.js'

const props = defineProps({
  currentJobId: {
    type: String,
    default: ''
  }
})

const emit = defineEmits(['restore-job', 'delete-job'])

const incompleteJobs = ref([])
const loading = ref(false)

// 加载未完成任务
async function loadIncompleteJobs() {
  loading.value = true
  try {
    const data = await TranscriptionService.getIncompleteJobs()
    incompleteJobs.value = data.jobs || []

    if (incompleteJobs.value.length > 0) {
      ElMessage.info(`发现 ${incompleteJobs.value.length} 个未完成任务`)
    }
  } catch (error) {
    console.error('加载未完成任务失败:', error)
  } finally {
    loading.value = false
  }
}

// 刷新任务列表
function refreshJobs() {
  loadIncompleteJobs()
}

// 继续任务
async function continueJob(job) {
  try {
    await ElMessageBox.confirm(
      `继续转录任务：${job.filename}？\n已完成 ${job.progress}%`,
      '确认操作',
      {
        confirmButtonText: '继续',
        cancelButtonText: '取消',
        type: 'info'
      }
    )

    emit('restore-job', job)
  } catch (error) {
    // 用户取消
  }
}

// 删除任务
async function deleteJob(job) {
  try {
    await ElMessageBox.confirm(
      `确定要删除任务：${job.filename}？\n这将删除所有相关数据，无法恢复！`,
      '确认删除',
      {
        confirmButtonText: '删除',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    // 取消任务并删除数据
    await TranscriptionService.cancelJob(job.job_id, true)
    ElMessage.success('任务已删除')

    // 刷新列表
    await loadIncompleteJobs()
    emit('delete-job', job.job_id)
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除任务失败:', error)
      ElMessage.error('删除任务失败')
    }
  }
}

// 组件挂载时加载
onMounted(() => {
  loadIncompleteJobs()
})

// 暴露方法给父组件
defineExpose({
  loadIncompleteJobs,
  refreshJobs
})
</script>

<style scoped>
.incomplete-jobs-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
}

.card-header span {
  display: flex;
  align-items: center;
  gap: 8px;
}

.job-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.job-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px;
  border: 1px solid #e4e7ed;
  border-radius: 6px;
  background: #fff;
  transition: all 0.3s;
}

.job-item:hover {
  border-color: #409eff;
  box-shadow: 0 2px 8px rgba(64, 158, 255, 0.2);
}

.job-item-active {
  border-color: #67c23a;
  background: #f0f9ff;
}

.job-info {
  flex: 1;
  min-width: 0;
}

.job-name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: bold;
  margin-bottom: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.job-progress {
  display: flex;
  align-items: center;
  gap: 12px;
}

.job-progress .el-progress {
  flex: 1;
}

.progress-text {
  font-size: 12px;
  color: #909399;
  white-space: nowrap;
}

.job-actions {
  display: flex;
  gap: 8px;
  margin-left: 12px;
}
</style>
