<template>
  <div class="store-test-view">
    <el-container>
      <el-header>
        <h1>状态管理层测试</h1>
      </el-header>

      <el-main>
        <el-tabs v-model="activeTab" type="border-card">
          <!-- ProjectStore 测试 -->
          <el-tab-pane label="ProjectStore 测试" name="project">
            <el-card header="项目元数据" class="test-card">
              <el-descriptions :column="2" border>
                <el-descriptions-item label="任务ID">{{ projectStore.meta.jobId || '未设置' }}</el-descriptions-item>
                <el-descriptions-item label="文件名">{{ projectStore.meta.filename || '未设置' }}</el-descriptions-item>
                <el-descriptions-item label="时长">{{ projectStore.meta.duration }}秒</el-descriptions-item>
                <el-descriptions-item label="字幕数">{{ projectStore.totalSubtitles }}</el-descriptions-item>
                <el-descriptions-item label="是否有修改">{{ projectStore.isDirty ? '是' : '否' }}</el-descriptions-item>
                <el-descriptions-item label="最后保存">{{ formatTime(projectStore.meta.lastSaved) }}</el-descriptions-item>
              </el-descriptions>
            </el-card>

            <el-card header="字幕操作" class="test-card">
              <el-space direction="vertical" style="width: 100%">
                <el-button-group>
                  <el-button @click="addTestSubtitle" type="primary">添加测试字幕</el-button>
                  <el-button @click="importTestSRT" type="success">导入测试SRT</el-button>
                  <el-button @click="clearSubtitles" type="danger">清空字幕</el-button>
                </el-button-group>

                <el-button-group>
                  <el-button @click="projectStore.undo" :disabled="!projectStore.canUndo">撤销</el-button>
                  <el-button @click="projectStore.redo" :disabled="!projectStore.canRedo">重做</el-button>
                </el-button-group>

                <el-button @click="exportSRT" type="warning">导出SRT</el-button>
              </el-space>
            </el-card>

            <el-card header="字幕列表" class="test-card">
              <el-table :data="projectStore.subtitles" max-height="300">
                <el-table-column prop="id" label="ID" width="180" />
                <el-table-column label="时间" width="200">
                  <template #default="{ row }">
                    {{ formatTimestamp(row.start) }} → {{ formatTimestamp(row.end) }}
                  </template>
                </el-table-column>
                <el-table-column prop="text" label="文本" />
                <el-table-column label="操作" width="150">
                  <template #default="{ row }">
                    <el-button size="small" @click="editSubtitle(row)">编辑</el-button>
                    <el-button size="small" type="danger" @click="projectStore.removeSubtitle(row.id)">删除</el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-card>

            <el-card header="问题检测" class="test-card">
              <el-alert
                v-if="projectStore.validationErrors.length === 0"
                title="没有发现问题"
                type="success"
                :closable="false"
              />
              <el-alert
                v-for="(error, index) in projectStore.validationErrors"
                :key="index"
                :title="error.message"
                :type="error.severity === 'error' ? 'error' : 'warning'"
                :closable="false"
                style="margin-bottom: 10px"
              />
            </el-card>

            <el-card header="播放器状态" class="test-card">
              <el-space direction="vertical" style="width: 100%">
                <div>
                  当前时间: {{ projectStore.player.currentTime.toFixed(2) }}秒
                  <el-slider v-model="projectStore.player.currentTime" :max="100" />
                </div>
                <div>
                  播放状态: {{ projectStore.player.isPlaying ? '播放中' : '已暂停' }}
                  <el-switch v-model="projectStore.player.isPlaying" />
                </div>
                <div>
                  播放速度: {{ projectStore.player.playbackRate }}x
                  <el-slider v-model="projectStore.player.playbackRate" :min="0.5" :max="2" :step="0.1" />
                </div>
              </el-space>
            </el-card>
          </el-tab-pane>

          <!-- UnifiedTaskStore 测试 -->
          <el-tab-pane label="UnifiedTaskStore 测试" name="task">
            <el-card header="任务统计" class="test-card">
              <el-descriptions :column="3" border>
                <el-descriptions-item label="总任务数">{{ taskStore.tasks.length }}</el-descriptions-item>
                <el-descriptions-item label="活跃任务">{{ taskStore.activeCount }}</el-descriptions-item>
                <el-descriptions-item label="运行中">{{ taskStore.hasRunningTask ? '是' : '否' }}</el-descriptions-item>
              </el-descriptions>
            </el-card>

            <el-card header="任务操作" class="test-card">
              <el-space>
                <el-button @click="addTestTask" type="primary">添加测试任务</el-button>
                <el-button @click="addMultipleTasks" type="success">批量添加任务</el-button>
                <el-button @click="taskStore.clearAllTasks" type="danger">清空所有任务</el-button>
                <el-button @click="taskStore.cleanupOldTasks" type="warning">清理过期任务</el-button>
              </el-space>
            </el-card>

            <el-card header="任务列表" class="test-card">
              <el-table :data="taskStore.tasks" max-height="400">
                <el-table-column prop="job_id" label="任务ID" width="180" />
                <el-table-column prop="filename" label="文件名" width="200" />
                <el-table-column label="状态" width="120">
                  <template #default="{ row }">
                    <el-tag :type="getStatusType(row.status)">{{ row.status }}</el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="阶段" width="120">
                  <template #default="{ row }">
                    <el-tag type="info">{{ row.phase }}</el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="进度" width="150">
                  <template #default="{ row }">
                    <el-progress :percentage="row.progress" :status="row.progress === 100 ? 'success' : null" />
                  </template>
                </el-table-column>
                <el-table-column prop="message" label="消息" />
                <el-table-column label="操作" width="250">
                  <template #default="{ row }">
                    <el-button-group>
                      <el-button size="small" @click="updateProgress(row.job_id)">更新进度</el-button>
                      <el-button size="small" type="success" @click="completeTask(row.job_id)">完成</el-button>
                      <el-button size="small" type="danger" @click="taskStore.deleteTask(row.job_id)">删除</el-button>
                    </el-button-group>
                  </template>
                </el-table-column>
              </el-table>
            </el-card>

            <el-card header="最近任务" class="test-card">
              <el-timeline>
                <el-timeline-item
                  v-for="task in taskStore.recentTasks"
                  :key="task.job_id"
                  :timestamp="formatTime(task.updatedAt)"
                >
                  <strong>{{ task.filename }}</strong> - {{ task.status }} ({{ task.progress }}%)
                </el-timeline-item>
              </el-timeline>
            </el-card>
          </el-tab-pane>

          <!-- 缓存服务测试 -->
          <el-tab-pane label="缓存服务测试" name="cache">
            <el-card header="缓存统计" class="test-card">
              <el-descriptions :column="2" border>
                <el-descriptions-item label="内存缓存命中率">{{ cacheStats.memory.hitRate.toFixed(2) }}%</el-descriptions-item>
                <el-descriptions-item label="内存缓存大小">{{ cacheStats.memory.size }}</el-descriptions-item>
                <el-descriptions-item label="总命中">{{ cacheStats.total.hits }}</el-descriptions-item>
                <el-descriptions-item label="总未命中">{{ cacheStats.total.misses }}</el-descriptions-item>
              </el-descriptions>
            </el-card>

            <el-card header="缓存操作" class="test-card">
              <el-space direction="vertical" style="width: 100%">
                <el-input v-model="cacheKey" placeholder="缓存键" />
                <el-input v-model="cacheValue" placeholder="缓存值" type="textarea" :rows="3" />
                <el-button-group>
                  <el-button @click="setCacheValue" type="primary">设置缓存</el-button>
                  <el-button @click="getCacheValue" type="success">获取缓存</el-button>
                  <el-button @click="deleteCacheValue" type="danger">删除缓存</el-button>
                  <el-button @click="clearCache" type="warning">清空缓存</el-button>
                </el-button-group>
                <el-alert v-if="cacheResult" :title="cacheResult" type="info" :closable="false" />
              </el-space>
            </el-card>
          </el-tab-pane>

          <!-- SSE服务测试 -->
          <el-tab-pane label="SSE服务测试" name="sse">
            <el-card header="SSE连接状态" class="test-card">
              <el-descriptions :column="2" border>
                <el-descriptions-item label="连接状态">
                  <el-tag :type="sseStats.connected ? 'success' : 'danger'">
                    {{ sseStats.status }}
                  </el-tag>
                </el-descriptions-item>
                <el-descriptions-item label="重连次数">{{ sseStats.reconnectCount }}</el-descriptions-item>
                <el-descriptions-item label="收到消息数">{{ sseStats.messagesReceived }}</el-descriptions-item>
                <el-descriptions-item label="运行时长">{{ (sseStats.uptime / 1000).toFixed(0) }}秒</el-descriptions-item>
              </el-descriptions>
            </el-card>

            <el-card header="SSE操作" class="test-card">
              <el-space>
                <el-input v-model="sseUrl" placeholder="SSE URL" style="width: 400px" />
                <el-button @click="connectSSE" type="primary" :disabled="sseStats.connected">连接</el-button>
                <el-button @click="disconnectSSE" type="danger" :disabled="!sseStats.connected">断开</el-button>
                <el-button @click="reconnectSSE" type="warning">重连</el-button>
              </el-space>
            </el-card>

            <el-card header="SSE消息日志" class="test-card">
              <el-button @click="sseMessages = []" size="small" style="margin-bottom: 10px">清空日志</el-button>
              <el-timeline>
                <el-timeline-item
                  v-for="(msg, index) in sseMessages.slice(-10)"
                  :key="index"
                  :timestamp="msg.timestamp"
                >
                  <strong>{{ msg.type }}</strong>: {{ JSON.stringify(msg.data, null, 2) }}
                </el-timeline-item>
              </el-timeline>
            </el-card>
          </el-tab-pane>
        </el-tabs>
      </el-main>
    </el-container>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import cacheService from '@/services/cacheService'
import sseService from '@/services/sseService'
import { ElMessage, ElMessageBox } from 'element-plus'

// Store 实例
const projectStore = useProjectStore()
const taskStore = useUnifiedTaskStore()

// Tab 控制
const activeTab = ref('project')

// ========== ProjectStore 测试 ==========
let subtitleCounter = 0

function addTestSubtitle() {
  subtitleCounter++
  projectStore.addSubtitle(projectStore.subtitles.length, {
    start: subtitleCounter * 2,
    end: subtitleCounter * 2 + 1.5,
    text: `测试字幕 ${subtitleCounter}`
  })
  ElMessage.success('已添加测试字幕')
}

function importTestSRT() {
  const testSRT = `1
00:00:01,000 --> 00:00:03,000
这是第一条测试字幕

2
00:00:04,000 --> 00:00:06,000
这是第二条测试字幕

3
00:00:07,000 --> 00:00:09,000
这是第三条测试字幕，内容比较长，超过了三十个字，用来测试字数过长的警告功能`

  projectStore.importSRT(testSRT, {
    jobId: 'test-job-001',
    filename: 'test-video.mp4',
    duration: 120,
    videoPath: '/api/media/test-job-001/video'
  })
  ElMessage.success('已导入测试SRT')
}

function clearSubtitles() {
  projectStore.resetProject()
  subtitleCounter = 0
  ElMessage.success('已清空字幕')
}

function editSubtitle(subtitle) {
  ElMessageBox.prompt('修改字幕文本', '编辑', {
    inputValue: subtitle.text,
    confirmButtonText: '确定',
    cancelButtonText: '取消'
  }).then(({ value }) => {
    projectStore.updateSubtitle(subtitle.id, { text: value })
    ElMessage.success('已更新字幕')
  }).catch(() => {})
}

function exportSRT() {
  const srtContent = projectStore.generateSRT()
  console.log('导出的SRT内容:\n', srtContent)
  ElMessage.success('SRT已导出到控制台')
}

function formatTimestamp(seconds) {
  return projectStore.formatTimestamp(seconds)
}

// ========== UnifiedTaskStore 测试 ==========
let taskCounter = 0

function addTestTask() {
  taskCounter++
  taskStore.addTask({
    job_id: `test-task-${Date.now()}`,
    filename: `test-video-${taskCounter}.mp4`,
    status: taskStore.TaskStatus.QUEUED,
    phase: taskStore.TaskPhase.TRANSCRIBING,
    progress: 0,
    message: '等待处理'
  })
  ElMessage.success('已添加测试任务')
}

function addMultipleTasks() {
  for (let i = 0; i < 5; i++) {
    taskCounter++
    taskStore.addTask({
      job_id: `test-task-${Date.now()}-${i}`,
      filename: `batch-video-${taskCounter}.mp4`,
      status: taskStore.TaskStatus.QUEUED,
      phase: taskStore.TaskPhase.TRANSCRIBING,
      progress: 0,
      message: '批量任务'
    })
  }
  ElMessage.success('已添加5个测试任务')
}

function updateProgress(jobId) {
  const task = taskStore.getTask(jobId)
  if (task) {
    const newProgress = Math.min(task.progress + 20, 100)
    taskStore.updateTaskProgress(jobId, newProgress, taskStore.TaskStatus.PROCESSING)
    ElMessage.info(`进度更新到 ${newProgress}%`)
  }
}

function completeTask(jobId) {
  taskStore.updateTaskStatus(jobId, taskStore.TaskStatus.FINISHED)
  taskStore.updateTaskProgress(jobId, 100)
  taskStore.updateTaskMessage(jobId, '处理完成')
  ElMessage.success('任务已完成')
}

function getStatusType(status) {
  const typeMap = {
    'created': 'info',
    'queued': 'warning',
    'processing': 'primary',
    'finished': 'success',
    'failed': 'danger',
    'canceled': 'info'
  }
  return typeMap[status] || 'info'
}

// ========== 缓存服务测试 ==========
const cacheKey = ref('test-key')
const cacheValue = ref('test-value')
const cacheResult = ref('')
const cacheStats = ref({
  memory: { hitRate: 0, size: 0 },
  total: { hits: 0, misses: 0 }
})

async function setCacheValue() {
  await cacheService.set(cacheKey.value, cacheValue.value)
  cacheResult.value = '缓存已设置'
  await updateCacheStats()
  ElMessage.success('缓存已设置')
}

async function getCacheValue() {
  const value = await cacheService.get(cacheKey.value)
  cacheResult.value = value ? `获取到: ${JSON.stringify(value)}` : '未找到缓存'
  await updateCacheStats()
}

async function deleteCacheValue() {
  await cacheService.delete(cacheKey.value)
  cacheResult.value = '缓存已删除'
  await updateCacheStats()
  ElMessage.success('缓存已删除')
}

async function clearCache() {
  await cacheService.clear()
  cacheResult.value = '所有缓存已清空'
  await updateCacheStats()
  ElMessage.success('所有缓存已清空')
}

async function updateCacheStats() {
  cacheStats.value = await cacheService.getStats()
}

// ========== SSE服务测试 ==========
const sseUrl = ref('/api/tasks/stream')
const sseStats = ref({
  status: 'disconnected',
  connected: false,
  reconnectCount: 0,
  messagesReceived: 0,
  uptime: 0
})
const sseMessages = ref([])

function connectSSE() {
  sseService.connect(sseUrl.value)
  ElMessage.info('正在连接SSE...')
}

function disconnectSSE() {
  sseService.disconnect()
  ElMessage.info('已断开SSE连接')
}

function reconnectSSE() {
  sseService.reconnect()
  ElMessage.info('正在重连SSE...')
}

function updateSSEStats() {
  sseStats.value = sseService.getStats()
}

// 通用工具
function formatTime(timestamp) {
  const date = new Date(timestamp)
  return date.toLocaleString('zh-CN')
}

// ========== 生命周期 ==========
onMounted(async () => {
  // 更新缓存统计
  await updateCacheStats()

  // 监听SSE事件
  sseService.on('status', updateSSEStats)
  sseService.on('message', (msg) => {
    sseMessages.value.push({
      timestamp: new Date().toLocaleString('zh-CN'),
      type: msg.type,
      data: msg.data
    })
  })

  // 定时更新统计
  const statsInterval = setInterval(async () => {
    await updateCacheStats()
    updateSSEStats()
  }, 2000)

  onUnmounted(() => {
    clearInterval(statsInterval)
    sseService.off('status', updateSSEStats)
    sseService.removeAllListeners('message')
  })
})
</script>

<style lang="scss" scoped>
.store-test-view {
  height: 100vh;
  overflow: hidden;

  .el-header {
    display: flex;
    align-items: center;
    background: var(--el-color-primary);
    color: white;
    padding: 0 20px;

    h1 {
      margin: 0;
      font-size: 20px;
    }
  }

  .el-main {
    padding: 20px;
    overflow-y: auto;
  }

  .test-card {
    margin-bottom: 20px;

    :deep(.el-card__header) {
      font-weight: bold;
    }
  }
}
</style>
