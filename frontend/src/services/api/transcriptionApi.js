/**
 * 转录任务 API
 *
 * 职责：管理转录任务的完整生命周期
 * - 上传文件并创建任务
 * - 启动、暂停、取消任务
 * - 获取任务状态和队列信息
 * - 下载转录结果
 */

import { apiClient } from './client'

class TranscriptionAPI {
  /**
   * 上传文件并创建转录任务
   * @param {File} file - 视频文件对象
   * @param {Function} onProgress - 上传进度回调 (percent) => void
   * @returns {Promise<{job_id: string, filename: string, message: string, queue_position: number}>}
   */
  async uploadFile(file, onProgress = null) {
    const formData = new FormData()
    formData.append('file', file)

    const config = {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    }

    // 添加上传进度监听
    if (onProgress) {
      config.onUploadProgress = (progressEvent) => {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        onProgress(percent)
      }
    }

    return apiClient.post('/api/upload', formData, config)
  }

  /**
   * 为本地 input 文件创建转录任务
   * @param {string} filename - 文件名
   * @returns {Promise<{job_id: string, filename: string}>}
   */
  async createJob(filename) {
    const formData = new FormData()
    formData.append('filename', filename)

    return apiClient.post('/api/create-job', formData)
  }

  /**
   * 启动转录任务（加入队列）
   * @param {string} jobId - 任务ID
   * @param {Object} settings - 转录设置
   * @param {string} settings.model - 模型名称 (tiny, base, small, medium, large-v2, large-v3)
   * @param {string} settings.compute_type - 计算类型 (float16, int8, etc.)
   * @param {string} settings.device - 设备 (cuda, cpu)
   * @param {number} settings.batch_size - 批次大小
   * @param {boolean} settings.word_timestamps - 是否生成词级时间戳
   * @returns {Promise<{job_id: string, started: boolean, queue_position: number}>}
   */
  async startJob(jobId, settings) {
    const formData = new FormData()
    formData.append('job_id', jobId)
    formData.append('settings', JSON.stringify(settings))

    // FormData 会自动设置正确的 Content-Type (multipart/form-data with boundary)
    return apiClient.post('/api/start', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  }

  /**
   * 取消任务
   * @param {string} jobId - 任务ID
   * @param {boolean} deleteData - 是否删除任务数据
   * @returns {Promise<{job_id: string, canceled: boolean, data_deleted: boolean}>}
   */
  async cancelJob(jobId, deleteData = false) {
    return apiClient.post(`/api/cancel/${jobId}`, null, {
      params: { delete_data: deleteData }
    })
  }

  /**
   * 暂停任务
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id: string, paused: boolean}>}
   */
  async pauseJob(jobId) {
    return apiClient.post(`/api/pause/${jobId}`)
  }

  /**
   * 恢复暂停的任务（重新加入队列）
   *
   * 与 restoreJob 不同：
   * - resumeJob: 恢复暂停的任务，重新加入队列尾部，状态变为 queued
   * - restoreJob: 从 checkpoint 断点续传
   *
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id: string, resumed: boolean, status: string, queue_position: number}>}
   */
  async resumeJob(jobId) {
    return apiClient.post(`/api/resume/${jobId}`)
  }

  /**
   * 任务插队
   * @param {string} jobId - 任务ID
   * @param {string} mode - 插队模式 ('gentle' | 'force')
   * @returns {Promise<{job_id: string, prioritized: boolean, mode: string, queue_position: number}>}
   */
  async prioritizeJob(jobId, mode = 'gentle') {
    return apiClient.post(`/api/prioritize/${jobId}`, null, {
      params: { mode }
    })
  }

  /**
   * 获取任务状态
   * @param {string} jobId - 任务ID
   * @param {boolean} includeMedia - 是否包含媒体状态信息
   * @returns {Promise<Object>} 完整任务对象
   */
  async getJobStatus(jobId, includeMedia = true) {
    return apiClient.get(`/api/status/${jobId}`, {
      params: { include_media: includeMedia }
    })
  }

  /**
   * 获取队列状态摘要
   * @returns {Promise<{queue: string[], running: string, interrupted: string, jobs: Object}>}
   */
  async getQueueStatus() {
    return apiClient.get('/api/queue-status')
  }

  /**
   * 获取队列设置
   * @returns {Promise<{default_prioritize_mode: string}>}
   */
  async getQueueSettings() {
    return apiClient.get('/api/queue-settings')
  }

  /**
   * 更新队列设置
   * @param {string} defaultPrioritizeMode - 默认插队模式 ('gentle' | 'force')
   * @returns {Promise<{success: boolean, settings: Object}>}
   */
  async updateQueueSettings(defaultPrioritizeMode) {
    return apiClient.post('/api/queue-settings', {
      default_prioritize_mode: defaultPrioritizeMode
    })
  }

  /**
   * 重新排序队列
   * @param {string[]} jobIds - 按新顺序排列的任务ID列表
   * @returns {Promise<{reordered: boolean, queue: string[]}>}
   */
  async reorderQueue(jobIds) {
    return apiClient.post('/api/reorder-queue', {
      job_ids: jobIds
    })
  }

  /**
   * 下载 SRT 文件
   * @param {string} jobId - 任务ID
   * @param {boolean} copyToSource - 是否复制到源文件目录
   * @returns {Promise<Blob>} SRT 文件 Blob
   */
  async downloadResult(jobId, copyToSource = false) {
    const response = await apiClient.get(`/api/download/${jobId}`, {
      params: { copy_to_source: copyToSource },
      responseType: 'blob'
    })
    return response
  }

  /**
   * 复制转录结果到源文件目录
   * @param {string} jobId - 任务ID
   * @returns {Promise<{success: boolean, message: string, target_path: string}>}
   */
  async copyResultToSource(jobId) {
    return apiClient.post(`/api/copy-result/${jobId}`)
  }

  /**
   * 同步所有任务（第一阶段修复：数据同步）
   *
   * 从后端获取所有实际存在的任务列表（处理中 + 已完成）
   * 用于在应用启动时同步前端的 localStorage 与后端 jobs 目录的一致性
   * 修复幽灵任务问题
   *
   * @returns {Promise<{success: boolean, tasks: Array, count: number, timestamp: number}>}
   */
  async syncTasks() {
    return apiClient.get('/api/sync-tasks')
  }

  /**
   * 获取所有未完成的任务
   * @returns {Promise<{jobs: Object[], count: number}>}
   */
  async getIncompleteJobs() {
    return apiClient.get('/api/incomplete-jobs')
  }

  /**
   * 检查任务是否可以断点续传
   * @param {string} jobId - 任务ID
   * @returns {Promise<{can_resume: boolean, progress: number, message: string}>}
   */
  async checkResume(jobId) {
    return apiClient.get(`/api/check-resume/${jobId}`)
  }

  /**
   * 从检查点恢复任务
   * @param {string} jobId - 任务ID
   * @returns {Promise<Object>} 任务对象
   */
  async restoreJob(jobId) {
    return apiClient.post(`/api/restore-job/${jobId}`)
  }

  /**
   * 获取 checkpoint 中保存的原始设置
   * @param {string} jobId - 任务ID
   * @returns {Promise<{has_checkpoint: boolean, original_settings: Object, progress: Object}>}
   */
  async getCheckpointSettings(jobId) {
    return apiClient.get(`/api/checkpoint-settings/${jobId}`)
  }

  /**
   * 获取已完成的转录文字（从checkpoint）
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id: string, segments: Array, progress: Object}>}
   */
  async getTranscriptionText(jobId) {
    return apiClient.get(`/api/transcription-text/${jobId}`)
  }

  /**
   * 校验恢复任务时的参数修改
   * @param {string} jobId - 任务ID
   * @param {Object} newSettings - 新设置
   * @returns {Promise<{valid: boolean, warnings: Array, errors: Array, force_original: Object}>}
   */
  async validateResumeSettings(jobId, newSettings) {
    const formData = new FormData()
    formData.append('job_id', jobId)
    formData.append('new_settings', JSON.stringify(newSettings))

    return apiClient.post('/api/validate-resume-settings', formData)
  }

  /**
   * 获取任务缩略图（任务卡片展示用）
   * @param {string} jobId - 任务ID
   * @returns {Promise<{thumbnail: string|null, message: string}>} Base64编码的JPEG缩略图或null
   */
  async getThumbnail(jobId) {
    return apiClient.get(`/api/media/${jobId}/thumbnail`)
  }

  /**
   * 重命名任务
   * @param {string} jobId - 任务ID
   * @param {string} title - 新的任务名称（为空时恢复使用 filename）
   * @returns {Promise<{success: boolean, job_id: string, title: string, message: string}>}
   */
  async renameJob(jobId, title) {
    return apiClient.post(`/api/rename-job/${jobId}`, {
      title
    })
  }
}

// 导出单例实例
export default new TranscriptionAPI()
