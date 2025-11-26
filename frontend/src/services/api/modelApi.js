/**
 * 模型管理 API
 *
 * 职责：管理 Whisper 和对齐模型的下载、删除
 */

import { apiClient } from './client'

class ModelAPI {
  /**
   * 获取 Whisper 模型列表
   * @returns {Promise<Array<{
   *   id: string,
   *   name: string,
   *   size: number,
   *   status: string,
   *   downloaded: boolean
   * }>>}
   */
  async listWhisperModels() {
    return apiClient.get('/api/models/whisper')
  }

  /**
   * 获取对齐模型列表
   * @returns {Promise<Array<{
   *   language: string,
   *   name: string,
   *   status: string,
   *   downloaded: boolean
   * }>>}
   */
  async listAlignModels() {
    return apiClient.get('/api/models/align')
  }

  /**
   * 下载 Whisper 模型
   * @param {string} modelId - 模型ID (tiny, base, small, medium, large-v2, large-v3)
   * @returns {Promise<{success: boolean, message: string}>}
   */
  async downloadWhisperModel(modelId) {
    return apiClient.post(`/api/models/whisper/${modelId}/download`)
  }

  /**
   * 下载对齐模型
   * @param {string} language - 语言代码 (zh, en, ja, ko, etc.)
   * @returns {Promise<{success: boolean, message: string}>}
   */
  async downloadAlignModel(language) {
    return apiClient.post(`/api/models/align/${language}/download`)
  }

  /**
   * 删除 Whisper 模型
   * @param {string} modelId - 模型ID
   * @returns {Promise<{success: boolean, message: string}>}
   */
  async deleteWhisperModel(modelId) {
    return apiClient.delete(`/api/models/whisper/${modelId}`)
  }

  /**
   * 删除对齐模型
   * @param {string} language - 语言代码
   * @returns {Promise<{success: boolean, message: string}>}
   */
  async deleteAlignModel(language) {
    return apiClient.delete(`/api/models/align/${language}`)
  }
}

// 导出单例实例
export default new ModelAPI()
