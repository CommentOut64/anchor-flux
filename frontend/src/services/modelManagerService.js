/**
 * 模型管理服务
 * 用于模型下载、删除和状态查询
 */

import axios from 'axios'

const API_BASE = '/api/models'

export const ModelManagerService = {
  /**
   * 获取所有Whisper模型列表
   * @returns {Promise<Array>} 模型列表
   */
  async getWhisperModels() {
    try {
      const response = await axios.get(`${API_BASE}/whisper`)
      return response.data
    } catch (error) {
      console.error('获取Whisper模型列表失败:', error)
      throw error
    }
  },

  /**
   * 获取所有对齐模型列表
   * @returns {Promise<Array>} 对齐模型列表
   */
  async getAlignModels() {
    try {
      const response = await axios.get(`${API_BASE}/align`)
      return response.data
    } catch (error) {
      console.error('获取对齐模型列表失败:', error)
      throw error
    }
  },

  /**
   * 下载Whisper模型
   * @param {string} modelId - 模型ID
   * @returns {Promise<Object>} 操作结果
   */
  async downloadWhisperModel(modelId) {
    try {
      const response = await axios.post(`${API_BASE}/whisper/${modelId}/download`)
      return response.data
    } catch (error) {
      console.error(`下载Whisper模型失败: ${modelId}`, error)
      throw error
    }
  },

  /**
   * 下载对齐模型
   * @param {string} language - 语言代码
   * @returns {Promise<Object>} 操作结果
   */
  async downloadAlignModel(language) {
    try {
      const response = await axios.post(`${API_BASE}/align/${language}/download`)
      return response.data
    } catch (error) {
      console.error(`下载对齐模型失败: ${language}`, error)
      throw error
    }
  },

  /**
   * 删除Whisper模型
   * @param {string} modelId - 模型ID
   * @returns {Promise<Object>} 操作结果
   */
  async deleteWhisperModel(modelId) {
    try {
      const response = await axios.delete(`${API_BASE}/whisper/${modelId}`)
      return response.data
    } catch (error) {
      console.error(`删除Whisper模型失败: ${modelId}`, error)
      throw error
    }
  },

  /**
   * 删除对齐模型
   * @param {string} language - 语言代码
   * @returns {Promise<Object>} 操作结果
   */
  async deleteAlignModel(language) {
    try {
      const response = await axios.delete(`${API_BASE}/align/${language}`)
      return response.data
    } catch (error) {
      console.error(`删除对齐模型失败: ${language}`, error)
      throw error
    }
  },

  /**
   * 获取下载进度（轮询模式）
   * @returns {Promise<Object>} 进度信息
   */
  async getDownloadProgress() {
    try {
      const response = await axios.get(`${API_BASE}/progress`)
      return response.data
    } catch (error) {
      console.error('获取下载进度失败:', error)
      throw error
    }
  },

  /**
   * 创建SSE连接以实时获取进度
   * @returns {EventSource} SSE连接对象
   */
  createProgressSSE() {
    return new EventSource(`${API_BASE}/events/progress`)
  }
}
