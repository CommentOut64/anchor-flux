/**
 * 硬件检测 API 服务
 */
import apiClient from './api'

export const hardwareService = {
  /**
   * 获取基础硬件信息
   */
  async getBasicInfo() {
    try {
      const response = await apiClient.get('/hardware/basic')
      return response.data
    } catch (error) {
      console.error('获取硬件信息失败:', error)
      throw error
    }
  },

  /**
   * 获取优化配置
   */
  async getOptimization() {
    try {
      const response = await apiClient.get('/hardware/optimize')
      return response.data
    } catch (error) {
      console.error('获取优化配置失败:', error)
      throw error
    }
  },

  /**
   * 获取完整硬件状态
   */
  async getStatus() {
    try {
      const response = await apiClient.get('/hardware/status')
      return response.data
    } catch (error) {
      console.error('获取硬件状态失败:', error)
      throw error
    }
  }
}