/**
 * 文件管理 API
 *
 * 职责：管理 input 目录中的文件
 */

import { apiClient } from './client'

class FileAPI {
  /**
   * 获取 input 目录中的所有媒体文件
   * @returns {Promise<{
   *   files: Array<{name: string, size: number, modified: string}>,
   *   input_dir: string,
   *   message?: string
   * }>}
   */
  async listFiles() {
    return apiClient.get('/api/files')
  }

  /**
   * 删除 input 目录中的文件
   * @param {string} filename - 文件名
   * @returns {Promise<{success: boolean, message: string}>}
   */
  async deleteFile(filename) {
    return apiClient.delete(`/api/files/${encodeURIComponent(filename)}`)
  }
}

// 导出单例实例
export default new FileAPI()
