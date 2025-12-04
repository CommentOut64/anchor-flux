/**
 * 系统管理 API
 * 包含系统关闭、客户端心跳等功能
 */

import { apiClient } from './client'

/**
 * 检查是否有活跃的客户端
 * @returns {Promise<{has_active: boolean, count: number}>}
 */
export async function hasActiveClients() {
  return apiClient.get('/api/system/has-active-clients')
}

/**
 * 客户端注册
 * @param {string} clientId - 客户端ID
 * @param {string} [userAgent] - 客户端User-Agent
 * @returns {Promise<{success: boolean, client_id: string}>}
 */
export async function registerClient(clientId, userAgent = null) {
  return apiClient.post('/api/system/register', {
    client_id: clientId,
    user_agent: userAgent
  })
}

/**
 * 客户端心跳
 * @param {string} clientId - 客户端ID
 * @returns {Promise<{success: boolean, active_clients: number}>}
 */
export async function heartbeat(clientId) {
  return apiClient.post('/api/system/heartbeat', {
    client_id: clientId
  })
}

/**
 * 客户端注销
 * @param {string} clientId - 客户端ID
 * @returns {Promise<{success: boolean}>}
 */
export async function unregisterClient(clientId) {
  return apiClient.post('/api/system/unregister', {
    client_id: clientId
  })
}

/**
 * 关闭系统
 * @param {Object} options - 关闭选项
 * @param {boolean} [options.cleanup_temp=false] - 是否清理临时文件
 * @param {boolean} [options.force=false] - 是否强制关闭
 * @returns {Promise<{success: boolean, message: string, cleanup_report: Object}>}
 */
export async function shutdownSystem(options = {}) {
  return apiClient.post('/api/system/shutdown', {
    cleanup_temp: options.cleanup_temp || false,
    force: options.force || false
  })
}

const systemApi = {
  hasActiveClients,
  registerClient,
  heartbeat,
  unregisterClient,
  shutdownSystem
}

export default systemApi
