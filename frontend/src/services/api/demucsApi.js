/**
 * Demucs分级人声分离配置API
 */
import { apiClient } from './client'

/**
 * Demucs质量预设
 * @typedef {Object} DemucsPreset
 * @property {string} description - 预设描述
 * @property {string} weak_model - 弱BGM使用的模型
 * @property {string} strong_model - 强BGM使用的模型
 * @property {string} fallback_model - 兜底模型
 */

/**
 * Demucs模型信息
 * @typedef {Object} DemucsModel
 * @property {string} description - 模型描述
 * @property {number} size_mb - 模型大小(MB)
 * @property {number} quality_score - 质量评分(1-5)
 * @property {number} speed_score - 速度评分(1-5)
 */

/**
 * Demucs用户配置
 * @typedef {Object} DemucsUserSettings
 * @property {boolean} enabled - 是否启用人声分离
 * @property {string} mode - 工作模式: "auto" | "always" | "never"
 * @property {string} quality_preset - 质量预设: "fast" | "balanced" | "quality"
 * @property {string} [weak_model] - 弱BGM模型（高级配置）
 * @property {string} [strong_model] - 强BGM模型（高级配置）
 * @property {string} [fallback_model] - 兜底模型（高级配置）
 * @property {boolean} [auto_escalation] - 是否允许自动升级模型
 * @property {number} [max_escalations] - 最大升级次数
 * @property {number} [bgm_light_threshold] - 轻微BGM阈值
 * @property {number} [bgm_heavy_threshold] - 强BGM阈值
 * @property {string} [on_break] - 熔断策略: "continue" | "fallback" | "fail" | "pause"
 */

/**
 * 获取完整的Demucs配置
 * @returns {Promise<{
 *   version: string,
 *   description: string,
 *   presets: Record<string, DemucsPreset>,
 *   models: Record<string, DemucsModel>,
 *   defaults: Object
 * }>}
 */
export async function getDemucsConfig() {
  const response = await apiClient.get('/api/demucs/config')
  return response.data
}

/**
 * 获取质量预设列表
 * @returns {Promise<Record<string, DemucsPreset>>}
 */
export async function getDemucsPresets() {
  const response = await apiClient.get('/api/demucs/presets')
  return response.data
}

/**
 * 获取可用模型信息
 * @returns {Promise<Record<string, DemucsModel>>}
 */
export async function getDemucsModels() {
  const response = await apiClient.get('/api/demucs/models')
  return response.data
}

/**
 * 获取默认配置参数
 * @returns {Promise<{
 *   bgm_light_threshold: number,
 *   bgm_heavy_threshold: number,
 *   consecutive_threshold: number,
 *   ratio_threshold: number,
 *   max_escalations: number
 * }>}
 */
export async function getDemucsDefaults() {
  const response = await apiClient.get('/api/demucs/defaults')
  return response.data
}

/**
 * 创建默认的Demucs用户配置
 * @returns {DemucsUserSettings}
 */
export function createDefaultDemucsSettings() {
  return {
    enabled: true,
    mode: 'auto',
    quality_preset: 'balanced',
    auto_escalation: true,
    max_escalations: 1
  }
}

/**
 * 应用质量预设到配置
 * @param {DemucsUserSettings} settings - 当前配置
 * @param {DemucsPreset} preset - 要应用的预设
 * @returns {DemucsUserSettings} 更新后的配置
 */
export function applyPresetToSettings(settings, preset) {
  return {
    ...settings,
    weak_model: preset.weak_model,
    strong_model: preset.strong_model,
    fallback_model: preset.fallback_model
  }
}
