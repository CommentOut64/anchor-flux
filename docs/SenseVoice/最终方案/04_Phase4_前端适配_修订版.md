# Phase 4: 前端适配（修订版）

> 目标：更新前端界面，支持 SenseVoice 引擎选择和硬件状态显示
>
> 工期：1-2天

---

## ⚠️ 重要修订

- ❌ **删除**：不创建使用 `hardware_detector.py` 的 API
- ✅ **复用**：使用现有的 `hardware_service.py` 和 `hardware_routes.py`
- ✅ **检查**：确认现有硬件 API 是否满足需求，不满足则扩展

---

## 一、任务清单

| 任务 | 文件 | 优先级 |
|------|------|--------|
| 引擎选择器 | `TaskListView.vue` | P0 |
| 硬件状态显示 | `TaskListView.vue` | P1 |
| 实时字幕预览 | `TaskListView.vue` | P1 |
| API 适配 | `api/index.js` | P0 |
| **检查现有硬件 API** | `hardware_routes.py` | P0 |

---

## 二、前置检查：现有硬件 API（⚠️ 重要）

在开始前端开发前，先检查现有的硬件 API 是否满足需求：

### 2.1 检查现有路由

```bash
# 查看现有硬件路由
cat backend/app/api/routes/hardware_routes.py
```

**预期发现**：
- 现有的 `/api/hardware` 端点
- 返回 `HardwareInfo` 数据

### 2.2 验证 API 响应格式

如果现有 API 返回格式类似：

```json
{
  "gpu_count": 1,
  "gpu_memory_mb": [8192],
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3060",
  "cpu_cores": 8,
  "cpu_name": "Intel Core i7-10700"
}
```

**则需要扩展**，添加 `OptimizationConfig` 信息。

### 2.3 扩展现有硬件 API（如需要）

**修改文件**: `backend/app/api/routes/hardware_routes.py`

在现有端点中添加优化配置信息：

```python
"""
硬件检测 API 路由（扩展）
"""
from fastapi import APIRouter
from ...services.hardware_service import get_hardware_detector, get_hardware_optimizer

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


@router.get("")
async def get_hardware_info():
    """
    获取硬件信息（扩展版）

    ✅ 复用现有服务，添加 SenseVoice 相关配置
    """
    # 使用现有服务
    detector = get_hardware_detector()
    hardware_info = detector.detect()

    optimizer = get_hardware_optimizer()
    optimization_config = optimizer.get_optimization_config(hardware_info)

    # 返回硬件信息 + 优化配置
    return {
        # 原有硬件信息
        "gpu_count": hardware_info.gpu_count,
        "gpu_memory_mb": hardware_info.gpu_memory_mb,
        "cuda_available": hardware_info.cuda_available,
        "gpu_name": hardware_info.gpu_name,
        "cpu_cores": hardware_info.cpu_cores,
        "cpu_name": hardware_info.cpu_name,
        "memory_total_mb": hardware_info.memory_total_mb,
        "memory_available_mb": hardware_info.memory_available_mb,

        # 新增：SenseVoice 优化配置
        "sensevoice_enabled": optimization_config.enable_sensevoice,
        "sensevoice_device": optimization_config.sensevoice_device,
        "demucs_enabled": optimization_config.enable_demucs,
        "demucs_model": optimization_config.demucs_model,
        "config_note": optimization_config.note,

        # 便捷字段（供前端直接使用）
        "gpu_memory_gb": max(hardware_info.gpu_memory_mb or [0]) / 1024 if hardware_info.gpu_memory_mb else 0,
        "has_gpu": hardware_info.cuda_available
    }
```

**注意**：
- ✅ 复用现有的 `get_hardware_detector()` 和 `get_hardware_optimizer()`
- ✅ 在现有端点上扩展，不创建新端点（除非现有端点不存在）
- ✅ 保持向后兼容，不删除原有字段

---

## 三、修改1：引擎选择器

**修改文件**: `frontend/src/views/TaskListView.vue`

### 3.1 添加引擎选择组件

在任务设置区域添加：

```vue
<template>
  <div class="task-settings">
    <!-- 现有设置 ... -->

    <!-- 引擎选择器（新增） -->
    <div class="setting-group">
      <label class="setting-label">转录引擎</label>
      <select v-model="taskSettings.engine" class="select-input" @change="onEngineChange">
        <option value="sensevoice">SenseVoice（推荐）</option>
        <option value="whisperx">WhisperX（传统）</option>
      </select>
      <p class="setting-hint">
        {{ engineHint }}
      </p>
    </div>

    <!-- SenseVoice 专用设置（新增） -->
    <div v-if="taskSettings.engine === 'sensevoice'" class="setting-group">
      <label class="setting-label">
        置信度阈值
        <span class="hint-text">（低于此值将触发 Whisper 补刀）</span>
      </label>
      <div class="slider-container">
        <input
          type="range"
          v-model.number="taskSettings.confidence_threshold"
          min="0.3"
          max="0.9"
          step="0.1"
          class="slider"
        />
        <span class="slider-value">
          {{ taskSettings.confidence_threshold.toFixed(1) }}
        </span>
      </div>
    </div>

    <!-- 硬件状态显示（新增） -->
    <div class="setting-group">
      <label class="setting-label">硬件状态</label>
      <div class="hardware-status">
        <div class="status-item">
          <span class="status-label">GPU:</span>
          <span :class="['status-value', hardwareStatus.has_gpu ? 'status-ok' : 'status-warn']">
            {{ hardwareStatus.gpu_name || '未检测到' }}
          </span>
        </div>
        <div class="status-item" v-if="hardwareStatus.has_gpu">
          <span class="status-label">显存:</span>
          <span class="status-value">
            {{ hardwareStatus.gpu_memory_gb.toFixed(1) }} GB
          </span>
        </div>
        <div class="status-item">
          <span class="status-label">推荐配置:</span>
          <span class="status-value status-note">
            {{ hardwareStatus.config_note || '检测中...' }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
```

### 3.2 添加 data 属性

```vue
<script>
export default {
  data() {
    return {
      taskSettings: {
        engine: 'sensevoice',               // 默认 SenseVoice
        confidence_threshold: 0.6,          // 置信度阈值
        // ... 其他设置
      },
      hardwareStatus: {
        has_gpu: false,
        gpu_name: '',
        gpu_memory_gb: 0,
        config_note: '检测中...'
      }
    }
  },
  computed: {
    engineHint() {
      const hints = {
        sensevoice: '高速转录，推理速度是 Whisper 的 15 倍，适合中文/英文/粤语',
        whisperx: '传统 Whisper 引擎，支持更多语言，速度较慢'
      }
      return hints[this.taskSettings.engine] || ''
    }
  },
  mounted() {
    // 加载硬件状态
    this.fetchHardwareStatus()
  },
  methods: {
    async fetchHardwareStatus() {
      try {
        const response = await this.$api.getHardwareStatus()
        // ✅ 使用扩展后的 API 响应
        this.hardwareStatus = {
          has_gpu: response.data.has_gpu,
          gpu_name: response.data.gpu_name,
          gpu_memory_gb: response.data.gpu_memory_gb,
          config_note: response.data.config_note
        }
      } catch (error) {
        console.error('获取硬件状态失败:', error)
        this.hardwareStatus.config_note = '获取硬件信息失败'
      }
    },
    onEngineChange() {
      // 引擎切换时可以做一些处理
      console.log('引擎已切换:', this.taskSettings.engine)
    }
  }
}
</script>
```

### 3.3 添加样式

```vue
<style scoped>
.slider-container {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.slider {
  flex: 1;
  height: 4px;
  background: #ddd;
  outline: none;
  border-radius: 2px;
}

.slider::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  background: #42b983;
  border-radius: 50%;
  cursor: pointer;
}

.slider-value {
  min-width: 3rem;
  text-align: center;
  font-weight: bold;
  color: #42b983;
}

.hardware-status {
  padding: 0.75rem;
  background: #f5f5f5;
  border-radius: 4px;
  font-size: 0.9rem;
}

.status-item {
  display: flex;
  align-items: center;
  margin-bottom: 0.5rem;
}

.status-item:last-child {
  margin-bottom: 0;
}

.status-label {
  min-width: 80px;
  font-weight: 500;
  color: #666;
}

.status-value {
  flex: 1;
  color: #333;
}

.status-ok {
  color: #42b983;
  font-weight: 500;
}

.status-warn {
  color: #e67e22;
  font-weight: 500;
}

.status-note {
  font-size: 0.85rem;
  color: #666;
  font-style: italic;
}

.hint-text {
  font-size: 0.85rem;
  color: #999;
  margin-left: 0.5rem;
}
</style>
```

---

## 四、修改2：实时字幕预览

在任务详情页面添加实时字幕预览：

### 4.1 添加预览组件

```vue
<template>
  <div class="task-detail">
    <!-- 现有内容 ... -->

    <!-- 实时字幕预览（新增） -->
    <div class="subtitle-preview" v-if="job.status === 'processing'">
      <h3>实时字幕预览</h3>
      <div class="preview-container">
        <div
          v-for="sentence in realtimeSubtitles"
          :key="sentence.id"
          class="subtitle-item"
        >
          <span class="subtitle-time">
            {{ formatTime(sentence.start) }} - {{ formatTime(sentence.end) }}
          </span>
          <span class="subtitle-text">{{ sentence.text }}</span>
          <span
            v-if="!sentence.is_final"
            class="subtitle-status status-processing"
          >
            处理中
          </span>
          <span
            v-else
            :class="['subtitle-confidence', getConfidenceClass(sentence.confidence)]"
          >
            {{ (sentence.confidence * 100).toFixed(0) }}%
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      realtimeSubtitles: []
    }
  },
  methods: {
    setupSSE() {
      // ... 现有 SSE 代码 ...

      // 新增：监听句级事件
      eventSource.addEventListener('sentence', (event) => {
        const sentence = JSON.parse(event.data)
        sentence.id = Date.now() + Math.random()
        this.realtimeSubtitles.push(sentence)

        // 限制预览数量（最多显示最近 20 条）
        if (this.realtimeSubtitles.length > 20) {
          this.realtimeSubtitles.shift()
        }
      })
    },
    formatTime(seconds) {
      const mins = Math.floor(seconds / 60)
      const secs = (seconds % 60).toFixed(1)
      return `${mins}:${secs.padStart(4, '0')}`
    },
    getConfidenceClass(confidence) {
      if (confidence >= 0.8) return 'confidence-high'
      if (confidence >= 0.6) return 'confidence-medium'
      return 'confidence-low'
    }
  }
}
</script>

<style scoped>
.subtitle-preview {
  margin-top: 2rem;
  padding: 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background: #fafafa;
}

.subtitle-preview h3 {
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 1.1rem;
  color: #333;
}

.preview-container {
  max-height: 400px;
  overflow-y: auto;
  padding: 0.5rem;
  background: white;
  border-radius: 4px;
}

.subtitle-item {
  display: flex;
  align-items: center;
  padding: 0.5rem;
  margin-bottom: 0.5rem;
  border-bottom: 1px solid #f0f0f0;
  font-size: 0.9rem;
}

.subtitle-item:last-child {
  border-bottom: none;
}

.subtitle-time {
  min-width: 120px;
  color: #666;
  font-family: monospace;
}

.subtitle-text {
  flex: 1;
  color: #333;
  margin: 0 1rem;
}

.subtitle-confidence {
  min-width: 50px;
  text-align: right;
  font-weight: 500;
  font-size: 0.85rem;
}

.confidence-high {
  color: #27ae60;
}

.confidence-medium {
  color: #f39c12;
}

.confidence-low {
  color: #e74c3c;
}

.status-processing {
  color: #3498db;
  font-size: 0.85rem;
}
</style>
```

---

## 五、修改3：API 适配（⚠️ 修订）

**修改文件**: `frontend/src/services/api/index.js`

### 5.1 新增硬件状态 API

```javascript
export default {
  // ... 现有方法 ...

  /**
   * 获取硬件状态
   *
   * ✅ 使用现有的 /api/hardware 端点（已扩展）
   */
  async getHardwareStatus() {
    try {
      const response = await axios.get('/api/hardware')
      return response
    } catch (error) {
      console.error('获取硬件状态失败:', error)
      throw error
    }
  },

  /**
   * 创建转录任务（更新参数）
   */
  async createTranscriptionJob(params) {
    const payload = {
      input_path: params.inputPath,
      output_path: params.outputPath,
      settings: {
        engine: params.engine || 'sensevoice',
        confidence_threshold: params.confidenceThreshold || 0.6,
        language: params.language || 'auto',
        // ... 其他设置
      }
    }

    try {
      const response = await axios.post('/api/transcription/jobs', payload)
      return response
    } catch (error) {
      console.error('创建任务失败:', error)
      throw error
    }
  }
}
```

---

## 六、快速测试

### 6.1 测试硬件状态显示

1. 启动后端服务
2. 打开前端页面
3. 检查硬件状态区域是否显示正确信息
4. ✅ 验证数据来自现有 API

### 6.2 测试引擎选择

1. 切换引擎选项
2. 检查提示文字是否变化
3. 创建任务，验证引擎参数是否正确传递

### 6.3 测试实时字幕

1. 创建一个转录任务
2. 观察实时字幕预览区域
3. 验证字幕是否实时更新

---

## 七、验收标准

- [ ] 引擎选择器可正常切换
- [ ] 硬件状态正确显示
- [ ] 实时字幕预览可正常工作
- [ ] API 参数正确传递到后端
- [ ] **使用现有硬件 API，不创建重复端点**
- [ ] **硬件信息包含 SenseVoice 优化配置**

---

## 八、注意事项

1. **兼容性**：
   - 确保新界面不影响现有 WhisperX 功能
   - 保留原有配置项

2. **用户体验**：
   - 硬件状态加载失败时显示友好提示
   - 实时字幕预览限制显示数量（避免卡顿）

3. **样式一致性**：
   - 新组件样式与现有界面保持一致
   - 使用项目现有的颜色方案

4. **⚠️ API 复用原则**（关键）：
   - ✅ 使用现有的 `/api/hardware` 端点
   - ✅ 扩展现有端点而非创建新端点
   - ✅ 保持向后兼容

---

## 九、下一步

完成 Phase 4（修订版）后，进入 [Phase 5: 整合测试](./05_Phase5_整合测试.md)

（Phase 5 无需修订，测试脚本可直接使用）
