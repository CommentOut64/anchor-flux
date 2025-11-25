# PlaybackControls - 播放控制组件

## 概述

PlaybackControls 是媒体播放控制面板组件，提供播放、暂停、进度控制、倍速调节、音量控制等功能。它与 ProjectStore 的播放器状态双向绑定，为用户提���直观的媒体控制界面。

## 功能特性

1. **基础播放控制**
   - 播放/暂停切换
   - 进度条拖拽定位
   - 时间显示（当前/总时长）

2. **高级控制功能**
   - 倍速播放（0.5x - 4.0x）
   - 音量调节（带静音）
   - 循环播放设置
   - 快进/快退按钮

3. **键盘快捷键**
   - 空格：播放/暂停
   - ←/→：快退/快进
   - ↑/↓：音量调节
   - Shift + ←/→：微调时间

4. **状态同步**
   - 与 ProjectStore 实时同步
   - 支持外部控制响应

## 技术依赖

```json
{
  "vue": "^3.4.0",
  "element-plus": "^2.11.0",
  "pinia": "^3.0.0",
  "@vueuse/core": "^10.0.0"
}
```

## 组件属性

```typescript
interface PlaybackControlsProps {
  // 布局配置
  compact?: boolean         // 紧凑模式，默认false
  vertical?: boolean       // 垂直布局，默认false

  // 功能开关
  showSpeed?: boolean      // 显示倍速控制，默认true
  showVolume?: boolean     // 显示音量控制，默认true
  showLoop?: boolean       // 显示循环按钮，默认true
  showFullscreen?: boolean // 显示全屏按钮，默认true

  // 快进配置
  seekStep?: number       // 快进/快退步长（秒），默认5
  microSeekStep?: number  // 微调步长（秒），默认0.1
}

interface PlaybackControlsEmits {
  'play': () => void
  'pause': () => void
  'seek': (time: number) => void
  'speed-change': (rate: number) => void
  'volume-change': (volume: number) => void
}
```

## 核心实现

```vue
<!-- components/PlaybackControls/index.vue -->
<template>
  <div
    class="playback-controls"
    :class="{
      'compact': compact,
      'vertical': vertical
    }"
  >
    <!-- 主控制区 -->
    <div class="main-controls">
      <!-- 播放/暂停 -->
      <el-button
        @click="togglePlay"
        circle
        :icon="isPlaying ? VideoPause : VideoPlay"
      />

      <!-- 快退 -->
      <el-button
        @click="seek(-seekStep)"
        circle
        :icon="DArrowLeft"
      />

      <!-- 进度条 -->
      <div class="progress-wrapper">
        <span class="time-current">{{ formatTime(currentTime) }}</span>
        <el-slider
          v-model="currentTime"
          :max="duration"
          :step="0.1"
          :show-tooltip="false"
          @change="handleSeek"
        />
        <span class="time-total">{{ formatTime(duration) }}</span>
      </div>

      <!-- 快进 -->
      <el-button
        @click="seek(seekStep)"
        circle
        :icon="DArrowRight"
      />
    </div>

    <!-- 辅助控制区 -->
    <div class="extra-controls">
      <!-- 音量控制 -->
      <div v-if="showVolume" class="volume-control">
        <el-button
          @click="toggleMute"
          circle
          size="small"
          :icon="volumeIcon"
        />
        <el-slider
          v-model="volume"
          :max="1"
          :step="0.01"
          :show-tooltip="false"
          style="width: 100px"
          @change="handleVolumeChange"
        />
      </div>

      <!-- 倍速控制 -->
      <el-dropdown v-if="showSpeed" trigger="click">
        <el-button size="small">
          {{ playbackRate }}x
        </el-button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item
              v-for="speed in speedOptions"
              :key="speed"
              @click="setSpeed(speed)"
              :class="{ active: playbackRate === speed }"
            >
              {{ speed }}x
            </el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>

      <!-- 循环播放 -->
      <el-button
        v-if="showLoop"
        @click="toggleLoop"
        circle
        size="small"
        :type="isLooping ? 'primary' : 'default'"
        :icon="RefreshRight"
      />

      <!-- 全屏 -->
      <el-button
        v-if="showFullscreen"
        @click="requestFullscreen"
        circle
        size="small"
        :icon="FullScreen"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import {
  VideoPlay,
  VideoPause,
  DArrowLeft,
  DArrowRight,
  RefreshRight,
  FullScreen,
  Mute,
  Microphone
} from '@element-plus/icons-vue'

// Props & Emits
const props = defineProps({
  compact: {
    type: Boolean,
    default: false
  },
  vertical: {
    type: Boolean,
    default: false
  },
  showSpeed: {
    type: Boolean,
    default: true
  },
  showVolume: {
    type: Boolean,
    default: true
  },
  showLoop: {
    type: Boolean,
    default: true
  },
  showFullscreen: {
    type: Boolean,
    default: true
  },
  seekStep: {
    type: Number,
    default: 5
  },
  microSeekStep: {
    type: Number,
    default: 0.1
  }
})

const emit = defineEmits([
  'play',
  'pause',
  'seek',
  'speed-change',
  'volume-change'
])

// Store
const projectStore = useProjectStore()

// State
const isLooping = ref(false)
const isMuted = ref(false)
const previousVolume = ref(1)

// Speed options
const speedOptions = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 3, 4]

// Computed from store
const currentTime = computed({
  get: () => projectStore.player.currentTime,
  set: (val) => projectStore.player.currentTime = val
})

const duration = computed(() => projectStore.meta.duration)
const isPlaying = computed(() => projectStore.player.isPlaying)
const playbackRate = computed(() => projectStore.player.playbackRate)

const volume = computed({
  get: () => isMuted.value ? 0 : projectStore.player.volume,
  set: (val) => {
    projectStore.player.volume = val
    if (val > 0) isMuted.value = false
  }
})

const volumeIcon = computed(() => {
  if (isMuted.value || volume.value === 0) return Mute
  return Microphone
})

// Methods
function togglePlay() {
  projectStore.player.isPlaying = !projectStore.player.isPlaying
  if (projectStore.player.isPlaying) {
    emit('play')
  } else {
    emit('pause')
  }
}

function handleSeek(time) {
  projectStore.seekTo(time)
  emit('seek', time)
}

function seek(seconds) {
  const newTime = Math.max(0, Math.min(duration.value, currentTime.value + seconds))
  handleSeek(newTime)
}

function setSpeed(rate) {
  projectStore.player.playbackRate = rate
  emit('speed-change', rate)
}

function toggleMute() {
  if (isMuted.value) {
    projectStore.player.volume = previousVolume.value
    isMuted.value = false
  } else {
    previousVolume.value = projectStore.player.volume
    projectStore.player.volume = 0
    isMuted.value = true
  }
}

function handleVolumeChange(val) {
  emit('volume-change', val)
}

function toggleLoop() {
  isLooping.value = !isLooping.value
  // TODO: Implement loop logic
}

function requestFullscreen() {
  // Emit to parent to handle fullscreen
  // Parent should have video container reference
  document.querySelector('.video-container')?.requestFullscreen()
}

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)

  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }
  return `${m}:${s.toString().padStart(2, '0')}`
}

// Keyboard shortcuts
function handleKeyboard(e) {
  // Ignore if typing in input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

  switch(e.code) {
    case 'Space':
      e.preventDefault()
      togglePlay()
      break
    case 'ArrowLeft':
      e.preventDefault()
      if (e.shiftKey) {
        seek(-props.microSeekStep)
      } else {
        seek(-props.seekStep)
      }
      break
    case 'ArrowRight':
      e.preventDefault()
      if (e.shiftKey) {
        seek(props.microSeekStep)
      } else {
        seek(props.seekStep)
      }
      break
    case 'ArrowUp':
      e.preventDefault()
      volume.value = Math.min(1, volume.value + 0.1)
      break
    case 'ArrowDown':
      e.preventDefault()
      volume.value = Math.max(0, volume.value - 0.1)
      break
  }
}

// Loop handling
watch(currentTime, (time) => {
  if (isLooping.value && time >= duration.value - 0.1) {
    handleSeek(0)
    if (!isPlaying.value) {
      togglePlay()
    }
  }
})

// Lifecycle
onMounted(() => {
  document.addEventListener('keydown', handleKeyboard)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyboard)
})
</script>
```

## 样���定义

```scss
.playback-controls {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: 8px;

  // 主控制区
  .main-controls {
    display: flex;
    align-items: center;
    gap: 12px;

    .progress-wrapper {
      flex: 1;
      display: flex;
      align-items: center;
      gap: 12px;

      .time-current,
      .time-total {
        min-width: 50px;
        font-size: 14px;
        color: var(--text-muted);
      }

      .el-slider {
        flex: 1;
      }
    }
  }

  // 辅助控制区
  .extra-controls {
    display: flex;
    align-items: center;
    gap: 12px;

    .volume-control {
      display: flex;
      align-items: center;
      gap: 8px;
    }
  }

  // 紧凑模式
  &.compact {
    padding: 8px;

    .main-controls {
      gap: 8px;
    }

    .extra-controls {
      gap: 8px;
    }
  }

  // 垂直布局
  &.vertical {
    width: 200px;

    .main-controls {
      flex-direction: column;

      .progress-wrapper {
        flex-direction: column;
        width: 100%;

        .el-slider {
          width: 100%;
        }
      }
    }

    .extra-controls {
      flex-direction: column;
      width: 100%;

      .volume-control {
        width: 100%;

        .el-slider {
          flex: 1;
        }
      }
    }
  }
}
```

## 与其他组件的关系

### 依赖关系
- 强依赖 `ProjectStore` 的播放器状态
- 使用 Element Plus 的 UI 组件

### 被依赖关系
- 被 `EditorView` 包含使用
- 与 `VideoStage` 协同控制播放
- 与 `WaveformTimeline` 时间同步

## 键盘快捷键

| 快捷键 | 功能 |
|--------|------|
| 空格 | 播放/暂停 |
| ← | 快退5秒 |
| → | 快进5秒 |
| Shift + ← | 微调后退0.1秒 |
| Shift + → | 微调前进0.1秒 |
| ↑ | 增加音量 |
| ↓ | 减少音量 |

## 性能优化

1. **防抖处理**
   - 进度条拖拽使用防抖
   - 避免频繁更新Store

2. **事件优化**
   - 键盘事件使用事件委托
   - 组件销毁时清理监听器

## 测试要点

1. 播放控制响应性
2. 进度条精确度
3. 倍速切换功能
4. 音量控制和静音
5. 键盘快捷键
6. 循环播放逻辑

## 未来扩展

1. 添加A-B循环功能
2. 帧级精确控制
3. 播放列表支持
4. 手势控制
5. 自定义快捷键