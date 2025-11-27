/**
 * ProjectStore - 项目数据管理
 *
 * 负责管理字幕编辑器的核心数据，包括字幕数据、播放器状态、视图配置等
 * 实现了撤销/重做、自动保存、智能问题检测等功能
 */
import { defineStore } from "pinia";
import { ref, computed, watch, toRaw } from "vue";
import { useRefHistory } from "@vueuse/core";
import localforage from "localforage";
import smartSaver from "@/services/SmartSaver";

export const useProjectStore = defineStore("project", () => {
  // ========== 1. 项目元数据 ==========
  const meta = ref({
    jobId: null, // 转录任务ID
    videoPath: null, // 视频文件路径
    audioPath: null, // 音频文件路径
    peaksPath: null, // 波形峰值数据路径
    duration: 0, // 视频总时长（秒）
    filename: "", // 源文件名
    title: "", // 用户自定义任务名称
    videoFormat: null, // 视频格式
    hasProxyVideo: false, // 是否有 Proxy 视频
    lastSaved: Date.now(), // 最后保存时间
    isDirty: false, // 是否有未保存修改
  });

  // ========== 2. 字幕数据（Single Source of Truth） ==========
  const subtitles = ref([]);

  // ========== 3. Undo/Redo 历史记录 ==========
  const {
    history,
    undo,
    redo,
    canUndo,
    canRedo,
    clear: clearHistory,
  } = useRefHistory(subtitles, {
    deep: true,
    capacity: 50, // 限制历史记录步数
    clone: true, // 深拷贝，确保历史记录独立
  });

  // ========== 4. 播放器全局状态 ==========
  const player = ref({
    currentTime: 0, // 当前播放时间（秒）
    isPlaying: false, // 是否正在播放
    playbackRate: 1.0, // 播放速度（0.5-4.0）
    volume: 1.0, // 音量（0.0-1.0）
  });

  // ========== 5. 视图状态 ==========
  const view = ref({
    theme: "dark", // 'dark' | 'light'
    zoomLevel: 100, // 波形缩放比例（%）
    autoScroll: true, // 列表自动跟随播放
    selectedSubtitleId: null, // 当前选中的字幕ID
  });

  // ========== 6. 计算属性 ==========
  const totalSubtitles = computed(() => subtitles.value.length);

  const currentSubtitle = computed(() => {
    return subtitles.value.find(
      (s) =>
        player.value.currentTime >= s.start && player.value.currentTime < s.end
    );
  });

  const isDirty = computed(() => {
    return meta.value.isDirty || subtitles.value.some((s) => s.isDirty);
  });

  // 智能问题检测
  const validationErrors = computed(() => {
    const errors = [];

    // 检测时间重叠
    for (let i = 0; i < subtitles.value.length - 1; i++) {
      if (subtitles.value[i].end > subtitles.value[i + 1].start) {
        errors.push({
          type: "overlap",
          index: i,
          message: `字幕 #${i + 1} 与 #${i + 2} 时间重叠`,
          severity: "error",
        });
      }
    }

    // 检测字数过长
    subtitles.value.forEach((sub, i) => {
      if (sub.text.length > 30) {
        errors.push({
          type: "too_long",
          index: i,
          message: `字幕 #${i + 1} 超过30字`,
          severity: "warning",
        });
      }
    });

    // 检测显示时长异常
    subtitles.value.forEach((sub, i) => {
      const duration = sub.end - sub.start;
      if (duration < 0.5) {
        errors.push({
          type: "too_short",
          index: i,
          message: `字幕 #${i + 1} 显示时长过短（<0.5秒）`,
          severity: "warning",
        });
      } else if (duration > 7) {
        errors.push({
          type: "too_long_duration",
          index: i,
          message: `字幕 #${i + 1} 显示时长过长（>7秒）`,
          severity: "warning",
        });
      }
    });

    // 检测空字幕
    subtitles.value.forEach((sub, i) => {
      if (!sub.text || sub.text.trim() === "") {
        errors.push({
          type: "empty",
          index: i,
          message: `字幕 #${i + 1} 文本为空`,
          severity: "error",
        });
      }
    });

    return errors;
  });

  // ========== 7. 智能保存系统 ==========
  // 内存缓存（热数据）
  const memoryCache = new Map();
  const MAX_MEMORY_CACHE = 10; // 最多缓存10个任务的数据

  // 标记是否正在进行保存后的状态更新（避免循环触发）
  let isUpdatingAfterSave = false;

  // 配置智能保存回调
  smartSaver.onSaveSuccess = (jobId) => {
    console.log("[ProjectStore] 自动保存成功:", jobId);
    // 使用标记避免循环触发 watch
    isUpdatingAfterSave = true;
    meta.value.lastSaved = Date.now();
    meta.value.isDirty = false;
    // 重置每个字幕的 isDirty 标记
    subtitles.value.forEach((s) => (s.isDirty = false));
    isUpdatingAfterSave = false;
  };

  smartSaver.onSaveError = (error, jobId) => {
    console.error("[ProjectStore] 自动保存失败:", jobId, error);
  };

  // 监听数据变化，触发智能保存
  watch(
    [subtitles, meta],
    () => {
      // 跳过保存后的状态更新触发
      if (isUpdatingAfterSave) return;
      if (!meta.value.jobId) return;

      // 更新内存缓存
      memoryCache.set(meta.value.jobId, {
        subtitles: toRaw(subtitles.value),
        meta: toRaw(meta.value),
      });

      // 限制内存缓存大小（LRU淘汰）
      if (memoryCache.size > MAX_MEMORY_CACHE) {
        const firstKey = memoryCache.keys().next().value;
        memoryCache.delete(firstKey);
      }

      // 触发智能保存
      smartSaver.save({
        jobId: meta.value.jobId,
        subtitles: subtitles.value,
        meta: meta.value,
      });
    },
    { deep: true }
  );

  // ========== 8. Actions ==========

  /**
   * 导入SRT字幕（从转录结果加载）
   */
  function importSRT(srtContent, metadata) {
    const parsed = parseSRT(srtContent);
    subtitles.value = parsed.map((item, idx) => ({
      id: `subtitle-${Date.now()}-${idx}`,
      start: item.start,
      end: item.end,
      text: item.text,
      isDirty: false,
    }));

    meta.value = {
      ...meta.value,
      ...metadata,
      lastSaved: Date.now(),
      isDirty: false,
    };

    // 清除历史记录，避免撤销到空状态
    clearHistory();
  }

  /**
   * 从缓存/存储恢复项目
   */
  async function restoreProject(jobId) {
    try {
      // 优先从内存缓存获取
      if (memoryCache.has(jobId)) {
        const cached = memoryCache.get(jobId);
        subtitles.value = cached.subtitles;
        meta.value = cached.meta;
        console.log("[ProjectStore] 项目已从内存缓存恢复");
        return true;
      }

      // 使用智能保存系统恢复（支持 IndexedDB + localStorage 备份）
      const saved = await smartSaver.restoreFromBackup(jobId);
      if (saved) {
        subtitles.value = saved.subtitles;
        meta.value = saved.meta;
        console.log("[ProjectStore] 项目已从存储恢复");
        return true;
      }
      return false;
    } catch (error) {
      console.error("[ProjectStore] 恢复项目失败:", error);
      return false;
    }
  }

  /**
   * 更新字幕内容
   */
  function updateSubtitle(id, payload) {
    const index = subtitles.value.findIndex((s) => s.id === id);
    if (index === -1) return;

    subtitles.value[index] = {
      ...subtitles.value[index],
      ...payload,
      isDirty: true,
    };
    meta.value.isDirty = true;
  }

  /**
   * 添加字幕
   */
  function addSubtitle(insertIndex, payload) {
    const newSubtitle = {
      id: `subtitle-${Date.now()}`,
      start: payload.start || 0,
      end: payload.end || 0,
      text: payload.text || "",
      isDirty: true,
    };
    subtitles.value.splice(insertIndex, 0, newSubtitle);
    meta.value.isDirty = true;
  }

  /**
   * 删除字幕
   */
  function removeSubtitle(id) {
    const index = subtitles.value.findIndex((s) => s.id === id);
    if (index !== -1) {
      subtitles.value.splice(index, 1);
      meta.value.isDirty = true;
    }
  }

  /**
   * 导出SRT字符串
   */
  function generateSRT() {
    let srtContent = "";
    subtitles.value.forEach((sub, index) => {
      srtContent += `${index + 1}\n`;
      srtContent += `${formatTimestamp(sub.start)} --> ${formatTimestamp(
        sub.end
      )}\n`;
      srtContent += `${sub.text}\n\n`;
    });
    return srtContent;
  }

  /**
   * 同步播放器时间
   */
  function seekTo(time) {
    player.value.currentTime = time;
  }

  /**
   * 保存项目（持久化到后端 + 本地强制保存）
   */
  async function saveProject() {
    // TODO: 调用后端API保存编辑后的字幕
    const srtContent = generateSRT();
    // await api.saveSubtitle(meta.value.jobId, srtContent)

    // 强制立即保存到本地存储
    await smartSaver.forceSave({
      jobId: meta.value.jobId,
      subtitles: subtitles.value,
      meta: meta.value,
    });

    meta.value.lastSaved = Date.now();
    meta.value.isDirty = false;
    subtitles.value.forEach((s) => (s.isDirty = false));
    console.log("[ProjectStore] 项目已保存");
  }

  /**
   * 重置项目状态
   */
  function resetProject() {
    subtitles.value = [];
    meta.value = {
      jobId: null,
      videoPath: null,
      audioPath: null,
      peaksPath: null,
      duration: 0,
      filename: "",
      videoFormat: null,
      hasProxyVideo: false,
      lastSaved: Date.now(),
      isDirty: false,
    };
    player.value = {
      currentTime: 0,
      isPlaying: false,
      playbackRate: 1.0,
      volume: 1.0,
    };
    clearHistory();
    console.log("[ProjectStore] 项目已重置");
  }

  // ========== 9. 辅助函数 ==========

  /**
   * 解析SRT字符串
   */
  function parseSRT(srtContent) {
    const blocks = srtContent.trim().split(/\n\n+/);
    return blocks
      .map((block) => {
        const lines = block.split("\n");
        const timeMatch = lines[1]?.match(
          /(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})/
        );
        if (!timeMatch) return null;

        return {
          start: parseTimestamp(timeMatch[1]),
          end: parseTimestamp(timeMatch[2]),
          text: lines.slice(2).join("\n"),
        };
      })
      .filter(Boolean);
  }

  /**
   * 解析时间戳字符串
   */
  function parseTimestamp(ts) {
    // "00:01:23,456" => 83.456 秒
    const [h, m, s] = ts.replace(",", ".").split(":");
    return parseInt(h) * 3600 + parseInt(m) * 60 + parseFloat(s);
  }

  /**
   * 格式化时间戳
   */
  function formatTimestamp(sec) {
    // 83.456 => "00:01:23,456"
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.round((sec % 1) * 1000);
    return `${h.toString().padStart(2, "0")}:${m
      .toString()
      .padStart(2, "0")}:${s.toString().padStart(2, "0")},${ms
      .toString()
      .padStart(3, "0")}`;
  }

  return {
    // 状态
    meta,
    subtitles,
    player,
    view,

    // 计算属性
    totalSubtitles,
    currentSubtitle,
    isDirty,
    validationErrors,

    // 历史记录
    canUndo,
    canRedo,
    undo,
    redo,
    clearHistory,

    // 操作方法
    importSRT,
    restoreProject,
    updateSubtitle,
    addSubtitle,
    removeSubtitle,
    generateSRT,
    seekTo,
    saveProject,
    resetProject,

    // 辅助方法
    formatTimestamp,
    parseTimestamp,
  };
});
