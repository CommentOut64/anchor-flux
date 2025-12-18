"""
VAD (Voice Activity Detection) 服务

从 transcription_service.py 中提取的 VAD 功能，封装为独立服务。
支持 Silero VAD（默认）和 Pyannote VAD（可选）。

Phase 2 实现 - 2025-12-10
"""

import os
import logging
import tempfile
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import torch


class VADMethod(Enum):
    """
    VAD模型选择枚举
    用于选择语音活动检测（Voice Activity Detection）模型
    """
    SILERO = "silero"      # 默认，无需认证，速度快
    PYANNOTE = "pyannote"  # 可选，需要HF Token，精度更高


@dataclass
class VADConfig:
    """
    VAD配置数据类
    用于配置语音活动检测的参数

    参数说明：
    - onset (0.0-1.0)：语音开始阈值，越高越严格，推荐0.5-0.6以避免截断语音起始
    - offset (0.0-1.0)：语音结束阈值，通常为onset的80%左右
    - min_speech_duration_ms：最小语音段长度，避免误检碎片音（推荐300-500ms）
    - min_silence_duration_ms：最小静音长度，越长越能过滤背景音乐（推荐300-500ms）

    Post-VAD合并参数（2025-12-17 迁移自旧架构 _merge_vad_segments）：
    - merge_max_gap：允许合并的最大静音间隔，超过1.5秒通常意味着换气或换话题
    - merge_max_duration：合并后的最大时长，12秒是甜蜜点，避免Whisper幻觉和对齐算法爆炸
    - merge_min_fragment：短于此时长的片段强制尝试合并（碎片保护）

    修改历史：
    - 2025-12: onset 从 0.7 降低至 0.5，offset 从 0.5 降低至 0.4
      原因：避免语音起始被截断，提高时间戳准确性
    - 2025-12-17: 新增 Post-VAD 合并参数，迁移自旧架构
      max_duration 从 25.0 改为 12.0，max_gap 从 1.0 改为 1.5
    """
    method: VADMethod = VADMethod.SILERO  # 默认使用Silero
    hf_token: Optional[str] = None         # Pyannote需要的HF Token
    onset: float = 0.4                     # 语音开始阈值（恢复旧值，保持向后兼容）
    offset: float = 0.4                    # 语音结束阈值（恢复旧值）
    chunk_size: int = 30                   # 最大段长（秒）
    min_speech_duration_ms: int = 250      # 最小语音段长度（恢复旧值400ms）
    min_silence_duration_ms: int = 400     # 最小静音长度（恢复旧值400ms）
    speech_pad_ms: int = 300
    # Post-VAD 合并参数
    merge_max_gap: float = 1.0             # 允许合并的最大静音间隔（秒），恢复旧值1.0s
    merge_max_duration: float = 12.0       # 合并后的最大时长（秒），恢复旧值25.0s
    merge_min_fragment: float = 1.0        # 短于此时长的片段强制尝试合并（碎片保护）

    def validate(self) -> bool:
        """验证配置有效性"""
        if self.method == VADMethod.PYANNOTE and not self.hf_token:
            return False  # Pyannote需要Token
        if not (0.0 <= self.onset <= 1.0) or not (0.0 <= self.offset <= 1.0):
            return False  # 阈值必须在0-1之间
        return True


class VADService:
    """
    VAD 服务类

    提供语音活动检测功能，支持多种 VAD 模型。
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化 VAD 服务

        Args:
            logger: 日志记录器，如果为 None 则创建新的
        """
        self.logger = logger or logging.getLogger(__name__)

    def detect_speech_segments(
        self,
        audio_array: np.ndarray,
        sr: int,
        config: VADConfig,
        enable_merge: bool = True
    ) -> List[Dict]:
        """
        检测语音段

        Args:
            audio_array: 音频数组
            sr: 采样率
            config: VAD 配置
            enable_merge: 是否启用 Post-VAD 智能合并（默认启用）

        Returns:
            List[Dict]: 分段元数据列表，每个元素包含:
                - index: 段索引
                - start: 起始时间（秒）
                - end: 结束时间（秒）
                - mode: 处理模式（"memory"）

        Raises:
            ValueError: 配置无效时抛出
        """
        if not config.validate():
            raise ValueError("VAD 配置无效")

        try:
            if config.method == VADMethod.SILERO:
                segments = self._vad_silero(audio_array, sr, config)
            elif config.method == VADMethod.PYANNOTE:
                segments = self._vad_pyannote(audio_array, sr, config)
            else:
                raise ValueError(f"不支持的 VAD 方法: {config.method}")
        except Exception as e:
            self.logger.error(f"VAD 检测失败: {e}")
            self.logger.warning("降级到能量检测分段")
            segments = self._energy_based_split(audio_array, sr, config.chunk_size)

        # Post-VAD 智能合并（默认启用）
        if enable_merge and len(segments) > 1:
            segments = self.merge_vad_segments(segments, config)

        return segments

    def _vad_silero(
        self,
        audio_array: np.ndarray,
        sr: int,
        vad_config: VADConfig
    ) -> List[Dict]:
        """
        Silero VAD分段（使用内置ONNX模型，无需下载）

        优点：
        - 使用项目内置ONNX模型，无需网络下载
        - 使用 onnxruntime 推理，跨平台兼容性好
        - 速度快，内存占用低（~2MB）

        Args:
            audio_array: 音频数组
            sr: 采样率
            vad_config: VAD配置

        Returns:
            List[Dict]: 分段元数据列表
        """
        self.logger.info("加载Silero VAD模型（内置ONNX）...")

        # 使用 silero-vad 库（基于 onnxruntime）
        from silero_vad import get_speech_timestamps
        from silero_vad.utils_vad import OnnxWrapper

        # 使用项目内置的 ONNX 模型
        builtin_model_path = Path(__file__).parent.parent.parent / "assets" / "silero" / "silero_vad.onnx"

        if not builtin_model_path.exists():
            raise FileNotFoundError(
                f"内置Silero VAD模型不存在: {builtin_model_path}\n"
                "请确保项目完整，或重新从源码仓库获取"
            )

        self.logger.info(f"使用内置模型: {builtin_model_path}")

        # 加载ONNX模型（直接从本地路径）
        model = OnnxWrapper(str(builtin_model_path), force_onnx_cpu=False)

        # 转换为torch tensor（silero-vad 需要）
        audio_tensor = torch.from_numpy(audio_array)

        # 获取语音时间戳
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sr,
            threshold=vad_config.onset,                    # 检测阈值（从config读取，默认0.4）
            min_speech_duration_ms=vad_config.min_speech_duration_ms,   # 最小语音段长度（默认250ms）
            min_silence_duration_ms=vad_config.min_silence_duration_ms, # 最小静音长度（默认400ms）
            speech_pad_ms=vad_config.speech_pad_ms,        # 语音段前后padding（默认300ms）
            return_seconds=False  # 返回采样点而非秒数
        )

        # VAD detection complete

        # 合并分段（确保每段不超过chunk_size秒）
        segments_metadata = []
        current_start = None
        current_end = None

        for ts in speech_timestamps:
            start_sec = ts['start'] / sr
            end_sec = ts['end'] / sr

            if current_start is None:
                current_start = start_sec
                current_end = end_sec
            elif (end_sec - current_start) <= vad_config.chunk_size:
                # 可以合并
                current_end = end_sec
            else:
                # 保存当前段，开始新段
                segments_metadata.append({
                    "index": len(segments_metadata),
                    "start": current_start,
                    "end": current_end,
                    "mode": "memory"
                })
                current_start = start_sec
                current_end = end_sec

        # 保存最后一段
        if current_start is not None:
            segments_metadata.append({
                "index": len(segments_metadata),
                "start": current_start,
                "end": current_end,
                "mode": "memory"
            })

        # 如果没有检测到任何语音段，按固定时长分段
        if len(segments_metadata) == 0:
            self.logger.warning("VAD未检测到语音，使用固定时长分段")
            return self._energy_based_split(audio_array, sr, vad_config.chunk_size)

        self.logger.info(f"Silero VAD 检测完成: {len(segments_metadata)} 个语音段")
        return segments_metadata

    def _vad_pyannote(
        self,
        audio_array: np.ndarray,
        sr: int,
        vad_config: VADConfig
    ) -> List[Dict]:
        """
        Pyannote VAD分段（高精度方案，需要HF Token）

        优点：
        - 精度更高
        - 支持更复杂的语音活动检测

        注意：
        - 需要HuggingFace Token
        - 首次使用需要接受模型使用协议

        Args:
            audio_array: 音频数组
            sr: 采样率
            vad_config: VAD配置

        Returns:
            List[Dict]: 分段元数据列表

        Raises:
            ValueError: 未配置HF Token时抛出
        """
        if not vad_config.hf_token:
            raise ValueError("Pyannote VAD需要HuggingFace Token，请在设置中配置")

        self.logger.info("加载Pyannote VAD模型（需要HF Token）...")

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise RuntimeError("Pyannote未安装，请使用Silero VAD或安装pyannote-audio")

        # 初始化Pyannote VAD Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=vad_config.hf_token
        )

        # 准备输入（Pyannote需要特定格式）
        # 创建临时文件用于Pyannote处理
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_array, sr)

        try:
            # 执行VAD
            vad_result = pipeline(temp_path)

            # 合并分段
            segments_metadata = []
            current_start = None
            current_end = None

            for speech in vad_result.get_timeline().support():
                start_sec = speech.start
                end_sec = speech.end

                if current_start is None:
                    current_start = start_sec
                    current_end = end_sec
                elif (end_sec - current_start) <= vad_config.chunk_size:
                    current_end = end_sec
                else:
                    segments_metadata.append({
                        "index": len(segments_metadata),
                        "start": current_start,
                        "end": current_end,
                        "mode": "memory"
                    })
                    current_start = start_sec
                    current_end = end_sec

            # 保存最后一段
            if current_start is not None:
                segments_metadata.append({
                    "index": len(segments_metadata),
                    "start": current_start,
                    "end": current_end,
                    "mode": "memory"
                })

            self.logger.info(f"Pyannote VAD 检测完成: {len(segments_metadata)} 个语音段")
            return segments_metadata

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _energy_based_split(
        self,
        audio_array: np.ndarray,
        sr: int,
        chunk_size: int = 30
    ) -> List[Dict]:
        """
        基于能量的简易分段（降级方案）

        当VAD模型加载失败时使用，按固定时长分段。
        会尝试在静音处分割以避免切断语音。

        Args:
            audio_array: 音频数组
            sr: 采样率
            chunk_size: 每段最大长度（秒）

        Returns:
            List[Dict]: 分段元数据列表
        """
        self.logger.warning("使用能量检测降级分段（固定时长）")

        total_duration = len(audio_array) / sr
        segments_metadata = []
        pos = 0.0

        while pos < total_duration:
            # 计算理想结束位置
            ideal_end = min(pos + chunk_size, total_duration)

            # 尝试在静音处分割（在理想结束点前后1秒范围内寻找）
            if ideal_end < total_duration:
                search_start = max(pos, ideal_end - 1.0)
                search_end = min(total_duration, ideal_end + 1.0)

                # 计算搜索范围内的能量
                start_sample = int(search_start * sr)
                end_sample = int(search_end * sr)
                search_audio = audio_array[start_sample:end_sample]

                if len(search_audio) > 0:
                    # 计算短时能量（每100ms一个窗口）
                    window_size = int(0.1 * sr)
                    energies = []
                    for i in range(0, len(search_audio) - window_size, window_size):
                        window = search_audio[i:i + window_size]
                        energy = np.sum(window ** 2)
                        energies.append((i, energy))

                    if energies:
                        # 找到能量最低的点
                        min_energy_idx = min(energies, key=lambda x: x[1])[0]
                        actual_end = search_start + (min_energy_idx / sr)
                        # 确保分段至少有1秒
                        if actual_end - pos >= 1.0:
                            ideal_end = actual_end

            segments_metadata.append({
                "index": len(segments_metadata),
                "start": pos,
                "end": ideal_end,
                "mode": "memory"
            })
            pos = ideal_end

        self.logger.info(f"能量检测分段完成: {len(segments_metadata)}段")
        return segments_metadata

    def merge_vad_segments(
        self,
        segments: List[Dict],
        config: VADConfig
    ) -> List[Dict]:
        """
        Post-VAD 智能合并层

        迁移自旧架构 transcription_service.py:_merge_vad_segments (2025-12-17)

        策略：宁可错合（依赖 SentenceSplitter 分句），不可错分（导致 ASR 丢失上下文）。

        合并条件1：基础合并 - 间隔小且总长不超标
        合并条件2：碎片保护 - 当前段极短（可能是被切断的单词），强制合并

        Args:
            segments: VAD 切分后的原始片段列表 [{start, end, index, mode}, ...]
            config: VAD 配置，包含合并参数

        Returns:
            合并后的片段列表
        """
        if not segments:
            return []

        max_gap = config.merge_max_gap            # 默认 1.5s
        max_duration = config.merge_max_duration  # 默认 12.0s
        min_fragment = config.merge_min_fragment  # 默认 1.0s

        merged = []
        current = segments[0].copy()

        for next_seg in segments[1:]:
            gap = next_seg['start'] - current['end']
            current_duration = current['end'] - current['start']
            combined_duration = next_seg['end'] - current['start']

            should_merge = False

            # 条件 1: 基础合并（间隔小且总长不超标）
            if gap <= max_gap and combined_duration <= max_duration:
                should_merge = True

            # 条件 2: 碎片保护（当前段极短，可能是被切断的单词）
            # 例如: "It's" (0.5s) ... [gap 1.5s] ... "only..."
            elif current_duration < min_fragment and combined_duration <= max_duration:
                # 限制 gap 不超过 3s，避免引入过长静音
                if gap < 3.0:
                    self.logger.debug(
                        f"碎片强制合并: fragment={current_duration:.2f}s, gap={gap:.2f}s"
                    )
                    should_merge = True

            if should_merge:
                current['end'] = next_seg['end']
            else:
                merged.append(current)
                current = next_seg.copy()

        merged.append(current)

        # 重新编号
        for i, seg in enumerate(merged):
            seg['index'] = i

        self.logger.info(
            f"Post-VAD 智能合并: 原始 {len(segments)} -> 合并后 {len(merged)} 段 "
            f"(max_gap={max_gap}s, max_dur={max_duration}s)"
        )
        return merged


# 便捷函数
def get_vad_service(logger: Optional[logging.Logger] = None) -> VADService:
    """
    获取 VAD 服务实例

    Args:
        logger: 日志记录器

    Returns:
        VADService 实例
    """
    return VADService(logger=logger)
