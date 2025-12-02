"""
Demucs 人声分离服务
使用 Hybrid Transformer Demucs (htdemucs) 模型进行高质量人声提取
"""

import os
import gc
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass

import torch
import numpy as np
import soundfile as sf


class BGMLevel(Enum):
    """背景音乐强度级别"""
    NONE = "none"      # 无背景音乐
    LIGHT = "light"    # 轻微背景音乐（按需分离）
    HEAVY = "heavy"    # 强背景音乐（全局分离）


@dataclass
class DemucsConfig:
    """Demucs配置"""
    model_name: str = "htdemucs"          # 模型名称
    device: str = "cuda"                   # 设备 (cuda/cpu)
    shifts: int = 1                        # 增强次数（1=快速，5=高质量）
    overlap: float = 0.25                  # 分段重叠率
    segment_length: int = 10               # 每段处理长度（秒）

    # 按需分离的缓冲区
    segment_buffer_sec: float = 2.0        # 分离时前后各加的缓冲（秒）

    # BGM检测参数（分位数采样策略）
    bgm_sample_duration: float = 10.0      # 每个采样片段的长度（秒）
    bgm_light_threshold: float = 0.2       # 轻微BGM阈值（BGM能量占比）
    bgm_heavy_threshold: float = 0.6       # 强BGM阈值（只要有一处超过此值即为Heavy）


class DemucsService:
    """
    Demucs人声分离服务

    支持三种使用模式：
    1. 全局分离：处理整个音频文件，返回纯人声
    2. 按需分离：只处理指定的时间段
    3. BGM检测：快速检测背景音乐强度
    """

    _instance = None
    _model = None
    _model_lock = None

    def __new__(cls):
        if cls._instance is None:
            import threading
            cls._instance = super().__new__(cls)
            cls._model_lock = threading.Lock()
        return cls._instance

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = DemucsConfig()
        self._cache_dir = Path("models/demucs")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self, device: str = None):
        """
        懒加载Demucs模型

        模型首次加载时会自动下载（~80MB）
        """
        if device:
            self.config.device = device

        with self._model_lock:
            if self._model is not None:
                return self._model

            self.logger.info(f"加载Demucs模型: {self.config.model_name}")

            try:
                from demucs.pretrained import get_model
                from demucs.apply import apply_model

                # 加载预训练模型
                self._model = get_model(self.config.model_name)

                # 移动到指定设备
                if self.config.device == "cuda" and torch.cuda.is_available():
                    self._model.cuda()
                    self.logger.info("Demucs模型已加载到GPU")
                else:
                    self._model.cpu()
                    self.config.device = "cpu"
                    self.logger.info("Demucs模型已加载到CPU")

                self._model.eval()
                return self._model

            except ImportError:
                raise RuntimeError(
                    "Demucs未安装，请运行: pip install demucs"
                )

    def unload_model(self):
        """卸载模型释放显存"""
        with self._model_lock:
            if self._model is not None:
                del self._model
                self._model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.info("Demucs模型已卸载")

    def separate_vocals(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        全局人声分离（处理整个音频文件）

        Args:
            audio_path: 输入音频路径
            output_path: 输出路径（可选，默认在同目录生成 xxx_vocals.wav）
            progress_callback: 进度回调 callback(progress: float, message: str)

        Returns:
            str: 分离后的人声文件路径
        """
        from demucs.apply import apply_model
        from demucs.audio import AudioFile, save_audio

        self.logger.info(f"开始全局人声分离: {audio_path}")

        # 生成输出路径
        if output_path is None:
            audio_dir = Path(audio_path).parent
            audio_stem = Path(audio_path).stem
            output_path = str(audio_dir / f"{audio_stem}_vocals.wav")

        # 检查缓存
        cache_key = self._get_cache_key(audio_path, "full")
        cached_path = self._cache_dir / f"{cache_key}_vocals.wav"
        if cached_path.exists():
            self.logger.info(f"使用缓存的分离结果: {cached_path}")
            return str(cached_path)

        model = self._load_model()

        if progress_callback:
            progress_callback(0.1, "加载音频...")

        # 加载音频
        wav = AudioFile(audio_path).read(
            streams=0,
            samplerate=model.samplerate,
            channels=model.audio_channels
        )

        # 添加batch维度
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = wav.unsqueeze(0)  # (1, channels, samples)

        if self.config.device == "cuda":
            wav = wav.cuda()

        if progress_callback:
            progress_callback(0.2, "分离人声中...")

        # 执行分离
        with torch.no_grad():
            sources = apply_model(
                model,
                wav,
                shifts=self.config.shifts,
                overlap=self.config.overlap,
                progress=True,
                device=self.config.device
            )

        # 提取人声（htdemucs输出顺序：drums, bass, other, vocals）
        source_names = model.sources
        vocals_idx = source_names.index('vocals')
        vocals = sources[0, vocals_idx]  # (channels, samples)

        # 恢复原始scale
        vocals = vocals * ref.std() + ref.mean()

        if progress_callback:
            progress_callback(0.9, "保存文件...")

        # 保存人声
        vocals = vocals.cpu().numpy()
        sf.write(output_path, vocals.T, model.samplerate)

        # 保存到缓存
        sf.write(str(cached_path), vocals.T, model.samplerate)

        if progress_callback:
            progress_callback(1.0, "人声分离完成")

        self.logger.info(f"人声分离完成: {output_path}")
        return output_path

    def separate_vocals_segment(
        self,
        audio_array: np.ndarray,
        sr: int,
        start_sec: float,
        end_sec: float,
        buffer_sec: float = None
    ) -> np.ndarray:
        """
        按需分离指定时间段的人声（内存模式）

        Args:
            audio_array: 完整音频数组 (samples,) 或 (channels, samples)
            sr: 采样率
            start_sec: 开始时间（秒）
            end_sec: 结束时间（秒）
            buffer_sec: 前后缓冲区（秒），默认使用配置值

        Returns:
            np.ndarray: 分离后的人声片段（不含缓冲区）
        """
        from demucs.apply import apply_model

        if buffer_sec is None:
            buffer_sec = self.config.segment_buffer_sec

        model = self._load_model()

        # 计算采样点范围（含缓冲区）
        buffer_samples = int(buffer_sec * sr)
        start_sample = max(0, int(start_sec * sr) - buffer_samples)
        end_sample = min(len(audio_array), int(end_sec * sr) + buffer_samples)

        # 提取片段
        if audio_array.ndim == 1:
            segment = audio_array[start_sample:end_sample]
            segment = np.stack([segment, segment])  # 转为立体声
        else:
            segment = audio_array[:, start_sample:end_sample]

        # 重采样到模型要求的采样率（如果需要）
        if sr != model.samplerate:
            import librosa
            segment = librosa.resample(segment, orig_sr=sr, target_sr=model.samplerate)
            target_sr = model.samplerate
        else:
            target_sr = sr

        # 转为tensor
        wav = torch.from_numpy(segment).float()
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = wav.unsqueeze(0)

        if self.config.device == "cuda":
            wav = wav.cuda()

        # 执行分离
        with torch.no_grad():
            sources = apply_model(
                model,
                wav,
                shifts=1,  # 按需分离使用快速模式
                overlap=self.config.overlap,
                progress=False,
                device=self.config.device
            )

        # 提取人声
        source_names = model.sources
        vocals_idx = source_names.index('vocals')
        vocals = sources[0, vocals_idx]
        vocals = vocals * ref.std() + ref.mean()
        vocals = vocals.cpu().numpy()

        # 重采样回原始采样率
        if target_sr != sr:
            vocals = librosa.resample(vocals, orig_sr=target_sr, target_sr=sr)

        # 去除缓冲区，返回原始时间段
        original_start = int(buffer_sec * sr) if start_sec > buffer_sec else int(start_sec * sr)
        original_duration = int((end_sec - start_sec) * sr)
        vocals = vocals[:, original_start:original_start + original_duration]

        # 转为单声道（Whisper要求）
        if vocals.ndim > 1:
            vocals = vocals.mean(axis=0)

        return vocals

    def detect_background_music_level(
        self,
        audio_path: str,
        audio_array: Optional[np.ndarray] = None,
        sr: int = 16000,
        duration_sec: Optional[float] = None
    ) -> Tuple[BGMLevel, List[float]]:
        """
        快速检测背景音乐强度（分位数采样策略）

        采样策略：取音频时长的 15%、50%、85% 处各截取 10 秒
        - 15%：捕获 Intro 结束后的主歌背景音
        - 50%：捕获中间部分
        - 85%：捕获结尾前的部分

        Args:
            audio_path: 音频文件路径
            audio_array: 音频数组（可选，用于内存模式）
            sr: 采样率
            duration_sec: 音频总时长（可选，如果audio_array提供则自动计算）

        Returns:
            Tuple[BGMLevel, List[float]]: (背景音乐强度级别, 各采样点的BGM比例列表)
        """
        self.logger.info("检测背景音乐强度（分位数采样）...")

        # 加载音频
        if audio_array is None:
            import librosa
            audio_array, sr = librosa.load(audio_path, sr=sr)

        if duration_sec is None:
            duration_sec = len(audio_array) / sr

        # 分位数采样位置
        sample_positions = [0.15, 0.50, 0.85]
        sample_duration = self.config.bgm_sample_duration  # 默认10秒

        # 检查音频是否足够长
        if duration_sec < sample_duration * 2:
            self.logger.warning(f"音频太短({duration_sec:.1f}s)，无法可靠检测BGM")
            return BGMLevel.LIGHT, []  # 保守起见返回LIGHT

        ratios = []

        for pos in sample_positions:
            start_time = duration_sec * pos

            # 确保不超出边界
            if start_time + sample_duration > duration_sec:
                start_time = duration_sec - sample_duration
            if start_time < 0:
                start_time = 0

            try:
                # 分离这一段
                vocals = self.separate_vocals_segment(
                    audio_array, sr,
                    start_sec=start_time,
                    end_sec=start_time + sample_duration,
                    buffer_sec=0.5  # 检测时用较短缓冲
                )

                # 获取原始片段
                start_sample = int(start_time * sr)
                end_sample = int((start_time + sample_duration) * sr)
                original = audio_array[start_sample:end_sample]

                # 计算BGM能量比（使用改进的算法）
                bgm_ratio = self._calculate_bgm_ratio(original, vocals)
                ratios.append(bgm_ratio)

                self.logger.debug(
                    f"采样点 {pos*100:.0f}% ({start_time:.1f}s): BGM比例={bgm_ratio:.2f}"
                )

            except Exception as e:
                self.logger.warning(f"采样点 {pos*100:.0f}% 检测失败: {e}")
                continue

        if not ratios:
            return BGMLevel.LIGHT, []  # 默认假设有轻微BGM

        # 决策逻辑：使用最大值判断（只要有一处BGM很重，就视为Heavy）
        avg_ratio = np.mean(ratios)
        max_ratio = np.max(ratios)

        self.logger.info(
            f"BGM检测完成: 比例={ratios}, 平均={avg_ratio:.2f}, 最大={max_ratio:.2f}"
        )

        # 使用max_ratio作为主要判断依据
        if max_ratio > self.config.bgm_heavy_threshold:  # 默认0.6
            return BGMLevel.HEAVY, ratios
        elif max_ratio > self.config.bgm_light_threshold:  # 默认0.2
            return BGMLevel.LIGHT, ratios
        else:
            return BGMLevel.NONE, ratios

    def _calculate_bgm_ratio(
        self,
        original: np.ndarray,
        vocals: np.ndarray
    ) -> float:
        """
        计算 BGM 能量占比

        逻辑：(原音频能量 - 人声能量) / 原音频能量

        注意：Demucs分离出的vocals能量可能与原始不完全一致，
        这里使用RMS能量比作为近似估算。

        Args:
            original: 原始混合音频
            vocals: 分离后的人声

        Returns:
            float: BGM能量占比 (0.0-1.0)，越高表示BGM越强
        """
        # 确保长度一致
        min_len = min(len(original), len(vocals))
        original = original[:min_len]
        vocals = vocals[:min_len]

        # 计算均方根能量 (RMS)
        rms_orig = np.sqrt(np.mean(original ** 2))
        rms_voc = np.sqrt(np.mean(vocals ** 2))

        # 如果原音就很小（静音片段），返回0
        if rms_orig < 0.01:
            return 0.0

        # 计算非人声部分的能量占比（近似背景音）
        # 假设 Energy_Total ≈ Energy_Vocal + Energy_BGM
        # BGM_ratio ≈ 1 - (Vocal_RMS / Total_RMS)
        bgm_ratio = 1.0 - (rms_voc / (rms_orig + 1e-6))

        return max(0.0, min(1.0, bgm_ratio))  # 钳制在0-1范围

    def _get_cache_key(self, audio_path: str, mode: str) -> str:
        """生成缓存键"""
        path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:16]
        mtime = int(os.path.getmtime(audio_path))
        return f"{path_hash}_{mtime}_{mode}"


# 全局单例
_demucs_service: Optional[DemucsService] = None


def get_demucs_service() -> DemucsService:
    """获取Demucs服务单例"""
    global _demucs_service
    if _demucs_service is None:
        _demucs_service = DemucsService()
    return _demucs_service
