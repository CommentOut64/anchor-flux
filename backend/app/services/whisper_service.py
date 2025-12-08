"""
Faster-Whisper 转录服务

职责：
- Whisper 补刀（后处理增强阶段）
- 仅提供文本，时间戳由 SenseVoice 确定，使用伪对齐
"""
from faster_whisper import WhisperModel
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import logging
import gc

from app.core import config

logger = logging.getLogger(__name__)


class WhisperService:
    """Faster-Whisper 转录服务"""

    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self._model_name: str = ""
        self._device: str = "cuda"
        self._compute_type: str = "float16"

    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    @property
    def model_name(self) -> str:
        """获取当前加载的模型名称"""
        return self._model_name

    @property
    def device(self) -> str:
        """获取当前设备"""
        return self._device

    @property
    def compute_type(self) -> str:
        """获取当前计算类型"""
        return self._compute_type

    def load_model(
        self,
        model_name: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        download_root: str = None,
        local_files_only: bool = False
    ) -> "WhisperService":
        """
        加载 Faster-Whisper 模型

        Args:
            model_name: 模型名称 (tiny, base, small, medium, large-v2, large-v3)
            device: 设备 (cuda, cpu)
            compute_type: 计算类型 (float16, int8, int8_float16)
            download_root: 模型下载目录
            local_files_only: 是否仅使用本地文件

        Returns:
            self: 支持链式调用
        """
        # 如果已加载相同模型，跳过
        if self.model and self._model_name == model_name:
            logger.debug(f"模型 {model_name} 已加载，跳过")
            return self

        # 卸载旧模型
        self.unload_model()

        # 设置下载目录
        if download_root is None:
            download_root = str(config.HF_CACHE_DIR)

        logger.info(f"加载 Faster-Whisper 模型: {model_name} (device={device}, compute_type={compute_type})")

        try:
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
                local_files_only=local_files_only
            )

            self._model_name = model_name
            self._device = device
            self._compute_type = compute_type

            logger.info(f"Faster-Whisper 模型加载完成: {model_name}")

        except Exception as e:
            logger.error(f"加载 Faster-Whisper 模型失败: {e}")
            raise

        return self

    def unload_model(self):
        """卸载模型释放显存"""
        if self.model:
            logger.info(f"卸载 Faster-Whisper 模型: {self._model_name}")
            del self.model
            self.model = None
            self._model_name = ""

            # 清理内存
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: str = None,
        initial_prompt: str = None,
        word_timestamps: bool = False,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: dict = None,
        temperature: float = 0.0,
        condition_on_previous_text: bool = True
    ) -> Dict[str, Any]:
        """
        转录音频

        Args:
            audio: 音频文件路径或 numpy 数组 (16kHz, mono)
            language: 语言代码 (zh, en, ja 等)，None 表示自动检测
            initial_prompt: 上下文提示（提高准确性）
            word_timestamps: 是否生成词级时间戳
            beam_size: beam search 大小
            vad_filter: 是否启用内置 VAD 过滤
            vad_parameters: VAD 参数
            temperature: 采样温度
            condition_on_previous_text: 是否基于前文条件生成

        Returns:
            dict: {
                "text": str,              # 完整文本
                "segments": [...],        # 分段结果
                "language": str,          # 检测到的语言
                "language_probability": float  # 语言检测置信度
            }
        """
        if not self.model:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 执行转录
        segments_generator, info = self.model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text
        )

        # 转换生成器为列表
        segment_list = list(segments_generator)

        # 构建统一格式的返回结果
        result = {
            "text": " ".join(seg.text.strip() for seg in segment_list),
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "avg_logprob": seg.avg_logprob,
                    "no_speech_prob": seg.no_speech_prob,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability
                        }
                        for w in (seg.words or [])
                    ] if word_timestamps and seg.words else []
                }
                for seg in segment_list
            ],
            "language": info.language,
            "language_probability": info.language_probability
        }

        return result

    def transcribe_segment(
        self,
        audio: Union[str, np.ndarray],
        start_time: float,
        end_time: float,
        language: str = None,
        initial_prompt: str = None
    ) -> Dict[str, Any]:
        """
        转录指定时间段的音频（用于补刀场景）

        Args:
            audio: 完整音频数组 (16kHz)
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            language: 语言代码
            initial_prompt: 上下文提示

        Returns:
            dict: 转录结果
        """
        if isinstance(audio, np.ndarray):
            # 切片音频
            sr = 16000
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = audio[start_sample:end_sample]
        else:
            # 如果是文件路径，需要先加载再切片
            import librosa
            full_audio, _ = librosa.load(audio, sr=16000, mono=True)
            sr = 16000
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = full_audio[start_sample:end_sample]

        return self.transcribe(
            audio=audio_segment,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=False,  # 补刀场景使用伪对齐，不需要词级时间戳
            beam_size=5,
            vad_filter=False  # 已经是切片，不需要 VAD
        )

    def warmup(self):
        """预热模型（空跑一次确保完全加载到显存）"""
        if not self.model:
            logger.warning("模型未加载，无法预热")
            return

        logger.debug("开始 Faster-Whisper 模型预热")

        # 创建 1 秒静音音频
        dummy_audio = np.zeros(16000, dtype=np.float32)

        try:
            segments, _ = self.model.transcribe(dummy_audio)
            _ = list(segments)  # 触发生成器执行
            logger.debug("Faster-Whisper 模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")

    def estimate_confidence(self, result: Dict[str, Any]) -> float:
        """
        估算转录结果的置信度

        Args:
            result: transcribe() 返回的结果

        Returns:
            float: 0-1 之间的置信度分数
        """
        segments = result.get("segments", [])
        if not segments:
            return 0.7  # 默认置信度

        # 基于 avg_logprob 和 no_speech_prob 计算
        total_logprob = sum(s.get("avg_logprob", -0.5) for s in segments)
        avg_logprob = total_logprob / len(segments)

        avg_no_speech = sum(s.get("no_speech_prob", 0.1) for s in segments) / len(segments)

        # 转换为 0-1 置信度
        # logprob 范围大约 -1 到 0，越接近 0 越好
        confidence = min(1.0, max(0.0, 1.0 + avg_logprob))
        # no_speech_prob 越低越好
        confidence *= (1.0 - avg_no_speech)

        return round(confidence, 3)


# ========== 音频加载工具函数 ==========

def load_audio(audio_path: str, sr: int = 16000) -> np.ndarray:
    """
    加载音频文件为 numpy 数组

    Args:
        audio_path: 音频文件路径
        sr: 采样率（默认 16000）

    Returns:
        np.ndarray: 音频数组
    """
    import librosa
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    return audio.astype(np.float32)


# ========== 单例访问 ==========

_whisper_service_instance: Optional[WhisperService] = None


def get_whisper_service() -> WhisperService:
    """获取 Whisper 服务单例"""
    global _whisper_service_instance
    if _whisper_service_instance is None:
        _whisper_service_instance = WhisperService()
    return _whisper_service_instance


def reset_whisper_service():
    """重置 Whisper 服务（用于测试）"""
    global _whisper_service_instance
    if _whisper_service_instance:
        _whisper_service_instance.unload_model()
    _whisper_service_instance = None
