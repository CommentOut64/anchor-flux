"""
SenseVoice ONNX 推理服务

基于 funasr-onnx 库实现的 SenseVoice 模型推理服务
支持多语言语音识别、情感识别和事件检测

模型来源：
- ModelScope: https://www.modelscope.cn/models/iic/SenseVoiceSmall
- HuggingFace: https://huggingface.co/FunAudioLLM/SenseVoiceSmall

依赖安装：
pip install funasr-onnx
"""
import os
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union
import threading

logger = logging.getLogger(__name__)


class SenseVoiceONNXService:
    """SenseVoice ONNX 推理服务"""

    def __init__(self, config = None):
        """
        初始化 SenseVoice 服务

        Args:
            config: SenseVoice 配置
        """
        if config is None:
            from models.sensevoice_models import SenseVoiceConfig
            config = SenseVoiceConfig()
        self.config = config
        self.model = None
        self.is_loaded = False
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # 模型路径（支持 ModelScope 和本地路径）
        self.model_path = self._resolve_model_path()

    def _resolve_model_path(self) -> str:
        """
        解析模型路径

        优先级：
        1. 本地路径（如果存在）
        2. ModelScope 缓存路径
        3. 使用 ModelScope ID 自动下载

        Returns:
            模型路径
        """
        model_dir = self.config.model_dir

        # 检查是否为本地路径
        if os.path.exists(model_dir):
            self.logger.info(f"使用本地模型: {model_dir}")
            return model_dir

        # 检查 ModelScope 缓存
        home = Path.home()
        modelscope_cache = home / ".cache" / "modelscope" / "hub" / model_dir
        if modelscope_cache.exists():
            self.logger.info(f"使用 ModelScope 缓存: {modelscope_cache}")
            return str(modelscope_cache)

        # 使用 ModelScope ID（会自动下载）
        self.logger.info(f"将从 ModelScope 下载模型: {model_dir}")
        return model_dir

    def load_model(self):
        """
        加载 SenseVoice ONNX 模型

        使用 funasr-onnx 库加载模型
        """
        with self._lock:
            if self.is_loaded:
                self.logger.info("SenseVoice 模型已加载")
                return

            try:
                self.logger.info("开始加载 SenseVoice ONNX 模型...")
                self.logger.info(f"模型路径: {self.model_path}")
                self.logger.info(f"批处理大小: {self.config.batch_size}")
                self.logger.info(f"量化: {self.config.quantize}")

                # 导入 funasr-onnx
                try:
                    from funasr_onnx import SenseVoiceSmall
                except ImportError as e:
                    raise ImportError(
                        "funasr-onnx 未安装。请运行: pip install funasr-onnx\n"
                        "详见: https://github.com/modelscope/FunASR"
                    ) from e

                # 加载模型
                self.model = SenseVoiceSmall(
                    self.model_path,
                    batch_size=self.config.batch_size,
                    quantize=self.config.quantize
                )

                self.is_loaded = True
                self.logger.info("SenseVoice ONNX 模型加载成功")

            except Exception as e:
                self.logger.error(f"加载 SenseVoice 模型失败: {e}", exc_info=True)
                self.is_loaded = False
                raise

    def unload_model(self):
        """卸载模型，释放内存"""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
                self.is_loaded = False
                self.logger.info("SenseVoice 模型已卸载")

    def transcribe(
        self,
        audio_path: Union[str, List[str]],
        language: str = None,
        use_itn: bool = None,
        ban_emo_unk: bool = None
    ) -> List[Dict]:
        """
        转录音频文件

        Args:
            audio_path: 音频文件路径或路径列表
            language: 语言代码 (auto, zh, en, yue, ja, ko, nospeech)
            use_itn: 是否使用逆文本正则化
            ban_emo_unk: 是否禁用未知情感标签

        Returns:
            转录结果列表，每个结果包含：
            - text: 转录文本（带特殊标签）
            - text_clean: 清洗后的文本
            - language: 检测到的语言
            - emotion: 情感标签
            - event: 事件标签
            - confidence: 置信度（如果可用）
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 使用配置的默认值
        language = language or self.config.language
        use_itn = use_itn if use_itn is not None else self.config.use_itn
        ban_emo_unk = ban_emo_unk if ban_emo_unk is not None else self.config.ban_emo_unk

        # 确保输入为列表
        if isinstance(audio_path, str):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        try:
            self.logger.info(f"开始转录 {len(audio_paths)} 个音频文件")
            self.logger.debug(f"语言: {language}, ITN: {use_itn}")

            # 调用模型推理
            results = self.model(
                audio_paths,
                language=language,
                use_itn=use_itn,
                ban_emo_unk=ban_emo_unk
            )

            # 处理结果
            processed_results = []
            for i, raw_text in enumerate(results):
                # 提取标签信息
                from ..services.text_normalizer import get_text_normalizer
                normalizer = get_text_normalizer()

                # 提取标签和清洗文本
                process_result = normalizer.process(raw_text, extract_info=True)

                result = {
                    "text": raw_text,
                    "text_clean": process_result["text_clean"],
                    "language": process_result["tags"]["language"] if process_result["tags"] else None,
                    "emotion": process_result["tags"]["emotion"] if process_result["tags"] else None,
                    "event": process_result["tags"]["event"] if process_result["tags"] else None,
                    "confidence": 1.0,  # funasr-onnx 不提供置信度，默认为1.0
                    "audio_path": audio_paths[i]
                }

                processed_results.append(result)

            self.logger.info(f"转录完成，共 {len(processed_results)} 个结果")
            return processed_results

        except Exception as e:
            self.logger.error(f"转录失败: {e}", exc_info=True)
            raise

    def transcribe_audio_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: str = None,
        use_itn: bool = None
    ) -> Dict:
        """
        转录音频数组（内存中的音频数据）

        Args:
            audio_array: 音频数组 (numpy array)
            sample_rate: 采样率
            language: 语言代码
            use_itn: 是否使用逆文本正则化

        Returns:
            转录结果字典

        Note:
            funasr-onnx 主要支持文件路径输入
            如需处理音频数组，需要先保存为临时文件
        """
        import tempfile
        import soundfile as sf

        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # 保存音频到临时文件
            sf.write(tmp_path, audio_array, sample_rate)

            # 转录
            results = self.transcribe(tmp_path, language=language, use_itn=use_itn)

            return results[0] if results else {}

        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_model_info(self) -> Dict:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            "model_name": "SenseVoice-Small",
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "batch_size": self.config.batch_size,
            "quantize": self.config.quantize,
            "device": self.config.device,
            "supported_languages": ["zh", "en", "yue", "ja", "ko"],
            "features": [
                "多语言识别",
                "情感识别",
                "事件检测",
                "逆文本正则化"
            ]
        }


# ========== 单例模式 ==========

_sensevoice_service_instance: Optional[SenseVoiceONNXService] = None
_instance_lock = threading.Lock()


def get_sensevoice_service(config = None) -> SenseVoiceONNXService:
    """
    获取 SenseVoice 服务单例

    Args:
        config: SenseVoice 配置（仅在首次创建时使用）

    Returns:
        SenseVoiceONNXService 实例
    """
    global _sensevoice_service_instance

    if _sensevoice_service_instance is None:
        with _instance_lock:
            if _sensevoice_service_instance is None:
                _sensevoice_service_instance = SenseVoiceONNXService(config)
                logger.info("SenseVoice 服务单例已创建")

    return _sensevoice_service_instance


def reset_sensevoice_service():
    """重置 SenseVoice 服务单例（用于测试）"""
    global _sensevoice_service_instance

    with _instance_lock:
        if _sensevoice_service_instance is not None:
            _sensevoice_service_instance.unload_model()
            _sensevoice_service_instance = None
            logger.info("SenseVoice 服务单例已重置")
