"""
统一模型与数据集管理服务 - 改进版
- 下载管理（支持进度追踪）
- 完整性验证
- 下载队列管理（一次只下载一个）
- 缓存管理
- 自动检测语言并下载
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Callable
from pathlib import Path
import threading
import logging
import os
import shutil
import time

from models.model_models import ModelInfo, AlignModelInfo
from core.config import config
from services.model_validator import ModelValidator


class ModelManagerService:
    """
    模型管理服务
    统一管理Whisper模型和对齐模型的下载、缓存、删除
    """

    # 支持的Whisper模型
    WHISPER_MODELS = {
        "tiny": {"size_mb": 75, "desc": "最快，精度较低"},
        "base": {"size_mb": 145, "desc": "快速，精度一般"},
        "small": {"size_mb": 490, "desc": "平衡速度与精度"},
        "medium": {"size_mb": 1500, "desc": "较慢，精度较高"},
        "large-v2": {"size_mb": 3100, "desc": "最慢，精度最高"},
        "large-v3": {"size_mb": 3100, "desc": "最新版本，精度最高"},
    }

    # 支持的语言（对齐模型）
    SUPPORTED_LANGUAGES = {
        "zh": "中文 (Chinese)",
        "en": "英语 (English)",
        "ja": "日语 (Japanese)",
        "ko": "韩语 (Korean)",
        "es": "西班牙语 (Spanish)",
        "fr": "法语 (French)",
        "de": "德语 (German)",
        "ru": "俄语 (Russian)",
        "pt": "葡萄牙语 (Portuguese)",
        "it": "意大利语 (Italian)",
        "ar": "阿拉伯语 (Arabic)",
        "hi": "印地语 (Hindi)",
    }

    # Whisper模型推荐的对齐模型（默认为中文）
    WHISPER_RECOMMENDED_ALIGN_MODELS = {
        "tiny": "zh",
        "base": "zh",
        "small": "zh",
        "medium": "zh",
        "large-v2": "zh",
        "large-v3": "zh",
    }

    def __init__(self, models_dir: Path = None):
        """
        初始化模型管理服务

        Args:
            models_dir: 模型目录路径，默认使用config中的配置
        """
        self.models_dir = models_dir or config.MODELS_DIR
        self.logger = logging.getLogger(__name__)

        # 模型状态跟踪
        self.whisper_models: Dict[str, ModelInfo] = {}
        self.align_models: Dict[str, AlignModelInfo] = {}

        # 下载队列和锁 - 确保一次只下载一个模型（改进版）
        self.download_lock = threading.Lock()
        # 跟踪正在下载的模型（使用字典而不是简单布尔值）
        self.downloading_models: Dict[str, bool] = {}  # key: "whisper/model_id" 或 "align/language"

        # 进度回调函数列表（用于 SSE 推送）
        self.progress_callbacks: List[Callable] = []

        # 初始化模型信息
        self._init_model_info()

        # 启动后台验证任务
        threading.Thread(target=self._background_validate_models, daemon=True).start()

    def _init_model_info(self):
        """扫描本地已有模型并验证完整性"""
        self.logger.info("🔍 扫描本地已有模型...")

        # 初始化Whisper模型信息
        for model_id, info in self.WHISPER_MODELS.items():
            status, local_path, validation_msg = self._check_whisper_model_exists(model_id)
            
            self.whisper_models[model_id] = ModelInfo(
                model_id=model_id,
                size_mb=info["size_mb"],
                status=status,
                download_progress=100.0 if status == "ready" else 0.0,
                local_path=str(local_path) if local_path else None,
                description=info["desc"]
            )
            
            if status == "ready":
                self.logger.info(f"✅ 发现完整的Whisper模型: {model_id}")
            elif status == "incomplete":
                self.logger.warning(f"⚠️ Whisper模型不完整: {model_id}\n{validation_msg}")

        # 初始化对齐模型信息
        for lang, name in self.SUPPORTED_LANGUAGES.items():
            status, local_path, validation_msg = self._check_align_model_exists(lang)
            
            self.align_models[lang] = AlignModelInfo(
                language=lang,
                language_name=name,
                status=status,
                download_progress=100.0 if status == "ready" else 0.0,
                local_path=str(local_path) if local_path else None
            )
            
            if status == "ready":
                self.logger.info(f"✅ 发现完整的对齐模型: {lang} ({name})")
            elif status == "incomplete":
                self.logger.warning(f"⚠️ 对齐模型不完整: {lang}\n{validation_msg}")

    def _check_whisper_model_exists(self, model_id: str) -> tuple[str, Optional[Path], str]:
        """
        检查Whisper模型是否存在并验证完整性

        Args:
            model_id: 模型ID

        Returns:
            tuple: (状态, 本地路径, 验证信息)
            状态可以是: "ready"(完整), "incomplete"(不完整), "not_downloaded"(不存在)
        """
        # WhisperX模型缓存在HuggingFace缓存目录中
        hf_cache = config.HF_CACHE_DIR / "hub"

        # 检查可能的模型缓存路径
        possible_paths = [
            hf_cache / f"models--Systran--faster-whisper-{model_id}",
            hf_cache / f"models--guillaumekln--faster-whisper-{model_id}",
        ]

        self.logger.debug(f"🔍 查找模型 {model_id}，候选路径: {[str(p) for p in possible_paths]}")

        for model_dir in possible_paths:
            self.logger.debug(f"  检查路径: {model_dir}")
            if not model_dir.exists():
                self.logger.debug(f"    ✗ 路径不存在")
                continue

            # 查找快照目录
            snapshots = ModelValidator.find_model_snapshots(hf_cache, model_dir.name)
            self.logger.debug(f"    找到 {len(snapshots)} 个快照")
            if not snapshots:
                continue

            # 检查最新的快照
            latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
            self.logger.debug(f"    最新快照: {latest_snapshot}")

            # 验证完整性
            is_complete, missing_files, detail = ModelValidator.validate_whisper_model(latest_snapshot)

            if is_complete:
                self.logger.debug(f"    ✓ 验证成功")
                return ("ready", latest_snapshot, detail)
            else:
                self.logger.debug(f"    ✗ 验证失败: {missing_files}")
                return ("incomplete", latest_snapshot, f"缺失文件: {', '.join(missing_files)}\n{detail}")

        self.logger.debug(f"  未找到任何有效的模型路径")
        return ("not_downloaded", None, "模型未下载")

    def _check_align_model_exists(self, language: str) -> tuple[str, Optional[Path], str]:
        """
        检查对齐模型是否存在并验证完整性

        Args:
            language: 语言代码

        Returns:
            tuple: (状态, 本地路径, 验证信息)
        """
        # 对齐模型也缓存在HuggingFace目录中
        hf_cache = config.HF_CACHE_DIR / "hub"

        # 不同语言的模型名称可能不同
        model_patterns = [
            f"models--jonatasgrosman--wav2vec2-large-xlsr-53-{language}",
            f"models--facebook--wav2vec2-large-xlsr-53-{language}",
        ]

        for pattern in model_patterns:
            model_dir = hf_cache / pattern
            if not model_dir.exists():
                continue
            
            # 查找快照目录
            snapshots = ModelValidator.find_model_snapshots(hf_cache, pattern)
            if not snapshots:
                continue
            
            # 检查最新的快照
            latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
            
            # 验证完整性
            is_complete, missing_files, detail = ModelValidator.validate_align_model(latest_snapshot)
            
            if is_complete:
                return ("ready", latest_snapshot, detail)
            else:
                return ("incomplete", latest_snapshot, f"缺失文件: {', '.join(missing_files)}\n{detail}")

        return ("not_downloaded", None, "模型未下载")

    def list_whisper_models(self) -> List[ModelInfo]:
        """列出所有Whisper模型状态"""
        return list(self.whisper_models.values())

    def list_align_models(self) -> List[AlignModelInfo]:
        """列出所有对齐模型状态"""
        return list(self.align_models.values())
    
    def register_progress_callback(self, callback: Callable):
        """注册进度回调函数（用于SSE推送）"""
        if callback not in self.progress_callbacks:
            self.progress_callbacks.append(callback)
    
    def unregister_progress_callback(self, callback: Callable):
        """取消注册进度回调函数"""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    def _notify_progress(self, model_type: str, model_id: str, progress: float, status: str, message: str = ""):
        """通知所有注册的回调函数"""
        for callback in self.progress_callbacks:
            try:
                callback(model_type, model_id, progress, status, message)
            except Exception as e:
                self.logger.error(f"进度回调失败: {e}")
    
    def _background_validate_models(self):
        """后台异步验证所有模型完整性"""
        time.sleep(10)  # 启动后延迟10秒再验证
        
        self.logger.info("🔍 开始后台验证模型完整性...")
        
        # 验证 Whisper 模型
        for model_id, model in self.whisper_models.items():
            if model.status == "ready":
                status, local_path, detail = self._check_whisper_model_exists(model_id)
                if status != "ready":
                    self.logger.warning(f"⚠️ 后台验证发现模型不完整: {model_id}")
                    model.status = "incomplete"
                    self._notify_progress("whisper", model_id, 0, "incomplete", "模型文件不完整，请重新下载")
        
        # 验证对齐模型
        for lang, model in self.align_models.items():
            if model.status == "ready":
                status, local_path, detail = self._check_align_model_exists(lang)
                if status != "ready":
                    self.logger.warning(f"⚠️ 后台验证发现对齐模型不完整: {lang}")
                    model.status = "incomplete"
                    self._notify_progress("align", lang, 0, "incomplete", "模型文件不完整，请重新下载")
        
        self.logger.info("✅ 后台模型验证完成")

    def download_whisper_model(self, model_id: str) -> bool:
        """
        下载Whisper模型（支持队列管理 + 双重检查锁定）

        Args:
            model_id: 模型ID

        Returns:
            bool: 是否成功加入下载队列
        """
        if model_id not in self.whisper_models:
            self.logger.warning(f"❌ 不支持的模型: {model_id}")
            return False

        model = self.whisper_models[model_id]
        model_key = f"whisper/{model_id}"

        # 第一次检查（快速失败，无锁）
        if model_key in self.downloading_models and self.downloading_models[model_key]:
            self.logger.warning(f"⏳ 模型正在下载中: {model_id}")
            self._notify_progress("whisper", model_id, 0, "waiting", f"模型正在下载中，请等待")
            return False

        # 检查当前模型状态
        if model.status == "downloading":
            self.logger.info(f"⏳ 模型正在下载中: {model_id}")
            return False

        # 检查模型是否存在且完整
        status, local_path, detail = self._check_whisper_model_exists(model_id)
        if status == "ready":
            self.logger.info(f"✅ 模型已存在且完整: {model_id}")
            model.status = "ready"
            model.download_progress = 100.0
            if local_path:
                model.local_path = str(local_path)
            self._notify_progress("whisper", model_id, 100, "ready", "模型已就绪")
            return True  # 不需要下载

        # 双重检查锁定（确保原子性）
        with self.download_lock:
            # 第二次检查（锁内，确保原子性）
            if model_key in self.downloading_models and self.downloading_models[model_key]:
                self.logger.warning(f"⏳ 模型正在下载中（锁内检查）: {model_id}")
                return False

            # 标记为下载中
            self.downloading_models[model_key] = True

        # 如果模型不完整，清理旧文件
        if status == "incomplete" and local_path:
            self.logger.warning(f"🗑️ 清理不完整的模型文件: {model_id}")
            try:
                # 删除整个模型目录
                model_parent = local_path.parent.parent.parent
                if model_parent.exists():
                    shutil.rmtree(model_parent)
                    self.logger.info(f"✅ 已清理: {model_parent}")
            except Exception as e:
                self.logger.error(f"清理失败: {e}")

        model.status = "downloading"
        model.download_progress = 0.0

        self._notify_progress("whisper", model_id, 0, "downloading", "开始下载...")

        # 启动下载线程
        threading.Thread(
            target=self._download_whisper_model_task,
            args=(model_id,),
            daemon=True,
            name=f"DownloadWhisper-{model_id}"
        ).start()

        self.logger.info(f"🚀 开始下载Whisper模型: {model_id}")
        return True

    def download_align_model(self, language: str) -> bool:
        """
        下载对齐模型（支持并发控制 + 双重检查锁定）

        Args:
            language: 语言代码

        Returns:
            bool: 是否成功加入下载队列
        """
        if language not in self.align_models:
            self.logger.warning(f"❌ 不支持的语言: {language}")
            return False

        model = self.align_models[language]
        model_key = f"align/{language}"

        # 第一次检查（快速失败，无锁）
        if model_key in self.downloading_models and self.downloading_models[model_key]:
            self.logger.warning(f"⏳ 对齐模型正在下载中: {language}")
            return False

        if model.status == "downloading":
            self.logger.info(f"⏳ 对齐模型正在下载中: {language}")
            return False

        # 双重检查锁定（确保原子性）
        with self.download_lock:
            # 第二次检查（锁内，确保原子性）
            if model_key in self.downloading_models and self.downloading_models[model_key]:
                self.logger.warning(f"⏳ 对齐模型正在下载中（锁内检查）: {language}")
                return False

            # 标记为下载中
            self.downloading_models[model_key] = True

        # 标记为下载中
        model.status = "downloading"
        model.download_progress = 0.0

        self._notify_progress("align", language, 0, "downloading", "开始下载...")

        # 启动下载线程
        threading.Thread(
            target=self._download_align_model_task,
            args=(language,),
            daemon=True,
            name=f"DownloadAlign-{language}"
        ).start()

        self.logger.info(f"🚀 开始下载对齐模型: {language}")
        return True

    def auto_download_for_language(self, language: str) -> bool:
        """
        自动下载指定语言所需的对齐模型
        用于断点续传恢复时自动补齐模型

        Args:
            language: 语言代码

        Returns:
            bool: 是否需要下载（True）或已存在（False）
        """
        if language not in self.align_models:
            self.logger.warning(f"⚠️ 不支持的语言: {language}")
            return False

        model = self.align_models[language]

        if model.status == "ready":
            self.logger.info(f"✅ 对齐模型已存在: {language}")
            return False

        self.logger.info(f"🔍 检测到新语言 {language}，开始自动下载对齐模型")
        return self.download_align_model(language)

    def delete_whisper_model(self, model_id: str) -> bool:
        """
        删除Whisper模型

        Args:
            model_id: 模型ID

        Returns:
            bool: 是否删除成功
        """
        if model_id not in self.whisper_models:
            return False

        model = self.whisper_models[model_id]

        if model.status != "ready" or not model.local_path:
            self.logger.warning(f"⚠️ 模型未下载或路径不存在: {model_id}")
            return False

        try:
            # 删除模型目录
            local_path = Path(model.local_path)
            if local_path.exists():
                shutil.rmtree(local_path)
                self.logger.info(f"🗑️ 已删除Whisper模型: {model_id}")

            # 更新状态
            model.status = "not_downloaded"
            model.download_progress = 0.0
            model.local_path = None

            return True

        except Exception as e:
            self.logger.error(f"❌ 删除模型失败: {model_id} - {e}")
            return False

    def delete_align_model(self, language: str) -> bool:
        """
        删除对齐模型

        Args:
            language: 语言代码

        Returns:
            bool: 是否删除成功
        """
        if language not in self.align_models:
            return False

        model = self.align_models[language]

        if model.status != "ready" or not model.local_path:
            self.logger.warning(f"⚠️ 对齐模型未下载或路径不存在: {language}")
            return False

        try:
            # 删除模型目录
            local_path = Path(model.local_path)
            if local_path.exists():
                shutil.rmtree(local_path)
                self.logger.info(f"🗑️ 已删除对齐模型: {language}")

            # 更新状态
            model.status = "not_downloaded"
            model.download_progress = 0.0
            model.local_path = None

            return True

        except Exception as e:
            self.logger.error(f"❌ 删除对齐模型失败: {language} - {e}")
            return False

    def get_download_progress(self) -> Dict:
        """获取所有下载进度"""
        return {
            "whisper": {
                mid: {
                    "status": m.status,
                    "progress": m.download_progress
                }
                for mid, m in self.whisper_models.items()
            },
            "align": {
                lang: {
                    "status": m.status,
                    "progress": m.download_progress
                }
                for lang, m in self.align_models.items()
            }
        }

    def is_model_downloading(self, model_type: str, model_id: str) -> bool:
        """
        检查指定模型是否正在下载

        Args:
            model_type: "whisper" 或 "align"
            model_id: 模型ID或语言代码

        Returns:
            bool: 是否正在下载
        """
        model_key = f"{model_type}/{model_id}"
        with self.download_lock:
            return self.downloading_models.get(model_key, False)

    def wait_for_download_complete(
        self,
        model_type: str,
        model_id: str,
        timeout: int = 600,
        check_interval: float = 2.0
    ) -> bool:
        """
        等待模型下载完成（带超时）

        Args:
            model_type: 模型类型 ("whisper" 或 "align")
            model_id: 模型ID或语言代码
            timeout: 超时时间（秒）
            check_interval: 检查间隔（秒）

        Returns:
            bool: 是否成功完成（True）或超时/失败（False）
        """
        start_time = time.time()
        model_key = f"{model_type}/{model_id}"

        self.logger.info(f"⏳ 等待模型下载完成: {model_key} (超时: {timeout}秒)")

        while time.time() - start_time < timeout:
            # 检查下载状态
            with self.download_lock:
                if model_key not in self.downloading_models or \
                   not self.downloading_models[model_key]:
                    # 下载已结束，检查结果
                    if model_type == "whisper":
                        model = self.whisper_models.get(model_id)
                    else:
                        model = self.align_models.get(model_id)

                    if model and model.status == "ready":
                        self.logger.info(f"✅ 模型下载完成: {model_key}")
                        return True
                    elif model and model.status == "error":
                        self.logger.error(f"❌ 模型下载失败: {model_key}")
                        return False

            # 等待一段时间后重试
            time.sleep(check_interval)

        self.logger.warning(f"⏰ 等待模型下载超时: {model_key}")
        return False

    def _download_whisper_model_task(self, model_id: str):
        """下载Whisper模型任务（后台线程）- 改进版"""
        model = None
        try:
            model = self.whisper_models[model_id]
            self.logger.info(f"📥 正在下载Whisper模型: {model_id}")
            self.logger.info(f"📁 下载目录: {config.HF_CACHE_DIR}")
            
            # 更新进度: 准备下载
            self._notify_progress("whisper", model_id, 5, "downloading", "准备下载...")
            model.download_progress = 5.0
            
            # 策略: 优先镜像站，失败后尝试官方源
            use_mirror = os.getenv('USE_HF_MIRROR', 'false').lower() == 'true'
            download_success = False
            last_error = None
            local_dir = None  # 初始化下载路径变量
            
            # 方式1: 使用 huggingface_hub 直接下载（更可控）
            if not download_success:
                try:
                    self.logger.info(f"🔄 方式1: 使用 huggingface_hub 下载...")
                    self._notify_progress("whisper", model_id, 10, "downloading", "连接下载源...")
                    
                    from huggingface_hub import snapshot_download
                    
                    repo_id = f"Systran/faster-whisper-{model_id}"
                    cache_dir = str(config.HF_CACHE_DIR)
                    
                    if use_mirror:
                        self.logger.info(f"📦 从镜像站下载: {config.HF_ENDPOINT}")
                    else:
                        self.logger.info(f"📦 从官方源下载: {repo_id}")
                    
                    self._notify_progress("whisper", model_id, 20, "downloading", "正在下载模型文件...")
                    model.download_progress = 20.0
                    
                    local_dir = snapshot_download(
                        repo_id=repo_id,
                        cache_dir=cache_dir,
                        local_files_only=False,
                    )
                    
                    self.logger.info(f"✅ 方式1成功下载到: {local_dir}")
                    self._notify_progress("whisper", model_id, 80, "downloading", "验证模型文件...")
                    model.download_progress = 80.0
                    download_success = True
                    
                except Exception as e1:
                    last_error = e1
                    self.logger.warning(f"⚠️ 方式1失败: {e1}")
                    self._notify_progress("whisper", model_id, 10, "downloading", f"方式1失败，尝试其他方式...")
            
            # 方式2: 如果方式1失败且使用了镜像，尝试切换到官方源
            if not download_success and use_mirror:
                try:
                    self.logger.info(f"🔄 方式2: 切换到官方源重试...")
                    self._notify_progress("whisper", model_id, 15, "downloading", "切换到官方源...")
                    
                    # 临时切换到官方源
                    old_endpoint = os.environ.get('HF_ENDPOINT')
                    if 'HF_ENDPOINT' in os.environ:
                        del os.environ['HF_ENDPOINT']
                    
                    try:
                        from huggingface_hub import snapshot_download
                        
                        repo_id = f"Systran/faster-whisper-{model_id}"
                        cache_dir = str(config.HF_CACHE_DIR)
                        
                        self.logger.info(f"📦 从官方源下载: https://huggingface.co")
                        self._notify_progress("whisper", model_id, 25, "downloading", "正在从官方源下载...")
                        model.download_progress = 25.0
                        
                        local_dir = snapshot_download(
                            repo_id=repo_id,
                            cache_dir=cache_dir,
                            local_files_only=False,
                        )
                        
                        self.logger.info(f"✅ 方式2成功")
                        self._notify_progress("whisper", model_id, 80, "downloading", "验证模型文件...")
                        model.download_progress = 80.0
                        download_success = True
                        
                    finally:
                        # 恢复镜像源设置
                        if old_endpoint:
                            os.environ['HF_ENDPOINT'] = old_endpoint
                    
                except Exception as e2:
                    last_error = e2
                    self.logger.error(f"❌ 方式2也失败: {e2}")
                    self._notify_progress("whisper", model_id, 15, "downloading", "方式2失败，尝试最后方式...")
            
            # 方式3: 使用 whisperx 加载（会触发下载）
            if not download_success:
                try:
                    self.logger.info(f"🔄 方式3: 使用 whisperx 加载模型...")
                    self._notify_progress("whisper", model_id, 30, "downloading", "使用备用方式下载...")
                    model.download_progress = 30.0
                    
                    import whisperx
                    _ = whisperx.load_model(
                        model_id,
                        device="cpu",
                        compute_type="int8",
                        download_root=str(config.HF_CACHE_DIR)
                    )
                    
                    self.logger.info(f"✅ 方式3成功")
                    self._notify_progress("whisper", model_id, 85, "downloading", "验证模型文件...")
                    model.download_progress = 85.0
                    download_success = True
                    
                except Exception as e3:
                    last_error = e3
                    self.logger.error(f"❌ 方式3也失败: {e3}")
            
            # 检查下载是否成功
            if not download_success:
                raise Exception(f"所有下载方式均失败。最后错误: {str(last_error)[:200]}")

            # 验证模型完整性（使用下载返回的路径）
            self._notify_progress("whisper", model_id, 90, "downloading", "验证模型完整性...")
            model.download_progress = 90.0

            # 使用 snapshot_download 返回的路径直接验证
            if local_dir:
                download_path = Path(local_dir)
                self.logger.info(f"📂 验证下载路径: {download_path}")

                # 直接验证返回的路径
                is_complete, missing_files, detail = ModelValidator.validate_whisper_model(download_path)

                if is_complete:
                    self.logger.info(f"✅ 下载路径验证成功")
                else:
                    self.logger.warning(f"⚠️ 下载路径验证失败，尝试标准查找...")
                    # 回退到标准查找
                    status, local_path, detail = self._check_whisper_model_exists(model_id)
                    if status != "ready":
                        raise Exception(f"模型下载后验证失败: {detail}")
                    download_path = local_path
            else:
                # 没有返回路径，使用标准查找
                status, download_path, detail = self._check_whisper_model_exists(model_id)
                if status != "ready":
                    raise Exception(f"模型下载后验证失败: {detail}")
            
            # 下载完成，更新状态
            model.status = "ready"
            model.download_progress = 100.0
            if download_path:
                model.local_path = str(download_path)

            self._notify_progress("whisper", model_id, 100, "ready", "下载完成！")
            self.logger.info(f"✅ Whisper模型下载完成: {model_id}")
            self.logger.info(f"📂 模型位置: {download_path}")
            self.logger.info(f"📋 文件验证:\n{detail}")

            # 自动下载对应的对齐模型（串行策略）
            self._auto_download_align_model_for_whisper(model_id)

        except Exception as e:
            if model:
                model.status = "error"
                model.download_progress = 0.0
            error_msg = f"下载失败: {str(e)[:200]}"
            self._notify_progress("whisper", model_id, 0, "error", error_msg)
            self.logger.error(f"❌ Whisper模型下载失败: {model_id} - {e}", exc_info=True)

        finally:
            # 释放下载锁
            model_key = f"whisper/{model_id}"
            with self.download_lock:
                if model_key in self.downloading_models:
                    del self.downloading_models[model_key]
            self.logger.info(f"🔓 下载锁已释放: {model_key}")

    def _auto_download_align_model_for_whisper(self, model_id: str):
        """
        自动下载Whisper模型对应的对齐模型（串行执行）

        Args:
            model_id: Whisper模型ID
        """
        # 获取推荐的对齐模型语言
        align_language = self.WHISPER_RECOMMENDED_ALIGN_MODELS.get(model_id)
        if not align_language:
            self.logger.warning(f"⚠️ 未找到模型 {model_id} 的推荐对齐模型")
            return

        # 检查对齐模型是否已存在
        status, local_path, detail = self._check_align_model_exists(align_language)
        if status == "ready":
            self.logger.info(f"✅ 对齐模型 {align_language} 已存在，无需下载")
            return

        self.logger.info(f"🔄 开始自动下载对齐模型: {align_language}")
        self._notify_progress("align", align_language, 0, "downloading", f"自动下载对齐模型（关联模型: {model_id}）")

        # 直接调用下载对齐模型函数（会自动处理并发控制）
        success = self.download_align_model(align_language)
        if success:
            self.logger.info(f"✅ 对齐模型 {align_language} 已加入下载队列")
        else:
            self.logger.warning(f"⚠️ 对齐模型 {align_language} 下载失败或已在下载中")

    def _download_align_model_task(self, language: str):
        """下载对齐模型任务（后台线程）"""
        model = None
        try:
            model = self.align_models[language]

            import whisperx

            self.logger.info(f"📥 正在下载对齐模型: {language}")
            self._notify_progress("align", language, 10, "downloading", "开始下载...")

            # 加载对齐模型会自动触发下载
            _, _ = whisperx.load_align_model(
                language_code=language,
                device="cpu",
                model_dir=str(config.HF_CACHE_DIR)
            )

            # 下载完成，更新状态
            model.status = "ready"
            model.download_progress = 100.0

            # 重新检查路径
            status, local_path, validation_msg = self._check_align_model_exists(language)
            if local_path:
                model.local_path = str(local_path)

            self._notify_progress("align", language, 100, "ready", "下载完成！")
            self.logger.info(f"✅ 对齐模型下载完成: {language}")

        except Exception as e:
            if model:
                model.status = "error"
                model.download_progress = 0.0
            error_msg = f"下载失败: {str(e)[:200]}"
            self._notify_progress("align", language, 0, "error", error_msg)
            self.logger.error(f"❌ 对齐模型下载失败: {language} - {e}", exc_info=True)

        finally:
            # 释放下载锁
            model_key = f"align/{language}"
            with self.download_lock:
                if model_key in self.downloading_models:
                    del self.downloading_models[model_key]
            self.logger.info(f"🔓 下载锁已释放: {model_key}")


# ========== 单例模式 ==========

_model_manager_instance: Optional[ModelManagerService] = None


def get_model_manager() -> ModelManagerService:
    """
    获取模型管理器实例（单例模式）

    Returns:
        ModelManagerService: 模型管理器实例
    """
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManagerService()
    return _model_manager_instance
