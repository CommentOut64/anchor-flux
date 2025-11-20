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

    # 对齐模型的实际路径名称映射（HuggingFace仓库中的完整名称）
    ALIGN_MODEL_PATHS = {
        "zh": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn",
            "models--facebook--wav2vec2-large-xlsr-53-chinese-zh-cn",
        ],
        "en": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-english",
            "models--facebook--wav2vec2-large-xlsr-53-english",
        ],
        "ja": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-japanese",
            "models--facebook--wav2vec2-large-xlsr-53-japanese",
        ],
        "ko": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-korean",
            "models--facebook--wav2vec2-large-xlsr-53-korean",
        ],
        "es": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-spanish",
            "models--facebook--wav2vec2-large-xlsr-53-spanish",
        ],
        "fr": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-french",
            "models--facebook--wav2vec2-large-xlsr-53-french",
        ],
        "de": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-german",
            "models--facebook--wav2vec2-large-xlsr-53-german",
        ],
        "ru": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-russian",
            "models--facebook--wav2vec2-large-xlsr-53-russian",
        ],
        "pt": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-portuguese",
            "models--facebook--wav2vec2-large-xlsr-53-portuguese",
        ],
        "it": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-italian",
            "models--facebook--wav2vec2-large-xlsr-53-italian",
        ],
        "ar": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-arabic",
            "models--facebook--wav2vec2-large-xlsr-53-arabic",
        ],
        "hi": [
            "models--jonatasgrosman--wav2vec2-large-xlsr-53-hindi",
            "models--facebook--wav2vec2-large-xlsr-53-hindi",
        ],
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

        print("="*80)
        print("🚀 ModelManagerService 初始化开始")
        print(f"📁 模型目录: {self.models_dir}")
        print("="*80)

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
        print("🔍 开始快速扫描本地模型...")
        self._init_model_info()
        print(f"✅ 模型扫描完成: Whisper={len([m for m in self.whisper_models.values() if m.status == 'ready'])}/{len(self.whisper_models)}, Align={len([m for m in self.align_models.values() if m.status == 'ready'])}/{len(self.align_models)}")

        # 启动后台验证任务
        print("🔧 启动后台验证线程...")
        threading.Thread(target=self._background_validate_models, daemon=True).start()
        print("="*80)

    def _init_model_info(self):
        """快速扫描本地已有模型（不进行完整性验证，留给后台任务）"""
        self.logger.info("🔍 快速扫描本地模型...")
        print("🔍 快速扫描本地模型...")

        # 初始化Whisper模型信息（仅检查目录是否存在）
        for model_id, info in self.WHISPER_MODELS.items():
            # 快速检测：只检查目录是否存在
            exists, local_path = self._quick_check_whisper_model(model_id)

            if exists:
                # 目录存在，先标记为ready，后台会验证完整性
                self.whisper_models[model_id] = ModelInfo(
                    model_id=model_id,
                    size_mb=info["size_mb"],
                    status="ready",
                    download_progress=100.0,
                    local_path=str(local_path) if local_path else None,
                    description=info["desc"]
                )
                print(f"   ✅ Whisper模型 {model_id}: 已下载 (路径: {local_path})")
                self.logger.info(f"✅ 发现Whisper模型目录: {model_id}（待后台验证）")
            else:
                # 目录不存在，标记为未下载
                self.whisper_models[model_id] = ModelInfo(
                    model_id=model_id,
                    size_mb=info["size_mb"],
                    status="not_downloaded",
                    download_progress=0.0,
                    local_path=None,
                    description=info["desc"]
                )
                print(f"   ⚪ Whisper模型 {model_id}: 未下载")

        # 初始化对齐模型信息（仅检查目录是否存在）
        for lang, name in self.SUPPORTED_LANGUAGES.items():
            # 快速检测：只检查目录是否存在
            exists, local_path = self._quick_check_align_model(lang)

            if exists:
                # 目录存在，先标记为ready，后台会验证完整性
                self.align_models[lang] = AlignModelInfo(
                    language=lang,
                    language_name=name,
                    status="ready",
                    download_progress=100.0,
                    local_path=str(local_path) if local_path else None
                )
                print(f"   ✅ 对齐模型 {lang} ({name}): 已下载 (路径: {local_path})")
                self.logger.info(f"✅ 发现对齐模型目录: {lang} ({name})（待后台验证）")
            else:
                # 目录不存在，标记为未下载
                self.align_models[lang] = AlignModelInfo(
                    language=lang,
                    language_name=name,
                    status="not_downloaded",
                    download_progress=0.0,
                    local_path=None
                )
                print(f"   ⚪ 对齐模型 {lang} ({name}): 未下载")

    def _get_latest_snapshot(self, model_dir: Path) -> Optional[Path]:
        """
        获取模型的最新快照（优先使用refs/main指向的快照）

        Args:
            model_dir: 模型目录

        Returns:
            最新快照路径，如果不存在则返回None
        """
        # 优先使用refs/main指向的快照
        main_ref = model_dir / "refs" / "main"
        if main_ref.exists():
            try:
                snapshot_id = main_ref.read_text().strip()
                snapshot_path = model_dir / "snapshots" / snapshot_id
                if snapshot_path.exists() and snapshot_path.is_dir():
                    self.logger.debug(f"   使用main引用的快照: {snapshot_id}")
                    return snapshot_path
            except Exception as e:
                self.logger.warning(f"   读取main引用失败: {e}")

        # 回退到最新修改时间的快照
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshots:
            return None

        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        self.logger.debug(f"   使用最新快照（按时间）: {latest.name}")
        return latest

    def _quick_check_whisper_model(self, model_id: str) -> tuple[bool, Optional[Path]]:
        """
        快速检查Whisper模型目录是否存在（不验证完整性）

        Args:
            model_id: 模型ID

        Returns:
            tuple: (是否存在, 本地路径)
        """
        # 模型实际存储在 HF_CACHE_DIR 直接目录下（不是hub子目录）
        hf_cache = config.HF_CACHE_DIR

        print(f"      🔍 检查 {model_id}")
        print(f"         HF缓存: {hf_cache}")
        print(f"         是否存在: {hf_cache.exists()}")

        # 检查可能的模型缓存路径
        possible_paths = [
            hf_cache / f"models--Systran--faster-whisper-{model_id}",
            hf_cache / f"models--guillaumekln--faster-whisper-{model_id}",
        ]

        for model_dir in possible_paths:
            print(f"         检查路径: {model_dir}")
            print(f"         路径存在: {model_dir.exists()}")
            if not model_dir.exists():
                continue

            # 使用统一的快照选择逻辑
            latest_snapshot = self._get_latest_snapshot(model_dir)
            if latest_snapshot:
                print(f"         ✅ 找到快照: {latest_snapshot}")
                self.logger.info(f"   ✓ 检测到Whisper模型: {model_id} (路径: {latest_snapshot.name})")
                return (True, latest_snapshot)

        self.logger.debug(f"   ✗ 未检测到Whisper模型: {model_id}")
        return (False, None)

    def _quick_check_align_model(self, language: str) -> tuple[bool, Optional[Path]]:
        """
        快速检查对齐模型目录是否存在（不验证完整性）

        Args:
            language: 语言代码

        Returns:
            tuple: (是否存在, 本地路径)
        """
        # 模型实际存储在 HF_CACHE_DIR 直接目录下（不是hub子目录）
        hf_cache = config.HF_CACHE_DIR

        # 使用预定义的路径映射
        model_patterns = self.ALIGN_MODEL_PATHS.get(language, [])

        if not model_patterns:
            self.logger.warning(f"⚠️ 语言 {language} 没有对应的模型路径映射")
            return (False, None)

        print(f"      🔍 检查对齐模型 {language}")
        print(f"         候选路径: {model_patterns}")

        for pattern in model_patterns:
            model_dir = hf_cache / pattern
            print(f"         检查: {model_dir}")
            print(f"         存在: {model_dir.exists()}")

            if not model_dir.exists():
                continue

            # 使用统一的快照选择逻辑
            latest_snapshot = self._get_latest_snapshot(model_dir)
            if latest_snapshot:
                print(f"         ✅ 找到快照: {latest_snapshot}")
                self.logger.info(f"   ✓ 检测到对齐模型: {language} (路径: {latest_snapshot.name})")
                return (True, latest_snapshot)

        self.logger.debug(f"   ✗ 未检测到对齐模型: {language}")
        return (False, None)

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
        hf_cache = config.HF_CACHE_DIR  # 修复：直接使用 HF_CACHE_DIR，不加 hub

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

            # 使用统一的快照选择逻辑
            latest_snapshot = self._get_latest_snapshot(model_dir)
            if not latest_snapshot:
                self.logger.debug(f"    ✗ 未找到有效快照")
                continue

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
        hf_cache = config.HF_CACHE_DIR  # 修复：直接使用 HF_CACHE_DIR，不加 hub

        # 使用预定义的路径映射
        model_patterns = self.ALIGN_MODEL_PATHS.get(language, [])

        if not model_patterns:
            return ("not_downloaded", None, f"语言 {language} 没有对应的模型路径映射")

        for pattern in model_patterns:
            model_dir = hf_cache / pattern
            if not model_dir.exists():
                continue

            # 使用统一的快照选择逻辑
            latest_snapshot = self._get_latest_snapshot(model_dir)
            if not latest_snapshot:
                continue

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

    def get_largest_ready_model(self) -> Optional[str]:
        """
        获取体积最大的已就绪（ready）的Whisper模型

        Returns:
            Optional[str]: 模型ID，如果没有ready的模型则返回None
        """
        ready_models = [
            (model_id, model.size_mb)
            for model_id, model in self.whisper_models.items()
            if model.status == "ready"
        ]

        if not ready_models:
            return None

        # 按体积排序，返回最大的
        largest_model = max(ready_models, key=lambda x: x[1])
        self.logger.debug(f"📊 最大的ready模型: {largest_model[0]} ({largest_model[1]}MB)")
        return largest_model[0]

    def get_ready_whisper_models(self) -> List[str]:
        """
        获取所有已就绪（ready）的Whisper模型ID列表

        Returns:
            List[str]: 模型ID列表
        """
        ready_models = [
            model_id
            for model_id, model in self.whisper_models.items()
            if model.status == "ready"
        ]
        return ready_models
    
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
        """后台异步验证所有模型完整性（立即启动，无延迟）"""
        self.logger.info("🔍 开始后台验证模型完整性...")

        # 验证 Whisper 模型
        for model_id, model in self.whisper_models.items():
            if model.status == "ready":
                self.logger.debug(f"🔍 验证Whisper模型: {model_id}")
                status, local_path, detail = self._check_whisper_model_exists(model_id)

                if status != "ready":
                    # 验证失败，立即更新状态并通知前端
                    self.logger.warning(f"⚠️ 后台验证发现模型不完整: {model_id}\n{detail}")
                    model.status = "incomplete"
                    model.download_progress = 0.0

                    # 通过SSE通知前端
                    self._notify_progress(
                        "whisper",
                        model_id,
                        0,
                        "incomplete",
                        f"模型文件不完整：{detail}"
                    )
                else:
                    self.logger.info(f"✅ Whisper模型验证通过: {model_id}")

        # 验证对齐模型
        for lang, model in self.align_models.items():
            if model.status == "ready":
                self.logger.debug(f"🔍 验证对齐模型: {lang}")
                status, local_path, detail = self._check_align_model_exists(lang)

                if status != "ready":
                    # 验证失败，立即更新状态并通知前端
                    self.logger.warning(f"⚠️ 后台验证发现对齐模型不完整: {lang}\n{detail}")
                    model.status = "incomplete"
                    model.download_progress = 0.0

                    # 通过SSE通知前端
                    self._notify_progress(
                        "align",
                        lang,
                        0,
                        "incomplete",
                        f"模型文件不完整：{detail}"
                    )
                else:
                    self.logger.info(f"✅ 对齐模型验证通过: {lang}")

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

        # 如果模型不完整，清理旧文件（只清理该模型，不影响其他模型）
        if status == "incomplete" and local_path:
            self.logger.warning(f"🗑️ 清理不完整的模型文件: {model_id}")
            try:
                # 获取该模型的根目录：snapshots的上两级
                # 结构: models--Systran--faster-whisper-xxx/snapshots/hash/
                # 需要删除: models--Systran--faster-whisper-xxx/
                model_root = local_path.parent.parent
                if model_root.exists() and model_root.name.startswith("models--"):
                    self.logger.info(f"🗑️ 删除不完整模型目录: {model_root}")
                    shutil.rmtree(model_root)
                    self.logger.info(f"✅ 已清理不完整模型: {model_root.name}")
                else:
                    self.logger.warning(f"⚠️ 模型路径异常，跳过清理: {model_root}")
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
        删除Whisper模型（只删除指定模型，不影响其他模型）

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
            # 获取模型的根目录（删除整个模型缓存，而非仅快照）
            local_path = Path(model.local_path)
            # 结构: models--Systran--faster-whisper-xxx/snapshots/hash/
            # 需要删除: models--Systran--faster-whisper-xxx/
            model_root = local_path.parent.parent

            if model_root.exists() and model_root.name.startswith("models--"):
                self.logger.info(f"🗑️ 删除Whisper模型目录: {model_root}")
                shutil.rmtree(model_root)
                self.logger.info(f"✅ 已删除Whisper模型: {model_id} ({model_root.name})")
            else:
                self.logger.warning(f"⚠️ 模型路径异常: {model_root}")
                return False

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
        删除对齐模型（只删除指定模型，不影响其他模型）

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
            # 获取模型的根目录（删除整个模型缓存，而非仅快照）
            local_path = Path(model.local_path)
            # 结构: models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn/snapshots/hash/
            # 需要删除: models--jonatasgrosman--wav2vec2-large-xlsr-53-chinese-zh-cn/
            model_root = local_path.parent.parent

            if model_root.exists() and model_root.name.startswith("models--"):
                self.logger.info(f"🗑️ 删除对齐模型目录: {model_root}")
                shutil.rmtree(model_root)
                self.logger.info(f"✅ 已删除对齐模型: {language} ({model_root.name})")
            else:
                self.logger.warning(f"⚠️ 模型路径异常: {model_root}")
                return False

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
            
            # 方式1: 使用 requests 手动下载（完全控制，带实时进度追踪）
            if not download_success:
                try:
                    self.logger.info(f"🔄 方式1: 使用手动下载方式...")
                    self._notify_progress("whisper", model_id, 0, "downloading", "准备下载...")

                    import requests
                    from huggingface_hub import hf_hub_url, list_repo_files
                    from pathlib import Path as PathlibPath

                    repo_id = f"Systran/faster-whisper-{model_id}"
                    cache_dir = str(config.HF_CACHE_DIR)

                    if use_mirror:
                        self.logger.info(f"📦 从镜像站下载: {config.HF_ENDPOINT}")
                    else:
                        self.logger.info(f"📦 从官方源下载: {repo_id}")

                    # 获取文件列表
                    self.logger.info("📋 获取模型文件列表...")
                    files = list_repo_files(repo_id, repo_type="model")

                    # 分类文件：小文件和大文件（model.bin）
                    small_files = [f for f in files if not f.endswith('.bin')]
                    large_files = [f for f in files if f.endswith('.bin')]

                    self.logger.info(f"📦 需要下载 {len(small_files)} 个配置文件和 {len(large_files)} 个模型文件")

                    # 确定保存路径（使用HuggingFace的标准路径结构）
                    # 格式：models--Systran--faster-whisper-{model_id}
                    repo_path = repo_id.replace("/", "--")
                    storage_folder = PathlibPath(cache_dir) / f"models--{repo_path}"

                    # 创建snapshots目录
                    snapshots_dir = storage_folder / "snapshots"
                    # 使用时间戳作为snapshot ID
                    import hashlib
                    snapshot_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
                    snapshot_dir = snapshots_dir / snapshot_id
                    snapshot_dir.mkdir(parents=True, exist_ok=True)

                    # 阶段1：下载小文件（0% -> 10%）
                    for idx, filename in enumerate(small_files):
                        try:
                            progress = (idx / len(small_files)) * 10
                            self._notify_progress("whisper", model_id, progress, "downloading", f"下载配置文件 ({idx+1}/{len(small_files)})")
                            model.download_progress = progress

                            url = hf_hub_url(repo_id, filename, repo_type="model")
                            response = requests.get(url, timeout=(30, 300))  # 连接30秒，读取5分钟
                            response.raise_for_status()

                            file_path = snapshot_dir / filename
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            file_path.write_bytes(response.content)

                            self.logger.info(f"  ✓ {filename}")
                        except Exception as e:
                            self.logger.warning(f"  ✗ {filename}: {e}")
                            raise  # 小文件下载失败就终止

                    self._notify_progress("whisper", model_id, 10, "downloading", "配置文件下载完成，开始下载模型文件...")
                    model.download_progress = 10.0

                    # 阶段2：下载大文件（10% -> 100%）
                    for filename in large_files:
                        self.logger.info(f"📥 开始下载大文件: {filename}")

                        url = hf_hub_url(repo_id, filename, repo_type="model")

                        # 获取文件大小
                        head_response = requests.head(url, allow_redirects=True, timeout=30)
                        total_size = int(head_response.headers.get('content-length', 0))

                        self.logger.info(f"📊 文件大小: {total_size / 1024 / 1024:.1f} MB")

                        # 流式下载并追踪进度
                        response = requests.get(url, stream=True, timeout=(30, 600))  # 连接30秒，读取10分钟
                        response.raise_for_status()

                        # 创建临时文件
                        temp_file = snapshot_dir / f"{filename}.download"
                        final_file = snapshot_dir / filename
                        temp_file.parent.mkdir(parents=True, exist_ok=True)

                        downloaded = 0
                        chunk_size = 1024 * 1024  # 1MB chunks
                        last_reported_progress = 0  # 记录上次报告的进度

                        with open(temp_file, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)

                                    # 计算进度：10% + (当前下载/总大小) * 90%
                                    if total_size > 0:
                                        file_progress = (downloaded / total_size) * 90
                                        total_progress = 10 + file_progress

                                        # 优化：只在进度变化超过0.5%时才更新（避免频繁更新）
                                        if total_progress - last_reported_progress >= 0.5 or downloaded == total_size:
                                            self._notify_progress("whisper", model_id, total_progress, "downloading",
                                                                f"下载模型文件 {downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB ({total_progress:.1f}%)")
                                            model.download_progress = total_progress
                                            last_reported_progress = total_progress

                        # 下载完成，重命名文件
                        temp_file.rename(final_file)
                        self.logger.info(f"  ✓ {filename} 下载完成")

                    # 创建 refs/main 指向当前snapshot
                    refs_dir = storage_folder / "refs"
                    refs_dir.mkdir(parents=True, exist_ok=True)
                    (refs_dir / "main").write_text(snapshot_id)

                    local_dir = snapshot_dir
                    self.logger.info(f"✅ 方式1成功下载到: {local_dir}")
                    self._notify_progress("whisper", model_id, 100, "downloading", "下载完成，验证文件...")
                    model.download_progress = 100.0
                    download_success = True

                except Exception as e1:
                    last_error = e1
                    self.logger.warning(f"⚠️ 方式1失败: {e1}")
                    self._notify_progress("whisper", model_id, 0, "downloading", f"方式1失败，尝试其他方式...")
            
            # 方式2: 如果方式1失败且使用了镜像，尝试切换到官方源
            if not download_success and use_mirror:
                try:
                    self.logger.info(f"🔄 方式2: 切换到官方源重试...")
                    self._notify_progress("whisper", model_id, 0, "downloading", "切换到官方源...")

                    # 临时切换到官方源
                    old_endpoint = os.environ.get('HF_ENDPOINT')
                    if 'HF_ENDPOINT' in os.environ:
                        del os.environ['HF_ENDPOINT']

                    try:
                        import requests
                        from huggingface_hub import hf_hub_url, list_repo_files
                        from pathlib import Path as PathlibPath

                        repo_id = f"Systran/faster-whisper-{model_id}"
                        cache_dir = str(config.HF_CACHE_DIR)

                        self.logger.info(f"📦 从官方源下载: https://huggingface.co")

                        # 获取文件列表
                        files = list_repo_files(repo_id, repo_type="model")
                        small_files = [f for f in files if not f.endswith('.bin')]
                        large_files = [f for f in files if f.endswith('.bin')]

                        # 确定保存路径
                        repo_path = repo_id.replace("/", "--")
                        storage_folder = PathlibPath(cache_dir) / f"models--{repo_path}"
                        snapshots_dir = storage_folder / "snapshots"

                        import hashlib
                        snapshot_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]
                        snapshot_dir = snapshots_dir / snapshot_id
                        snapshot_dir.mkdir(parents=True, exist_ok=True)

                        # 下载小文件（0% -> 10%）
                        for idx, filename in enumerate(small_files):
                            progress = (idx / len(small_files)) * 10
                            self._notify_progress("whisper", model_id, progress, "downloading", f"下载配置文件 ({idx+1}/{len(small_files)})")
                            model.download_progress = progress

                            url = hf_hub_url(repo_id, filename, repo_type="model")
                            response = requests.get(url, timeout=(30, 300))
                            response.raise_for_status()

                            file_path = snapshot_dir / filename
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            file_path.write_bytes(response.content)

                            self.logger.info(f"  ✓ {filename}")

                        self._notify_progress("whisper", model_id, 10, "downloading", "配置文件下载完成，开始下载模型文件...")
                        model.download_progress = 10.0

                        # 下载大文件（10% -> 100%）
                        for filename in large_files:
                            self.logger.info(f"📥 开始下载大文件: {filename}")

                            url = hf_hub_url(repo_id, filename, repo_type="model")

                            # 获取文件大小
                            head_response = requests.head(url, allow_redirects=True, timeout=30)
                            total_size = int(head_response.headers.get('content-length', 0))
                            self.logger.info(f"📊 文件大小: {total_size / 1024 / 1024:.1f} MB")

                            # 流式下载
                            response = requests.get(url, stream=True, timeout=(30, 600))
                            response.raise_for_status()

                            temp_file = snapshot_dir / f"{filename}.download"
                            final_file = snapshot_dir / filename
                            temp_file.parent.mkdir(parents=True, exist_ok=True)

                            downloaded = 0
                            chunk_size = 1024 * 1024
                            last_reported_progress = 0  # 记录上次报告的进度

                            with open(temp_file, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=chunk_size):
                                    if chunk:
                                        f.write(chunk)
                                        downloaded += len(chunk)

                                        if total_size > 0:
                                            file_progress = (downloaded / total_size) * 90
                                            total_progress = 10 + file_progress

                                            # 优化：只在进度变化超过0.5%时才更新（避免频繁更新）
                                            if total_progress - last_reported_progress >= 0.5 or downloaded == total_size:
                                                self._notify_progress("whisper", model_id, total_progress, "downloading",
                                                                    f"下载模型文件 {downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB ({total_progress:.1f}%)")
                                                model.download_progress = total_progress
                                                last_reported_progress = total_progress

                            temp_file.rename(final_file)
                            self.logger.info(f"  ✓ {filename} 下载完成")

                        # 创建 refs/main
                        refs_dir = storage_folder / "refs"
                        refs_dir.mkdir(parents=True, exist_ok=True)
                        (refs_dir / "main").write_text(snapshot_id)

                        local_dir = snapshot_dir
                        self.logger.info(f"✅ 方式2成功")
                        self._notify_progress("whisper", model_id, 100, "downloading", "下载完成，验证文件...")
                        model.download_progress = 100.0
                        download_success = True

                    finally:
                        # 恢复镜像源设置
                        if old_endpoint:
                            os.environ['HF_ENDPOINT'] = old_endpoint

                except Exception as e2:
                    last_error = e2
                    self.logger.error(f"❌ 方式2也失败: {e2}")
                    self._notify_progress("whisper", model_id, 0, "downloading", "方式2失败，尝试最后方式...")
            
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
