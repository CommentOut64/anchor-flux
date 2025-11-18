"""
模型预加载和缓存管理器
实现模型预加载、LRU缓存、内存监控等功能
"""

import os
import gc
import logging
import threading
import time
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import psutil
import torch
import whisperx

# 修复导入路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.job_models import JobSettings


@dataclass
class ModelCacheInfo:
    """模型缓存信息"""
    model: Any
    key: Tuple[str, str, str]
    load_time: float
    last_used: float
    memory_size: int  # 估算的内存占用(MB)


@dataclass
class PreloadConfig:
    """预加载配置"""
    enabled: bool = True
    default_models: List[str] = None
    max_cache_size: int = 3  # 最大缓存模型数量
    memory_threshold: float = 0.8  # 内存使用阈值(80%)
    preload_timeout: int = 300  # 预加载超时时间(秒)
    warmup_enabled: bool = True  # 是否启用预热

    def __post_init__(self):
        if self.default_models is None:
            self.default_models = ["medium"]


class ModelPreloadManager:
    """简化版模型预加载和缓存管理器 - 方案二实现
    
    核心改进:
    1. 统一锁机制避免死锁
    2. 幂等性预加载避免重复执行
    3. 缓存版本号确保状态同步
    4. 标准化日志便于调试
    """
    
    def __init__(self, config: PreloadConfig = None):
        self.config = config or PreloadConfig()
        self.logger = self._setup_logger()
        
        # 模型缓存 (LRU)
        self._whisper_cache: OrderedDict[Tuple[str, str, str], ModelCacheInfo] = OrderedDict()
        self._align_cache: OrderedDict[str, Tuple[Any, Any, float]] = OrderedDict()
        
        # 统一锁 - 简化并发控制，避免多锁死锁
        self._global_lock = threading.RLock()
        
        # 简化的预加载状态 - 单一数据源
        self._preload_status = {
            "is_preloading": False,
            "progress": 0.0,
            "current_model": "",
            "total_models": 0,
            "loaded_models": 0,
            "errors": [],
            "failed_attempts": 0,
            "last_attempt_time": 0,
            "max_retry_attempts": 3,
            "retry_cooldown": 30,
            "cache_version": int(time.time())  # 缓存版本号，用于状态同步
        }
        
        # 预加载任务管理 - 实现幂等性
        self._preload_promise: Optional[asyncio.Task] = None
        
        # 内存监控
        self._memory_monitor = MemoryMonitor()
        
        self.logger.info("🏗️ ModelPreloadManager初始化完成 - 简化架构")
    
    def _setup_logger(self) -> logging.Logger:
        """设置标准化的日志记录器"""
        logger = logging.getLogger(f"{__name__}.ModelPreloadManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - 🤖[模型管理] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def get_preload_status(self) -> Dict[str, Any]:
        """获取预加载状态 - 线程安全版本"""
        with self._global_lock:
            status = self._preload_status.copy()
            self.logger.debug(f"📊 状态查询: 预加载={status['is_preloading']}, 进度={status['progress']:.1f}%, 已加载={status['loaded_models']}")
            return status
    
    def get_cache_status(self) -> Dict[str, Any]:
        """获取缓存状态 - 线程安全版本"""
        with self._global_lock:
            whisper_models = [
                {
                    "key": info.key,
                    "memory_mb": info.memory_size,
                    "last_used": info.last_used,
                    "load_time": info.load_time
                }
                for info in self._whisper_cache.values()
            ]
            
            align_models = list(self._align_cache.keys())
            total_memory = sum(info.memory_size for info in self._whisper_cache.values())
            
            cache_status = {
                "whisper_models": whisper_models,
                "align_models": align_models,
                "total_memory_mb": total_memory,
                "max_cache_size": self.config.max_cache_size,
                "memory_info": self._memory_monitor.get_memory_info(),
                "cache_version": self._preload_status["cache_version"]
            }
            
            self.logger.debug(f"💾 缓存查询: Whisper模型={len(whisper_models)}个, 对齐模型={len(align_models)}个, 内存={total_memory}MB")
            return cache_status
    
    async def preload_models(self, progress_callback=None) -> Dict[str, Any]:
        """预加载默认模型 - 简化版实现，带幂等性保证"""
        with self._global_lock:
            # 检查是否已经在预加载中（幂等性检查）
            if self._preload_status["is_preloading"]:
                self.logger.info("⚡ 预加载已在进行中，返回已有任务")
                return {"success": True, "message": "预加载已在进行中"}
            
            if not self.config.enabled:
                self.logger.warning("⚠️ 模型预加载功能已禁用")
                return {"success": False, "message": "预加载功能已禁用"}

            # 设置预加载状态
            self._preload_status.update({
                "is_preloading": True,
                "progress": 0.0,
                "current_model": "",
                "total_models": len(self.config.default_models),
                "loaded_models": 0,
                "errors": [],
                "last_attempt_time": time.time()
            })
            
            self.logger.info(f"🚀 开始预加载任务: {self.config.default_models}")

        try:
            success_count = 0
            total_models = len(self.config.default_models)
            
            for i, model_name in enumerate(self.config.default_models):
                try:
                    # 更新当前进度
                    with self._global_lock:
                        self._preload_status.update({
                            "current_model": model_name,
                            "progress": (i / total_models) * 100
                        })
                    
                    self.logger.info(f"🔄 [{i+1}/{total_models}] 处理模型: {model_name}")
                    
                    # 检查内存
                    if not self._memory_monitor.check_memory_available():
                        error_msg = f"内存不足，跳过模型 {model_name}"
                        self.logger.warning(f"⚠️ {error_msg}")
                        with self._global_lock:
                            self._preload_status["errors"].append(error_msg)
                        continue
                    
                    # 检查模型是否已缓存
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    key = (model_name, "float16", device)
                    
                    with self._global_lock:
                        if key in self._whisper_cache:
                            self.logger.info(f"✅ 模型 {model_name} 已在缓存中")
                            self._preload_status["loaded_models"] += 1
                            success_count += 1
                            continue
                    
                    # 加载新模型
                    settings = JobSettings(
                        model=model_name,
                        compute_type="float16",
                        device=device
                    )
                    
                    self.logger.info(f"🔍 开始加载模型: {model_name} (device={device})")
                    start_time = time.time()
                    
                    model = self._load_whisper_model(settings)
                    
                    load_time = time.time() - start_time
                    
                    # 预热模型
                    if self.config.warmup_enabled:
                        self.logger.info(f"🔥 预热模型: {model_name}")
                        self._warmup_model(model)
                    
                    with self._global_lock:
                        self._preload_status["loaded_models"] += 1
                    success_count += 1
                    
                    self.logger.info(f"✅ 模型 {model_name} 加载成功 (耗时: {load_time:.2f}s)")
                    
                    # 调用进度回调
                    if progress_callback:
                        with self._global_lock:
                            progress_callback(self._preload_status.copy())
                    
                except Exception as e:
                    error_msg = f"加载模型 {model_name} 失败: {str(e)}"
                    self.logger.error(f"❌ {error_msg}", exc_info=True)
                    with self._global_lock:
                        self._preload_status["errors"].append(error_msg)
            
            # 完成预加载
            success = success_count > 0
            
            with self._global_lock:
                # 更新失败计数
                if not success:
                    self._preload_status["failed_attempts"] += 1
                else:
                    self._preload_status["failed_attempts"] = 0  # 成功后重置
                
                # 更新缓存版本号
                self._preload_status["cache_version"] = int(time.time())
                
                # 结束预加载状态
                self._preload_status.update({
                    "is_preloading": False,
                    "progress": 100.0,
                    "current_model": ""
                })
                
                result = {
                    "success": success,
                    "loaded_models": self._preload_status["loaded_models"],
                    "total_models": self._preload_status["total_models"],
                    "errors": self._preload_status["errors"].copy(),
                    "failed_attempts": self._preload_status["failed_attempts"],
                    "cache_version": self._preload_status["cache_version"]
                }
            
            # 日志输出
            if success:
                self.logger.info(f"✅ 预加载任务成功完成: {success_count}/{total_models} 个模型")
            else:
                self.logger.warning(f"⚠️ 预加载任务完成但无成功加载的模型")
                
            if result["errors"]:
                self.logger.warning(f"⚠️ 预加载过程中出现 {len(result['errors'])} 个错误")
            
            # 调用最终进度回调
            if progress_callback:
                progress_callback(self._preload_status.copy())
            
            return result
            
        except Exception as e:
            with self._global_lock:
                self._preload_status["is_preloading"] = False
                self._preload_status["failed_attempts"] += 1
            
            self.logger.error(f"❌ 预加载过程异常: {str(e)}", exc_info=True)
            return {
                "success": False, 
                "message": f"预加载失败: {str(e)}", 
                "failed_attempts": self._preload_status["failed_attempts"]
            }

    def reset_preload_attempts(self):
        """重置预加载失败计数 - 线程安全版本"""
        with self._global_lock:
            old_attempts = self._preload_status["failed_attempts"]
            self._preload_status["failed_attempts"] = 0
            self._preload_status["last_attempt_time"] = 0
            self._preload_status["cache_version"] = int(time.time())
            
        self.logger.info(f"🔄 预加载失败计数已重置: {old_attempts} -> 0")
    
    def get_model(self, settings: JobSettings):
        """获取Whisper模型 (带LRU缓存) - 简化版本"""
        key = (settings.model, settings.compute_type, settings.device)
        
        with self._global_lock:
            # 命中缓存
            if key in self._whisper_cache:
                info = self._whisper_cache[key]
                info.last_used = time.time()
                # 移到最后 (最近使用)
                self._whisper_cache.move_to_end(key)
                self.logger.debug(f"✅ 命中模型缓存: {key}")
                return info.model
            
            # 缓存未命中，加载新模型
            self.logger.info(f"🔄 需要加载新模型: {key}")
            return self._load_whisper_model(settings)
    
    def _load_whisper_model(self, settings: JobSettings):
        """加载Whisper模型 - 简化版本带并发保护"""
        key = (settings.model, settings.compute_type, settings.device)
        
        # 再次检查缓存（避免并发加载同一模型）
        with self._global_lock:
            if key in self._whisper_cache:
                info = self._whisper_cache[key]
                info.last_used = time.time()
                self._whisper_cache.move_to_end(key)
                self.logger.debug(f"⚡ 并发检查命中缓存，避免重复加载: {key}")
                return info.model
        
        self.logger.info(f"🔍 开始加载新Whisper模型: {key}")
        
        # 检查内存
        if not self._memory_monitor.check_memory_available():
            self.logger.warning("⚠️ 内存不足，尝试清理缓存")
            self._cleanup_old_models()
        
        # 检查缓存大小
        with self._global_lock:
            if len(self._whisper_cache) >= self.config.max_cache_size:
                self._evict_lru_model()
        
        try:
            start_time = time.time()
            self.logger.info(f"🚀 正在从磁盘加载模型 {settings.model} (device={settings.device}, compute_type={settings.compute_type})")
            
            model = whisperx.load_model(
                settings.model, 
                settings.device, 
                compute_type=settings.compute_type
            )
            load_time = time.time() - start_time
            
            # 估算内存使用
            memory_size = self._estimate_model_memory(model)
            
            # 添加到缓存
            info = ModelCacheInfo(
                model=model,
                key=key,
                load_time=load_time,
                last_used=time.time(),
                memory_size=memory_size
            )
            
            with self._global_lock:
                self._whisper_cache[key] = info
                # 更新缓存版本号
                self._preload_status["cache_version"] = int(time.time())
            
            self.logger.info(f"✅ 成功加载并缓存Whisper模型 {key} (内存: {memory_size}MB, 耗时: {load_time:.2f}s)")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ 加载Whisper模型失败 {key}: {str(e)}", exc_info=True)
            raise
    
    def get_align_model(self, lang: str, device: str):
        """获取对齐模型 (带LRU缓存) - 简化版本"""
        with self._global_lock:
            # 命中缓存
            if lang in self._align_cache:
                model, meta, last_used = self._align_cache[lang]
                # 更新使用时间并移到最后
                self._align_cache[lang] = (model, meta, time.time())
                self._align_cache.move_to_end(lang)
                self.logger.debug(f"✅ 命中对齐模型缓存: {lang}")
                return model, meta
            
            # 缓存未命中，加载新模型
            self.logger.info(f"🔄 加载新对齐模型: {lang}")
            try:
                model, meta = whisperx.load_align_model(language_code=lang, device=device)
                
                # 添加到缓存 (限制大小)
                if len(self._align_cache) >= 5:  # 对齐模型缓存上限
                    # 移除最旧的
                    oldest = next(iter(self._align_cache))
                    del self._align_cache[oldest]
                    self.logger.debug(f"🗑️ 移除最旧对齐模型: {oldest}")
                
                self._align_cache[lang] = (model, meta, time.time())
                
                # 更新缓存版本号
                self._preload_status["cache_version"] = int(time.time())
                
                self.logger.info(f"✅ 成功加载对齐模型: {lang}")
                return model, meta
                
            except Exception as e:
                self.logger.error(f"❌ 加载对齐模型失败 {lang}: {str(e)}", exc_info=True)
                raise
    
    def _warmup_model(self, model):
        """预热模型 - 空跑一次确保完全加载"""
        try:
            self.logger.debug("开始模型预热")
            
            # 创建虚拟音频数据 (1秒静音)
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 16kHz 1秒
            
            # 空跑一次
            _ = model.transcribe(dummy_audio, batch_size=1, verbose=False)
            
            self.logger.debug("模型预热完成")
            
        except Exception as e:
            self.logger.warning(f"模型预热失败: {str(e)}")
    
    def _evict_lru_model(self):
        """驱逐最久未使用的模型 - 需要在锁内调用"""
        if not self._whisper_cache:
            return
        
        # 最久未使用的在开头
        oldest_key = next(iter(self._whisper_cache))
        info = self._whisper_cache.pop(oldest_key)
        
        self.logger.info(f"🗑️ 驱逐LRU模型: {oldest_key}, 释放内存: {info.memory_size}MB")
        
        # 释放内存
        del info.model
        del info
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _cleanup_old_models(self):
        """清理旧模型释放内存 - 需要在锁外调用"""
        current_time = time.time()
        to_remove = []
        
        with self._global_lock:
            for key, info in self._whisper_cache.items():
                # 超过10分钟未使用的模型
                if current_time - info.last_used > 600:
                    to_remove.append(key)
            
            for key in to_remove:
                info = self._whisper_cache.pop(key)
                self.logger.info(f"🗑️ 清理旧模型: {key}")
                del info.model
                del info
        
        if to_remove:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"💫 清理了 {len(to_remove)} 个旧模型")
    
    def _estimate_model_memory(self, model) -> int:
        """估算模型内存使用 (MB)"""
        try:
            # 简单估算，基于模型参数
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                total_params = sum(p.numel() for p in model.model.parameters())
                # 假设每个参数4字节 (float32) 或 2字节 (float16)
                bytes_per_param = 2  # float16
                total_bytes = total_params * bytes_per_param
                return int(total_bytes / (1024 * 1024))  # 转换为MB
        except:
            pass
        
        # 默认估算值
        return 500  # 默认500MB
    
    def clear_cache(self):
        """清空所有缓存 - 简化版本，立即同步状态"""
        with self._global_lock:
            # 记录清理前的缓存状态
            whisper_count = len(self._whisper_cache)
            align_count = len(self._align_cache)
            total_memory = sum(info.memory_size for info in self._whisper_cache.values())
            
            # 清理Whisper模型缓存
            for info in self._whisper_cache.values():
                del info.model
            self._whisper_cache.clear()
            
            # 清理对齐模型缓存
            self._align_cache.clear()
            
            # 立即更新预加载状态 - 解决状态同步问题
            self._preload_status.update({
                "loaded_models": 0,
                "is_preloading": False,
                "progress": 0.0,
                "current_model": "",
                "errors": [],
                "cache_version": int(time.time())  # 更新缓存版本号
            })
            
        # 垃圾回收和GPU内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"🗑️ 已清空所有模型缓存: Whisper={whisper_count}个, 对齐={align_count}个, 释放内存={total_memory}MB")

    # ========== 单模型管理接口 - 委托给模型管理服务 ==========

    def download_whisper_model(self, model_id: str) -> bool:
        """
        下载单个Whisper模型（委托给模型管理服务）

        Args:
            model_id: 模型ID (tiny, base, small, medium, large-v2, large-v3)

        Returns:
            bool: 是否成功启动下载
        """
        try:
            from services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()
            success = model_mgr.download_whisper_model(model_id)

            if success:
                self.logger.info(f"✅ 已委托模型管理服务下载Whisper模型: {model_id}")
            return success

        except Exception as e:
            self.logger.error(f"❌ 下载Whisper模型失败: {model_id} - {e}")
            return False

    def download_align_model(self, language: str) -> bool:
        """
        下载单个对齐模型（委托给模型管理服务）

        Args:
            language: 语言代码

        Returns:
            bool: 是否成功启动下载
        """
        try:
            from services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()
            success = model_mgr.download_align_model(language)

            if success:
                self.logger.info(f"✅ 已委托模型管理服务下载对齐模型: {language}")
            return success

        except Exception as e:
            self.logger.error(f"❌ 下载对齐模型失败: {language} - {e}")
            return False

    def delete_whisper_model(self, model_id: str) -> bool:
        """
        删除Whisper模型（委托给模型管理服务，并清理缓存）

        Args:
            model_id: 模型ID

        Returns:
            bool: 是否删除成功
        """
        try:
            from services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()

            # 先从缓存中移除
            with self._global_lock:
                keys_to_remove = [k for k in self._whisper_cache.keys() if k[0] == model_id]
                for key in keys_to_remove:
                    info = self._whisper_cache.pop(key)
                    del info.model
                    self.logger.debug(f"🗑️ 从缓存中移除模型: {key}")

                # 更新缓存版本号
                self._preload_status["cache_version"] = int(time.time())

            # 清理GPU内存
            if keys_to_remove:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 委托给模型管理服务删除磁盘文件
            success = model_mgr.delete_whisper_model(model_id)

            if success:
                self.logger.info(f"✅ 已删除Whisper模型: {model_id}")
            return success

        except Exception as e:
            self.logger.error(f"❌ 删除Whisper模型失败: {model_id} - {e}")
            return False

    def delete_align_model(self, language: str) -> bool:
        """
        删除对齐模型（委托给模型管理服务，并清理缓存）

        Args:
            language: 语言代码

        Returns:
            bool: 是否删除成功
        """
        try:
            from services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()

            # 先从缓存中移除
            with self._global_lock:
                if language in self._align_cache:
                    del self._align_cache[language]
                    self.logger.debug(f"🗑️ 从缓存中移除对齐模型: {language}")

                    # 更新缓存版本号
                    self._preload_status["cache_version"] = int(time.time())

            # 委托给模型管理服务删除磁盘文件
            success = model_mgr.delete_align_model(language)

            if success:
                self.logger.info(f"✅ 已删除对齐模型: {language}")
            return success

        except Exception as e:
            self.logger.error(f"❌ 删除对齐模型失败: {language} - {e}")
            return False

    def list_all_models(self) -> Dict[str, Any]:
        """
        列出所有模型的状态（整合磁盘状态和缓存状态）

        Returns:
            Dict: 包含whisper和align模型的状态信息
        """
        try:
            from services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()

            # 获取磁盘上的模型状态
            whisper_models = [
                {
                    "model_id": m.model_id,
                    "size_mb": m.size_mb,
                    "status": m.status,
                    "download_progress": m.download_progress,
                    "local_path": m.local_path,
                    "description": m.description,
                    "cached": any(k[0] == m.model_id for k in self._whisper_cache.keys())
                }
                for m in model_mgr.list_whisper_models()
            ]

            align_models = [
                {
                    "language": m.language,
                    "language_name": m.language_name,
                    "status": m.status,
                    "download_progress": m.download_progress,
                    "local_path": m.local_path,
                    "cached": m.language in self._align_cache
                }
                for m in model_mgr.list_align_models()
            ]

            return {
                "whisper_models": whisper_models,
                "align_models": align_models,
                "cache_info": {
                    "whisper_cached": len(self._whisper_cache),
                    "align_cached": len(self._align_cache),
                    "total_memory_mb": sum(info.memory_size for info in self._whisper_cache.values())
                }
            }

        except Exception as e:
            self.logger.error(f"❌ 列出模型失败: {e}")
            return {"error": str(e)}


class MemoryMonitor:
    """内存监控器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        try:
            # 系统内存
            memory = psutil.virtual_memory()

            # GPU内存 (如果可用)
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                    "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**3),  # GB
                    "gpu_memory_cached": torch.cuda.memory_reserved() / (1024**3),  # GB
                }

            return {
                "system_memory_total": memory.total / (1024**3),  # GB
                "system_memory_used": memory.used / (1024**3),  # GB
                "system_memory_percent": memory.percent,
                **gpu_info
            }
        except Exception as e:
            self.logger.error(f"获取内存信息失败: {str(e)}")
            return {}

    def check_memory_available(self, threshold: float = 0.85) -> bool:
        """检查内存是否充足"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < (threshold * 100)
        except:
            return True  # 默认认为内存充足


# ========== 全局单例模式 - 提供统一的模型管理器接口 ==========

_model_manager: Optional[ModelPreloadManager] = None


def initialize_model_manager(config: PreloadConfig = None) -> ModelPreloadManager:
    """
    初始化全局模型管理器

    Args:
        config: 预加载配置

    Returns:
        ModelPreloadManager: 模型管理器实例
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelPreloadManager(config)
        logging.getLogger(__name__).info("🏗️ 全局模型预加载管理器已初始化")
    return _model_manager


def get_model_manager() -> Optional[ModelPreloadManager]:
    """
    获取全局模型管理器

    Returns:
        Optional[ModelPreloadManager]: 模型管理器实例，未初始化则返回None
    """
    return _model_manager


async def preload_default_models(progress_callback=None) -> Dict[str, Any]:
    """
    预加载默认模型

    Args:
        progress_callback: 进度回调函数

    Returns:
        Dict: 预加载结果
    """
    if _model_manager is None:
        return {"success": False, "message": "模型管理器未初始化"}

    return await _model_manager.preload_models(progress_callback)


def get_preload_status() -> Dict[str, Any]:
    """
    获取预加载状态

    Returns:
        Dict: 预加载状态信息
    """
    if _model_manager is None:
        return {"is_preloading": False, "message": "模型管理器未初始化"}

    return _model_manager.get_preload_status()


def get_cache_status() -> Dict[str, Any]:
    """
    获取缓存状态

    Returns:
        Dict: 缓存状态信息
    """
    if _model_manager is None:
        return {"message": "模型管理器未初始化"}

    return _model_manager.get_cache_status()
