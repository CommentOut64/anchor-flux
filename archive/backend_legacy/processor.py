"""
模型预加载管理接口
保留与模型管理相关的全局函数，供main.py使用
转录处理逻辑已迁移到 services/transcription_service.py
"""

import logging
from typing import Optional, Dict
from services.model_preload_manager import ModelPreloadManager, PreloadConfig

# 全局模型预加载管理器
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
        logging.getLogger(__name__).info("模型预加载管理器已初始化")
    return _model_manager


def get_model_manager() -> Optional[ModelPreloadManager]:
    """
    获取全局模型管理器

    Returns:
        Optional[ModelPreloadManager]: 模型管理器实例，未初始化则返回None
    """
    return _model_manager


async def preload_default_models(progress_callback=None) -> Dict[str, any]:
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


def get_preload_status() -> Dict[str, any]:
    """
    获取预加载状态

    Returns:
        Dict: 预加载状态信息
    """
    if _model_manager is None:
        return {"is_preloading": False, "message": "模型管理器未初始化"}

    return _model_manager.get_preload_status()


def get_cache_status() -> Dict[str, any]:
    """
    获取缓存状态

    Returns:
        Dict: 缓存状态信息
    """
    if _model_manager is None:
        return {"message": "模型管理器未初始化"}

    return _model_manager.get_cache_status()
