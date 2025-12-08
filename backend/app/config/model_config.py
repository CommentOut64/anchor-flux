"""
模型预加载配置文件
"""

import os
from typing import List


class ModelPreloadConfig:
    """模型预加载配置"""
    
    # 基础配置
    ENABLED = os.getenv("MODEL_PRELOAD_ENABLED", "true").lower() == "true"
    
    # 默认预加载的模型列表
    DEFAULT_MODELS = os.getenv("MODEL_PRELOAD_MODELS", "medium").split(",")
    
    # 缓存配置
    MAX_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", "3"))
    MEMORY_THRESHOLD = float(os.getenv("MODEL_MEMORY_THRESHOLD", "0.8"))
    
    # 预加载配置
    PRELOAD_TIMEOUT = int(os.getenv("MODEL_PRELOAD_TIMEOUT", "300"))
    WARMUP_ENABLED = os.getenv("MODEL_WARMUP_ENABLED", "true").lower() == "true"

  
    # 内存监控配置
    MEMORY_CHECK_INTERVAL = int(os.getenv("MEMORY_CHECK_INTERVAL", "60"))  # 秒
    
    @classmethod
    def get_preload_config(cls):
        """获取预加载配置对象"""
        from app.services.model_preload_manager import PreloadConfig
        
        return PreloadConfig(
            enabled=cls.ENABLED,
            default_models=cls.DEFAULT_MODELS,
            max_cache_size=cls.MAX_CACHE_SIZE,
            memory_threshold=cls.MEMORY_THRESHOLD,
            preload_timeout=cls.PRELOAD_TIMEOUT,
            warmup_enabled=cls.WARMUP_ENABLED
        )
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("模型预加载配置:")
        print(f"  启用预加载: {cls.ENABLED}")
        print(f"  默认模型: {cls.DEFAULT_MODELS}")
        print(f"  最大缓存大小: {cls.MAX_CACHE_SIZE}")
        print(f"  内存阈值: {cls.MEMORY_THRESHOLD}")
        print(f"  预加载超时: {cls.PRELOAD_TIMEOUT}s")
        print(f"  启用预热: {cls.WARMUP_ENABLED}")


# 常用模型配置
WHISPER_MODELS = {
    "tiny": {"size": "~39MB", "speed": "~32x", "memory": "~1GB"},
    "base": {"size": "~74MB", "speed": "~16x", "memory": "~1GB"},
    "small": {"size": "~244MB", "speed": "~6x", "memory": "~2GB"},
    "medium": {"size": "~769MB", "speed": "~2x", "memory": "~5GB"},
    "large": {"size": "~1550MB", "speed": "~1x", "memory": "~10GB"},
    "large-v2": {"size": "~1550MB", "speed": "~1x", "memory": "~10GB"},
    "large-v3": {"size": "~1550MB", "speed": "~1x", "memory": "~10GB"}
}

def get_model_info(model_name: str) -> dict:
    """获取模型信息"""
    return WHISPER_MODELS.get(model_name, {"size": "Unknown", "speed": "Unknown", "memory": "Unknown"})

def recommend_models_by_memory(total_memory_gb: float) -> List[str]:
    """根据可用内存推荐模型"""
    if total_memory_gb < 4:
        return ["tiny", "base"]
    elif total_memory_gb < 8:
        return ["tiny", "base", "small"]
    elif total_memory_gb < 16:
        return ["base", "small", "medium"]
    else:
        return ["medium", "large"]
