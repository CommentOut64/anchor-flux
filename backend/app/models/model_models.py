"""
模型管理数据模型
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class ModelInfo:
    """Whisper模型信息"""
    model_id: str           # 模型ID，例如: "large-v2"
    size_mb: int            # 模型大小(MB)
    status: str             # 状态: "not_downloaded", "downloading", "ready", "error"
    download_progress: float  # 下载进度 0-100
    local_path: Optional[str] = None  # 本地路径
    description: str = ""   # 模型描述

    def to_dict(self):
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "size_mb": self.size_mb,
            "status": self.status,
            "download_progress": self.download_progress,
            "local_path": str(self.local_path) if self.local_path else None,
            "description": self.description
        }


