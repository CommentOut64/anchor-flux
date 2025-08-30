"""
硬件检测相关数据模型
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging


@dataclass
class HardwareInfo:
    """核心硬件信息结构"""
    # GPU关键信息
    gpu_count: int = 0
    gpu_memory_mb: List[int] = field(default_factory=list)  # 每个GPU显存容量
    cuda_available: bool = False
    
    # CPU关键信息  
    cpu_cores: int = 1
    cpu_threads: int = 1
    
    # 内存关键信息
    memory_total_mb: int = 0
    memory_available_mb: int = 0
    
    # 存储关键信息
    temp_space_available_gb: int = 0
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "gpu": {
                "count": self.gpu_count,
                "memory_mb": self.gpu_memory_mb,
                "cuda_available": self.cuda_available,
                "total_memory_mb": sum(self.gpu_memory_mb) if self.gpu_memory_mb else 0
            },
            "cpu": {
                "cores": self.cpu_cores,
                "threads": self.cpu_threads
            },
            "memory": {
                "total_mb": self.memory_total_mb,
                "available_mb": self.memory_available_mb,
                "usage_percent": round((1 - self.memory_available_mb / max(1, self.memory_total_mb)) * 100, 1)
            },
            "storage": {
                "temp_space_gb": self.temp_space_available_gb
            }
        }


@dataclass
class OptimizationConfig:
    """基于硬件的优化配置"""
    # 转录优化配置
    batch_size: int = 16
    concurrency: int = 1
    use_memory_mapping: bool = False
    cpu_affinity_cores: List[int] = field(default_factory=list)
    
    # 推荐设备选择
    recommended_device: str = "cpu"
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "transcription": {
                "batch_size": self.batch_size,
                "concurrency": self.concurrency,
                "device": self.recommended_device
            },
            "system": {
                "use_memory_mapping": self.use_memory_mapping,
                "cpu_affinity_cores": self.cpu_affinity_cores
            }
        }