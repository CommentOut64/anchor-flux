"""
实时硬件监测服务

提供 GPU/CPU/内存的实时监测和预警功能
"""
import logging
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


class ResourceLevel(Enum):
    """资源使用等级"""
    NORMAL = "normal"       # 正常 < 70%
    WARNING = "warning"     # 警告 70-85%
    CRITICAL = "critical"   # 危险 > 85%


@dataclass
class GPUStatus:
    """GPU 状态"""
    device_id: int = 0
    name: str = "Unknown"
    total_memory_mb: int = 0
    used_memory_mb: int = 0
    free_memory_mb: int = 0
    utilization_percent: float = 0.0
    temperature: Optional[float] = None

    @property
    def memory_usage_percent(self) -> float:
        if self.total_memory_mb == 0:
            return 0.0
        return (self.used_memory_mb / self.total_memory_mb) * 100

    @property
    def level(self) -> ResourceLevel:
        usage = self.memory_usage_percent
        if usage > 85:
            return ResourceLevel.CRITICAL
        elif usage > 70:
            return ResourceLevel.WARNING
        return ResourceLevel.NORMAL

    def to_dict(self) -> Dict:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "total_memory_mb": self.total_memory_mb,
            "used_memory_mb": self.used_memory_mb,
            "free_memory_mb": self.free_memory_mb,
            "memory_usage_percent": round(self.memory_usage_percent, 1),
            "utilization_percent": round(self.utilization_percent, 1),
            "temperature": self.temperature,
            "level": self.level.value
        }


@dataclass
class CPUStatus:
    """CPU 状态"""
    usage_percent: float = 0.0
    core_count: int = 1
    thread_count: int = 1
    frequency_mhz: Optional[float] = None
    per_core_usage: List[float] = field(default_factory=list)

    @property
    def level(self) -> ResourceLevel:
        if self.usage_percent > 90:
            return ResourceLevel.CRITICAL
        elif self.usage_percent > 75:
            return ResourceLevel.WARNING
        return ResourceLevel.NORMAL

    def to_dict(self) -> Dict:
        return {
            "usage_percent": round(self.usage_percent, 1),
            "core_count": self.core_count,
            "thread_count": self.thread_count,
            "frequency_mhz": self.frequency_mhz,
            "per_core_usage": [round(u, 1) for u in self.per_core_usage],
            "level": self.level.value
        }


@dataclass
class MemoryStatus:
    """内存状态"""
    total_mb: int = 0
    used_mb: int = 0
    available_mb: int = 0

    @property
    def usage_percent(self) -> float:
        if self.total_mb == 0:
            return 0.0
        return ((self.total_mb - self.available_mb) / self.total_mb) * 100

    @property
    def level(self) -> ResourceLevel:
        usage = self.usage_percent
        if usage > 90:
            return ResourceLevel.CRITICAL
        elif usage > 80:
            return ResourceLevel.WARNING
        return ResourceLevel.NORMAL

    def to_dict(self) -> Dict:
        return {
            "total_mb": self.total_mb,
            "used_mb": self.used_mb,
            "available_mb": self.available_mb,
            "usage_percent": round(self.usage_percent, 1),
            "level": self.level.value
        }


@dataclass
class SystemStatus:
    """系统整体状态"""
    timestamp: float = 0.0
    gpu: Optional[GPUStatus] = None
    cpu: CPUStatus = field(default_factory=CPUStatus)
    memory: MemoryStatus = field(default_factory=MemoryStatus)

    @property
    def overall_level(self) -> ResourceLevel:
        """获取整体资源等级（取最严重的）"""
        levels = [self.cpu.level, self.memory.level]
        if self.gpu:
            levels.append(self.gpu.level)

        if ResourceLevel.CRITICAL in levels:
            return ResourceLevel.CRITICAL
        elif ResourceLevel.WARNING in levels:
            return ResourceLevel.WARNING
        return ResourceLevel.NORMAL

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "gpu": self.gpu.to_dict() if self.gpu else None,
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "overall_level": self.overall_level.value
        }


class HardwareMonitor:
    """
    实时硬件监测器

    功能:
    - 实时监测 GPU/CPU/内存使用情况
    - 支持周期性采样
    - 支持阈值预警回调
    - 提供历史数据查询
    """

    def __init__(
        self,
        sample_interval: float = 1.0,
        history_size: int = 60,
        warning_callback: Optional[Callable[[SystemStatus], None]] = None
    ):
        """
        初始化硬件监测器

        Args:
            sample_interval: 采样间隔（秒）
            history_size: 历史记录数量
            warning_callback: 预警回调函数
        """
        self.logger = logging.getLogger(__name__)
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.warning_callback = warning_callback

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._history: List[SystemStatus] = []
        self._last_status: Optional[SystemStatus] = None

        # GPU 设备信息缓存
        self._gpu_count = 0
        self._gpu_names: List[str] = []
        self._gpu_total_memory: List[int] = []

        self._init_gpu_info()

    def _init_gpu_info(self):
        """初始化 GPU 信息"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch 未安装，GPU 监测不可用")
            return

        try:
            if torch.cuda.is_available():
                self._gpu_count = torch.cuda.device_count()
                for i in range(self._gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self._gpu_names.append(props.name)
                    self._gpu_total_memory.append(props.total_memory // (1024 * 1024))
                self.logger.info(f"检测到 {self._gpu_count} 个 GPU")
        except Exception as e:
            self.logger.error(f"初始化 GPU 信息失败: {e}")

    def get_gpu_status(self, device_id: int = 0) -> Optional[GPUStatus]:
        """获取 GPU 状态"""
        if not TORCH_AVAILABLE or device_id >= self._gpu_count:
            return None

        try:
            # 获取显存使用情况
            memory_allocated = torch.cuda.memory_allocated(device_id) // (1024 * 1024)
            memory_reserved = torch.cuda.memory_reserved(device_id) // (1024 * 1024)
            total_memory = self._gpu_total_memory[device_id]
            free_memory = total_memory - memory_reserved

            return GPUStatus(
                device_id=device_id,
                name=self._gpu_names[device_id],
                total_memory_mb=total_memory,
                used_memory_mb=memory_reserved,
                free_memory_mb=free_memory,
                utilization_percent=0.0,  # 需要 nvidia-smi 或 pynvml
                temperature=None
            )
        except Exception as e:
            self.logger.error(f"获取 GPU {device_id} 状态失败: {e}")
            return None

    def get_cpu_status(self) -> CPUStatus:
        """获取 CPU 状态"""
        if not PSUTIL_AVAILABLE:
            return CPUStatus()

        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            freq = psutil.cpu_freq()

            return CPUStatus(
                usage_percent=cpu_percent,
                core_count=psutil.cpu_count(logical=False) or 1,
                thread_count=psutil.cpu_count(logical=True) or 1,
                frequency_mhz=freq.current if freq else None,
                per_core_usage=per_core or []
            )
        except Exception as e:
            self.logger.error(f"获取 CPU 状态失败: {e}")
            return CPUStatus()

    def get_memory_status(self) -> MemoryStatus:
        """获取内存状态"""
        if not PSUTIL_AVAILABLE:
            return MemoryStatus()

        try:
            mem = psutil.virtual_memory()
            return MemoryStatus(
                total_mb=mem.total // (1024 * 1024),
                used_mb=mem.used // (1024 * 1024),
                available_mb=mem.available // (1024 * 1024)
            )
        except Exception as e:
            self.logger.error(f"获取内存状态失败: {e}")
            return MemoryStatus()

    def get_current_status(self) -> SystemStatus:
        """获取当前系统状态"""
        gpu_status = self.get_gpu_status(0) if self._gpu_count > 0 else None

        status = SystemStatus(
            timestamp=time.time(),
            gpu=gpu_status,
            cpu=self.get_cpu_status(),
            memory=self.get_memory_status()
        )

        self._last_status = status
        return status

    def get_available_vram(self, device_id: int = 0) -> int:
        """
        获取可用显存（MB）

        Args:
            device_id: GPU 设备 ID

        Returns:
            可用显存（MB），无 GPU 返回 0
        """
        gpu_status = self.get_gpu_status(device_id)
        if gpu_status:
            return gpu_status.free_memory_mb
        return 0

    def get_total_vram(self, device_id: int = 0) -> int:
        """获取总显存（MB）"""
        if device_id < len(self._gpu_total_memory):
            return self._gpu_total_memory[device_id]
        return 0

    async def start(self):
        """启动周期性监测"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        self.logger.info(f"硬件监测已启动，采样间隔: {self.sample_interval}s")

    async def stop(self):
        """停止监测"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("硬件监测已停止")

    async def _monitor_loop(self):
        """监测循环"""
        while self._running:
            try:
                status = self.get_current_status()

                # 添加到历史记录
                self._history.append(status)
                if len(self._history) > self.history_size:
                    self._history.pop(0)

                # 检查预警
                if status.overall_level != ResourceLevel.NORMAL:
                    if self.warning_callback:
                        self.warning_callback(status)

                await asyncio.sleep(self.sample_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"监测循环异常: {e}")
                await asyncio.sleep(self.sample_interval)

    def get_history(self, count: Optional[int] = None) -> List[SystemStatus]:
        """获取历史记录"""
        if count is None:
            return self._history.copy()
        return self._history[-count:]

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_gpu(self) -> bool:
        return self._gpu_count > 0

    @property
    def gpu_count(self) -> int:
        return self._gpu_count


# 单例实例
_monitor_instance: Optional[HardwareMonitor] = None


def get_hardware_monitor() -> HardwareMonitor:
    """获取硬件监测器单例"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = HardwareMonitor()
    return _monitor_instance
