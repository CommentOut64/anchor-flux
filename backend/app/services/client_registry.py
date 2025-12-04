"""
客户端注册表服务
用于追踪活跃的浏览器标签页，实现标签页重用机制
"""

import time
import threading
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClientInfo:
    """客户端信息"""
    client_id: str
    last_heartbeat: float
    user_agent: Optional[str] = None
    registered_at: float = None

    def __post_init__(self):
        if self.registered_at is None:
            self.registered_at = time.time()


class ClientRegistry:
    """客户端注册表，追踪活跃的浏览器标签页"""

    def __init__(self, heartbeat_timeout: int = 15):
        """
        初始化客户端注册表

        Args:
            heartbeat_timeout: 心跳超时时间（秒），超过此时间无心跳视为断开
        """
        self._clients: Dict[str, ClientInfo] = {}
        self._lock = threading.RLock()
        self.heartbeat_timeout = heartbeat_timeout
        logger.info(f"客户端注册表已初始化 (心跳超时: {heartbeat_timeout}秒)")

    def register(self, client_id: str, user_agent: str = None) -> bool:
        """
        注册新客户端

        Args:
            client_id: 客户端唯一标识
            user_agent: 客户端User-Agent字符串

        Returns:
            是否注册成功
        """
        with self._lock:
            self._clients[client_id] = ClientInfo(
                client_id=client_id,
                last_heartbeat=time.time(),
                user_agent=user_agent
            )
            logger.info(f"客户端已注册: {client_id}")
            return True

    def heartbeat(self, client_id: str) -> bool:
        """
        更新客户端心跳时间

        Args:
            client_id: 客户端唯一标识

        Returns:
            是否更新成功（如果客户端不存在则返回False）
        """
        with self._lock:
            if client_id in self._clients:
                self._clients[client_id].last_heartbeat = time.time()
                return True
            return False

    def unregister(self, client_id: str):
        """
        注销客户端

        Args:
            client_id: 客户端唯一标识
        """
        with self._lock:
            if client_id in self._clients:
                self._clients.pop(client_id)
                logger.info(f"客户端已注销: {client_id}")

    def has_active_clients(self) -> bool:
        """
        检查是否有活跃的客户端

        Returns:
            是否有活跃客户端
        """
        self._cleanup_stale_clients()
        with self._lock:
            return len(self._clients) > 0

    def get_active_count(self) -> int:
        """
        获取活跃客户端数量

        Returns:
            活跃客户端数量
        """
        self._cleanup_stale_clients()
        with self._lock:
            return len(self._clients)

    def get_all_clients(self) -> Dict[str, ClientInfo]:
        """
        获取所有活跃客户端信息

        Returns:
            客户端信息字典
        """
        self._cleanup_stale_clients()
        with self._lock:
            return self._clients.copy()

    def _cleanup_stale_clients(self):
        """清理过期的客户端"""
        now = time.time()
        with self._lock:
            stale = [
                cid for cid, info in self._clients.items()
                if now - info.last_heartbeat > self.heartbeat_timeout
            ]
            for cid in stale:
                logger.info(f"客户端心跳超时，自动注销: {cid}")
                del self._clients[cid]


# 全局单例
_client_registry: Optional[ClientRegistry] = None
_registry_lock = threading.Lock()


def get_client_registry() -> ClientRegistry:
    """
    获取客户端注册表单例

    Returns:
        ClientRegistry实例
    """
    global _client_registry
    if _client_registry is None:
        with _registry_lock:
            if _client_registry is None:
                _client_registry = ClientRegistry()
    return _client_registry
