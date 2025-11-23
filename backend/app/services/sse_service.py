"""
统一SSE (Server-Sent Events) 管理服务
支持多频道、多客户端、线程安全的实时事件推送

使用场景：
1. 模型下载进度推送（频道：models）
2. 转录任务进度推送（频道：job:{job_id}）
3. 转录文字流式输出（频道：job:{job_id}）

核心原则：
- 单通道原则：每个资源只建立一个SSE连接
- 轻量推送原则：只推送小数据和信号，大数据用HTTP GET
- 重连即全量：断线重连后，推送initial_state，客户端全量拉取
- 线程安全：支持从后台线程广播消息到异步事件循环
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from fastapi import Request

logger = logging.getLogger(__name__)


class SSEManager:
    """统一SSE连接管理器"""

    def __init__(self, heartbeat_interval: int = 10, max_queue_size: int = 1000):
        """
        初始化SSE管理器

        Args:
            heartbeat_interval: 心跳间隔（秒）
            max_queue_size: 每个连接的消息队列最大容量
        """
        # 连接池：{channel_id: [queue1, queue2, ...]}
        self.connections: Dict[str, List[asyncio.Queue]] = defaultdict(list)

        # 配置
        self.heartbeat_interval = heartbeat_interval
        self.max_queue_size = max_queue_size

        # 统计信息
        self.total_connections = 0
        self.total_messages_sent = 0

        # 主事件循环引用（在应用启动时设置）
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(f"✅ SSE管理器已初始化 (心跳: {heartbeat_interval}s, 队列: {max_queue_size})")

    async def subscribe(
        self,
        channel_id: str,
        request: Request,
        initial_state_callback: Optional[Callable] = None
    ):
        """
        订阅指定频道的SSE事件流

        Args:
            channel_id: 频道ID（如 "models", "job:abc123"）
            request: FastAPI请求对象
            initial_state_callback: 初始状态回调函数（可选，用于推送initial_state）

        Yields:
            SSE格式的消息字符串
        """
        # 创建此连接的专用队列
        event_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.connections[channel_id].append(event_queue)
        self.total_connections += 1

        connection_id = f"{channel_id}#{len(self.connections[channel_id])}"

        try:
            logger.info(f"✅ SSE连接已建立: {connection_id} (频道: {channel_id}, 总连接: {self._get_total_active_connections()})")

            # 1. 发送连接成功消息
            yield self._format_sse("connected", {
                "channel_id": channel_id,
                "message": "SSE连接已建立",
                "timestamp": time.time()
            })

            # 2. 发送初始状态（如果提供了回调）
            if initial_state_callback:
                try:
                    initial_state = initial_state_callback()
                    if initial_state:
                        yield self._format_sse("initial_state", initial_state)
                        logger.debug(f"📤 发送初始状态: {connection_id}")
                except Exception as e:
                    logger.error(f"获取初始状态失败: {e}")

            # 3. 持续推送消息
            heartbeat_counter = 0
            while True:
                # 检查客户端是否断开
                if await request.is_disconnected():
                    logger.info(f"⚠️ 客户端已断开: {connection_id}")
                    break

                try:
                    # 等待新消息（超时后发送心跳）
                    message = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=self.heartbeat_interval
                    )

                    # 发送事件
                    logger.info(f"📨 subscribe取到消息，准备yield: {channel_id}/{message['event']}")
                    formatted = self._format_sse(message["event"], message["data"])
                    logger.info(f"📨 yield消息: {formatted[:100]}...")
                    yield formatted
                    self.total_messages_sent += 1

                except asyncio.TimeoutError:
                    # 超时，发送心跳
                    heartbeat_counter += 1
                    yield self._format_sse("ping", {
                        "timestamp": time.time(),
                        "count": heartbeat_counter
                    })

        except asyncio.CancelledError:
            logger.info(f"🔌 SSE连接被取消: {connection_id}")
        except Exception as e:
            logger.error(f"❌ SSE错误: {connection_id} - {e}")
        finally:
            # 清理连接
            try:
                self.connections[channel_id].remove(event_queue)
                if not self.connections[channel_id]:
                    del self.connections[channel_id]
                logger.info(f"🔌 SSE连接已断开: {connection_id} (剩余连接: {self._get_total_active_connections()})")
            except (ValueError, KeyError):
                pass

    async def broadcast(self, channel_id: str, event: str, data: dict):
        """
        向指定频道的所有订阅者广播消息（异步安全）

        Args:
            channel_id: 频道ID
            event: 事件类型（如 "progress", "fragment", "signal"）
            data: 事件数据（字典）
        """
        logger.info(f"📥 broadcast被调用: {channel_id}/{event}, 连接数: {len(self.connections.get(channel_id, []))}")

        if channel_id not in self.connections:
            logger.warning(f"频道无连接，跳过广播: {channel_id}")
            return

        message = {
            "event": event,
            "data": data
        }

        success_count = 0
        failed_count = 0

        for queue in self.connections[channel_id][:]:  # 使用切片避免遍历时修改
            try:
                # 检查队列容量，避免阻塞
                if queue.qsize() >= self.max_queue_size * 0.95:
                    # 队列接近满，跳过此次更新
                    failed_count += 1
                    logger.warning(f"队列已满，跳过更新: {channel_id}")
                    continue

                # 非阻塞放入队列
                queue.put_nowait(message)
                success_count += 1
                logger.info(f"✅ 消息已放入队列: {channel_id}/{event}, 队列大小: {queue.qsize()}")

            except asyncio.QueueFull:
                failed_count += 1
                logger.warning(f"队列满，放入失败: {channel_id}")
            except Exception as e:
                failed_count += 1
                logger.error(f"广播失败: {channel_id} - {e}")

        if success_count > 0:
            logger.info(f"📤 广播完成: {channel_id} - {event} (成功: {success_count}, 失败: {failed_count})")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """
        设置主事件循环引用（在应用启动时调用）

        Args:
            loop: uvicorn/FastAPI 的主事件循环
        """
        self.loop = loop
        logger.info("✅ SSE管理器已绑定主事件循环")

    def broadcast_sync(self, channel_id: str, event: str, data: dict):
        """
        从同步上下文（后台线程）广播消息（线程安全）

        用于从后台线程（如转录任务、模型下载任务）向SSE推送消息

        Args:
            channel_id: 频道ID
            event: 事件类型
            data: 事件数据

        注意：此方法依赖于在应用启动时设置的主事件循环
        """
        # 优先使用预先保存的主事件循环
        loop = self.loop

        if loop is None:
            logger.warning(f"SSE主事件循环未设置，跳过推送: {channel_id}/{event}")
            # 回退：尝试获取事件循环（可能不可靠）
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        logger.warning("事件循环已关闭，无法推送SSE消息")
                        return
                except Exception:
                    logger.warning("无法获取事件循环，跳过SSE推送")
                    return

        if loop is None or loop.is_closed():
            logger.warning("事件循环不可用，跳过SSE推送")
            return

        # 检查频道是否有连接
        if channel_id not in self.connections or not self.connections[channel_id]:
            logger.info(f"频道无活跃连接，跳过推送: {channel_id}")
            return

        # 使用 run_coroutine_threadsafe 从后台线程安全地调度协程
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.broadcast(channel_id, event, data),
                loop
            )
            # 记录成功调度
            logger.info(f"📤 SSE消息已调度: {channel_id}/{event}")
        except Exception as e:
            logger.warning(f"SSE推送调度失败: {e}")

    def _format_sse(self, event: str, data: dict) -> str:
        """
        格式化为标准SSE消息格式

        Args:
            event: 事件名称
            data: 数据字典

        Returns:
            SSE格式字符串: "event: xxx\ndata: {...}\n\n"
        """
        json_data = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {json_data}\n\n"

    def _get_total_active_connections(self) -> int:
        """获取当前活跃连接总数"""
        return sum(len(queues) for queues in self.connections.values())

    def get_channel_stats(self, channel_id: str) -> Dict:
        """
        获取指定频道的统计信息

        Args:
            channel_id: 频道ID

        Returns:
            统计信息字典
        """
        if channel_id not in self.connections:
            return {
                "channel_id": channel_id,
                "active_connections": 0,
                "exists": False
            }

        queues = self.connections[channel_id]
        return {
            "channel_id": channel_id,
            "active_connections": len(queues),
            "queue_sizes": [q.qsize() for q in queues],
            "exists": True
        }

    def get_global_stats(self) -> Dict:
        """
        获取全局统计信息

        Returns:
            全局统计字典
        """
        return {
            "total_channels": len(self.connections),
            "total_connections": self._get_total_active_connections(),
            "total_messages_sent": self.total_messages_sent,
            "channels": list(self.connections.keys())
        }


# ========== 单例模式 ==========

_sse_manager_instance: Optional[SSEManager] = None


def get_sse_manager() -> SSEManager:
    """
    获取SSE管理器单例

    Returns:
        SSEManager: SSE管理器实例
    """
    global _sse_manager_instance
    if _sse_manager_instance is None:
        # 从配置中读取参数（如果需要）
        from core.config import config
        heartbeat_interval = getattr(config, 'SSE_HEARTBEAT_INTERVAL', 10)
        max_queue_size = getattr(config, 'SSE_MAX_QUEUE_SIZE', 1000)

        _sse_manager_instance = SSEManager(
            heartbeat_interval=heartbeat_interval,
            max_queue_size=max_queue_size
        )
    return _sse_manager_instance
