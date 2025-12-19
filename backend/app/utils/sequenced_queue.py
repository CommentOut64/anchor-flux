"""
智能序列化异步队列

实现乱序放入、顺序取出的队列机制，支持失败跳号、超时处理和缓冲区背压控制。
用于AsyncDualPipeline的FastWorker并发处理场景。
"""

import asyncio
import time
import logging
from typing import Dict, Generic, TypeVar, Optional, Set
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class QueueStatistics:
    """队列统计信息"""
    buffer_size: int
    next_expected: int
    failed_count: int
    inner_queue_size: int
    total_processed: int
    max_buffer_seen: int


class SequencedAsyncQueue(Generic[T]):
    """
    智能序列化异步队列

    功能：
    - 乱序放入，顺序取出：无论上游完成顺序如何，保证下游接收到的是严格按index排序的数据
    - 失败跳号：通过mark_failed()标记失败的index，自动跳过不阻塞
    - 超时处理：长时间等待的index自动标记失败
    - 缓冲区背压：限制缓冲区大小，满时阻塞put操作

    使用场景：
    AsyncDualPipeline中FastWorker并发处理chunk，但SlowWorker需要按顺序接收
    """

    def __init__(
        self,
        maxsize: int = 0,
        max_buffer_size: int = 100,
        timeout_seconds: float = 60.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化队列

        Args:
            maxsize: 内部队列最大大小（0表示无限制）
            max_buffer_size: 乱序缓冲区最大大小
            timeout_seconds: 单个index的超时时间（秒）
            logger: 日志记录器
        """
        self._inner_queue = asyncio.Queue(maxsize=maxsize)
        self._buffer: Dict[int, T] = {}  # 乱序暂存区
        self._next_expected_index = 0  # 下一个期望的index
        self._failed_indices: Set[int] = set()  # 失败的index集合
        self._put_timestamps: Dict[int, float] = {}  # 记录put时间

        self._max_buffer_size = max_buffer_size
        self._timeout_seconds = timeout_seconds
        self._total_expected: Optional[int] = None  # 总共期望的元素数量

        self._lock = asyncio.Lock()  # 保护内部状态的锁
        self._total_processed = 0  # 已处理的元素总数
        self._max_buffer_seen = 0  # 观察到的最大缓冲区大小

        self.logger = logger or logging.getLogger(__name__)

    def set_total(self, total: int):
        """
        设置总元素数量

        Args:
            total: 总共期望的元素数量
        """
        self._total_expected = total
        self.logger.debug(f"设置总元素数量: {total}")

    async def put(self, item: T):
        """
        放入元素（乱序）

        Args:
            item: 要放入的元素，必须有chunk_index属性

        Raises:
            ValueError: 如果item没有chunk_index属性
        """
        # 获取index（不需要锁）
        index = getattr(item, 'chunk_index', None)
        if index is None:
            raise ValueError("item must have chunk_index attribute")

        # 背压控制：缓冲区满时等待（在锁外等待，避免死锁）
        while True:
            async with self._lock:
                if not self._is_buffer_full():
                    # 存入缓冲区
                    self._buffer[index] = item
                    self._put_timestamps[index] = time.time()

                    # 更新统计
                    self._max_buffer_seen = max(self._max_buffer_seen, len(self._buffer))

                    self.logger.debug(f"放入 index={index}，缓冲区大小={len(self._buffer)}")

                    # 尝试刷新到内部队列
                    await self._try_flush()
                    return

            # 缓冲区满，释放锁后等待
            self.logger.warning(
                f"缓冲区已满 ({self._max_buffer_size})，等待..."
            )
            await asyncio.sleep(0.1)

    async def put_direct(self, item: T, is_end_signal: bool = True):
        """
        直接放入内部队列（不经过排序）

        用于特殊场景，如发送结束信号。

        当 is_end_signal=True 时（默认）：
        - 先尝试正常flush
        - 如果buffer仍非空（说明有chunk丢失），强制将剩余数据按index排序全部吐出
        - 这样可以避免死锁，同时尽可能保留已到达的数据

        死锁预防原理：
        - 场景: Chunk 0,1 到了，Chunk 2 丢了，Chunk 3 在 Buffer 里
        - 旧方案: put_direct 会无限等待 Chunk 2，导致死锁
        - 新方案: 强制将 Chunk 3 吐出，然后放入结束信号

        Args:
            item: 要放入的元素
            is_end_signal: 是否是结束信号（默认True，会触发强制Flush）
        """
        async with self._lock:
            if is_end_signal:
                # 首先尝试正常flush（处理已经可以按顺序输出的数据）
                await self._try_flush()

                # 检查buffer是否仍有剩余数据
                if self._buffer:
                    # 上游说结束了，但 Buffer 里还有数据
                    # 说明中间有 Chunk 丢了，或者逻辑错位
                    # 策略：将剩余数据全部强制按 Key 排序发出，不再等待缺失的序号
                    orphaned_indices = sorted(self._buffer.keys())
                    self.logger.warning(
                        f"收到结束信号时 Buffer 非空 (len={len(self._buffer)})，"
                        f"强制Flush剩余数据: {orphaned_indices}，"
                        f"缺失的序号: {self._next_expected_index} 到 {min(orphaned_indices)-1 if orphaned_indices else '无'}"
                    )

                    for idx in orphaned_indices:
                        orphaned_item = self._buffer.pop(idx)
                        self._put_timestamps.pop(idx, None)
                        # 使用put_nowait避免在锁内阻塞
                        # 如果inner_queue满了，这里会抛异常，但这种情况不应该发生
                        # 因为结束信号意味着不会有新数据了
                        try:
                            self._inner_queue.put_nowait(orphaned_item)
                            self._total_processed += 1
                            self.logger.info(f"强制输出孤儿 Chunk index={idx}")
                        except asyncio.QueueFull:
                            # inner_queue满了，等待后重试
                            # 释放锁后等待，避免死锁
                            self.logger.warning(f"inner_queue满，等待后重试放入 index={idx}")
                            # 把数据放回buffer，稍后在锁外处理
                            self._buffer[idx] = orphaned_item

                    # 如果还有数据因为队列满没放进去，在锁外处理
                    if self._buffer:
                        # 需要在锁外处理，先释放锁
                        pass

            # 更新 next_expected_index（虽然此时已无意义，但保持一致性）
            if self._buffer:
                # 还有数据没放进去，需要在锁外处理
                remaining = dict(self._buffer)
                self._buffer.clear()
            else:
                remaining = {}

        # 在锁外处理因队列满而未能放入的数据
        for idx in sorted(remaining.keys()):
            orphaned_item = remaining[idx]
            await self._inner_queue.put(orphaned_item)  # 使用await版本，会等待
            self._total_processed += 1
            self.logger.info(f"(锁外)强制输出孤儿 Chunk index={idx}")

        # 最后放入结束信号
        await self._inner_queue.put(item)
        self.logger.debug(
            f"直接放入元素（is_end_signal={is_end_signal}）"
        )

    async def mark_failed(self, index: int, reason: str = ""):
        """
        标记某个index为失败（跳号）

        Args:
            index: 失败的index
            reason: 失败原因
        """
        async with self._lock:
            self._failed_indices.add(index)
            self.logger.warning(f"标记 index={index} 为失败: {reason}")

            # 尝试刷新
            await self._try_flush()

    async def _try_flush(self):
        """
        尝试将连续的元素推送到内部队列

        核心逻辑：
        1. 如果下一个期望的index在buffer中，取出并放入inner_queue
        2. 如果下一个期望的index在failed_indices中，跳过
        3. 否则，停止刷新

        注意：使用put_nowait避免在持有锁时阻塞
        """
        while True:
            if self._next_expected_index in self._buffer:
                # 检查inner_queue是否有空间
                if self._inner_queue.full():
                    # inner_queue满了，停止刷新，等待消费者取走数据
                    break

                # 找到了期望的index
                item = self._buffer.pop(self._next_expected_index)
                self._put_timestamps.pop(self._next_expected_index, None)

                # 放入内部队列（使用put_nowait避免阻塞）
                self._inner_queue.put_nowait(item)

                self._total_processed += 1
                self.logger.debug(
                    f"刷新 index={self._next_expected_index} 到内部队列，"
                    f"已处理={self._total_processed}"
                )

                self._next_expected_index += 1

            elif self._next_expected_index in self._failed_indices:
                # 跳过失败的index
                self._failed_indices.discard(self._next_expected_index)
                self.logger.info(f"跳过失败的 index={self._next_expected_index}")
                self._next_expected_index += 1

            else:
                # 下一个期望的index既不在buffer中，也不在failed_indices中
                # 停止刷新
                break

    async def _check_timeout(self):
        """
        检查并处理超时的index

        如果期望的index超过timeout_seconds未到达，标记为失败
        """
        now = time.time()

        # 如果期望的index已经在buffer中，不需要超时检查
        if self._next_expected_index in self._buffer:
            return

        # 检查是否有旧的任务在buffer中但期望的index一直不来
        if self._put_timestamps:
            oldest_index = min(self._put_timestamps.keys())
            oldest_time = self._put_timestamps[oldest_index]
            elapsed = now - oldest_time

            # 如果最老的任务等待时间超过2倍timeout，说明可能期望的index卡住了
            if elapsed > self._timeout_seconds * 2:
                self.logger.warning(
                    f"等待 index={self._next_expected_index} 超时 ({elapsed:.1f}s)，"
                    f"最老的buffer index={oldest_index}，标记为失败"
                )
                await self.mark_failed(self._next_expected_index, "timeout")

    async def get(self, timeout: Optional[float] = None) -> T:
        """
        获取下一个有序元素

        Args:
            timeout: 超时时间（秒），None表示使用默认超时

        Returns:
            下一个元素

        Raises:
            asyncio.TimeoutError: 超时
        """
        timeout = timeout or self._timeout_seconds
        return await asyncio.wait_for(
            self._inner_queue.get(),
            timeout=timeout
        )

    def _is_buffer_full(self) -> bool:
        """
        检查缓冲区是否已满

        Returns:
            True表示缓冲区已满
        """
        return len(self._buffer) >= self._max_buffer_size

    def buffer_size(self) -> int:
        """
        获取当前缓冲区大小

        Returns:
            缓冲区中的元素数量
        """
        return len(self._buffer)

    def inner_queue_size(self) -> int:
        """
        获取内部队列大小

        Returns:
            内部队列中的元素数量
        """
        return self._inner_queue.qsize()

    def is_complete(self) -> bool:
        """
        检查是否已完成所有处理

        Returns:
            True表示所有元素都已处理或标记失败
        """
        if self._total_expected is None:
            return False

        total_handled = self._total_processed + len(self._failed_indices)
        return total_handled >= self._total_expected

    def get_statistics(self) -> QueueStatistics:
        """
        获取队列统计信息

        Returns:
            统计信息对象
        """
        return QueueStatistics(
            buffer_size=len(self._buffer),
            next_expected=self._next_expected_index,
            failed_count=len(self._failed_indices),
            inner_queue_size=self._inner_queue.qsize(),
            total_processed=self._total_processed,
            max_buffer_seen=self._max_buffer_seen
        )
