"""
AsyncDualPipeline - 三级异步流水线控制器

核心架构：
    AudioChunk → [FastWorker (CPU)]
                   ↓ Queue1 (maxsize=5)
                 [SlowWorker (GPU)]
                   ↓ Queue2 (maxsize=5)
                 [AlignmentWorker (CPU)]
                   ↓ 完成

设计决策：
- 生产者-消费者模型：数据单向流动
- 队列背压：asyncio.Queue(maxsize=5) 防止内存溢出
- 错位并行：当 SlowWorker 处理 Chunk N 时，FastWorker 同时处理 Chunk N+1
- 异常传播：任何 Worker 的异常都会传播到 run() 方法
- 结束信号：使用 ProcessingContext.is_end 通知下游停止
"""
import asyncio
import logging
from typing import List, Optional, Any

from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.services.sse_service import get_sse_manager
from app.pipelines.workers import FastWorker, SlowWorker, AlignmentWorker


class AsyncDualPipeline:
    """
    三级异步流水线控制器

    职责：
    1. 编排三个 Worker 的生命周期
    2. 管理队列和背压
    3. 处理异常传播
    4. 推送 SSE 事件
    """

    def __init__(
        self,
        job_id: str,
        queue_maxsize: int = 5,
        sensevoice_language: str = "auto",
        whisper_language: str = "auto",
        user_glossary: Optional[list] = None,
        enable_semantic_grouping: bool = True,
        alignment_score_threshold: float = 0.3,
        enable_fallback: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化流水线

        Args:
            job_id: 任务 ID
            queue_maxsize: 队列最大长度（背压控制）
            sensevoice_language: SenseVoice 语言设置
            whisper_language: Whisper 语言设置
            user_glossary: 用户词表
            enable_semantic_grouping: 是否启用语义分组
            alignment_score_threshold: 对齐质量阈值
            enable_fallback: 是否启用降级策略
            logger: 日志记录器
        """
        self.job_id = job_id
        self.logger = logger or logging.getLogger(__name__)

        # 创建队列（带背压）
        self.queue_inter = asyncio.Queue(maxsize=queue_maxsize)  # FastWorker → SlowWorker
        self.queue_final = asyncio.Queue(maxsize=queue_maxsize)  # SlowWorker → AlignmentWorker

        # 实例化三个 Worker
        self.fast_worker = FastWorker(
            job_id=job_id,
            sensevoice_language=sensevoice_language,
            enable_semantic_grouping=enable_semantic_grouping,
            logger=self.logger
        )

        self.slow_worker = SlowWorker(
            whisper_language=whisper_language,
            user_glossary=user_glossary,
            logger=self.logger
        )

        self.alignment_worker = AlignmentWorker(
            job_id=job_id,
            enable_semantic_grouping=enable_semantic_grouping,
            alignment_score_threshold=alignment_score_threshold,
            enable_fallback=enable_fallback,
            logger=self.logger
        )

        # 获取 SSE 管理器
        self.sse_manager = get_sse_manager()

        # 错误收集
        self.errors: List[Exception] = []

    async def run(
        self,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000
    ) -> List[ProcessingContext]:
        """
        运行三级流水线

        流程：
        1. 启动三个并行任务（FastWorker, SlowWorker, AlignmentWorker）
        2. FastWorker 遍历 audio_chunks，每个 chunk 包装为 ProcessingContext
        3. 数据通过两个队列单向流动
        4. 等待所有任务完成
        5. 检查异常

        Args:
            audio_chunks: AudioChunk 列表
            full_audio_array: 完整音频数组（用于 Audio Overlap）
            full_audio_sr: 完整音频采样率

        Returns:
            List[ProcessingContext]: 处理结果列表
        """
        self.logger.info(f"开始三级流水线: {len(audio_chunks)} 个 Chunk")

        # 存储结果
        results: List[ProcessingContext] = []

        # 启动三个并行任务
        task_fast = asyncio.create_task(
            self._fast_loop(audio_chunks, full_audio_array, full_audio_sr)
        )
        task_slow = asyncio.create_task(self._slow_loop())
        task_align = asyncio.create_task(self._align_loop(results))

        # 等待所有任务结束（return_exceptions=True 确保一个挂了不会立刻抛出）
        await_results = await asyncio.gather(
            task_fast, task_slow, task_align,
            return_exceptions=True
        )

        # 检查是否有异常
        for res in await_results:
            if isinstance(res, Exception):
                self.logger.error(f"任务异常: {res}", exc_info=res)
                raise res

        # 检查收集的错误
        if self.errors:
            self.logger.error(f"流水线执行中发生 {len(self.errors)} 个错误")
            raise self.errors[0]

        self.logger.info(f"三级流水线完成: {len(results)} 个 Chunk 已处理")

        return results

    async def _fast_loop(
        self,
        chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000
    ):
        """
        FastWorker 循环（生产者）

        职责：
        1. 遍历所有 audio_chunks
        2. 每个 chunk 包装为 ProcessingContext
        3. 调用 FastWorker.process()
        4. 将 context 放入 queue_inter
        5. 发送结束信号

        Args:
            chunks: AudioChunk 列表
            full_audio_array: 完整音频数组（用于 Audio Overlap）
            full_audio_sr: 完整音频采样率
        """
        try:
            for i, chunk in enumerate(chunks):
                # 创建处理上下文（包含完整音频数组）
                ctx = ProcessingContext(
                    job_id=self.job_id,
                    chunk_index=i,
                    audio_chunk=chunk,
                    full_audio_array=full_audio_array,
                    full_audio_sr=full_audio_sr
                )

                # FastWorker 处理（SenseVoice 推理 + 分句 + 推送草稿）
                await self.fast_worker.process(ctx)

                # 放入队列（如果队列满了，会自动阻塞，实现背压）
                await self.queue_inter.put(ctx)

            # 发送结束信号
            end_ctx = ProcessingContext(
                job_id=self.job_id,
                chunk_index=-1,
                audio_chunk=None,
                is_end=True
            )
            await self.queue_inter.put(end_ctx)

            self.logger.info("FastWorker 循环完成")

        except Exception as e:
            self.logger.error(f"FastWorker 循环异常: {e}", exc_info=True)
            self.errors.append(e)

            # 发送错误信号
            error_ctx = ProcessingContext(
                job_id=self.job_id,
                chunk_index=-1,
                audio_chunk=None,
                is_end=True,
                error=e
            )
            await self.queue_inter.put(error_ctx)

    async def _slow_loop(self):
        """
        SlowWorker 循环（中间消费者-生产者）

        职责：
        1. 从 queue_inter 取 context
        2. 调用 SlowWorker.process()
        3. 将 context 放入 queue_final
        4. 透传结束/错误信号

        """
        try:
            while True:
                # 从队列取 context
                ctx = await self.queue_inter.get()

                # 检查结束信号或错误
                if ctx.is_end or ctx.error:
                    await self.queue_final.put(ctx)  # 透传
                    break

                # SlowWorker 处理（Whisper 推理 + 幻觉检测）
                await self.slow_worker.process(ctx)

                # 更新 SlowWorker 的 Prompt 缓存
                if ctx.whisper_result:
                    whisper_text = ctx.whisper_result.get('text', '')
                    self.slow_worker.update_prompt_cache(whisper_text)

                # 放入队列
                await self.queue_final.put(ctx)

            self.logger.info("SlowWorker 循环完成")

        except Exception as e:
            self.logger.error(f"SlowWorker 循环异常: {e}", exc_info=True)
            self.errors.append(e)

            # 发送错误信号
            error_ctx = ProcessingContext(
                job_id=self.job_id,
                chunk_index=-1,
                audio_chunk=None,
                is_end=True,
                error=e
            )
            await self.queue_final.put(error_ctx)

    async def _align_loop(self, results: List[ProcessingContext]):
        """
        AlignmentWorker 循环（最终消费者）

        职责：
        1. 从 queue_final 取 context
        2. 调用 AlignmentWorker.process()
        3. 收集结果到 results 列表
        4. 检测结束信号

        Args:
            results: 结果列表（用于收集 context）
        """
        try:
            while True:
                # 从队列取 context
                ctx = await self.queue_final.get()

                # 检查结束信号
                if ctx.is_end:
                    if ctx.error:
                        self.logger.error(f"上游错误: {ctx.error}")
                        raise ctx.error
                    break

                # AlignmentWorker 处理（双流对齐 + 分句 + 推送定稿）
                await self.alignment_worker.process(ctx)

                # 收集结果
                results.append(ctx)

            self.logger.info("AlignmentWorker 循环完成")

        except Exception as e:
            self.logger.error(f"AlignmentWorker 循环异常: {e}", exc_info=True)
            self.errors.append(e)

    def get_statistics(self) -> dict:
        """
        获取流水线统计信息

        Returns:
            dict: 统计信息
        """
        return {
            "queue_inter_size": self.queue_inter.qsize(),
            "queue_final_size": self.queue_final.qsize(),
            "errors": len(self.errors)
        }


# 便捷函数
def get_async_dual_pipeline(
    job_id: str,
    queue_maxsize: int = 5,
    logger: Optional[logging.Logger] = None
) -> AsyncDualPipeline:
    """
    获取异步双流流水线实例

    Args:
        job_id: 任务 ID
        queue_maxsize: 队列最大长度
        logger: 日志记录器

    Returns:
        AsyncDualPipeline 实例
    """
    return AsyncDualPipeline(
        job_id=job_id,
        queue_maxsize=queue_maxsize,
        logger=logger
    )
