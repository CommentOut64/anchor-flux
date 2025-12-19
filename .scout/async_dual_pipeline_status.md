<!-- 异步双流管道实现状态调查报告 -->

### Code Sections (The Evidence)

- `backend/app/pipelines/async_dual_pipeline.py` (AsyncDualPipeline): 三级异步流水线控制器，管理FastWorker、SlowWorker、AlignmentWorker的生命周期和队列。
- `backend/app/pipelines/async_dual_pipeline.py` (AsyncDualPipeline.run): 主入口方法，使用asyncio.create_task()启动三个并行任务，使用asyncio.gather()等待所有任务完成。
- `backend/app/pipelines/async_dual_pipeline.py` (AsyncDualPipeline._fast_loop): FastWorker循环（生产者），遍历chunks，调用FastWorker.process()，将context放入queue_inter，发送结束信号。
- `backend/app/pipelines/async_dual_pipeline.py` (AsyncDualPipeline._slow_loop): SlowWorker循环（中间消费者-生产者），从queue_inter取context，调用SlowWorker.process()，放入queue_final。
- `backend/app/pipelines/async_dual_pipeline.py` (AsyncDualPipeline._align_loop): AlignmentWorker循环（最终消费者），从queue_final取context，调用AlignmentWorker.process()，收集结果。
- `backend/app/pipelines/async_dual_pipeline.py` (AsyncDualPipeline.__init__): 初始化两个asyncio.Queue(maxsize=queue_maxsize)，默认maxsize=5用于背压控制。
- `backend/app/pipelines/workers/fast_worker.py` (FastWorker.process): 处理单个chunk，支持enable_fuse_breaker参数控制是否启用熔断回溯。
- `backend/app/pipelines/workers/fast_worker.py` (FastWorker._process_without_fuse): 不带熔断的处理流程，执行SenseVoice推理、分句、推送草稿。
- `backend/app/pipelines/workers/fast_worker.py` (FastWorker._process_with_fuse): 带熔断的处理流程，包含while True熔断循环，根据FuseBreakerV2决策升级分离。
- `backend/app/schemas/pipeline_context.py` (ProcessingContext): 流水线中流动的唯一数据对象，包含job_id、chunk_index、audio_chunk、sv_result、whisper_result、final_sentences等字段。

### Report (The Answers)

#### result

**1. _fast_loop 方法实现状态**

`_fast_loop` 方法（第158-219行）是一个串行处理的生产者循环，实现如下：
- 遍历所有audio_chunks（for循环）
- 对每个chunk创建ProcessingContext
- 调用 `await self.fast_worker.process(ctx)` 进行SenseVoice推理和分句
- 调用 `await self.queue_inter.put(ctx)` 将context放入队列
- 发送结束信号 `ProcessingContext(is_end=True)`
- 异常处理：捕获异常并发送错误信号

**2. 并发处理机制**

系统采用**异步并发**而非真正的并行处理：
- 使用 `asyncio.create_task()` 创建三个并行任务（第131-135行）
- 使用 `asyncio.gather(task_fast, task_slow, task_align, return_exceptions=True)` 等待所有任务完成（第138-141行）
- **未使用** asyncio.Semaphore 或其他并发限制机制
- 三个Worker在逻辑上并行运行，但在单线程事件循环中交错执行
- 没有使用ThreadPool或ProcessPool进行真正的并行处理

**3. 队列使用方式**

使用**标准asyncio.Queue**而非SequencedAsyncQueue：
- `self.queue_inter = asyncio.Queue(maxsize=queue_maxsize)` - FastWorker → SlowWorker（第70行）
- `self.queue_final = asyncio.Queue(maxsize=queue_maxsize)` - SlowWorker → AlignmentWorker（第71行）
- 默认maxsize=5（第43行）
- 队列中流动的是ProcessingContext对象
- 未发现项目中使用SequencedAsyncQueue的代码

**4. 背压控制机制**

实现了基于队列maxsize的背压控制：
- 当队列满时（达到maxsize=5），`await queue.put(ctx)` 会自动阻塞
- 这会阻塞FastWorker的_fast_loop，防止内存溢出
- 背压链条：FastWorker阻塞 → SlowWorker继续消费 → 队列腾出空间 → FastWorker恢复
- 注释明确说明："如果队列满了，会自动阻塞，实现背压"（第193行）
- 统计方法 `get_statistics()` 可查询队列当前大小（第306-317行）

#### conclusions

- **处理模式**：_fast_loop是串行处理（for循环逐个处理chunk），不是并发处理
- **并发架构**：三个Worker通过asyncio事件循环并发运行，但单个Worker内部是串行的
- **队列类型**：使用标准asyncio.Queue，不是SequencedAsyncQueue
- **背压机制**：通过asyncio.Queue的maxsize参数实现自动背压，当队列满时put()会阻塞
- **错误处理**：使用return_exceptions=True确保一个任务异常不会立刻中断其他任务
- **结束信号**：通过ProcessingContext.is_end标记通知下游Worker停止
- **熔断集成**：FastWorker支持enable_fuse_breaker参数，启用时包含while True循环进行升级分离重试

#### relations

- `AsyncDualPipeline.run()` 调用 `asyncio.create_task()` 启动三个循环任务
- `_fast_loop()` 调用 `FastWorker.process()` 处理每个chunk，然后 `await queue_inter.put(ctx)`
- `_slow_loop()` 从 `queue_inter` 取context，调用 `SlowWorker.process()`，然后 `await queue_final.put(ctx)`
- `_align_loop()` 从 `queue_final` 取context，调用 `AlignmentWorker.process()`，收集结果
- `FastWorker.process()` 根据 `enable_fuse_breaker` 参数选择 `_process_without_fuse()` 或 `_process_with_fuse()`
- `_process_with_fuse()` 包含while True循环，调用 `FuseBreakerV2.should_fuse()` 决策是否升级分离
- 三个队列（queue_inter、queue_final）和ProcessingContext对象形成数据流通道
- 异常通过ProcessingContext.error字段传播，或直接在asyncio.gather中抛出
