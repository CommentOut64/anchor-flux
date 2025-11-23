● SSE 设计方案完整分析

  一、现状分析

  1.1 已完成的工作

  基于之前的实现，我们已经完成：

  ✅ Phase 1（需求2 - 分离转录和对齐）：
  - _transcribe_segment_unaligned() - 只转录，返回粗略时间戳
  - _align_all_results() - 统一对齐所有片段
  - checkpoint.json 保存 unaligned_results 字段
  - 进度阶段新增：extract (5%) → split (5%) → transcribe (70%) → align (10%) → srt (10%)

  ✅ Phase 2（需求1 - 参数修改校验）：
  - GET /checkpoint-settings/{job_id} - 获取原始设置
  - POST /validate-resume-settings - 参数校验
  - POST /start - 自动强制禁止参数

  1.2 现有代码结构

  transcription_service.py 关键流程：
  _run_pipeline():
    ├─ 阶段1: extract_audio()
    ├─ 阶段2: split_audio() → 保存 checkpoint (segments)
    ├─ 阶段3: 转录循环
    │   └─ for seg in segments:
    │       ├─ _transcribe_segment_unaligned() → 返回 unaligned_result
    │       ├─ unaligned_results.append(result)
    │       └─ _save_checkpoint(unaligned_results)  # 每段保存一次
    ├─ 阶段4: _align_all_results(unaligned_results) → aligned_results
    └─ 阶段5: _generate_srt(aligned_results)

  checkpoint.json 数据结构（已实现）：
  {
    "job_id": "xxx",
    "phase": "transcribe",
    "total_segments": 100,
    "processed_indices": [0, 1, 2, ...],
    "segments": [{file, start_ms, duration_ms}],
    "unaligned_results": [
      {
        "segment_index": 0,
        "language": "zh",
        "segments": [
          {"id": 0, "start": 10.5, "end": 15.2, "text": "第一句话"}
        ]
      }
    ],
    "original_settings": {...}
  }

  1.3 现有问题

  1. 无实时推送机制 - 用户只能通过轮询 /status/{job_id} 查看进度
  2. 无法实时查看转录文本 - 必须等任务完成
  3. 前端无法提前编辑 - 对齐阶段是黑盒

---
  二、原文档设计 vs 新要求对比

  2.1 核心差异分析

  | 维度     | 原文档设计                                 | 新要求
                  | 影响              |
  |--------|---------------------------------------|--------------------------------------
  ------|-----------------|
  | 数据传输策略 | 对齐完成后通过SSE推送完整 aligned_results（可能几MB） |
  只推信号，前端HTTP GET拉取                          | ✅ 避免SSE阻塞，关键改进  |
  | 事件类型   | 4种：connected, segment, aligned, error | 5种：ping, progress, fragment,
  signal, error | ✅ 更清晰的职责分离      |
  | 重连机制   | 未明确                                   | "重连即全量" -
  不搞消息队列重放                         | ✅ 简化实现，避免复杂状态管理 |
  | 数据结构   | 未统一                                   | 明确 {index, start, end, text}
                 | ✅ 降低前后端转换成本     |

  2.2 关键设计原则对齐

  新要求提出的 4 条核心原则完全合理：

  1. 单通道原则 ✅ - 一个job只建立一个SSE连接，所有消息复用
  2. 轻量推送原则 ✅ - 最重要：SSE只传小数据和信号，大数据用HTTP GET
  3. 重连即全量 ✅ - 断线重连后调用 /transcription-text/{job_id} 拉取全量
  4. 用户修改优先 ✅ - 前端维护 is_modified 标记，智能合并

---
  三、需要修改的地方

  3.1 原文档的问题点

  ❌ 问题1：SSE传输大数据

  原文档代码（在 _run_pipeline 中）：
  # 对齐完成后推送
  aligned_results = self._align_all_results(...)

  # ❌ 错误：直接推送可能几MB的数据
  asyncio.create_task(self.sse_manager.broadcast(
      job.job_id,
      {
          "type": "aligned",
          "data": {
              "segments": aligned_results[0]['segments'],  # 可能有2000条
              "word_segments": aligned_results[0].get('word_segments', []),  #
  可能几万个词
              "message": "对齐完成"
          }
      }
  ))

  影响：
  - SSE通道阻塞，浏览器可能超时
  - 内存占用高
  - 违反SSE设计初衷（用于实时事件，不是大文件传输）

  ❌ 问题2：缺少心跳机制

  原文档代码（sse_service.py）：
  try:
      message = await asyncio.wait_for(queue.get(), timeout=10.0)
      yield self._format_sse(message)
  except asyncio.TimeoutError:
      # 发送心跳保持连接
      yield ": heartbeat\n\n"  # ✅ 这个是对的，但格式不规范

  问题：
  - 心跳格式应该是标准的 event: ping + data: {...}
  - 未明确心跳的数据载荷

  ❌ 问题3：事件类型混乱

  原文档使用：
  {"type": "segment", "data": {...}}
  {"type": "aligned", "data": {...}}

  新要求使用：
  event: fragment
  data: {...}

  event: signal
  data: {"code": "alignment_ready"}

  差异：
  - 原文档把事件类型放在 data.type 字段中
  - 新要求使用SSE标准的 event: 字段

  ❌ 问题4：缺少 /transcription-text/{job_id} 的多阶段支持

  原文档实现（在第二阶段文档中）：
  @router.get("/transcription-text/{job_id}")
  async def get_transcription_text(job_id: str):
      # 只读取 checkpoint.json
      checkpoint_path = job_dir / "checkpoint.json"
      if not checkpoint_path.exists():
          return {"has_checkpoint": False}
      # ...

  问题：
  - 无法处理对齐完成后的场景
  - 无法从内存读取（如果任务正在进行）
  - 无法读取最终的 aligned.json

---
  3.2 需要修改的文件和逻辑

  文件1：backend/app/services/sse_service.py（新建）

  需要改进：
  1. ✅ 保留简单的连接管理（无消息队列）
  2. ❌ 移除原文档中的 type 字段，改用 SSE 标准的 event: 字段
  3. ✅ 实现规范的心跳（event: ping）
  4. ✅ 简化 broadcast() 方法，接收 event 和 data 参数

  文件2：backend/app/services/transcription_service.py

  需要改进：
  1. 在转录循环中：
    - ✅ 发送 event: fragment（每完成一段）
    - ✅ 发送 event: progress（更新百分比）
  2. 在对齐完成后：
    - ❌ 移除直接推送 aligned_results 的代码
    - ✅ 新增将对齐结果保存到磁盘文件 jobs/{job_id}/aligned.json
    - ✅ 发送 event: signal, data: {code: "alignment_ready"}
  3. 修复异步调用问题：
    - ❌ 原文档使用 asyncio.create_task()，但 _run_pipeline 是同步函数
    - ✅ 需要改用 asyncio.run_coroutine_threadsafe() 或其他方式

  文件3：backend/app/api/routes/transcription_routes.py

  需要修改：
  1. 新增 /stream/{job_id} SSE端点
  2. 改造 /transcription-text/{job_id} 支持多数据源：
    优先级：内存 > aligned.json > checkpoint.json

  文件4：backend/app/core/config.py（可选）

  新增配置：
  # SSE配置
  SSE_HEARTBEAT_INTERVAL = 10  # 心跳间隔（秒）
  SSE_MAX_QUEUE_SIZE = 100     # 每个连接的消息队列大小

---
  四、完善后的第三阶段实施方案

  4.1 数据结构规范

  统一的 Fragment 结构

  后端生成（_transcribe_segment_unaligned 返回）：
  {
    "segment_index": 5,      // 对应 current_segments 的索引
    "language": "zh",
    "segments": [
      {
        "id": 0,             // 片段内的句子ID
        "start": 310.5,      // 全局时间戳（秒）
        "end": 315.2,
        "text": "这是第六个片段的第一句话"
      },
      {
        "id": 1,
        "start": 315.2,
        "end": 320.0,
        "text": "第二句话"
      }
    ]
  }

  SSE推送格式（event: fragment）：
  {
    "segment_index": 5,
    "segments": [
      {"id": 0, "start": 310.5, "end": 315.2, "text": "..."},
      {"id": 1, "start": 315.2, "end": 320.0, "text": "..."}
    ],
    "language": "zh"
  }

  前端展示时需要"扁平化"成全局列表：
  // 前端处理逻辑
  onFragmentReceived(fragment) {
    fragment.segments.forEach(seg => {
      // 生成全局唯一ID（用于虚拟滚动的key）
      const globalId = `${fragment.segment_index}_${seg.id}`;

      this.subtitles.push({
        id: globalId,
        start: seg.start,
        end: seg.end,
        text: seg.text,
        is_modified: false
      });
    });
  }

  对齐结果文件结构（新增）

  文件路径：jobs/{job_id}/aligned.json

  内容：
  {
    "job_id": "xxx",
    "language": "zh",
    "aligned_at": "2025-01-22T10:30:00",
    "segments": [
      {
        "id": 0,
        "start": 10.523,    // 精确时间戳
        "end": 15.187,
        "text": "第一句话"
      },
      {
        "id": 1,
        "start": 15.187,
        "end": 20.045,
        "text": "第二句话"
      }
    ],
    "word_segments": [...]  // 如果 word_timestamps=true
  }

---
  4.2 完整实施步骤

  Step 1: 创建 SSE 基础服务

  文件: backend/app/services/sse_service.py

  功能要求：
  1. ✅ 单例 SSEManager 类
  2. ✅ subscribe(job_id, request) 生成器
    - 发送 event: connected
    - 10秒心跳 event: ping
    - 客户端断开时自动清理
  3. ✅ broadcast(job_id, event, data) 方法
    - 向指定任务的所有订阅者发送消息
    - 使用标准SSE格式：event: {event}\ndata: {json}\n\n
  4. ❌ 不要实现消息队列和重放机制

  关键点：
  def _format_sse(self, event: str, data: dict) -> str:
      """格式化为SSE消息"""
      return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

  async def broadcast(self, job_id: str, event: str, data: dict):
      """广播消息到所有订阅者"""
      if job_id not in self.connections:
          return

      for queue in self.connections[job_id]:
          try:
              await queue.put({"event": event, "data": data})
          except Exception as e:
              logger.error(f"SSE推送失败: {e}")

---
  Step 2: 集成到 TranscriptionService

  文件: backend/app/services/transcription_service.py

  修改点1：初始化时获取 SSE Manager
  def __init__(self, jobs_root: str):
      # ... 现有代码 ...

      # 新增：集成SSE管理器（懒加载，避免循环导入）
      self.sse_manager = None
      self._sse_initialized = False

  def _get_sse_manager(self):
      """懒加载SSE管理器"""
      if not self._sse_initialized:
          try:
              from services.sse_service import get_sse_manager
              self.sse_manager = get_sse_manager()
              self._sse_initialized = True
          except Exception as e:
              self.logger.warning(f"SSE管理器加载失败: {e}")
              self.sse_manager = None
      return self.sse_manager

  修改点2：转录循环中推送 fragment
  def _run_pipeline(self, job: JobState):
      # ... 前置代码 ...

      # 获取SSE管理器
      sse = self._get_sse_manager()
    
      # 阶段3: 转录处理
      for idx, seg in enumerate(current_segments):
          if idx in processed_indices:
              continue
    
          # 转录
          seg['index'] = idx
          result = self._transcribe_segment_unaligned(seg, model, job)
    
          if result:
              unaligned_results.append(result)
    
              # ✅ 新增：立即推送到前端
              if sse:
                  self._broadcast_sse(sse, job.job_id, "fragment", {
                      "segment_index": idx,
                      "segments": result['segments'],
                      "language": result.get('language')
                  })
    
          # 更新进度
          processed_indices.add(idx)
          job.processed = len(processed_indices)
    
          progress_ratio = len(processed_indices) / len(current_segments)
          self._update_progress(job, 'transcribe', progress_ratio, ...)
    
          # ✅ 新增：推送进度更新
          if sse:
              self._broadcast_sse(sse, job.job_id, "progress", {
                  "phase": "transcribe",
                  "percent": round(job.progress, 2),
                  "message": job.message,
                  "processed": job.processed,
                  "total": job.total
              })
    
          # 保存checkpoint
          self._save_checkpoint(job_dir, {...}, job)

  修改点3：对齐完成后推送信号 + 保存文件
  def _run_pipeline(self, job: JobState):
      # ... 转录循环完成 ...

      # 阶段4: 统一对齐
      self._update_progress(job, 'align', 0, '准备对齐...')
      aligned_results = self._align_all_results(
          unaligned_results,
          job,
          str(audio_path)
      )
    
      # ✅ 新增：保存对齐结果到磁盘
      aligned_file = job_dir / "aligned.json"
      try:
          with open(aligned_file, 'w', encoding='utf-8') as f:
              json.dump({
                  "job_id": job.job_id,
                  "language": job.language,
                  "aligned_at": datetime.now().isoformat(),
                  "segments": aligned_results[0].get('segments', []),
                  "word_segments": aligned_results[0].get('word_segments', [])
              }, f, ensure_ascii=False, indent=2)
    
          self.logger.info(f"✅ 对齐结果已保存: {aligned_file}")
      except Exception as e:
          self.logger.error(f"保存对齐结果失败: {e}")
    
      # ✅ 新增：发送信号（不传输数据）
      if sse:
          self._broadcast_sse(sse, job.job_id, "signal", {
              "code": "alignment_ready",
              "message": "对齐完成，请刷新获取最新数据"
          })
    
      self._update_progress(job, 'align', 1, '对齐完成')
    
      # ... 继续生成SRT ...

  修改点4：添加 SSE 推送辅助方法
  def _broadcast_sse(self, sse_manager, job_id: str, event: str, data: dict):
      """
      在同步上下文中安全地推送SSE消息

      Args:
          sse_manager: SSE管理器实例
          job_id: 任务ID
          event: 事件类型
          data: 数据载荷
      """
      try:
          # 在后台线程中运行异步任务
          import asyncio
          loop = asyncio.new_event_loop()
          asyncio.set_event_loop(loop)
          loop.run_until_complete(
              sse_manager.broadcast(job_id, event, data)
          )
          loop.close()
      except Exception as e:
          self.logger.debug(f"SSE推送失败（非致命）: {e}")

---
  Step 3: 添加 API 端点

  文件: backend/app/api/routes/transcription_routes.py

  修改点1：新增 SSE 端点
  from fastapi import Request
  from fastapi.responses import StreamingResponse
  from services.sse_service import get_sse_manager

  sse_manager = get_sse_manager()

  @router.get("/stream/{job_id}")
  async def stream_transcription(job_id: str, request: Request):
      """
      订阅任务的SSE事件流

      事件类型：
      - ping: 心跳保活
      - progress: 进度更新
      - fragment: 单个片段转录完成
      - signal: 关键节点信号（如 alignment_ready）
      - error: 错误事件
      """
      job = transcription_service.get_job(job_id)
      if not job:
          raise HTTPException(status_code=404, detail="任务未找到")
    
      return StreamingResponse(
          sse_manager.subscribe(job_id, request),
          media_type="text/event-stream",
          headers={
              "Cache-Control": "no-cache",
              "Connection": "keep-alive",
              "X-Accel-Buffering": "no"  # 禁用nginx缓冲
          }
      )

  修改点2：增强 /transcription-text/{job_id}
  @router.get("/transcription-text/{job_id}")
  async def get_transcription_text(job_id: str):
      """
      获取转录文本（支持多数据源）

      优先级：
      1. 对齐结果文件（aligned.json）- 如果存在
      2. checkpoint中的未对齐结果（checkpoint.json）
      3. 返回空
      """
      from pathlib import Path
    
      job = transcription_service.get_job(job_id)
      if not job:
          raise HTTPException(status_code=404, detail="任务未找到")
    
      job_dir = Path(job.dir)
    
      # 优先级1：读取对齐结果
      aligned_path = job_dir / "aligned.json"
      if aligned_path.exists():
          try:
              with open(aligned_path, 'r', encoding='utf-8') as f:
                  data = json.load(f)
    
              return {
                  "job_id": job_id,
                  "source": "aligned",
                  "language": data.get("language", "unknown"),
                  "segments": data.get("segments", []),
                  "word_segments": data.get("word_segments", []),
                  "aligned_at": data.get("aligned_at")
              }
          except Exception as e:
              logger.error(f"读取对齐结果失败: {e}")
    
      # 优先级2：读取checkpoint中的未对齐结果
      checkpoint_path = job_dir / "checkpoint.json"
      if checkpoint_path.exists():
          try:
              with open(checkpoint_path, 'r', encoding='utf-8') as f:
                  data = json.load(f)
    
              unaligned_results = data.get("unaligned_results", [])
    
              # 合并所有segments
              all_segments = []
              for result in unaligned_results:
                  all_segments.extend(result.get('segments', []))
    
              # 按时间排序并重新编号
              all_segments.sort(key=lambda x: x.get('start', 0))
              for idx, seg in enumerate(all_segments):
                  seg['id'] = idx
    
              return {
                  "job_id": job_id,
                  "source": "checkpoint",
                  "language": unaligned_results[0].get("language", "unknown") if
  unaligned_results else "unknown",
                  "segments": all_segments,
                  "progress": {
                      "processed": len(data.get("processed_indices", [])),
                      "total": data.get("total_segments", 0),
                      "percentage": round(
                          len(data.get("processed_indices", [])) / max(1,
  data.get("total_segments", 1)) * 100,
                          2
                      )
                  }
              }
          except Exception as e:
              logger.error(f"读取checkpoint失败: {e}")

      # 优先级3：无数据
      return {
          "job_id": job_id,
          "source": "none",
          "message": "没有可用的转录数据"
      }

---
  Step 4: 更新配置文件（可选）

  文件: backend/app/core/config.py

  # ========== SSE配置 ==========
  self.SSE_HEARTBEAT_INTERVAL = 10  # 心跳间隔（秒）
  self.SSE_MAX_QUEUE_SIZE = 100     # 每个连接的消息队列大小

---
  4.3 测试验证步骤

  后端测试

  1. 启动服务
    cd backend
    uvicorn app.main:app --reload
  2. 使用curl测试SSE
  # 创建并启动任务（假设job_id=abc123）
  curl -X POST http://localhost:8000/api/start \
    -F "job_id=abc123" \
    -F 'settings={"model":"medium","device":"cuda"}'

  # 监听SSE流
  curl -N http://localhost:8000/api/stream/abc123
  3. 预期输出
    event: connected
    data: {"job_id": "abc123", "message": "SSE连接已建立"}

  event: progress
  data: {"phase": "extract", "percent": 5, "message": "提取音频中"}

  event: fragment
  data: {"segment_index": 0, "segments": [...], "language": "zh"}

  event: progress
  data: {"phase": "transcribe", "percent": 15, "message": "转录中 1/100"}

  ...

  event: signal
  data: {"code": "alignment_ready", "message": "对齐完成，请刷新获取最新数据"}
  4. 验证文件生成
  # 检查对齐结果文件
  cat jobs/abc123/aligned.json
  5. 测试拉取接口
  # 对齐完成后调用
  curl http://localhost:8000/api/transcription-text/abc123

  前端集成测试（概念验证）

  // 1. 创建EventSource
  const eventSource = new EventSource(`http://localhost:8000/api/stream/${jobId}`);

  // 2. 监听不同事件
  eventSource.addEventListener('fragment', (e) => {
    const data = JSON.parse(e.data);
    console.log('收到片段:', data.segment_index, data.segments);
    // 追加到字幕列表
  });

  eventSource.addEventListener('signal', async (e) => {
    const data = JSON.parse(e.data);
    if (data.code === 'alignment_ready') {
      console.log('对齐完成，开始拉取数据');
      // HTTP GET 拉取完整数据
      const response = await fetch(`/api/transcription-text/${jobId}`);
      const aligned = await response.json();
      console.log('对齐结果:', aligned.segments.length, '条');
      // 智能合并到现有列表
    }
  });

  eventSource.addEventListener('progress', (e) => {
    const data = JSON.parse(e.data);
    console.log('进度:', data.percent, '%', data.message);
  });

  eventSource.onerror = (e) => {
    console.error('SSE连接错误');
    eventSource.close();
  };

---
  五、风险点和注意事项

  5.1 技术风险

  | 风险          | 影响      | 缓解措施                                         |
  |-------------|---------|----------------------------------------------|
  | SSE连接频繁断开   | 用户体验差   | 1. 实现前端自动重连2. 重连后调用
  /transcription-text 全量刷新 |
  | 同步函数调用异步SSE | 推送失败    | 使用 asyncio.new_event_loop() 或线程池
   |
  | 对齐结果文件过大    | 内存/磁盘压力 | 1. 限制单任务最大时长（如2小时）2.
  使用流式JSON写入              |
  | 多客户端同时订阅    | 服务器压力   | 限制每个job的最大连接数（如5个）
            |

  5.2 开发建议

  1. 分步实施：
    - 先实现SSE基础服务，用静态数据测试
    - 再集成到转录服务，只推送 progress 和 fragment
    - 最后实现 signal + HTTP GET 模式
  2. 日志完善：
    - SSE连接建立/断开
    - 每次广播的事件类型和数据大小
    - 对齐结果文件的写入成功/失败
  3. 错误处理：
    - SSE推送失败不应影响转录主流程
    - 使用 try-except 包裹所有 _broadcast_sse 调用
    - 如果SSE推送失败，记录警告日志但不中断

---
  六、总结

  完善后的第三阶段任务清单

  - Task 1: 创建 sse_service.py
    - 实现 SSEManager 类（单例模式）
    - 实现 subscribe() 生成器（心跳、断线处理）
    - 实现 broadcast() 方法（标准SSE格式）
    - 移除消息队列重放逻辑
  - Task 2: 修改 transcription_service.py
    - 在 __init__ 中懒加载 sse_manager
    - 在转录循环中推送 fragment 和 progress
    - 对齐完成后保存 aligned.json + 推送 signal
    - 实现 _broadcast_sse() 辅助方法（处理同步/异步）
  - Task 3: 修改 transcription_routes.py
    - 新增 GET /stream/{job_id} SSE端点
    - 增强 GET /transcription-text/{job_id}（多数据源支持）
  - Task 4: 更新 config.py（可选）
    - 添加 SSE 相关配置项
  - Task 5: 测试验证
    - 后端SSE推送测试（curl）
    - 验证 aligned.json 生成
    - 验证 /transcription-text 多数据源逻辑
    - 前端EventSource集成测试

  与原文档的核心差异

  | 项目     | 原文档          | 新方案                           |
  |--------|--------------|-------------------------------|
  | 对齐结果传输 | SSE直接推送      | 保存到磁盘 + HTTP GET拉取            |
  | 事件格式   | {type, data} | 标准SSE event: xxx\ndata: {...} |
  | 重连机制   | 未明确          | 重连即全量拉取                       |
  | 心跳格式   | : heartbeat  | event: ping\ndata: {...}      |

---
  SSE统一优化方案分析

  一、现状分析

  1.1 模型下载进度（已使用SSE - 已完成）

  后端实现：
  - ✅ model_manager_service.py:462-478 - 实现了 progress_callbacks 机制
  - ✅ model_routes.py:155-343 - 完整的SSE实现
    - /events/progress - 全局进度推送（所有模型）
    - /events/{model_type}/{model_id} - 单模型进度推送
    - 使用 asyncio.Queue (容量1000)
    - 心跳：15秒

  前端实现：
  - ✅ modelStore.js:106-187 - 单例 EventSource 连接
  - ✅ 监听事件：initial_state, model_progress, model_complete, model_error, model_incomplete, heartbeat
  - ✅ 全局状态管理（Pinia store）

  现有问题：
  1. ⚠️ 队列溢出风险（line 224）：频繁更新时，队列使用率>90%会跳过更新
  2. ⚠️ 进度回调线程安全（line 206-244）：使用 call_soon_threadsafe 但降级日志较多
  3. ⚠️ 重复代码：手动下载逻辑重复（line 881-1125）

  1.2 转录进度（当前是轮询 - 待改造）

  后端实现：
  - ❌ 无SSE支持 - 只有 _update_progress() 更新内存
  - ❌ 无进度推送机制
  - ✅ 有完整的进度阶段划分：extract(5%) → split(5%) → transcribe(70%) → align(10%) → srt(10%)

  前端实现：
  - ❌ App.vue:538-595 - 轮询模式
    - 每1.5秒调用 /status/{job_id}
    - 成功任务继续轮询，失败后2.5秒重试
    - 占用大量网络资源

  问题：
  1. 高延迟：最多1.5秒才能看到进度更新
  2. 高负载：多个客户端 × 每秒0.67次请求
  3. 资源浪费：大部分轮询返回的数据没有变化

  1.3 转录文字流式输出（未实现 - 第三阶段需求）

  现状：
  - ❌ 完全未实现
  - 📝 已在"断点续传增强功能-开发文档.md"中规划

  ---
  二、需要修改的地方

  2.1 模型下载进度优化（小改进）

  虽然已经使用SSE，但需要以下改进：

  1. 统一事件格式
    - ❌ 当前：{type, model_id, progress, status, message, timestamp}
    - ✅ 统一为：使用标准SSE event: 字段
  2. 移除队列容量限制逻辑
    - 当前1000容量 + 90%跳过机制过于复杂
    - 改为更大的队列 + 更优雅的降级策略
  3. 整合到统一SSE管理器

  2.2 转录进度改造（核心改造）

  后端修改

  文件1：backend/app/services/transcription_service.py

  需要在以下位置添加SSE推送：

  | 位置                    | 当前代码       | 需要添加                                  |
  |-----------------------|------------|---------------------------------------|
  | _run_pipeline:607-614 | 音频提取进度     | 推送 event: progress, phase: extract    |
  | _run_pipeline:626-633 | 音频分段进度     | 推送 event: progress, phase: split      |
  | _run_pipeline:668-720 | 转录循环（每段完成） | 推送 event: progress, phase: transcribe |
  | _run_pipeline:729-738 | 对齐阶段       | 推送 event: progress, phase: align      |
  | _run_pipeline:745-751 | SRT生成      | 推送 event: progress, phase: srt        |
  | _run_pipeline:767-769 | 任务完成       | 推送 event: signal, code: job_complete  |
  | _run_pipeline:772-784 | 任务失败/取消/暂停 | 推送 event: error/signal                |

  文件2：backend/app/api/routes/transcription_routes.py

  新增SSE端点：
  @router.get("/stream/{job_id}")
  async def stream_job_progress(job_id: str, request: Request):
      """SSE端点：推送任务进度（包括转录进度、文字片段、对齐完成信号）"""

  前端修改

  文件1：frontend/src/services/transcriptionService.js

  新增SSE连接方法：
  static createProgressSSE(jobId) {
    return new EventSource(`/api/stream/${jobId}`)
  }

  文件2：frontend/src/App.vue

  替换轮询逻辑：
  // 删除 poll() 函数和 pollTimer
  // 改为在 startJob() 中建立SSE连接
  function connectJobSSE() {
    const es = TranscriptionService.createProgressSSE(jobId.value)

    es.addEventListener('progress', (e) => {
      const data = JSON.parse(e.data)
      progress.value = data.percent
      statusText.value = data.message
      phase.value = data.phase
    })

    // ... 其他事件监听
  }

  ---
  三、统一SSE架构设计

  3.1 核心原则（基于新要求）

  1. 单通道原则 ✅
    - 每个job只建立一个SSE连接 /stream/{job_id}
    - 所有事件（进度、文字、信号）通过同一通道
  2. 轻量推送原则 ✅
    - SSE只传小数据（progress, fragment）
    - 大数据（aligned_results）通过HTTP GET拉取
  3. 统一事件管理器 🆕
    - 复用模型下载的SSE基础设施
    - 统一事件格式和错误处理

  3.2 统一的SSE管理器设计

  创建 backend/app/services/sse_service.py（基础版 + 增强版）

  class SSEManager:
      """统一SSE管理器 - 支持模型下载 + 转录任务 + 文字流式输出"""

      def __init__(self):
          # 连接池：{channel_id: [queue1, queue2, ...]}
          self.connections: Dict[str, List[asyncio.Queue]] = defaultdict(list)

      async def subscribe(self, channel_id: str, request: Request):
          """订阅指定频道的SSE流"""
          # channel_id 可以是：
          # - "models" - 全局模型下载进度
          # - "job:{job_id}" - 特定任务的进度+文字

      async def broadcast(self, channel_id: str, event: str, data: dict):
          """向指定频道广播消息"""

  3.3 事件类型统一定义

  | 事件名      | 用途   | 数据示例                                               | 频道               |
  |----------|------|----------------------------------------------------|------------------|
  | ping     | 心跳保活 | {ts: 123456}                                       | 所有               |
  | progress | 进度更新 | {phase: "transcribe", percent: 45, message: "..."} | job:{id}, models |
  | fragment | 文字片段 | {index: 5, segments: [...]}                        | job:{id}         |
  | signal   | 关键节点 | {code: "alignment_ready" / "job_complete"}         | job:{id}         |
  | error    | 错误事件 | {code: 500, message: "..."}                        | 所有               |

  3.4 改造后的架构图

  ┌─────────────────────────────────────────────────────────────┐
  │                    统一SSE管理器                              │
  │  SSEManager (backend/app/services/sse_service.py)           │
  ├─────────────────────────────────────────────────────────────┤
  │                                                               │
  │  频道1: "models"                                              │
  │  ├─ /api/models/events/progress → 所有模型下载进度           │
  │  └─ 推送：progress, model_complete, model_error              │
  │                                                               │
  │  频道2: "job:{job_id}"                                        │
  │  ├─ /api/stream/{job_id} → 转录进度+文字流                   │
  │  └─ 推送：progress, fragment, signal, error                  │
  │                                                               │
  └─────────────────────────────────────────────────────────────┘
           ↓                                    ↓
     前端EventSource                       前端EventSource
     modelStore.js                         App.vue / jobStore.js

  ---
  四、补充到第三阶段的任务

  4.1 新增任务：统一SSE基础设施

  优先级：P0（前置任务）

  任务内容：
  1. 创建 backend/app/services/sse_service.py - 统一SSE管理器
  2. 定义标准事件格式（event + data）
  3. 实现频道订阅机制（支持模式匹配）
  4. 实现心跳和断线重连处理

  预计工时：4小时

  4.2 改造任务：转录进度改为SSE

  优先级：P1

  任务内容：
  1. 修改 transcription_service.py - 集成SSE推送
  2. 新增 /stream/{job_id} SSE端点
  3. 前端移除轮询逻辑，改为SSE监听
  4. 测试断线重连和状态恢复

  预计工时：6小时

  4.3 优化任务：模型下载SSE迁移到统一管理器

  优先级：P2

  任务内容：
  1. 将 model_routes.py 的SSE逻辑迁移到统一管理器
  2. 保持API兼容性（/api/models/events/progress 不变）
  3. 移除 sse_queues 全局变量
  4. 统一事件格式

  预计工时：3小时

  4.4 原第三阶段任务：文字流式输出

  保持不变，但基于统一SSE管理器实现。

  预计工时：6小时（原8小时，因为基础设施已完成）

  ---
  五、实施顺序建议

  方案A：渐进式改造（推荐）

  阶段1（核心）：统一SSE基础设施
    └─ 创建 sse_service.py
    └─ 测试基础功能

  阶段2（转录）：转录进度改SSE
    └─ 后端集成SSE推送
    └─ 前端替换轮询
    └─ 测试断点续传场景

  阶段3（模型）：模型下载迁移（可选）
    └─ 迁移到统一管理器
    └─ 保持API兼容
    └─ 性能测试

  阶段4（文字）：文字流式输出
    └─ fragment推送
    └─ HTTP GET拉取对齐结果
    └─ 前端智能合并

  方案B：一步到位（风险较高）

  直接按照"断点续传增强功能-开发文档.md"实施，同时改造转录进度和模型下载。

  ---
  六、关键技术问题

  6.1 同步函数调用异步SSE的问题

  问题：_run_pipeline() 是同步函数（运行在后台线程），但SSE推送是异步的。

  解决方案（3种）：

  方案1：使用 asyncio.run_coroutine_threadsafe()（推荐）

  def _broadcast_sse_sync(self, sse_manager, job_id, event, data):
      """在同步上下文中推送SSE（线程安全）"""
      try:
          loop = asyncio.get_event_loop()  # 获取主事件循环
          asyncio.run_coroutine_threadsafe(
              sse_manager.broadcast(f"job:{job_id}", event, data),
              loop
          )
      except Exception as e:
          self.logger.debug(f"SSE推送失败（非致命）: {e}")

  方案2：使用线程池（模型下载当前方案）

  # 类似 model_routes.py:206
  loop = get_event_loop()
  if loop:
      loop.call_soon_threadsafe(queue.put_nowait, event_data)

  方案3：改造为异步（工作量大，不推荐）

  将整个 _run_pipeline() 改为异步函数。

  6.2 SSE断线后的状态恢复

  问题：客户端断线重连后，如何恢复当前状态？

  解决方案：
  1. SSE连接建立时，后端立即推送 initial_state 事件（包含当前进度）
  2. 前端收到 initial_state 后，调用 /status/{job_id} 获取完整状态
  3. 使用"重连即全量"原则，不搞消息队列重放

  6.3 多客户端同时监听的并发问题

  问题：一个job有多个客户端监听时，如何高效广播？

  解决方案（已在模型下载中实现）：
  # model_routes.py:218-230
  for q in sse_queues[:]:  # 使用切片避免遍历时修改
      if q.qsize() >= q.maxsize * 0.9:
          # 跳过繁忙队列，避免阻塞
          continue
      loop.call_soon_threadsafe(q.put_nowait, event_data)

  ---

  八、总结

  需要改造的文件清单

  后端（新建）：
  1. ✨ backend/app/services/sse_service.py - 统一SSE管理器

  后端（修改）：
  2. 🔧 backend/app/services/transcription_service.py - 集成SSE推送
  3. 🔧 backend/app/api/routes/transcription_routes.py - 新增 /stream/{job_id}
  4. 🔧 backend/app/api/routes/model_routes.py - 迁移到统一管理器（可选）
  5. 🔧 backend/app/core/config.py - 新增SSE配置（心跳间隔等）

  前端（修改）：
  6. 🔧 frontend/src/services/transcriptionService.js - 新增SSE方法
  7. 🔧 frontend/src/App.vue - 替换轮询为SSE
  8. 🔧 frontend/src/stores/modelStore.js - 可选：迁移到统一管理器

  推荐实施路线

  第一步：实施"4.1 统一SSE基础设施"（4小时）
  第二步：实施"4.2 转录进度改SSE"（6小时）
  第三步：实施原第三阶段"文字流式输出"（6小时）
  第四步（可选）：实施"4.3 模型下载迁移"（3小时）