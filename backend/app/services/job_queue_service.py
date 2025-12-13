"""
任务队列管理服务 - V2.4
核心功能: 串行执行，防止并发OOM，队列持久化，插队功能
"""
import threading
import time
import logging
import gc
import json
import os
from collections import deque
from typing import Dict, Optional, Literal
from pathlib import Path
import torch

from app.models.job_models import JobState
from app.services.sse_service import get_sse_manager

logger = logging.getLogger(__name__)

# 插队模式类型
PrioritizeMode = Literal["gentle", "force"]


class JobQueueService:
    """
    任务队列管理器

    职责:
    1. 维护任务队列 (FIFO)
    2. 单线程Worker循环
    3. 串行执行任务（同一时间只有1个running）
    4. 支持两种插队模式：温和插队、强制插队
    """

    def __init__(self, transcription_service):
        """
        初始化队列服务

        Args:
            transcription_service: 转录服务实例
        """
        # 核心数据结构
        self.jobs: Dict[str, JobState] = {}  # 任务注册表 {job_id: JobState}
        self.queue: deque = deque()           # 等待队列 [job_id1, job_id2, ...]
        self.running_job_id: Optional[str] = None  # 当前正在执行的任务ID

        # 强制插队相关：记录被中断的任务，用于自动恢复
        self.interrupted_job_id: Optional[str] = None  # 被强制中断的任务ID

        # 插队设置
        self._default_prioritize_mode: PrioritizeMode = "gentle"  # 默认插队模式

        # 依赖服务
        self.transcription_service = transcription_service
        self.sse_manager = get_sse_manager()

        # 控制信号
        self.stop_event = threading.Event()
        self.lock = threading.RLock()  # 使用可重入锁，避免嵌套调用死锁

        # 持久化文件路径
        from app.core.config import config
        self.queue_file = Path(config.JOBS_DIR) / "queue_state.json"
        self.settings_file = Path(config.JOBS_DIR) / "queue_settings.json"

        # 加载设置
        self._load_settings()

        # 启动时恢复队列
        self._load_state()

        # 启动Worker线程
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="JobQueueWorker"
        )
        self.worker_thread.start()
        logger.info("任务队列Worker线程已启动")

    def add_job(self, job: JobState):
        """
        添加任务到队列

        Args:
            job: 任务状态对象
        """
        with self.lock:
            self.jobs[job.job_id] = job
            self.queue.append(job.job_id)
            job.status = "queued"
            job.message = f"排队中 (位置: {len(self.queue)})"

        logger.info(f"任务已加入队列: {job.job_id} (队列长度: {len(self.queue)})")

        # 保存队列状态和任务元信息
        self._save_state()
        self.transcription_service.save_job_meta(job)

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job.job_id, job.status)

    def get_job(self, job_id: str) -> Optional[JobState]:
        """获取任务状态"""
        return self.jobs.get(job_id)

    def pause_job(self, job_id: str) -> bool:
        """
        暂停任务

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否成功设置暂停标志
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        with self.lock:
            if job_id == self.running_job_id:
                # 正在执行的任务：设置暂停标志（pipeline会自己检测并保存checkpoint）
                job.paused = True
                job.status = "paused"  # 立即更新状态，确保SSE推送正确的状态
                job.message = "暂停中..."
                logger.info(f"设置暂停标志: {job_id}")
            elif job_id in self.queue:
                # 还在排队的任务：直接从队列移除
                self.queue.remove(job_id)
                job.status = "paused"
                job.message = "已暂停（未开始）"
                logger.info(f"从队列移除: {job_id}")

        # 保存队列状态和任务元信息
        self._save_state()
        self.transcription_service.save_job_meta(job)

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)

        # 同时推送到单任务频道，确保 EditorView 能收到
        self._notify_job_signal(job_id, "job_paused")

        return True

    def resume_job(self, job_id: str) -> bool:
        """
        恢复暂停的任务（重新加入队列）

        与 restore_job 不同：
        - resume_job: 恢复暂停的任务，重新加入队列尾部
        - restore_job: 从 checkpoint 断点续传

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否成功
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status != "paused":
            logger.warning(f"任务未暂停，无法恢复: {job_id}, status={job.status}")
            return False

        with self.lock:
            # 重新加入队列
            if job_id not in self.queue:
                self.queue.append(job_id)

            job.status = "queued"
            job.paused = False
            job.message = f"已恢复，排队中 (位置: {len(self.queue)})"
            logger.info(f"恢复暂停任务: {job_id}")

        # 保存队列状态和任务元信息
        self._save_state()
        self.transcription_service.save_job_meta(job)

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)

        # 同时推送到单任务频道
        self._notify_job_signal(job_id, "job_resumed")

        return True

    def cancel_job(self, job_id: str, delete_data: bool = False) -> bool:
        """
        取消任务（支持删除已完成的任务）

        Args:
            job_id: 任务ID
            delete_data: 是否删除任务数据

        Returns:
            bool: 是否成功
        """
        job = self.jobs.get(job_id)

        # 如果任务不在队列服务中（可能是已完成的任务），直接调用transcription_service删除
        if not job:
            if delete_data:
                # 尝试通过transcription_service删除已完成的任务
                try:
                    result = self.transcription_service.cancel_job(job_id, delete_data=True)
                    if result:
                        # 推送全局SSE通知（通知前端任务已删除）
                        self._notify_job_status(job_id, "canceled")
                        return True
                except Exception as e:
                    logger.warning(f"删除任务 {job_id} 失败: {e}")
            return False

        with self.lock:
            # 设置取消标志
            job.canceled = True
            job.message = "取消中..."

            # 如果在队列中，移除
            if job_id in self.queue:
                self.queue.remove(job_id)
                job.status = "canceled"
                job.message = "已取消（未开始）"

        # 如果需要删除数据，调用transcription_service的清理逻辑
        if delete_data:
            result = self.transcription_service.cancel_job(job_id, delete_data=True)
        else:
            result = True
            # 不删除数据时，保存任务元信息
            self.transcription_service.save_job_meta(job)

        # 保存队列状态
        self._save_state()

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)

        # 同时推送到单任务频道，确保 EditorView 能收到
        self._notify_job_signal(job_id, "job_canceled")

        return result

    def _worker_loop(self):
        """
        Worker线程主循环

        核心逻辑:
        1. 从队列取任务
        2. 执行任务（阻塞）
        3. 清理资源
        4. 循环
        """
        logger.info("Worker循环已启动")

        while not self.stop_event.is_set():
            try:
                # 1. 检查队列是否为空
                with self.lock:
                    if not self.queue:
                        # 队列为空，休眠1秒
                        pass
                    else:
                        # 取队头任务（不移除，防止出错丢失）
                        job_id = self.queue[0]
                        job = self.jobs.get(job_id)

                        # 验证任务有效性
                        if not job:
                            logger.warning(f"⚠️ 任务不存在，跳过: {job_id}")
                            self.queue.popleft()
                            continue

                        if job.status in ["paused", "canceled"]:
                            logger.info(f"⏭️ 跳过已暂停/取消的任务: {job_id}")
                            self.queue.popleft()
                            continue

                        # 正式从队列移除
                        self.queue.popleft()
                        self.running_job_id = job_id
                        job.status = "processing"
                        job.message = "开始处理"

                        # 推送队列变化和任务状态通知（在lock内，避免数据不一致）
                        self._notify_queue_change()
                        self._notify_job_status(job_id, "processing")
                        # 推送初始进度（让前端立即知道任务的初始状态）
                        self._notify_job_progress(job_id)

                # 任务开始执行前保存状态（确保断电后能恢复 running 任务）
                if self.running_job_id:
                    self._save_state()
                    # 同时保存任务元信息（记录 processing 状态）
                    job = self.jobs.get(self.running_job_id)
                    if job:
                        self.transcription_service.save_job_meta(job)

                # 2. 如果没有任务，休眠后继续
                if self.running_job_id is None:
                    time.sleep(1)
                    continue

                # 3. 执行任务（阻塞，直到完成/失败/暂停/取消）
                job = self.jobs[self.running_job_id]
                logger.info(f" 开始执行任务: {self.running_job_id}")

                try:
                    # 根据引擎和预设选择流水线
                    engine = getattr(job.settings, 'engine', 'sensevoice')
                    preset_id = getattr(job.settings.sensevoice, 'preset_id', 'default') if hasattr(job.settings, 'sensevoice') else 'default'

                    # Phase 5: 对于非 default 预设，使用双流对齐流水线
                    # default: SenseVoice Only (旧流水线)
                    # preset1+: SenseVoice + Whisper 双流对齐 (新流水线)
                    use_dual_alignment = preset_id != 'default' and engine == 'sensevoice'

                    if use_dual_alignment:
                        # 双流对齐流水线 (V3.0+ 新架构)
                        logger.info(f"使用双流对齐流水线 (preset={preset_id})")
                        import asyncio
                        asyncio.run(self._run_dual_alignment_pipeline(job, preset_id))
                    elif engine == 'sensevoice':
                        # SenseVoice 流水线（旧架构，仅 default 预设）
                        logger.info(f"使用 SenseVoice 流水线 (极速模式)")
                        import asyncio
                        asyncio.run(self.transcription_service._process_video_sensevoice(job))
                    else:
                        # Whisper 流水线（同步，使用三点采样）
                        logger.info(f"使用 Whisper 流水线")
                        self.transcription_service._run_pipeline(job)

                    # 检查最终状态
                    if job.canceled:
                        job.status = "canceled"
                        job.message = "已取消"
                    elif job.paused:
                        job.status = "paused"
                        job.message = "已暂停"
                    else:
                        job.status = "finished"
                        job.message = "完成"
                        logger.info(f"任务完成: {self.running_job_id}")

                except Exception as e:
                    job.status = "failed"
                    job.message = f"失败: {e}"
                    job.error = str(e)
                    logger.error(f"❌ 任务执行失败: {self.running_job_id} - {e}", exc_info=True)

                finally:
                    # 4. 清理资源（关键！）
                    finished_job_id = self.running_job_id
                    with self.lock:
                        self.running_job_id = None

                    # 资源大清洗
                    self._cleanup_resources()

                    # 保存任务最终状态到 job_meta.json
                    self.transcription_service.save_job_meta(job)

                    # 推送任务结束信号（单任务频道）
                    # 使用统一的命名空间前缀格式：signal.{signal_type}
                    signal_type = "job_complete" if job.status == "finished" else f"job_{job.status}"
                    self.sse_manager.broadcast_sync(
                        f"job:{job.job_id}",
                        f"signal.{signal_type}",
                        {
                            "signal": signal_type,
                            "job_id": job.job_id,
                            "message": job.message,
                            "status": job.status,
                            "percent": round(job.progress, 1)
                        }
                    )

                    # 推送全局SSE通知
                    self._notify_job_status(job.job_id, job.status)
                    self._notify_queue_change()

                    # 5. 检查是否需要恢复被中断的任务（强制插队后的自动恢复）
                    self._try_restore_interrupted_job(finished_job_id, job.status)

                    # 保存队列状态
                    self._save_state()

            except Exception as e:
                logger.error(f"Worker循环异常: {e}", exc_info=True)
                time.sleep(1)

        logger.info("Worker循环已停止")

    async def _run_dual_alignment_pipeline(self, job: 'JobState', preset_id: str):
        """
        运行双流对齐流水线 (V3.0+ 新架构)

        Args:
            job: 任务状态对象
            preset_id: 预设 ID
        """
        from app.pipelines import (
            AudioProcessingPipeline,
            AudioProcessingConfig,
            DualAlignmentPipeline,
            DualAlignmentConfig,
            get_audio_processing_pipeline,
            get_dual_alignment_pipeline
        )
        from app.services.streaming_subtitle import get_streaming_subtitle_manager, remove_streaming_subtitle_manager
        from app.services.progress_tracker import get_progress_tracker, remove_progress_tracker, ProcessPhase
        from app.services.sse_service import get_sse_manager
        from pathlib import Path

        def push_signal_event(sse_manager, job_id: str, signal_code: str, message: str = ""):
            """推送信号事件"""
            sse_manager.broadcast_sync(
                f"job:{job_id}",
                f"signal.{signal_code}",
                {"signal": signal_code, "message": message}
            )

        # 初始化管理器
        subtitle_manager = get_streaming_subtitle_manager(job.job_id)
        progress_tracker = get_progress_tracker(job.job_id, preset_id)
        sse_manager = get_sse_manager()

        try:
            logger.info(f"[双流对齐] 开始处理任务: {job.job_id}, preset={preset_id}")

            # === 预触发 Proxy 生成（不阻塞主流程）===
            await self._maybe_trigger_proxy_generation(job)

            # 阶段 1: 音频前处理
            progress_tracker.start_phase(ProcessPhase.EXTRACT, 1, "音频前处理...")

            audio_config = AudioProcessingConfig()
            audio_pipeline = get_audio_processing_pipeline(
                job_id=job.job_id,
                config=audio_config,
                logger=logger
            )

            audio_result = await audio_pipeline.process(job.input_path)

            # 保存音频文件供波形图使用
            import soundfile as sf
            audio_path = Path(job.dir) / "audio.wav"
            sf.write(str(audio_path), audio_result.audio_array, audio_result.sample_rate)
            logger.info(f"音频文件已保存: {audio_path}")

            progress_tracker.complete_phase(ProcessPhase.EXTRACT)

            # 阶段 2: 双流对齐处理
            total_chunks = len(audio_result.chunks)
            progress_tracker.start_phase(ProcessPhase.SENSEVOICE, total_chunks, "双流对齐...")

            dual_config = DualAlignmentConfig()
            dual_pipeline = get_dual_alignment_pipeline(
                job_id=job.job_id,
                config=dual_config,
                logger=logger
            )

            # 处理所有 Chunks (使用 run 方法)
            results = await dual_pipeline.run(audio_result.chunks)

            progress_tracker.complete_phase(ProcessPhase.SENSEVOICE)

            # 阶段 3: 生成字幕文件
            progress_tracker.start_phase(ProcessPhase.SRT, 1, "生成字幕...")

            all_sentences = []
            for result in results:
                all_sentences.extend(result.sentences)

            # 按时间排序
            all_sentences.sort(key=lambda s: s.start)

            # 生成 SRT
            output_path = str(Path(job.dir) / f"{job.job_id}.srt")
            self.transcription_service._generate_subtitle_from_sentences(
                all_sentences,
                output_path,
                include_translation=False
            )

            progress_tracker.complete_phase(ProcessPhase.SRT)

            # 完成
            job.status = 'completed'
            push_signal_event(sse_manager, job.job_id, "job_complete", "处理完成")

            logger.info(f"[双流对齐] 任务完成: {job.job_id}")

        except Exception as e:
            logger.error(f"[双流对齐] 任务失败: {e}", exc_info=True)
            job.status = 'failed'
            job.error = str(e)
            push_signal_event(sse_manager, job.job_id, "job_failed", str(e))
            raise

        finally:
            # 清理资源
            remove_streaming_subtitle_manager(job.job_id)
            remove_progress_tracker(job.job_id)

    async def _maybe_trigger_proxy_generation(self, job: 'JobState'):
        """
        预检视频格式，提前触发 Proxy 生成（不阻塞主流程）

        在转录任务开始时调用，让 H265 等不兼容格式的视频提前开始转码，
        用户打开编辑器时可能已完成转码。
        """
        from app.services.media_prep_service import get_media_prep_service
        from app.api.routes.media_routes import (
            _get_video_codec, NEED_TRANSCODE_CODECS, NEED_TRANSCODE_FORMATS,
            BROWSER_COMPATIBLE_FORMATS, _find_video_file
        )

        try:
            job_dir = Path(job.dir)

            # 查找视频文件
            video_file = _find_video_file(job_dir)
            if not video_file:
                return

            # 检查是否需要转码
            needs_transcode = False
            if video_file.suffix.lower() in NEED_TRANSCODE_FORMATS:
                needs_transcode = True
            elif video_file.suffix.lower() in BROWSER_COMPATIBLE_FORMATS:
                codec = _get_video_codec(video_file)
                if codec and codec in NEED_TRANSCODE_CODECS:
                    needs_transcode = True

            if needs_transcode:
                proxy_path = job_dir / "proxy.mp4"
                if not proxy_path.exists():
                    # 提前入队（低优先级20，不抢转录任务的资源）
                    media_prep = get_media_prep_service()
                    enqueued = media_prep.enqueue_proxy(
                        job.job_id, video_file, proxy_path, priority=20
                    )
                    if enqueued:
                        logger.info(f"[Proxy预生成] 检测到不兼容格式，提前入队: {job.job_id}")
        except Exception as e:
            # 预触发失败不影响主流程
            logger.warning(f"[Proxy预生成] 预触发失败，忽略: {e}")

    def _cleanup_resources(self):
        """
        资源大清洗（增强版）

        策略:
        1. 清理 Whisper 模型（1-3GB）
        2. 保留最近使用的3个对齐模型（LRU，共~600MB）
        3. GC + CUDA 清理
        """
        logger.info("开始资源清理（增强版）...")

        # 1. 清空 Whisper 模型缓存
        try:
            self.transcription_service.clear_model_cache()
        except Exception as e:
            logger.warning(f"清空模型缓存失败: {e}")

        # 2. Python垃圾回收
        gc.collect()
        logger.debug("  - Python GC 完成")

        # 3. CUDA显存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # 记录显存状态（调试用）
            try:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.debug(f"  - 显存: 已分配 {memory_allocated:.2f}GB, 已保留 {memory_reserved:.2f}GB")
            except:
                pass

            logger.debug("  - CUDA缓存已清空")

        # 4. 等待资源释放
        time.sleep(1)

        logger.info("资源清理完成")

    def _try_restore_interrupted_job(self, finished_job_id: str, finished_status: str):
        """
        尝试恢复被强制中断的任务

        当插队任务完成后，自动将被中断的任务重新加入队列头部

        Args:
            finished_job_id: 刚完成的任务ID
            finished_status: 刚完成任务的状态
        """
        with self.lock:
            # 检查是否有被中断的任务需要恢复
            if not self.interrupted_job_id:
                return

            interrupted_job = self.jobs.get(self.interrupted_job_id)
            if not interrupted_job:
                logger.warning(f"被中断的任务不存在: {self.interrupted_job_id}")
                self.interrupted_job_id = None
                return

            # 只有插队任务正常完成时才自动恢复
            # 如果插队任务失败或被取消，不自动恢复（让用户决定）
            if finished_status == "finished":
                # 将被中断的任务重新加入队列头部
                if self.interrupted_job_id not in self.queue:
                    self.queue.appendleft(self.interrupted_job_id)
                    interrupted_job.status = "queued"
                    interrupted_job.paused = False
                    interrupted_job.message = "插队任务已完成，自动恢复执行"
                    logger.info(f"[自动恢复] 被中断的任务已恢复到队头: {self.interrupted_job_id}")
            else:
                # 插队任务未正常完成，被中断任务保持暂停状态
                interrupted_job.message = f"插队任务{finished_status}，需手动恢复"
                logger.info(f"[未恢复] 插队任务状态={finished_status}，被中断任务需手动恢复: {self.interrupted_job_id}")

            # 清除中断标记
            self.interrupted_job_id = None

    # ========== 全局SSE通知方法 (V3.0) ==========

    def _notify_queue_change(self):
        """推送队列变化事件到全局SSE"""
        with self.lock:
            data = {
                "queue": list(self.queue),
                "running": self.running_job_id,
                "interrupted": self.interrupted_job_id,
                "timestamp": time.time()
            }

        self.sse_manager.broadcast_sync("global", "queue_update", data)
        logger.debug(f"[全局SSE] 推送队列变化: queue={len(data['queue'])}个, running={data['running']}")

    def _notify_job_status(self, job_id: str, status: str):
        """推送任务状态变化到全局SSE"""
        job = self.jobs.get(job_id)
        if not job:
            return

        data = {
            "id": job_id,
            "status": status,
            "percent": round(job.progress, 1),  # 统一字段名为 percent，保留1位小数
            "message": job.message,
            "filename": job.filename,
            "phase": job.phase,  # 新增：阶段信息
            "timestamp": time.time()
        }

        self.sse_manager.broadcast_sync("global", "job_status", data)
        logger.debug(f"[全局SSE] 推送任务状态: {job_id[:8]}... -> {status}")

    def _notify_job_progress(self, job_id: str):
        """推送任务进度更新到全局SSE（低频调用，节省带宽）"""
        job = self.jobs.get(job_id)
        if not job:
            return

        data = {
            "id": job_id,
            "percent": round(job.progress, 1),  # 统一字段名为 percent，保留1位小数
            "phase": job.phase,
            "phase_percent": round(job.phase_percent, 1),  # 新增：阶段内进度
            "message": job.message,
            "processed": job.processed,
            "total": job.total,
            "timestamp": time.time()
        }

        self.sse_manager.broadcast_sync("global", "job_progress", data)

    def _notify_job_signal(self, job_id: str, signal: str):
        """
        推送关键信号到单任务SSE频道

        用于暂停/取消/恢复等关键操作，确保 EditorView 能收到状态变更通知

        Args:
            job_id: 任务ID
            signal: 信号类型 (job_paused, job_canceled, job_resumed)
        """
        job = self.jobs.get(job_id)
        if not job:
            return

        data = {
            "signal": signal,
            "job_id": job_id,
            "status": job.status,
            "message": job.message,
            "percent": round(job.progress, 1)
        }

        self.sse_manager.broadcast_sync(f"job:{job_id}", f"signal.{signal}", data)
        logger.debug(f"[单任务SSE] 推送信号: {job_id[:8]}... -> signal.{signal}")

    def _load_settings(self):
        """加载队列设置"""
        if not self.settings_file.exists():
            logger.info("无队列设置文件，使用默认设置")
            return

        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            self._default_prioritize_mode = settings.get("default_prioritize_mode", "gentle")
            logger.info(f"加载队列设置: 默认插队模式={self._default_prioritize_mode}")
        except Exception as e:
            logger.warning(f"加载队列设置失败: {e}")

    def _save_settings(self):
        """保存队列设置"""
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)

            settings = {
                "default_prioritize_mode": self._default_prioritize_mode,
                "timestamp": time.time()
            }

            temp_path = self.settings_file.with_suffix(".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2)

            temp_path.replace(self.settings_file)
            logger.debug("队列设置已保存")
        except Exception as e:
            logger.error(f"保存队列设置失败: {e}")

    def get_settings(self) -> dict:
        """获取队列设置"""
        return {
            "default_prioritize_mode": self._default_prioritize_mode
        }

    def update_settings(self, default_prioritize_mode: Optional[str] = None) -> dict:
        """
        更新队列设置

        Args:
            default_prioritize_mode: 默认插队模式 ("gentle" 或 "force")

        Returns:
            更新后的设置
        """
        if default_prioritize_mode is not None:
            if default_prioritize_mode not in ("gentle", "force"):
                raise ValueError(f"无效的插队模式: {default_prioritize_mode}")
            self._default_prioritize_mode = default_prioritize_mode
            logger.info(f"更新默认插队模式: {default_prioritize_mode}")

        self._save_settings()
        return self.get_settings()

    def _save_state(self):
        """
        持久化队列状态到磁盘

        格式:
        {
          "queue": ["job_id1", "job_id2"],
          "running": "job_id3",
          "interrupted": "job_id4",  // 被强制中断的任务
          "paused": ["job_id5", "job_id6"],  // 暂停的任务列表
          "timestamp": 1234567890.0
        }
        """
        with self.lock:
            # 收集所有暂停状态的任务
            paused_jobs = [
                job_id for job_id, job in self.jobs.items()
                if job.status == "paused" or job.paused
            ]
            state = {
                "queue": list(self.queue),
                "running": self.running_job_id,
                "interrupted": self.interrupted_job_id,
                "paused": paused_jobs,  # 新增：保存暂停的任务列表
                "timestamp": time.time()
            }

        try:
            # 确保目录存在
            self.queue_file.parent.mkdir(parents=True, exist_ok=True)

            # 原子写入（临时文件 + rename）
            temp_path = self.queue_file.with_suffix(".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)

            # 原子替换
            temp_path.replace(self.queue_file)
            logger.debug("队列状态已保存")
        except Exception as e:
            logger.error(f"保存队列状态失败: {e}")

    def _load_job_for_recovery(self, job_id: str) -> Optional[JobState]:
        """
        加载任务用于恢复（优先从已有缓存获取，其次从 job_meta.json 加载，最后从 checkpoint 加载）

        这是重启恢复的核心方法，确保能正确恢复任务状态

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 恢复的任务状态对象
        """
        # 0. 优先检查 transcription_service.jobs（可能已经被 _load_all_jobs_from_disk 加载）
        if job_id in self.transcription_service.jobs:
            job = self.transcription_service.jobs[job_id]
            logger.info(f"从 transcription_service 缓存获取任务: {job_id}")
            return job

        # 1. 从 job_meta.json 加载（包含完整的任务元信息）
        job = self.transcription_service.load_job_meta(job_id)
        if job:
            logger.info(f"从 job_meta.json 恢复任务: {job_id}")
            return job

        # 2. 降级：从 checkpoint 恢复（兼容旧版本）
        job = self.transcription_service.restore_job_from_checkpoint(job_id)
        if job:
            logger.info(f"从 checkpoint 恢复任务（旧版兼容）: {job_id}")
            # 同时保存 job_meta.json 以便下次直接加载
            self.transcription_service.save_job_meta(job)
            return job

        logger.warning(f"无法恢复任务: {job_id}")
        return None

    def _load_state(self):
        """
        启动时恢复队列状态

        恢复逻辑:
        1. 读取 queue_state.json
        2. 优先从 job_meta.json 恢复任务（包含完整状态）
        3. 如果有 running 任务，自动加入队列头部继续执行
        4. 恢复队列中的其他任务
        5. 恢复 interrupted 任务（被强制中断的任务）
        """
        if not self.queue_file.exists():
            logger.info("无队列状态文件，从空队列启动")
            return

        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            logger.info(f"加载队列状态: {state}")

            # 1. 恢复 running 任务（如果有）- 意外断电/崩溃场景
            running_id = state.get("running")
            if running_id:
                job = self._load_job_for_recovery(running_id)
                if job:
                    # 自动恢复：设为 queued 状态，放队列头部继续执行
                    job.status = "queued"
                    job.paused = False
                    job.message = "程序重启，任务自动恢复"
                    self.jobs[running_id] = job
                    self.queue.appendleft(running_id)  # 放队头
                    # 同步到 transcription_service.jobs（确保 SSE 路由能找到任务）
                    self.transcription_service.jobs[running_id] = job
                    # 更新 job_meta.json 中的状态
                    self.transcription_service.save_job_meta(job)
                    logger.info(f"恢复中断任务到队头（自动继续）: {running_id}")

            # 2. 恢复队列中的任务
            for job_id in state.get("queue", []):
                # 避免重复（running任务已经加入队列了）
                if job_id == running_id:
                    continue

                job = self._load_job_for_recovery(job_id)
                if job:
                    job.status = "queued"
                    job.paused = False
                    job.message = f"排队中 (位置: {len(self.queue) + 1})"
                    self.jobs[job_id] = job
                    self.queue.append(job_id)
                    # 同步到 transcription_service.jobs（确保 SSE 路由能找到任务）
                    self.transcription_service.jobs[job_id] = job
                    # 更新 job_meta.json 中的状态
                    self.transcription_service.save_job_meta(job)
                    logger.info(f"恢复排队任务: {job_id}")

            # 3. 恢复 interrupted 任务（被强制中断的任务）
            interrupted_id = state.get("interrupted")
            if interrupted_id and interrupted_id not in self.jobs:
                job = self._load_job_for_recovery(interrupted_id)
                if job:
                    job.status = "queued"
                    job.paused = False
                    job.message = "程序重启，被中断任务自动恢复"
                    self.jobs[interrupted_id] = job
                    self.queue.append(interrupted_id)
                    # 同步到 transcription_service.jobs（确保 SSE 路由能找到任务）
                    self.transcription_service.jobs[interrupted_id] = job
                    # 更新 job_meta.json 中的状态
                    self.transcription_service.save_job_meta(job)
                    logger.info(f"恢复被中断任务到队列: {interrupted_id}")

            # 4. 恢复暂停的任务（保持暂停状态）
            for job_id in state.get("paused", []):
                # 避免重复（可能已经被前面的逻辑处理过）
                if job_id in self.jobs:
                    continue

                job = self._load_job_for_recovery(job_id)
                if job:
                    # 保持暂停状态，不加入队列
                    job.status = "paused"
                    job.paused = True
                    job.message = "程序重启，暂停任务已恢复"
                    self.jobs[job_id] = job
                    # 同步到 transcription_service.jobs（确保 SSE 路由能找到任务）
                    self.transcription_service.jobs[job_id] = job
                    # 更新 job_meta.json 中的状态
                    self.transcription_service.save_job_meta(job)
                    logger.info(f"恢复暂停任务（保持暂停）: {job_id}")

            # 统计恢复情况
            paused_count = len([j for j in self.jobs.values() if j.status == "paused"])
            logger.info(f"队列恢复完成: {len(self.queue)}个排队任务, {paused_count}个暂停任务")

        except Exception as e:
            logger.error(f"恢复队列状态失败: {e}")

    def prioritize_job(self, job_id: str, mode: Optional[str] = None) -> dict:
        """
        将任务移到队列头部（插队）

        Args:
            job_id: 要优先的任务ID
            mode: 插队模式
                - "gentle": 温和插队，放到队列头部，等当前任务完成后执行
                - "force": 强制插队，暂停当前任务A -> 执行B -> B完成后自动恢复A
                - None: 使用默认模式

        Returns:
            dict: 操作结果
                - success: 是否成功
                - mode: 实际使用的模式
                - interrupted_job_id: 被中断的任务ID（仅force模式）
        """
        # 使用默认模式
        if mode is None:
            mode = self._default_prioritize_mode

        if mode not in ("gentle", "force"):
            return {"success": False, "error": f"无效的插队模式: {mode}"}

        job = self.jobs.get(job_id)
        if not job:
            return {"success": False, "error": "任务不存在"}

        with self.lock:
            # 1. 如果任务已经在跑，无法插队
            if job_id == self.running_job_id:
                logger.info(f"任务已在执行，无需插队: {job_id}")
                return {"success": False, "error": "任务已在执行中"}

            # 2. 如果任务在队列中，移除
            if job_id in self.queue:
                self.queue.remove(job_id)

            # 3. 插到队头
            self.queue.appendleft(job_id)
            job.status = "queued"

            result = {
                "success": True,
                "mode": mode,
                "job_id": job_id,
                "interrupted_job_id": None
            }

            if mode == "gentle":
                # 温和插队：只放队头，不影响当前任务
                job.message = "优先执行（队列第1位）"
                logger.info(f"[温和插队] 任务已插队到队头: {job_id}")

            elif mode == "force":
                # 强制插队：暂停当前任务，记录以便自动恢复
                if self.running_job_id:
                    current_job = self.jobs.get(self.running_job_id)
                    if current_job:
                        current_job.paused = True
                        current_job.message = "被强制插队暂停，稍后自动恢复..."
                        # 记录被中断的任务，用于自动恢复
                        self.interrupted_job_id = self.running_job_id
                        result["interrupted_job_id"] = self.running_job_id
                        logger.info(f"[强制插队] 暂停当前任务: {self.running_job_id}, 插队任务: {job_id}")

                job.message = "强制插队（等待当前任务暂停）"

        # 保存队列状态
        self._save_state()

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)
        if mode == "force" and result.get("interrupted_job_id"):
            # 通知被中断的任务状态变化
            self._notify_job_status(result["interrupted_job_id"], "pausing")

        return result

    def reorder_queue(self, job_ids: list) -> bool:
        """
        重新排序队列

        Args:
            job_ids: 按新顺序排列的任务ID列表

        Returns:
            bool: 是否成功
        """
        with self.lock:
            # 验证所有job_id都在队列中
            current_queue_set = set(self.queue)
            new_queue_set = set(job_ids)

            if current_queue_set != new_queue_set:
                logger.warning(f"队列重排失败：任务ID不匹配")
                return False

            # 更新队列顺序
            self.queue.clear()
            for job_id in job_ids:
                self.queue.append(job_id)

            # 更新每个任务的消息
            for idx, job_id in enumerate(self.queue):
                job = self.jobs.get(job_id)
                if job:
                    job.message = f"排队中 (位置: {idx + 1})"

            logger.info(f"队列已重新排序: {list(self.queue)}")

        # 保存队列状态
        self._save_state()

        # 推送全局SSE通知
        self._notify_queue_change()

        return True

    def get_queue_status(self) -> dict:
        """
        获取队列状态摘要

        Returns:
            dict: 队列状态信息
        """
        with self.lock:
            return {
                "queue": list(self.queue),
                "running": self.running_job_id,
                "queue_length": len(self.queue),
                "jobs": {
                    job_id: {
                        "status": job.status,
                        "message": job.message,
                        "filename": job.filename,
                        "progress": job.progress
                    }
                    for job_id, job in self.jobs.items()
                }
            }

    def shutdown(self):
        """停止Worker线程"""
        logger.info("停止队列服务...")
        self.stop_event.set()
        self.worker_thread.join(timeout=5)
        logger.info("队列服务已停止")


# ========== 单例模式 ==========

_queue_service_instance: Optional[JobQueueService] = None


def get_queue_service(transcription_service=None) -> JobQueueService:
    """
    获取队列服务单例

    Args:
        transcription_service: 首次调用时必须提供

    Returns:
        JobQueueService: 队列服务实例
    """
    global _queue_service_instance
    if _queue_service_instance is None:
        if transcription_service is None:
            raise RuntimeError("首次调用必须提供transcription_service")
        _queue_service_instance = JobQueueService(transcription_service)
    return _queue_service_instance