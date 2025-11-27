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

from models.job_models import JobState
from services.sse_service import get_sse_manager

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
        from core.config import config
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

        # 保存队列状态
        self._save_state()

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
                job.message = "暂停中..."
                logger.info(f"设置暂停标志: {job_id}")
            elif job_id in self.queue:
                # 还在排队的任务：直接从队列移除
                self.queue.remove(job_id)
                job.status = "paused"
                job.message = "已暂停（未开始）"
                logger.info(f"从队列移除: {job_id}")

        # 保存队列状态
        self._save_state()

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)
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

        # 保存队列状态
        self._save_state()

        # 推送全局SSE通知
        self._notify_queue_change()
        self._notify_job_status(job_id, job.status)
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

                # 2. 如果没有任务，休眠后继续
                if self.running_job_id is None:
                    time.sleep(1)
                    continue

                # 3. 执行任务（阻塞，直到完成/失败/暂停/取消）
                job = self.jobs[self.running_job_id]
                logger.info(f" 开始执行任务: {self.running_job_id}")

                try:
                    # 调用原有的转录流程（会阻塞到任务结束）
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

                    # 推送任务结束信号（单任务频道）
                    self.sse_manager.broadcast_sync(
                        f"job:{job.job_id}",
                        "signal",
                        {
                            "code": f"job_{job.status}",
                            "message": job.message,
                            "status": job.status
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

        logger.info("🛑 Worker循环已停止")

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
            "progress": job.progress,
            "message": job.message,
            "filename": job.filename,
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
            "progress": job.progress,
            "message": job.message,
            "phase": job.phase,
            "processed": job.processed,
            "total": job.total,
            "timestamp": time.time()
        }

        self.sse_manager.broadcast_sync("global", "job_progress", data)

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
          "timestamp": 1234567890.0
        }
        """
        with self.lock:
            state = {
                "queue": list(self.queue),
                "running": self.running_job_id,
                "interrupted": self.interrupted_job_id,
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

    def _load_state(self):
        """
        启动时恢复队列状态

        恢复逻辑:
        1. 读取queue_state.json
        2. 如果有running任务，检查checkpoint是否存在
        3. 恢复running任务为paused，放队列头部
        4. 恢复队列中的其他任务
        5. 恢复interrupted任务（被强制中断的任务）
        """
        if not self.queue_file.exists():
            logger.info("无队列状态文件，从空队列启动")
            return

        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            logger.info(f"加载队列状态: {state}")

            # 1. 恢复running任务（如果有）
            running_id = state.get("running")
            if running_id:
                # 尝试从checkpoint恢复
                job = self.transcription_service.restore_job_from_checkpoint(running_id)
                if job:
                    # 安全起见，改为paused，不自动开始
                    job.status = "paused"
                    job.message = "程序重启，任务已暂停"
                    self.jobs[running_id] = job
                    self.queue.appendleft(running_id)  # 放队头
                    logger.info(f"恢复中断任务到队头: {running_id}")
                else:
                    logger.warning(f"无法恢复running任务: {running_id}")

            # 2. 恢复队列中的任务
            for job_id in state.get("queue", []):
                # 避免重复（running任务已经加入队列了）
                if job_id == running_id:
                    continue

                # 尝试恢复任务
                job = self.transcription_service.restore_job_from_checkpoint(job_id)
                if job:
                    self.jobs[job_id] = job
                    job.status = "queued"
                    job.message = f"排队中 (位置: {len(self.queue) + 1})"
                    self.queue.append(job_id)
                    logger.info(f"恢复排队任务: {job_id}")
                else:
                    logger.warning(f"跳过无效任务: {job_id}")

            # 3. 恢复interrupted任务（被强制中断的任务）
            interrupted_id = state.get("interrupted")
            if interrupted_id and interrupted_id not in self.jobs:
                job = self.transcription_service.restore_job_from_checkpoint(interrupted_id)
                if job:
                    job.status = "paused"
                    job.message = "程序重启，被中断任务已暂停"
                    self.jobs[interrupted_id] = job
                    # 不加入队列，等用户手动恢复
                    logger.info(f"恢复被中断任务: {interrupted_id}")

            logger.info(f"队列恢复完成: {len(self.queue)}个任务")

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