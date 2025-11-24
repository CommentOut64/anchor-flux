"""
任务队列管理服务 - V2.3
核心功能: 串行执行，防止并发OOM，队列持久化
"""
import threading
import time
import logging
import gc
import json
import os
from collections import deque
from typing import Dict, Optional
from pathlib import Path
import torch

from models.job_models import JobState
from services.sse_service import get_sse_manager

logger = logging.getLogger(__name__)


class JobQueueService:
    """
    任务队列管理器

    职责:
    1. 维护任务队列 (FIFO)
    2. 单线程Worker循环
    3. 串行执行任务（同一时间只有1个running）
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

        # 依赖服务
        self.transcription_service = transcription_service
        self.sse_manager = get_sse_manager()

        # 控制信号
        self.stop_event = threading.Event()
        self.lock = threading.Lock()  # 保护queue和running_job_id

        # 持久化文件路径
        from core.config import config
        self.queue_file = Path(config.JOBS_DIR) / "queue_state.json"

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
        return True

    def cancel_job(self, job_id: str, delete_data: bool = False) -> bool:
        """
        取消任务

        Args:
            job_id: 任务ID
            delete_data: 是否删除任务数据

        Returns:
            bool: 是否成功
        """
        job = self.jobs.get(job_id)
        if not job:
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
            # 这里复用原有的清理逻辑
            result = self.transcription_service.cancel_job(job_id, delete_data=True)
        else:
            result = True

        # 保存队列状态
        self._save_state()
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
        logger.info("🔄 Worker循环已启动")

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

                # 2. 如果没有任务，休眠后继续
                if self.running_job_id is None:
                    time.sleep(1)
                    continue

                # 3. 执行任务（阻塞，直到完成/失败/暂停/取消）
                job = self.jobs[self.running_job_id]
                logger.info(f"🚀 开始执行任务: {self.running_job_id}")

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
                        logger.info(f"✅ 任务完成: {self.running_job_id}")

                except Exception as e:
                    job.status = "failed"
                    job.message = f"失败: {e}"
                    job.error = str(e)
                    logger.error(f"❌ 任务执行失败: {self.running_job_id} - {e}", exc_info=True)

                finally:
                    # 4. 清理资源（关键！）
                    with self.lock:
                        self.running_job_id = None

                    # 资源大清洗
                    self._cleanup_resources()

                    # 推送任务结束信号
                    self.sse_manager.broadcast_sync(
                        f"job:{job.job_id}",
                        "signal",
                        {
                            "code": f"job_{job.status}",
                            "message": job.message,
                            "status": job.status
                        }
                    )

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
        logger.info("🧹 开始资源清理（增强版）...")

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

    def _save_state(self):
        """
        持久化队列状态到磁盘

        格式:
        {
          "queue": ["job_id1", "job_id2"],
          "running": "job_id3",
          "timestamp": 1234567890.0
        }
        """
        with self.lock:
            state = {
                "queue": list(self.queue),
                "running": self.running_job_id,
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

            logger.info(f"队列恢复完成: {len(self.queue)}个任务")

        except Exception as e:
            logger.error(f"恢复队列状态失败: {e}")

    def shutdown(self):
        """停止Worker线程"""
        logger.info("🛑 停止队列服务...")
        self.stop_event.set()
        self.worker_thread.join(timeout=5)
        logger.info("✅ 队列服务已停止")


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