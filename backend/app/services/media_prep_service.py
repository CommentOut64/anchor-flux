"""
媒体准备服务 - 独立于转录流水线
职责: 波形图生成、缩略图生成、Proxy 转码
特点: 独立线程执行，不阻塞主流水线
"""
import threading
import queue
import time
import logging
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from concurrent.futures import ThreadPoolExecutor

from app.core.config import config

logger = logging.getLogger(__name__)

# 任务类型
TaskType = Literal["proxy_720p", "waveform", "thumbnail"]


class MediaPrepService:
    """
    媒体准备服务

    职责:
    1. Proxy 视频转码 (H265 -> H264)
    2. 波形图生成
    3. 缩略图生成

    特点:
    - 独立线程池执行，不阻塞转录流水线
    - 任务队列 + 优先级排序
    - SSE 进度推送
    """

    def __init__(self, max_workers: int = 1):
        """
        初始化媒体准备服务

        Args:
            max_workers: 最大并行工作线程数（默认1，避免多个FFmpeg抢CPU）
        """
        # 任务队列（优先级队列：priority越小越优先）
        self.task_queue = queue.PriorityQueue()

        # 任务状态 { job_id: { type: status_dict } }
        self.task_status: Dict[str, Dict[str, Any]] = {}

        # 线程池（独立于主事件循环）
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="MediaPrep"
        )

        # 控制信号
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # 启动消费线程
        self.consumer_thread = threading.Thread(
            target=self._consumer_loop,
            daemon=True,
            name="MediaPrepConsumer"
        )
        self.consumer_thread.start()

        logger.info(f"MediaPrepService 已启动 (workers={max_workers})")

    def enqueue_proxy(self, job_id: str, video_path: Path, output_path: Path,
                      priority: int = 10) -> bool:
        """
        将 Proxy 转码任务加入队列

        Args:
            job_id: 任务ID
            video_path: 源视频路径
            output_path: 输出路径 (proxy.mp4)
            priority: 优先级 (0最高，默认10)

        Returns:
            bool: 是否成功加入队列
        """
        # 检查是否已在队列或执行中
        with self.lock:
            if job_id in self.task_status:
                status = self.task_status[job_id].get("proxy_720p", {})
                if status.get("status") in ["queued", "processing"]:
                    logger.info(f"[MediaPrep] Proxy任务已存在，跳过: {job_id}")
                    return False

            # 初始化状态
            if job_id not in self.task_status:
                self.task_status[job_id] = {}

            self.task_status[job_id]["proxy_720p"] = {
                "status": "queued",
                "progress": 0,
                "video_path": str(video_path),
                "output_path": str(output_path)
            }

        # 加入优先级队列 (priority, timestamp, task_data)
        task = (priority, time.time(), {
            "type": "proxy_720p",
            "job_id": job_id,
            "video_path": video_path,
            "output_path": output_path
        })
        self.task_queue.put(task)

        logger.info(f"[MediaPrep] Proxy任务已入队: {job_id} (priority={priority})")
        return True

    def get_proxy_status(self, job_id: str) -> Optional[Dict]:
        """获取 Proxy 任务状态"""
        with self.lock:
            if job_id in self.task_status:
                return self.task_status[job_id].get("proxy_720p")
        return None

    def is_proxy_in_progress(self, job_id: str) -> bool:
        """检查 Proxy 任务是否正在处理中"""
        status = self.get_proxy_status(job_id)
        if status:
            return status.get("status") in ["queued", "processing"]
        return False

    def _consumer_loop(self):
        """任务消费循环"""
        logger.info("[MediaPrep] 消费线程已启动")

        while not self.stop_event.is_set():
            try:
                # 从队列获取任务（阻塞，超时1秒）
                try:
                    priority, timestamp, task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # 分发任务
                task_type = task.get("type")
                job_id = task.get("job_id")

                logger.info(f"[MediaPrep] 开始执行任务: {task_type} - {job_id}")

                if task_type == "proxy_720p":
                    self._execute_proxy_task(task)

                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"[MediaPrep] 消费循环异常: {e}", exc_info=True)

        logger.info("[MediaPrep] 消费线程已停止")

    def _execute_proxy_task(self, task: dict):
        """
        执行 Proxy 转码任务（在独立线程中）
        """
        job_id = task["job_id"]
        video_path = Path(task["video_path"])
        output_path = Path(task["output_path"])

        # 更新状态
        with self.lock:
            self.task_status[job_id]["proxy_720p"]["status"] = "processing"

        try:
            # 获取视频时长
            duration = self._get_video_duration(video_path)

            # 构建 FFmpeg 命令
            ffmpeg_cmd = config.get_ffmpeg_command()
            cmd = [
                ffmpeg_cmd,
                '-i', str(video_path),
                '-vf', 'scale=-2:720',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-g', '30',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-progress', 'pipe:1',
                '-y',
                str(output_path)
            ]

            logger.info(f"[MediaPrep] 开始转码: {video_path.name} -> proxy.mp4")

            # 同步执行 FFmpeg
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags
            )

            # 解析进度
            for line in process.stdout:
                line_str = line.decode().strip()
                if line_str.startswith('out_time_ms='):
                    try:
                        out_time_ms = int(line_str.split('=')[1])
                        if duration > 0:
                            progress = min(100, (out_time_ms / 1000000) / duration * 100)
                            progress = round(progress, 1)

                            # 更新状态
                            with self.lock:
                                self.task_status[job_id]["proxy_720p"]["progress"] = progress

                            # 推送 SSE 进度
                            self._push_proxy_progress(job_id, progress)
                    except:
                        pass

            process.wait()

            if process.returncode == 0 and output_path.exists():
                # 成功
                with self.lock:
                    self.task_status[job_id]["proxy_720p"]["status"] = "completed"
                    self.task_status[job_id]["proxy_720p"]["progress"] = 100

                logger.info(f"[MediaPrep] 转码完成: {output_path}")
                self._push_proxy_progress(job_id, 100, completed=True)
            else:
                # 失败
                stderr = process.stderr.read().decode() if process.stderr else ""
                with self.lock:
                    self.task_status[job_id]["proxy_720p"]["status"] = "failed"
                    self.task_status[job_id]["proxy_720p"]["error"] = stderr[:500]

                logger.error(f"[MediaPrep] 转码失败: {stderr[:200]}")

        except Exception as e:
            with self.lock:
                self.task_status[job_id]["proxy_720p"]["status"] = "failed"
                self.task_status[job_id]["proxy_720p"]["error"] = str(e)

            logger.error(f"[MediaPrep] 转码异常: {e}", exc_info=True)

    def _get_video_duration(self, video_path: Path) -> float:
        """获取视频时长（秒）"""
        try:
            ffprobe_cmd = config.get_ffprobe_command()
            cmd = [
                ffprobe_cmd, '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]

            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                creationflags=creationflags
            )

            return float(result.stdout.strip())
        except:
            return 0

    def _push_proxy_progress(self, job_id: str, progress: float, completed: bool = False):
        """推送 Proxy 进度到 SSE"""
        try:
            from app.services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job_id}"
            event_type = "proxy_complete" if completed else "proxy_progress"

            data = {
                "job_id": job_id,
                "progress": progress,
                "completed": completed
            }

            if completed:
                data["video_url"] = f"/api/media/{job_id}/video"

            sse_manager.broadcast_sync(channel_id, event_type, data)
        except Exception as e:
            # SSE 推送失败不影响转码
            pass

    def shutdown(self):
        """关闭服务"""
        logger.info("[MediaPrep] 正在关闭...")
        self.stop_event.set()
        self.consumer_thread.join(timeout=5)
        self.executor.shutdown(wait=False)
        logger.info("[MediaPrep] 已关闭")


# ========== 单例模式 ==========

_media_prep_instance: Optional[MediaPrepService] = None


def get_media_prep_service() -> MediaPrepService:
    """获取媒体准备服务单例"""
    global _media_prep_instance
    if _media_prep_instance is None:
        _media_prep_instance = MediaPrepService()
    return _media_prep_instance


def shutdown_media_prep_service():
    """关闭媒体准备服务"""
    global _media_prep_instance
    if _media_prep_instance:
        _media_prep_instance.shutdown()
        _media_prep_instance = None
