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
from typing import Optional, Dict, Any, Literal, Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from app.core.config import config

logger = logging.getLogger(__name__)

# 任务类型
TaskType = Literal["preview_360p", "proxy_720p", "remux", "waveform", "thumbnail"]


class TranscodeDecision(Enum):
    """
    转码决策类型

    决策优先级:
    1. DIRECT_PLAY - 容器和编解码器都兼容，可直接播放
    2. REMUX_ONLY - 容器不兼容但编解码器兼容，仅需重封装（零转码）
    3. TRANSCODE_AUDIO - 仅音频编解码器不兼容
    4. TRANSCODE_VIDEO - 仅视频编解码器不兼容
    5. TRANSCODE_FULL - 完整转码
    """
    DIRECT_PLAY = "direct_play"          # 直接播放，无需任何处理
    REMUX_ONLY = "remux_only"            # 仅重封装（-c:v copy -c:a copy）
    TRANSCODE_AUDIO = "transcode_audio"  # 仅转码音频
    TRANSCODE_VIDEO = "transcode_video"  # 仅转码视频
    TRANSCODE_FULL = "transcode_full"    # 完整转码


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

    def enqueue_preview(self, job_id: str, video_path: Path, output_path: Path,
                        priority: int = 5) -> bool:
        """
        将 360p 预览视频任务加入队列（高优先级，快速生成）

        Args:
            job_id: 任务ID
            video_path: 源视频路径
            output_path: 输出路径 (preview_360p.mp4)
            priority: 优先级 (0最高，默认5，比720p更高)

        Returns:
            bool: 是否成功加入队列
        """
        # 检查是否已在队列或执行中
        with self.lock:
            if job_id in self.task_status:
                status = self.task_status[job_id].get("preview_360p", {})
                if status.get("status") in ["queued", "processing"]:
                    logger.info(f"[MediaPrep] 360p预览任务已存在，跳过: {job_id}")
                    return False

            # 初始化状态
            if job_id not in self.task_status:
                self.task_status[job_id] = {}

            self.task_status[job_id]["preview_360p"] = {
                "status": "queued",
                "progress": 0,
                "video_path": str(video_path),
                "output_path": str(output_path)
            }

        # 加入优先级队列 (priority越小越优先)
        task = (priority, time.time(), {
            "type": "preview_360p",
            "job_id": job_id,
            "video_path": video_path,
            "output_path": output_path
        })
        self.task_queue.put(task)

        logger.info(f"[MediaPrep] 360p预览任务已入队: {job_id} (priority={priority})")
        return True

    def get_proxy_status(self, job_id: str) -> Optional[Dict]:
        """获取 720p Proxy 任务状态"""
        with self.lock:
            if job_id in self.task_status:
                return self.task_status[job_id].get("proxy_720p")
        return None

    def get_preview_status(self, job_id: str) -> Optional[Dict]:
        """获取 360p 预览任务状态"""
        with self.lock:
            if job_id in self.task_status:
                return self.task_status[job_id].get("preview_360p")
        return None

    def is_proxy_in_progress(self, job_id: str) -> bool:
        """检查 Proxy 任务是否正在处理中"""
        status = self.get_proxy_status(job_id)
        if status:
            return status.get("status") in ["queued", "processing"]
        return False

    def analyze_transcode_decision(self, video_info: dict) -> TranscodeDecision:
        """
        智能分析转码决策

        决策优先级：
        1. 容器和编解码器都兼容 -> DIRECT_PLAY
        2. 容器不兼容但编解码器兼容 -> REMUX_ONLY（零转码，极速）
        3. 仅音频编解码器不兼容 -> TRANSCODE_AUDIO
        4. 视频编解码器不兼容 -> TRANSCODE_FULL

        Args:
            video_info: 媒体分析结果，包含 video.codec, audio.codec, container 等

        Returns:
            TranscodeDecision: 转码决策类型
        """
        compatibility = config.BROWSER_COMPATIBILITY

        # 获取编解码器信息
        video_codec = video_info.get('video', {}).get('codec', '').lower()
        audio_info = video_info.get('audio')
        audio_codec = audio_info.get('codec', '').lower() if audio_info else ''
        container = video_info.get('container', '').lower()

        # 确保容器格式以点号开头
        if container and not container.startswith('.'):
            container = f'.{container}'

        # 检查各项兼容性
        container_ok = container in compatibility['compatible_containers']
        video_ok = video_codec in compatibility['compatible_video_codecs']
        # 无音频流或音频编解码器兼容均视为音频OK
        audio_ok = not audio_codec or audio_codec in compatibility['compatible_audio_codecs']

        # 检查是否是必须转码的编解码器
        video_must_transcode = video_codec in compatibility['need_transcode_codecs']

        logger.debug(
            f"[MediaPrep] 转码决策分析: container={container}({container_ok}), "
            f"video={video_codec}({video_ok}), audio={audio_codec}({audio_ok}), "
            f"must_transcode={video_must_transcode}"
        )

        # 决策逻辑
        if video_must_transcode:
            # HEVC/VP9/AV1 等必须完整转码
            return TranscodeDecision.TRANSCODE_FULL

        if container_ok and video_ok and audio_ok:
            # 完全兼容，可直接播放
            return TranscodeDecision.DIRECT_PLAY

        if video_ok and audio_ok:
            # 编解码器兼容，仅需重封装容器（极速）
            return TranscodeDecision.REMUX_ONLY

        if video_ok and not audio_ok:
            # 仅音频不兼容
            return TranscodeDecision.TRANSCODE_AUDIO

        # 视频不兼容，需完整转码
        return TranscodeDecision.TRANSCODE_FULL

    def enqueue_remux(self, job_id: str, video_path: Path, output_path: Path,
                      priority: int = 3) -> bool:
        """
        将容器重封装任务加入队列（最高优先级，极速完成）

        容器重封装使用 -c:v copy -c:a copy，直接复制流，速度极快

        Args:
            job_id: 任务ID
            video_path: 源视频路径
            output_path: 输出路径 (remux.mp4)
            priority: 优先级 (0最高，默认3，比360p更高)

        Returns:
            bool: 是否成功加入队列
        """
        with self.lock:
            if job_id in self.task_status:
                status = self.task_status[job_id].get("remux", {})
                if status.get("status") in ["queued", "processing"]:
                    logger.info(f"[MediaPrep] 重封装任务已存在，跳过: {job_id}")
                    return False

            if job_id not in self.task_status:
                self.task_status[job_id] = {}

            self.task_status[job_id]["remux"] = {
                "status": "queued",
                "progress": 0,
                "video_path": str(video_path),
                "output_path": str(output_path)
            }

        task = (priority, time.time(), {
            "type": "remux",
            "job_id": job_id,
            "video_path": video_path,
            "output_path": output_path
        })
        self.task_queue.put(task)

        logger.info(f"[MediaPrep] 重封装任务已入队: {job_id} (priority={priority})")
        return True

    def get_remux_status(self, job_id: str) -> Optional[Dict]:
        """获取重封装任务状态"""
        with self.lock:
            if job_id in self.task_status:
                return self.task_status[job_id].get("remux")
        return None

    def get_full_task_status(self, job_id: str) -> Optional[Dict]:
        """
        获取任务的完整状态（用于前端刷新后恢复）

        Returns:
            {
                "state": "transcoding_720",  # 当前状态
                "progress": 45.5,            # 当前进度
                "decision": "transcode_full", # 转码决策
                "preview_360p": {...},
                "proxy_720p": {...},
                "remux": {...},
                "error": None
            }
        """
        with self.lock:
            if job_id not in self.task_status:
                return None

            status = self.task_status[job_id]
            preview = status.get("preview_360p", {})
            proxy = status.get("proxy_720p", {})
            remux = status.get("remux", {})

            # 确定当前状态
            state = "idle"
            progress = 0
            error = None

            if remux.get("status") == "processing":
                state = "remuxing"
                progress = remux.get("progress", 0)
            elif remux.get("status") == "completed":
                state = "ready_720p"
                progress = 100
            elif remux.get("status") == "failed":
                state = "error"
                error = remux.get("error")
            elif preview.get("status") == "processing":
                state = "transcoding_360"
                progress = preview.get("progress", 0)
            elif preview.get("status") == "completed":
                if proxy.get("status") == "processing":
                    state = "transcoding_720"
                    progress = proxy.get("progress", 0)
                elif proxy.get("status") == "completed":
                    state = "ready_720p"
                    progress = 100
                elif proxy.get("status") == "failed":
                    state = "error"
                    error = proxy.get("error")
                else:
                    state = "ready_360p"
                    progress = 100
            elif preview.get("status") == "failed":
                state = "error"
                error = preview.get("error")
            elif preview.get("status") == "queued":
                state = "analyzing"
                progress = 0

            return {
                "state": state,
                "progress": progress,
                "preview_360p": preview,
                "proxy_720p": proxy,
                "remux": remux,
                "error": error
            }

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

                if task_type == "preview_360p":
                    self._execute_preview_task(task)
                elif task_type == "proxy_720p":
                    self._execute_proxy_task(task)
                elif task_type == "remux":
                    self._execute_remux_task(task)

                self.task_queue.task_done()

            except Exception as e:
                logger.error(f"[MediaPrep] 消费循环异常: {e}", exc_info=True)

        logger.info("[MediaPrep] 消费线程已停止")

    def _execute_preview_task(self, task: dict):
        """
        执行 360p 预览视频转码任务（在独立线程中）
        使用配置中的编码参数，优先快速生成
        """
        job_id = task["job_id"]
        video_path = Path(task["video_path"])
        output_path = Path(task["output_path"])

        # 更新状态
        with self.lock:
            self.task_status[job_id]["preview_360p"]["status"] = "processing"

        try:
            # 获取视频时长
            duration = self._get_video_duration(video_path)

            # 从配置读取参数
            preview_config = config.PROXY_CONFIG.get('preview_360p', {})
            scale = preview_config.get('scale', 360)
            preset = preview_config.get('preset', 'ultrafast')
            crf = preview_config.get('crf', 28)
            gop = preview_config.get('gop', 30)
            keyint_min = preview_config.get('keyint_min', gop // 2)
            tune = preview_config.get('tune', 'fastdecode')
            include_audio = preview_config.get('audio', False)
            audio_bitrate = preview_config.get('audio_bitrate', '64k')

            # 构建 FFmpeg 命令 - 360p 快速预览（参数从配置读取）
            ffmpeg_cmd = config.get_ffmpeg_command()
            cmd = [
                ffmpeg_cmd,
                '-i', str(video_path),
                '-vf', f'scale=-2:{scale}',       # 分辨率
                '-c:v', 'libx264',
                '-preset', preset,                 # 编码预设
                '-tune', tune,                     # 解码优化
                '-crf', str(crf),                  # 质量
                '-g', str(gop),                    # 关键帧间隔
                '-keyint_min', str(keyint_min),    # 最小关键帧间隔
                '-sc_threshold', '0',              # 禁用场景检测，保证 GOP 稳定
            ]

            # 音频处理
            if include_audio:
                cmd.extend(['-c:a', 'aac', '-b:a', audio_bitrate])
            else:
                cmd.append('-an')  # 无音频

            # 输出优化
            cmd.extend([
                '-movflags', '+faststart',  # 优化拖动
                '-progress', 'pipe:1',
                '-y',
                str(output_path)
            ])

            logger.info(f"[MediaPrep] 开始转码 360p 预览: {video_path.name} -> preview_360p.mp4")

            # 同步执行 FFmpeg
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # 丢弃 stderr，避免缓冲区阻塞
                creationflags=creationflags
            )

            # 解析进度
            last_logged_progress = 0  # 记录上次日志输出的进度
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
                                self.task_status[job_id]["preview_360p"]["progress"] = progress

                            # 推送 SSE 进度
                            self._push_preview_progress(job_id, progress)

                            # 每10%输出一次日志
                            if progress >= last_logged_progress + 10:
                                logger.info(f"[MediaPrep] 360p 转码进度: {job_id} - {progress:.1f}%")
                                last_logged_progress = int(progress / 10) * 10
                    except:
                        pass

            process.wait()

            if process.returncode == 0 and output_path.exists():
                # 成功
                with self.lock:
                    self.task_status[job_id]["preview_360p"]["status"] = "completed"
                    self.task_status[job_id]["preview_360p"]["progress"] = 100

                logger.info(f"[MediaPrep] 360p预览转码完成: {output_path}")
                self._push_preview_progress(job_id, 100, completed=True)

                # 【新增】360p完成后，延迟10秒检查队列是否空闲，如果空闲则自动启动720p
                # 使用独立线程避免阻塞当前任务
                import threading
                def delayed_720p_check():
                    import time
                    time.sleep(10)  # 等待10秒

                    # 检查队列是否空闲（使用 busy 方法取反）
                    queue_idle = not self._is_transcription_queue_busy()
                    logger.info(f"[MediaPrep] 360p完成后10秒检查: 队列空闲={queue_idle}")

                    if queue_idle:
                        # 队列空闲，自动启动720p
                        proxy_720p = output_path.parent / "proxy_720p.mp4"
                        if not proxy_720p.exists():
                            # 检查720p是否已在队列中
                            proxy_status = self.get_proxy_status(job_id)
                            if not proxy_status or proxy_status.get("status") not in ["queued", "processing"]:
                                logger.info(f"[MediaPrep] 360p完成后队列空闲，自动启动720p: {job_id}")
                                self.enqueue_proxy(job_id, video_path, proxy_720p, priority=10)

                thread = threading.Thread(target=delayed_720p_check, daemon=True)
                thread.start()
            else:
                # 失败
                error_msg = f"FFmpeg 返回码: {process.returncode}"
                with self.lock:
                    self.task_status[job_id]["preview_360p"]["status"] = "failed"
                    self.task_status[job_id]["preview_360p"]["error"] = error_msg

                logger.error(f"[MediaPrep] 360p预览转码失败: {error_msg}")

        except Exception as e:
            with self.lock:
                self.task_status[job_id]["preview_360p"]["status"] = "failed"
                self.task_status[job_id]["preview_360p"]["error"] = str(e)

            logger.error(f"[MediaPrep] 360p预览转码异常: {e}", exc_info=True)

    def _execute_proxy_task(self, task: dict):
        """
        执行 720p Proxy 转码任务（在独立线程中）
        特点：检测转录队列状态，如有新任务则降低优先级，不阻塞转录
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

            # 检查转录队列是否繁忙（有新任务进入）
            queue_busy = self._is_transcription_queue_busy()

            # 从配置读取参数
            proxy_config = config.PROXY_CONFIG.get('proxy_720p', {})
            scale = proxy_config.get('scale', 720)
            preset = proxy_config.get('preset', 'fast')
            crf = proxy_config.get('crf', 23)
            gop = proxy_config.get('gop', 30)
            keyint_min = proxy_config.get('keyint_min', gop // 2)
            audio_bitrate = proxy_config.get('audio_bitrate', '128k')
            audio_sample_rate = proxy_config.get('audio_sample_rate', 44100)

            # 构建 FFmpeg 命令 - 720p 高清（参数从配置读取）
            ffmpeg_cmd = config.get_ffmpeg_command()
            cmd = [
                ffmpeg_cmd,
                '-i', str(video_path),
                '-threads', '0',                   # 自动线程
                '-vf', f'scale=-2:{scale}',        # 分辨率
                '-c:v', 'libx264',
                '-preset', preset,                  # 编码预设
                '-crf', str(crf),                   # 质量
                '-g', str(gop),                     # 关键帧间隔
                '-keyint_min', str(keyint_min),     # 最小关键帧间隔
                '-sc_threshold', '0',               # 禁用场景检测，保证 GOP 稳定
                '-c:a', 'aac',                      # 音频编码
                '-b:a', audio_bitrate,              # 音频码率
                '-ar', str(audio_sample_rate),      # 音频采样率
                '-movflags', '+faststart',          # 优化网络播放
                '-progress', 'pipe:1',
                '-y',
                str(output_path)
            ]

            if queue_busy:
                logger.info(f"[MediaPrep] 检测到队列繁忙，720p 转码将使用低优先级: {job_id}")
            else:
                logger.info(f"[MediaPrep] 开始 720p 转码: {video_path.name} -> proxy.mp4")

            # 启动 FFmpeg 进程
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # 丢弃 stderr，避免缓冲区阻塞
                creationflags=creationflags
            )

            # 如果队列繁忙，降低 FFmpeg 进程优先级
            if queue_busy:
                self._set_low_priority(process.pid)

            # 解析进度
            last_logged_progress = 0  # 记录上次日志输出的进度
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

                            # 每10%输出一次日志
                            if progress >= last_logged_progress + 10:
                                logger.info(f"[MediaPrep] 720p 转码进度: {job_id} - {progress:.1f}%")
                                last_logged_progress = int(progress / 10) * 10
                    except:
                        pass

            process.wait()

            if process.returncode == 0 and output_path.exists():
                # 成功
                with self.lock:
                    self.task_status[job_id]["proxy_720p"]["status"] = "completed"
                    self.task_status[job_id]["proxy_720p"]["progress"] = 100

                logger.info(f"[MediaPrep] 720p 转码完成: {output_path}")
                self._push_proxy_progress(job_id, 100, completed=True)
            else:
                # 失败
                error_msg = f"FFmpeg 返回码: {process.returncode}"
                with self.lock:
                    self.task_status[job_id]["proxy_720p"]["status"] = "failed"
                    self.task_status[job_id]["proxy_720p"]["error"] = error_msg

                logger.error(f"[MediaPrep] 720p 转码失败: {error_msg}")

        except Exception as e:
            with self.lock:
                self.task_status[job_id]["proxy_720p"]["status"] = "failed"
                self.task_status[job_id]["proxy_720p"]["error"] = str(e)

            logger.error(f"[MediaPrep] 720p 转码异常: {e}", exc_info=True)

    def _is_transcription_queue_busy(self) -> bool:
        """检查转录队列是否繁忙（有活跃任务）"""
        try:
            from app.services.job_queue_service import get_job_queue

            job_queue = get_job_queue()
            if not job_queue:
                return False

            # 检查是否有正在处理或等待的任务
            active_jobs = [
                j for j in job_queue.jobs.values()
                if j.status in ['pending', 'processing', 'queued']
            ]

            return len(active_jobs) > 0

        except Exception as e:
            logger.debug(f"[MediaPrep] 检查队列状态失败: {e}")
            return False

    def _set_low_priority(self, pid: int):
        """设置进程为低优先级（Windows 和 Linux）"""
        try:
            if os.name == 'nt':
                # Windows: 使用 psutil 设置为低于正常优先级
                try:
                    import psutil
                    p = psutil.Process(pid)
                    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    logger.info(f"[MediaPrep] 已设置进程 {pid} 为低优先级（Windows）")
                except ImportError:
                    # 如果没有 psutil，使用 wmic（备选方案）
                    subprocess.run(
                        ['wmic', 'process', 'where', f'ProcessId={pid}', 'CALL', 'setpriority', '16384'],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    logger.info(f"[MediaPrep] 已设置进程 {pid} 为低优先级（wmic）")
            else:
                # Linux/Unix: 使用 nice 值
                try:
                    import psutil
                    p = psutil.Process(pid)
                    p.nice(10)  # nice 值 10（较低优先级）
                    logger.info(f"[MediaPrep] 已设置进程 {pid} 为低优先级（nice=10）")
                except:
                    pass

        except Exception as e:
            logger.debug(f"[MediaPrep] 设置低优先级失败: {e}")

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

    def _push_preview_progress(self, job_id: str, progress: float, completed: bool = False):
        """推送 360p 预览进度到 SSE"""
        channel_id = f"job:{job_id}"
        event_type = "preview_360p_complete" if completed else "preview_360p_progress"

        data = {
            "job_id": job_id,
            "progress": progress,
            "completed": completed,
            "resolution": "360p"
        }

        if completed:
            data["video_url"] = f"/api/media/{job_id}/video/preview"

        self._broadcast_progress(job_id, event_type, data)

    def _push_proxy_progress(self, job_id: str, progress: float, completed: bool = False):
        """推送 Proxy 进度到 SSE"""
        channel_id = f"job:{job_id}"
        event_type = "proxy_complete" if completed else "proxy_progress"

        data = {
            "job_id": job_id,
            "progress": progress,
            "completed": completed
        }

        if completed:
            data["video_url"] = f"/api/media/{job_id}/video"

        self._broadcast_progress(job_id, event_type, data)

    def _push_remux_progress(self, job_id: str, progress: float, completed: bool = False):
        """推送重封装进度到 SSE"""
        channel_id = f"job:{job_id}"
        event_type = "remux_complete" if completed else "remux_progress"

        data = {
            "job_id": job_id,
            "progress": progress,
            "completed": completed,
            "type": "remux"
        }

        if completed:
            data["video_url"] = f"/api/media/{job_id}/video"

        self._broadcast_progress(job_id, event_type, data)

    def _broadcast_progress(
        self,
        job_id: str,
        event_type: str,
        data: dict,
        retry_count: int = None
    ) -> bool:
        """
        带重试机制的进度推送

        Args:
            job_id: 任务ID
            event_type: 事件类型
            data: 事件数据
            retry_count: 重试次数（默认从配置读取）

        Returns:
            是否推送成功
        """
        sse_config = config.PROXY_CONFIG.get('sse', {})
        retry_count = retry_count or sse_config.get('retry_count', 3)
        retry_delay = sse_config.get('retry_delay', 0.1)

        for attempt in range(retry_count):
            try:
                from app.services.sse_service import get_sse_manager
                sse_manager = get_sse_manager()
                channel_id = f"job:{job_id}"

                sse_manager.broadcast_sync(channel_id, event_type, data)
                return True

            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(
                        f"[MediaPrep] SSE推送失败 (尝试 {attempt + 1}/{retry_count}): "
                        f"job={job_id}, event={event_type}, error={e}"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"[MediaPrep] SSE推送最终失败: job={job_id}, event={event_type}, error={e}"
                    )

        return False

    def _execute_remux_task(self, task: dict):
        """
        执行容器重封装任务（零转码，极速完成）

        使用 -c:v copy -c:a copy 直接复制流，速度极快
        通常几秒内完成，比完整转码快 10-100 倍
        """
        job_id = task["job_id"]
        video_path = Path(task["video_path"])
        output_path = Path(task["output_path"])

        # 更新状态
        with self.lock:
            self.task_status[job_id]["remux"]["status"] = "processing"

        try:
            # 获取视频时长（用于进度计算）
            duration = self._get_video_duration(video_path)

            # 构建 FFmpeg 重封装命令（零转码）
            ffmpeg_cmd = config.get_ffmpeg_command()
            cmd = [
                ffmpeg_cmd,
                '-i', str(video_path),
                '-c:v', 'copy',             # 视频流直接复制
                '-c:a', 'copy',             # 音频流直接复制
                '-movflags', '+faststart',  # 优化网络播放
                '-progress', 'pipe:1',
                '-y',
                str(output_path)
            ]

            logger.info(f"[MediaPrep] 开始容器重封装: {video_path.name} -> {output_path.name}")

            # 执行 FFmpeg
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # 丢弃 stderr，避免缓冲区阻塞
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

                            with self.lock:
                                self.task_status[job_id]["remux"]["progress"] = progress

                            # 推送进度
                            self._push_remux_progress(job_id, progress)
                    except:
                        pass

            process.wait()

            if process.returncode == 0 and output_path.exists():
                # 成功
                with self.lock:
                    self.task_status[job_id]["remux"]["status"] = "completed"
                    self.task_status[job_id]["remux"]["progress"] = 100

                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                logger.info(f"[MediaPrep] 容器重封装完成: {output_path} ({file_size:.1f}MB)")
                self._push_remux_progress(job_id, 100, completed=True)
            else:
                # 失败
                error_msg = f"FFmpeg 返回码: {process.returncode}"
                with self.lock:
                    self.task_status[job_id]["remux"]["status"] = "failed"
                    self.task_status[job_id]["remux"]["error"] = error_msg

                logger.error(f"[MediaPrep] 容器重封装失败: {error_msg}")
                # 推送错误事件
                self._broadcast_progress(job_id, "proxy_error", {
                    "job_id": job_id,
                    "message": f"重封装失败: {error_msg}",
                    "type": "remux"
                })

        except Exception as e:
            with self.lock:
                self.task_status[job_id]["remux"]["status"] = "failed"
                self.task_status[job_id]["remux"]["error"] = str(e)

            logger.error(f"[MediaPrep] 容器重封装异常: {e}", exc_info=True)
            self._broadcast_progress(job_id, "proxy_error", {
                "job_id": job_id,
                "message": str(e),
                "type": "remux"
            })

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
