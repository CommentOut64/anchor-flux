"""
转录处理服务
整合了processor.py和原transcription_service.py的所有功能
"""
import os, subprocess, uuid, threading, json, math, gc, logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import OrderedDict  # 新增导入
from pydub import AudioSegment, silence
import whisperx
import torch
import shutil
import psutil
import numpy as np


class ProcessingMode(Enum):
    """
    处理模式枚举
    用于智能决策使用内存模式还是硬盘模式进行音频处理
    """
    MEMORY = "memory"  # 内存模式（默认，高性能）
    DISK = "disk"      # 硬盘模式（降级，稳定性优先）


class VADMethod(Enum):
    """
    VAD模型选择枚举
    用于选择语音活动检测（Voice Activity Detection）模型
    """
    SILERO = "silero"      # 默认，无需认证，速度快
    PYANNOTE = "pyannote"  # 可选，需要HF Token，精度更高


@dataclass
class VADConfig:
    """
    VAD配置数据类
    用于配置语音活动检测的参数
    """
    method: VADMethod = VADMethod.SILERO  # 默认使用Silero
    hf_token: Optional[str] = None         # Pyannote需要的HF Token
    onset: float = 0.5                     # 语音开始阈值
    offset: float = 0.363                  # 语音结束阈值
    chunk_size: int = 30                   # 最大段长（秒）

    def validate(self) -> bool:
        """验证配置有效性"""
        if self.method == VADMethod.PYANNOTE and not self.hf_token:
            return False  # Pyannote需要Token
        return True

from models.job_models import JobSettings, JobState
from models.hardware_models import HardwareInfo, OptimizationConfig
from services.hardware_service import get_hardware_detector, get_hardware_optimizer
from services.cpu_affinity_service import CPUAffinityManager, CPUAffinityConfig
from services.job_index_service import get_job_index_service
from core.config import config  # 导入统一配置

# 全局模型缓存 (按 (model, compute_type, device) 键)
_model_cache: Dict[Tuple[str, str, str], object] = {}

# 对齐模型缓存（改为OrderedDict，支持LRU）
_align_model_cache: OrderedDict[str, Tuple[object, object]] = OrderedDict()
_MAX_ALIGN_MODELS = 3  # 最多缓存3种语言的对齐模型

_model_lock = threading.Lock()
_align_lock = threading.Lock()


class TranscriptionService:
    """
    转录处理服务
    整合了所有转录相关功能
    """

    def __init__(self, jobs_root: str):
        """
        初始化转录服务

        Args:
            jobs_root: 任务工作目录根路径
        """
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

        self.jobs: Dict[str, JobState] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # 集成CPU亲和性管理器
        self.cpu_manager = CPUAffinityManager()

        # 集成硬件检测
        self.hardware_detector = get_hardware_detector()
        self.hardware_optimizer = get_hardware_optimizer()
        self._hardware_info: Optional[HardwareInfo] = None
        self._optimization_config: Optional[OptimizationConfig] = None

        # 集成任务索引服务
        self.job_index = get_job_index_service(jobs_root)
        # 启动时清理无效映射
        self.job_index.cleanup_invalid_mappings()

        # 集成SSE管理器（用于实时进度推送）
        from services.sse_service import get_sse_manager
        self.sse_manager = get_sse_manager()
        self.logger.info("SSE管理器已集成")

        # 记录CPU信息
        sys_info = self.cpu_manager.get_system_info()
        if sys_info.get('supported', False):
            self.logger.info(
                f" CPU信息: {sys_info['logical_cores']}个逻辑核心, "
                f"{sys_info.get('physical_cores', '?')}个物理核心, "
                f"平台: {sys_info.get('platform', '?')}"
            )
        else:
            self.logger.warning("CPU亲和性功能不可用")

        # 执行硬件检测
        self._detect_hardware()

        # 启动时扫描并加载所有任务（修复重启后无法打开旧任务的问题）
        self._load_all_jobs_from_disk()

    def _detect_hardware(self):
        """执行硬件检测并生成优化配置"""
        try:
            self.logger.info("开始硬件检测...")
            self._hardware_info = self.hardware_detector.detect()
            self._optimization_config = self.hardware_optimizer.get_optimization_config(self._hardware_info)

            # 记录检测结果
            hw = self._hardware_info
            opt = self._optimization_config
            self.logger.info(f"硬件检测完成GPU: {'' if hw.cuda_available else ''}, "
                           f"CPU: {hw.cpu_cores}核/{hw.cpu_threads}线程, "
                           f"内存: {hw.memory_total_mb}MB, "
                           f"优化配置: batch={opt.batch_size}, device={opt.recommended_device}")
        except Exception as e:
            self.logger.error(f"硬件检测失败: {e}")

    def _load_all_jobs_from_disk(self):
        """
        启动时扫描并加载所有任务到内存（修复重启后无法打开旧任务的问题）

        这个方法会扫描 jobs 目录中的所有任务，并加载到内存中，
        避免重启后因为内存为空导致无法访问旧任务
        """
        try:
            loaded_count = 0
            for job_dir in self.jobs_root.iterdir():
                if not job_dir.is_dir():
                    continue

                job_id = job_dir.name
                if job_id in self.jobs:
                    # 已加载，跳过
                    continue

                # 尝试加载任务
                job = self.get_job(job_id)
                if job:
                    loaded_count += 1

            if loaded_count > 0:
                self.logger.info(f"启动时已加载 {loaded_count} 个历史任务到内存")
        except Exception as e:
            self.logger.error(f"加载历史任务失败: {e}")
    
    def get_hardware_info(self) -> Optional[HardwareInfo]:
        """获取硬件信息"""
        return self._hardware_info
    
    def get_optimization_config(self) -> Optional[OptimizationConfig]:
        """获取优化配置"""  
        return self._optimization_config
    
    def get_optimized_job_settings(self, base_settings: Optional[JobSettings] = None) -> JobSettings:
        """获取基于硬件优化的任务设置"""
        # 使用硬件优化配置作为默认值
        if self._optimization_config:
            optimized = JobSettings(
                model=base_settings.model if base_settings else "medium",
                compute_type=base_settings.compute_type if base_settings else "float16",
                device=self._optimization_config.recommended_device,
                batch_size=self._optimization_config.batch_size,
                word_timestamps=base_settings.word_timestamps if base_settings else False
            )
            return optimized
        
        # 如果没有硬件信息，使用传入的设置或默认设置
        return base_settings or JobSettings()

    def create_job(
        self,
        filename: str,
        src_path: str,
        settings: JobSettings,
        job_id: Optional[str] = None
    ) -> JobState:
        """
        创建转录任务

        Args:
            filename: 文件名
            src_path: 源文件路径
            settings: 任务设置
            job_id: 任务ID（可选，不提供则自动生成）

        Returns:
            JobState: 创建的任务状态对象
        """
        job_id = job_id or uuid.uuid4().hex
        job_dir = self.jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        dest_path = job_dir / filename

        # 复制文件到任务目录
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            try:
                shutil.copyfile(src_path, dest_path)
                self.logger.debug(f"文件已复制: {src_path} -> {dest_path}")
            except Exception as e:
                self.logger.warning(f"文件复制失败: {e}")

        # 创建任务状态对象
        job = JobState(
            job_id=job_id,
            filename=filename,
            dir=str(job_dir),
            input_path=src_path,
            settings=settings,
            status="uploaded",
            phase="pending",
            message="文件已上传"
        )

        with self.lock:
            self.jobs[job_id] = job

        # 添加文件路径到任务ID的映射
        self.job_index.add_mapping(src_path, job_id)

        self.logger.info(f"任务已创建: {job_id} - {filename}")
        return job

    def get_job(self, job_id: str) -> Optional[JobState]:
        """
        获取任务状态

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 任务状态对象，不存在则返回None
        """
        with self.lock:
            # 首先从内存中查找
            if job_id in self.jobs:
                return self.jobs[job_id]

        # 如果内存中没有，尝试从 jobs 目录读取（修复重启后无法打开旧任务的问题）
        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            return None

        try:
            # 尝试找到原始文件
            filename = "未知文件"
            input_path = None

            # 从目录中查找视频/音频文件
            for ext in ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mp3', '.wav', '.m4a']:
                matches = list(job_dir.glob(f"*{ext}"))
                if matches:
                    filename = matches[0].name
                    input_path = str(matches[0])
                    break

            # 检查是否有 SRT 文件（表示已完成）
            srt_files = list(job_dir.glob("*.srt"))
            is_finished = len(srt_files) > 0

            # 创建 JobState 对象
            job = JobState(
                job_id=job_id,
                filename=filename,
                dir=str(job_dir),
                input_path=input_path,
                status='finished' if is_finished else 'processing',
                phase='editing' if is_finished else 'transcribing',
                progress=100 if is_finished else 0,
                message='已完成' if is_finished else '处理中',
                srt_path=str(srt_files[0]) if srt_files else None
            )

            # 尝试从 checkpoint 获取更详细的信息
            checkpoint_path = job_dir / "checkpoint.json"
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                        total_segments = checkpoint_data.get('total_segments', 0)
                        processed_indices = checkpoint_data.get('processed_indices', [])
                        if total_segments > 0:
                            job.progress = min((len(processed_indices) / total_segments) * 100, 100)
                        job.phase = checkpoint_data.get('phase', 'transcribing')
                        job.language = checkpoint_data.get('language')
                        # 从checkpoint恢复segments
                        if 'unaligned_results' in checkpoint_data:
                            job.segments = checkpoint_data['unaligned_results']
                except Exception as e:
                    self.logger.warning(f"读取checkpoint失败 {checkpoint_path}: {e}")

            # 缓存到内存
            with self.lock:
                self.jobs[job_id] = job

            self.logger.info(f"从磁盘恢复任务: {job_id}")
            return job

        except Exception as e:
            self.logger.error(f"从磁盘读取任务失败 {job_id}: {e}")
            return None

    def scan_incomplete_jobs(self) -> List[Dict]:
        """
        扫描所有未完成的任务（有checkpoint.json的任务）

        Returns:
            List[Dict]: 未完成任务列表
        """
        incomplete_jobs = []

        try:
            # 遍历所有任务目录
            for job_dir in self.jobs_root.iterdir():
                if not job_dir.is_dir():
                    continue

                checkpoint_path = job_dir / "checkpoint.json"
                if not checkpoint_path.exists():
                    continue

                try:
                    # 加载检查点数据
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)

                    job_id = checkpoint_data.get('job_id') or job_dir.name
                    total_segments = checkpoint_data.get('total_segments', 0)
                    processed_indices = checkpoint_data.get('processed_indices', [])
                    processed_count = len(processed_indices)

                    # 计算进度
                    if total_segments > 0:
                        progress = (processed_count / total_segments) * 100
                    else:
                        progress = 0

                    # 从索引中查找文件名
                    file_path = self.job_index.get_file_path(job_id)
                    filename = os.path.basename(file_path) if file_path else "未知文件"

                    incomplete_jobs.append({
                        'job_id': job_id,
                        'filename': filename,
                        'file_path': file_path,  # 添加文件路径
                        'progress': round(progress, 2),
                        'processed_segments': processed_count,
                        'total_segments': total_segments,
                        'phase': checkpoint_data.get('phase', 'unknown'),
                        'dir': str(job_dir)
                    })

                except Exception as e:
                    self.logger.warning(f"读取检查点失败 {checkpoint_path}: {e}")
                    continue

            self.logger.info(f"扫描到 {len(incomplete_jobs)} 个未完成任务")
            return incomplete_jobs

        except Exception as e:
            self.logger.error(f"扫描未完成任务失败: {e}")
            return []

    def restore_job_from_checkpoint(self, job_id: str) -> Optional[JobState]:
        """
        从检查点恢复任务状态

        Args:
            job_id: 任务ID

        Returns:
            Optional[JobState]: 恢复的任务状态对象
        """
        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            return None

        checkpoint = self._load_checkpoint(job_dir)
        if not checkpoint:
            return None

        try:
            # 查找原文件
            filename = "unknown"
            input_path = None

            # 从目录中查找视频/音频文件
            for ext in ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mp3', '.wav', '.m4a']:
                matches = list(job_dir.glob(f"*{ext}"))
                if matches:
                    filename = matches[0].name
                    input_path = str(matches[0])
                    break

            if not input_path:
                self.logger.warning(f"无法找到任务 {job_id} 的输入文件")
                return None

            # 创建默认的CPU亲和性配置
            from services.cpu_affinity_service import CPUAffinityConfig
            default_cpu_config = CPUAffinityConfig(
                enabled=True,
                strategy="auto",
                custom_cores=None,
                exclude_cores=None
            )

            # 创建任务状态对象
            job = JobState(
                job_id=job_id,
                filename=filename,
                dir=str(job_dir),
                input_path=input_path,
                settings=JobSettings(cpu_affinity=default_cpu_config),  # 提供默认的cpu_affinity
                status="paused",
                phase=checkpoint.get('phase', 'pending'),
                message=f"已暂停 ({len(checkpoint.get('processed_indices', []))}/{checkpoint.get('total_segments', 0)}段)",
                total=checkpoint.get('total_segments', 0),
                processed=len(checkpoint.get('processed_indices', [])),
                progress=round((len(checkpoint.get('processed_indices', [])) / max(1, checkpoint.get('total_segments', 1))) * 100, 2)
            )

            with self.lock:
                self.jobs[job_id] = job

            self.logger.info(f"从检查点恢复任务: {job_id}")
            return job

        except Exception as e:
            self.logger.error(f"从检查点恢复任务失败: {e}")
            return None

    def check_file_checkpoint(self, file_path: str) -> Optional[Dict]:
        """
        检查文件是否有可用的断点

        Args:
            file_path: 文件路径

        Returns:
            Optional[Dict]: 断点信息，无断点则返回None
        """
        # 从索引中查找任务ID
        job_id = self.job_index.get_job_id(file_path)
        if not job_id:
            return None

        # 检查任务目录和checkpoint是否存在
        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            # 清理无效映射
            self.job_index.remove_mapping(file_path)
            return None

        checkpoint = self._load_checkpoint(job_dir)
        if not checkpoint:
            return None

        # 返回断点信息
        total_segments = checkpoint.get('total_segments', 0)
        processed_indices = checkpoint.get('processed_indices', [])
        processed_count = len(processed_indices)

        if total_segments > 0:
            progress = (processed_count / total_segments) * 100
        else:
            progress = 0

        return {
            'job_id': job_id,
            'progress': round(progress, 2),
            'processed_segments': processed_count,
            'total_segments': total_segments,
            'phase': checkpoint.get('phase', 'unknown'),
            'can_resume': True
        }

    def start_job(self, job_id: str):
        """
        启动转录任务（V2.2: 废弃，由队列服务调用_run_pipeline）

        注意: 此方法保留是为了向后兼容，但不再自动创建线程

        Args:
            job_id: 任务ID
        """
        #  关键改动: 不再自动创建线程，由队列服务统一管理
        # 原有代码:
        # threading.Thread(target=self._run_pipeline, args=(job,), daemon=True).start()

        # 新逻辑: 只更新状态，实际执行由队列服务控制
        job = self.get_job(job_id)
        if not job:
            self.logger.warning(f"任务未找到: {job_id}")
            return

        if job.status not in ("uploaded", "failed", "paused", "created"):
            self.logger.warning(f"任务无法启动: {job_id}, 状态: {job.status}")
            return

        job.canceled = False
        job.paused = False
        job.error = None
        # 状态由队列服务设置，这里不改

        self.logger.warning(f"start_job已废弃，请使用队列服务: {job_id}")

    def pause_job(self, job_id: str) -> bool:
        """
        暂停转录任务（保存断点）

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否成功设置暂停标志
        """
        job = self.get_job(job_id)
        if not job:
            return False

        job.paused = True
        job.message = "暂停中..."
        self.logger.info(f"⏸️ 任务暂停请求: {job_id}")
        return True

    def cancel_job(self, job_id: str, delete_data: bool = False) -> bool:
        """
        取消转录任务

        Args:
            job_id: 任务ID
            delete_data: 是否删除任务数据

        Returns:
            bool: 是否成功设置取消标志
        """
        job = self.get_job(job_id)
        if not job:
            return False

        job.canceled = True
        job.message = "取消中..."
        self.logger.info(f"🛑 任务取消请求: {job_id}, 删除数据: {delete_data}")

        # 如果需要删除数据
        if delete_data:
            try:
                job_dir = Path(job.dir)
                # 移除文件路径映射
                if job.input_path:
                    self.job_index.remove_mapping(job.input_path)

                if job_dir.exists():
                    # 删除整个任务目录
                    shutil.rmtree(job_dir)
                    self.logger.info(f"已删除任务数据: {job_id}")
                    # 从内存中移除任务
                    with self.lock:
                        if job_id in self.jobs:
                            del self.jobs[job_id]
            except Exception as e:
                self.logger.error(f"删除任务数据失败: {e}")

        return True

    def _update_progress(
        self,
        job: JobState,
        phase: str,
        phase_ratio: float,
        message: str = ""
    ):
        """
        更新任务进度

        Args:
            job: 任务状态对象
            phase: 当前阶段 (extract/split/transcribe/srt)
            phase_ratio: 当前阶段完成比例 (0.0-1.0)
            message: 进度消息
        """
        job.phase = phase

        # 使用配置中的进度权重
        phase_weights = config.PHASE_WEIGHTS
        total_weight = config.TOTAL_WEIGHT

        # 计算累计进度
        done_weight = 0
        for p, w in phase_weights.items():
            if p == phase:
                break
            done_weight += w

        current_weight = phase_weights.get(phase, 0) * max(0.0, min(1.0, phase_ratio))
        job.progress = round((done_weight + current_weight) / total_weight * 100, 2)

        if message:
            job.message = message

        # 推送SSE进度更新（线程安全）
        self._push_sse_progress(job)

    def _push_sse_progress(self, job: JobState):
        """
        推送SSE进度更新（线程安全）

        Args:
            job: 任务状态对象
        """
        try:
            # 动态获取SSE管理器（确保获取到已设置loop的实例）
            from services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"

            sse_manager.broadcast_sync(
                channel_id,
                "progress",
                {
                    "job_id": job.job_id,
                    "phase": job.phase,
                    "percent": job.progress,
                    "message": job.message,
                    "status": job.status,
                    "processed": job.processed,
                    "total": job.total,
                    "language": job.language or ""
                }
            )
        except Exception as e:
            # SSE推送失败不应影响转录流程
            self.logger.debug(f"SSE推送失败: {e}")

    def _push_sse_signal(self, job: JobState, signal_code: str, message: str = ""):
        """
        推送SSE信号事件（用于关键节点通知）

        Args:
            job: 任务状态对象
            signal_code: 信号代码（如 "job_complete", "job_failed", "job_canceled"）
            message: 附加消息
        """
        try:
            # 动态获取SSE管理器（确保获取到已设置loop的实例）
            from services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"
            sse_manager.broadcast_sync(
                channel_id,
                "signal",
                {
                    "job_id": job.job_id,
                    "code": signal_code,
                    "message": message or job.message,
                    "status": job.status,
                    "progress": job.progress
                }
            )
        except Exception as e:
            self.logger.debug(f"SSE信号推送失败（非致命）: {e}")

    def _trigger_media_post_process(self, job_id: str):
        """
        异步触发媒体预处理（转录完成后调用）
        生成波形峰值、视频缩略图、Proxy视频等，为编辑器做准备

        Args:
            job_id: 任务ID
        """
        try:
            import asyncio
            import aiohttp

            async def do_post_process():
                """异步执行预处理请求"""
                try:
                    # 调用媒体预处理接口
                    async with aiohttp.ClientSession() as session:
                        url = f"http://127.0.0.1:8000/api/media/{job_id}/post-process"
                        async with session.post(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                self.logger.info(f"媒体预处理完成: peaks={result.get('peaks')}, thumbnails={result.get('thumbnails')}, proxy={result.get('proxy')}")
                            else:
                                self.logger.warning(f"媒体预处理请求失败: {resp.status}")
                except asyncio.TimeoutError:
                    self.logger.warning(f"媒体预处理超时: {job_id}")
                except Exception as e:
                    self.logger.warning(f"媒体预处理异常: {e}")

            # 尝试在现有事件循环中执行
            try:
                loop = asyncio.get_running_loop()
                asyncio.ensure_future(do_post_process(), loop=loop)
            except RuntimeError:
                # 没有运行中的事件循环，创建新线程执行
                import threading
                def run_in_thread():
                    asyncio.run(do_post_process())
                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()

            self.logger.info(f"已触发媒体预处理任务: {job_id}")

        except Exception as e:
            # 预处理失败不影响转录结果
            self.logger.warning(f"触发媒体预处理失败（非致命）: {e}")

    def _push_sse_segment(self, job: JobState, segment_result: dict, processed: int, total: int):
        """
        推送单个segment的转录结果（流式输出）

        Args:
            job: 任务状态对象
            segment_result: 单个segment的转录结果（未对齐）
            processed: 已处理的segment数量
            total: 总segment数量
        """
        try:
            # 动态获取SSE管理器
            from services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"
            sse_manager.broadcast_sync(
                channel_id,
                "segment",
                {
                    "segment_index": segment_result.get('segment_index', 0),
                    "segments": segment_result.get('segments', []),
                    "language": segment_result.get('language', job.language),
                    "progress": {
                        "processed": processed,
                        "total": total,
                        "percentage": round(processed / max(1, total) * 100, 2)
                    }
                }
            )
            self.logger.debug(f"推送segment #{segment_result.get('segment_index', 0)} 转录结果")
        except Exception as e:
            # SSE推送失败不应影响转录流程
            self.logger.debug(f"SSE segment推送失败（非致命）: {e}")

    def _push_sse_aligned(self, job: JobState, aligned_results: List[Dict]):
        """
        推送对齐完成事件（流式输出）

        Args:
            job: 任务状态对象
            aligned_results: 对齐后的结果列表
        """
        try:
            # 动态获取SSE管理器
            from services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"

            # 提取对齐后的segments
            segments = []
            word_segments = []
            if aligned_results and len(aligned_results) > 0:
                segments = aligned_results[0].get('segments', [])
                word_segments = aligned_results[0].get('word_segments', [])

            sse_manager.broadcast_sync(
                channel_id,
                "aligned",
                {
                    "segments": segments,
                    "word_segments": word_segments,
                    "message": "对齐完成"
                }
            )
            self.logger.info(f"推送对齐完成事件，共 {len(segments)} 条字幕")
        except Exception as e:
            # SSE推送失败不应影响转录流程
            self.logger.debug(f"SSE aligned推送失败（非致命）: {e}")

    def _save_checkpoint(self, job_dir: Path, data: dict, job: JobState):
        """
        原子性保存检查点
        使用"写临时文件 -> 重命名"策略，确保文件要么完整写入，要么保持原样

        Args:
            job_dir: 任务目录
            data: 检查点数据
            job: 任务状态对象（用于获取settings）
        """
        # 添加原始设置到checkpoint（用于校验参数兼容性）
        data["original_settings"] = {
            "model": job.settings.model,
            "device": job.settings.device,
            "word_timestamps": job.settings.word_timestamps,
            "compute_type": job.settings.compute_type,
            "batch_size": job.settings.batch_size
        }

        checkpoint_path = job_dir / "checkpoint.json"
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            # 1. 写入临时文件
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 2. 原子替换（Windows/Linux/macOS 均支持）
            # 如果程序在这里崩溃，checkpoint.json 依然是旧版本，不会损坏
            os.replace(temp_path, checkpoint_path)

        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")
            # 保存失败不应中断主流程，仅记录日志

    def _load_checkpoint(self, job_dir: Path) -> Optional[dict]:
        """
        加载检查点，如果文件损坏则返回 None

        Args:
            job_dir: 任务目录

        Returns:
            Optional[dict]: 检查点数据，不存在或损坏则返回 None
        """
        checkpoint_path = job_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"检查点文件损坏，将重新开始任务: {checkpoint_path} - {e}")
            return None

    def _flush_checkpoint_after_split(
        self,
        job_dir: Path,
        job: JobState,
        segments: List[Dict],
        processing_mode: ProcessingMode
    ):
        """
        分段完成后强制刷新checkpoint（确保断点续传一致性）

        这是断点续传的关键节点！
        只有分段元数据被持久化后，后续的转录索引才有意义。

        Args:
            job_dir: 任务目录
            job: 任务状态对象
            segments: 分段元数据列表
            processing_mode: 当前处理模式
        """
        import time

        checkpoint_data = {
            "job_id": job.job_id,
            "phase": "split_complete",  # 明确标记分段完成
            "processing_mode": processing_mode.value,  # 记录模式
            "total_segments": len(segments),
            "processed_indices": [],
            "segments": segments,
            "unaligned_results": [],
            "timestamp": time.time()  # 时间戳用于调试
        }

        # 强制同步写入（确保数据落盘）
        self._save_checkpoint(job_dir, checkpoint_data, job)

        # 验证写入成功
        saved_checkpoint = self._load_checkpoint(job_dir)
        if saved_checkpoint is None:
            raise RuntimeError("checkpoint write verification failed: file not readable")

        if saved_checkpoint.get('phase') != 'split_complete':
            raise RuntimeError("checkpoint write verification failed: phase mismatch")

        if len(saved_checkpoint.get('segments', [])) != len(segments):
            raise RuntimeError("checkpoint write verification failed: segments count mismatch")

        self.logger.info(f"checkpoint flushed and verified after split (mode: {processing_mode.value}, segments: {len(segments)})")

    def _run_pipeline(self, job: JobState):
        """
        执行转录处理管道（支持断点续传）

        Args:
            job: 任务状态对象
        """
        # 应用CPU亲和性设置
        cpu_applied = False
        if job.settings.cpu_affinity and job.settings.cpu_affinity.enabled:
            cpu_applied = self.cpu_manager.apply_cpu_affinity(
                job.settings.cpu_affinity
            )
            if cpu_applied:
                self.logger.info(f"📌 任务 {job.job_id} 已应用CPU亲和性设置")

        try:
            # 检查取消和暂停标志
            if job.canceled:
                job.status = 'canceled'
                job.message = '已取消'
                return

            if job.paused:
                job.status = 'paused'
                job.message = '已暂停'
                self.logger.info(f"⏸️ 任务已暂停: {job.job_id}")
                return

            job_dir = Path(job.dir)
            input_path = job_dir / job.filename
            audio_path = job_dir / 'audio.wav'

            # ==========================================
            # 1. 尝试恢复状态（断点续传核心）
            # ==========================================
            checkpoint = self._load_checkpoint(job_dir)

            # 初始化内存状态
            processed_indices = set()
            unaligned_results = []  # 未对齐的转录结果
            current_segments = []

            if checkpoint:
                self.logger.info(f"发现检查点，从 {checkpoint.get('phase', 'unknown')} 阶段恢复")
                # 恢复数据到内存
                processed_indices = set(checkpoint.get('processed_indices', []))

                # 【兼容性处理】支持旧格式checkpoint
                if 'unaligned_results' in checkpoint:
                    # 新格式：unaligned_results字段
                    unaligned_results = checkpoint.get('unaligned_results', [])
                    self.logger.info("检测到新格式checkpoint（未对齐结果）")
                elif 'results' in checkpoint:
                    # 旧格式：results字段（已对齐）
                    self.logger.warning("检测到旧版checkpoint格式，将直接使用已对齐结果")
                    # 将旧格式转换为新格式（跳过对齐阶段）
                    # 这种情况下我们直接使用results作为最终结果
                    pass

                current_segments = checkpoint.get('segments', [])
                # 恢复任务基本信息
                job.total = checkpoint.get('total_segments', 0)
                job.processed = len(processed_indices)
                self.logger.info(f"已处理 {job.processed}/{job.total} 段")

            # ==========================================
            # 2. 阶段1: 提取音频
            # ==========================================
            # 只有当音频文件不存在，或者从头开始时，才执行提取
            if not audio_path.exists() or (checkpoint is None):
                self._update_progress(job, 'extract', 0, '提取音频中')
                if job.canceled:
                    raise RuntimeError('任务已取消')

                if not self._extract_audio(str(input_path), str(audio_path)):
                    raise RuntimeError('FFmpeg 提取音频失败')

                self._update_progress(job, 'extract', 1, '音频提取完成')
            else:
                self.logger.info("跳过音频提取，使用已有文件")

            if job.canceled:
                raise RuntimeError('任务已取消')

            # ==========================================
            # 3. 阶段1.5: 智能模式决策（新增）
            # ==========================================
            processing_mode = None
            audio_array = None  # 内存模式下的音频数组

            # 从checkpoint恢复模式（如果存在）
            if checkpoint and 'processing_mode' in checkpoint:
                mode_value = checkpoint['processing_mode']
                processing_mode = ProcessingMode(mode_value)
                self.logger.info(f"从检查点恢复处理模式: {processing_mode.value}")

            # 如果没有检查点或没有模式信息，进行智能决策
            if processing_mode is None:
                processing_mode = self._decide_processing_mode(str(audio_path), job)
                self.logger.info(f"智能选择处理模式: {processing_mode.value}")

            # ==========================================
            # 4. 阶段1.6: 音频加载（内存模式）
            # ==========================================
            if processing_mode == ProcessingMode.MEMORY:
                # 内存模式：尝试加载完整音频到内存
                try:
                    audio_array = self._safe_load_audio(str(audio_path), job)
                    self.logger.info("音频已加载到内存（内存模式）")
                except RuntimeError as e:
                    # 加载失败，降级到硬盘模式
                    self.logger.warning(f"内存加载失败，降级到硬盘模式: {e}")
                    processing_mode = ProcessingMode.DISK
                    audio_array = None

            # ==========================================
            # 5. 阶段2: 智能分段（模式感知）
            # ==========================================
            # 如果检查点里没有分段信息，说明上次没跑到分段完成
            if not current_segments:
                self._update_progress(job, 'split', 0, '音频分段中')

                # 根据模式选择分段方法
                if processing_mode == ProcessingMode.MEMORY and audio_array is not None:
                    # 内存模式：VAD分段（不产生磁盘IO）
                    self.logger.info("使用内存VAD分段（高性能模式）")
                    from services.transcription_service import VADConfig
                    current_segments = self._split_audio_in_memory(
                        audio_array,
                        sr=16000,
                        vad_config=VADConfig()  # 使用默认Silero VAD
                    )
                else:
                    # 硬盘模式：传统pydub分段
                    self.logger.info("使用硬盘分段（稳定模式）")
                    current_segments = self._split_audio_to_disk(str(audio_path))

                if job.canceled:
                    raise RuntimeError('任务已取消')

                job.segments = current_segments
                job.total = len(current_segments)
                self._update_progress(job, 'split', 1, f'分段完成 共{job.total}段')

                # 【关键埋点1】分段完成后强制刷新checkpoint（使用新方法）
                self._flush_checkpoint_after_split(
                    job_dir,
                    job,
                    current_segments,
                    processing_mode
                )
                self.logger.info("检查点已强制刷新: 分段完成")
            else:
                self.logger.info(f"跳过分段，使用检查点数据（共{len(current_segments)}段）")
                job.segments = current_segments  # 恢复到 job 对象
                job.total = len(current_segments)

            # ==========================================
            # 6. 阶段3: 转录处理（双模式统一循环）
            # ==========================================
            self._update_progress(job, 'transcribe', 0, '加载模型中')
            if job.canceled:
                raise RuntimeError('任务已取消')

            model = self._get_model(job.settings, job)

            # 过滤出需要处理的段
            todo_segments = [
                seg for i, seg in enumerate(current_segments)
                if i not in processed_indices
            ]

            self.logger.info(f"剩余 {len(todo_segments)}/{len(current_segments)} 段需要转录")
            self.logger.info(f"处理模式: {processing_mode.value}")

            for idx, seg in enumerate(current_segments):
                # 如果已经在 processed_indices 里，直接跳过
                if idx in processed_indices:
                    self.logger.debug(f"⏭️ 跳过已处理段 {idx}")
                    continue

                # 检查取消和暂停标志
                if job.canceled:
                    raise RuntimeError('任务已取消')

                if job.paused:
                    raise RuntimeError('任务已暂停')

                # 【内存监控】定期检查内存状态（每10段检查一次）
                if idx % 10 == 0 and processing_mode == ProcessingMode.MEMORY:
                    if not self._check_memory_during_transcription(job):
                        # 内存严重不足，任务已暂停
                        raise RuntimeError('内存不足，任务已暂停')

                ratio = len(processed_indices) / max(1, len(current_segments))
                self._update_progress(
                    job,
                    'transcribe',
                    ratio,
                    f'转录 {len(processed_indices)+1}/{len(current_segments)}'
                )

                # 确保segment有index字段
                if 'index' not in seg:
                    seg['index'] = idx

                # 【统一入口】使用双模式转录（自动根据mode字段选择）
                seg_result = self._transcribe_segment(
                    seg,
                    model,
                    job,
                    audio_array=audio_array  # 内存模式传数组，硬盘模式为None
                )

                # --- 更新内存状态 ---
                if seg_result:
                    unaligned_results.append(seg_result)
                processed_indices.add(idx)
                job.processed = len(processed_indices)

                # --- 更新进度条 ---
                progress = len(processed_indices) / len(current_segments)
                self._update_progress(
                    job,
                    'transcribe',
                    progress,
                    f'转录中 {len(processed_indices)}/{len(current_segments)}'
                )

                # 【流式输出】立即推送单个segment的转录结果
                if seg_result:
                    self._push_sse_segment(job, seg_result, len(processed_indices), len(current_segments))

                # 【关键埋点2】每处理一段保存一次（保存未对齐结果）
                checkpoint_data = {
                    "job_id": job.job_id,
                    "phase": "transcribe",
                    "processing_mode": processing_mode.value,  # 保存模式信息
                    "total_segments": len(current_segments),
                    "processed_indices": list(processed_indices),  # set转list
                    "segments": current_segments,
                    "unaligned_results": unaligned_results  # 保存未对齐结果
                }
                self._save_checkpoint(job_dir, checkpoint_data, job)
                self.logger.debug(f"检查点已保存: {len(processed_indices)}/{len(current_segments)}")

            self._update_progress(job, 'transcribe', 1, '转录完成')
            if job.canceled:
                raise RuntimeError('任务已取消')

            # ==========================================
            # 7. 阶段4: 批次对齐（使用批次对齐+SSE进度推送）
            # ==========================================
            self._update_progress(job, 'align', 0, '准备对齐...')

            # 根据处理模式选择音频源
            if processing_mode == ProcessingMode.MEMORY and audio_array is not None:
                # 内存模式：复用内存数组（避免重新加载）
                audio_source = audio_array
                self.logger.info("对齐阶段：复用内存音频数组")
            else:
                # 硬盘模式：传递音频文件路径
                audio_source = str(audio_path)
                self.logger.info("对齐阶段：从磁盘加载音频")

            # 使用批次对齐方法（支持SSE进度推送）
            aligned_results = self._align_all_results_batched(
                unaligned_results,
                job,
                audio_source,
                processing_mode
            )

            # 【流式输出】推送对齐完成事件
            self._push_sse_aligned(job, aligned_results)

            if job.canceled:
                raise RuntimeError('任务已取消')

            # ==========================================
            # 6. 阶段5: 生成SRT
            # ==========================================
            base_name = os.path.splitext(job.filename)[0]
            srt_path = job_dir / f'{base_name}.srt'
            self._update_progress(job, 'srt', 0, '写入 SRT...')
            self._generate_srt(
                aligned_results,
                str(srt_path),
                job.settings.word_timestamps
            )
            self._update_progress(job, 'srt', 1, '处理完成')

            job.srt_path = str(srt_path)

            # 【清理】任务成功完成后，删除 checkpoint
            try:
                checkpoint_file = job_dir / "checkpoint.json"
                checkpoint_file.unlink(missing_ok=True)
                self.logger.info("检查点已清理")
            except Exception as e:
                self.logger.warning(f"清理检查点失败: {e}")

            if job.canceled:
                job.status = 'canceled'
                job.message = '已取消'
                # 推送取消信号
                self._push_sse_signal(job, "job_canceled", "任务已取消")
            else:
                job.status = 'finished'
                job.message = '完成'
                self.logger.info(f"任务完成: {job.job_id}")
                # 推送完成信号
                self._push_sse_signal(job, "job_complete", "转录完成")

                # 异步触发媒体预处理（生成波形、缩略图等，为编辑器做准备）
                self._trigger_media_post_process(job.job_id)

        except Exception as e:
            if job.canceled and '取消' in str(e):
                job.status = 'canceled'
                job.message = '已取消'
                self.logger.info(f"🛑 任务已取消: {job.job_id}")
                # 推送取消信号
                self._push_sse_signal(job, "job_canceled", "任务已取消")
            elif job.paused and '暂停' in str(e):
                job.status = 'paused'
                job.message = '已暂停'
                self.logger.info(f"⏸️ 任务已暂停: {job.job_id}")
                # 推送暂停信号
                self._push_sse_signal(job, "job_paused", "任务已暂停")
            else:
                job.status = 'failed'
                job.message = f'失败: {e}'
                job.error = str(e)
                self.logger.error(f"任务失败: {job.job_id} - {e}", exc_info=True)
                # 推送失败信号
                self._push_sse_signal(job, "job_failed", f"任务失败: {e}")

        finally:
            # 恢复CPU亲和性设置
            if cpu_applied:
                restored = self.cpu_manager.restore_cpu_affinity()
                if restored:
                    self.logger.info(f"任务 {job.job_id} 已恢复CPU亲和性设置")

            # 释放内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========== 核心处理方法 ==========

    def _get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频时长（秒）

        Args:
            audio_path: 音频文件路径

        Returns:
            float: 音频时长（秒）
        """
        try:
            # 方法1: 使用pydub（精确但较慢）
            audio = AudioSegment.from_wav(audio_path)
            duration = len(audio) / 1000.0
            self.logger.debug(f"音频时长（pydub）: {duration:.1f}秒")
            return duration
        except Exception as e:
            self.logger.warning(f"pydub获取时长失败，使用文件大小估算: {e}")
            # 方法2: 根据文件大小估算（16kHz, 16bit, mono ≈ 32KB/秒）
            try:
                file_size = os.path.getsize(audio_path)
                duration = file_size / 32000
                self.logger.debug(f"音频时长（估算）: {duration:.1f}秒")
                return duration
            except Exception as e2:
                self.logger.error(f"获取音频时长失败: {e2}")
                return 0.0

    def _decide_processing_mode(self, audio_path: str, job: JobState) -> ProcessingMode:
        """
        智能决策处理模式（内存模式 vs 硬盘模式）

        决策逻辑：
        1. 估算音频内存需求
        2. 检测系统可用内存
        3. 预留安全余量（模型、转录中间变量等）
        4. 决定使用哪种模式

        Args:
            audio_path: 音频文件路径
            job: 任务状态对象

        Returns:
            ProcessingMode: 处理模式
        """
        # 获取音频时长（秒）
        audio_duration_sec = self._get_audio_duration(audio_path)

        # 估算音频内存需求 (16kHz, float32)
        # 公式: duration * 16000 * 4 bytes
        estimated_audio_mb = (audio_duration_sec * 16000 * 4) / (1024 * 1024)

        # 预留额外内存（模型加载、VAD处理、转录中间变量等）
        # 保守估计：音频内存的2倍 + 500MB基础开销
        total_estimated_mb = estimated_audio_mb * 2 + 500

        # 获取系统可用内存
        mem_info = psutil.virtual_memory()
        available_mb = mem_info.available / (1024 * 1024)
        total_mb = mem_info.total / (1024 * 1024)

        # 安全阈值：至少保留系统总内存的20%或2GB（取较大值）
        safety_reserve_mb = max(total_mb * 0.2, 2048)
        usable_mb = available_mb - safety_reserve_mb

        self.logger.info(f"内存评估:")
        self.logger.info(f"音频时长: {audio_duration_sec/60:.1f}分钟")
        self.logger.info(f"预估需求: {total_estimated_mb:.0f}MB")
        self.logger.info(f"可用内存: {available_mb:.0f}MB")
        self.logger.info(f"安全余量: {safety_reserve_mb:.0f}MB")
        self.logger.info(f"可用于处理: {usable_mb:.0f}MB")

        # 决策
        if usable_mb >= total_estimated_mb:
            self.logger.info("选择【内存模式】- 内存充足，使用高性能模式")
            job.message = "内存充足，使用高性能模式"
            return ProcessingMode.MEMORY
        else:
            self.logger.warning(f"选择【硬盘模式】- 内存不足（需要{total_estimated_mb:.0f}MB，可用{usable_mb:.0f}MB）")
            job.message = "内存受限，使用稳定模式"
            return ProcessingMode.DISK

    def _safe_load_audio(self, audio_path: str, job: JobState) -> np.ndarray:
        """
        安全加载音频到内存（带异常处理）

        用于内存模式下将完整音频一次性加载到内存中。
        包含加载验证和详细的异常处理，加载失败时抛出RuntimeError触发降级。

        Args:
            audio_path: 音频文件路径
            job: 任务状态对象（用于更新状态消息）

        Returns:
            np.ndarray: 音频数组（float32, 16kHz采样率）

        Raises:
            RuntimeError: 音频加载失败时抛出，调用方可据此触发硬盘模式降级
        """
        try:
            self.logger.info(f"加载音频到内存: {audio_path}")
            audio_array = whisperx.load_audio(audio_path)

            # 验证加载结果
            if audio_array is None or len(audio_array) == 0:
                raise ValueError("音频数组为空")

            # 记录加载信息
            duration_sec = len(audio_array) / 16000
            memory_mb = audio_array.nbytes / (1024 * 1024)
            # Audio loaded: {duration_sec/60:.1f}分钟")
            self.logger.info(f"内存占用: {memory_mb:.1f}MB")
            self.logger.info(f"采样点数: {len(audio_array):,}")

            return audio_array

        except MemoryError as e:
            self.logger.error(f"内存不足，无法加载音频: {e}")
            job.message = "内存不足，自动切换到硬盘模式"
            raise RuntimeError(f"内存不足: {e}")

        except Exception as e:
            self.logger.error(f"音频加载失败: {e}")
            job.message = f"音频加载失败: {e}"
            raise RuntimeError(f"音频加载失败（可能文件损坏）: {e}")

    def _split_audio_in_memory(
        self,
        audio_array: np.ndarray,
        sr: int = 16000,
        vad_config: Optional[VADConfig] = None
    ) -> List[Dict]:
        """
        内存VAD分段（不产生磁盘IO）

        默认使用Silero VAD（无需认证），可通过配置切换到Pyannote VAD。
        当VAD模型加载失败时，自动降级到基于能量的简易分段。

        Args:
            audio_array: 完整音频数组 (np.ndarray, float32, 16kHz)
            sr: 采样率（默认16000Hz）
            vad_config: VAD配置（可选，默认使用Silero）

        Returns:
            List[Dict]: 分段元数据列表
            [
                {"index": 0, "start": 0.0, "end": 30.5, "mode": "memory"},
                {"index": 1, "start": 30.5, "end": 58.2, "mode": "memory"},
                ...
            ]
        """
        # 使用默认配置
        if vad_config is None:
            vad_config = VADConfig()

        self.logger.info(f"开始内存VAD分段 (模型: {vad_config.method.value})...")

        try:
            # 根据配置选择VAD模型
            if vad_config.method == VADMethod.SILERO:
                segments = self._vad_silero(audio_array, sr, vad_config)
            else:
                segments = self._vad_pyannote(audio_array, sr, vad_config)

            self.logger.info(f"VAD分段完成: {len(segments)}段 (模型: {vad_config.method.value})")
            return segments

        except Exception as e:
            self.logger.error(f"VAD分段失败: {e}")
            # 降级到简易能量检测
            self.logger.warning("尝试降级到能量检测分段...")
            return self._energy_based_split(audio_array, sr, vad_config.chunk_size)

    def _vad_silero(
        self,
        audio_array: np.ndarray,
        sr: int,
        vad_config: VADConfig
    ) -> List[Dict]:
        """
        Silero VAD分段（使用内置ONNX模型，无需下载）

        优点：
        - 使用项目内置ONNX模型，无需网络下载
        - 使用 onnxruntime 推理，跨平台兼容性好
        - 速度快，内存占用低（~2MB）

        Args:
            audio_array: 音频数组
            sr: 采样率
            vad_config: VAD配置

        Returns:
            List[Dict]: 分段元数据列表
        """
        self.logger.info("加载Silero VAD模型（内置ONNX）...")

        # 使用 silero-vad 库（基于 onnxruntime）
        from silero_vad import get_speech_timestamps
        from silero_vad.utils_vad import OnnxWrapper
        from pathlib import Path as PathlibPath

        # 使用项目内置的 ONNX 模型
        builtin_model_path = PathlibPath(__file__).parent.parent / "assets" / "silero" / "silero_vad.onnx"

        if not builtin_model_path.exists():
            raise FileNotFoundError(
                f"内置Silero VAD模型不存在: {builtin_model_path}\n"
                "请确保项目完整，或重新从源码仓库获取"
            )

        self.logger.info(f"使用内置模型: {builtin_model_path}")

        # 加载ONNX模型（直接从本地路径）
        model = OnnxWrapper(str(builtin_model_path), force_onnx_cpu=False)

        # 转换为torch tensor（silero-vad 需要）
        audio_tensor = torch.from_numpy(audio_array)

        # 获取语音时间戳
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sr,
            threshold=vad_config.onset,      # 检测阈值
            min_speech_duration_ms=250,       # 最小语音段长度
            min_silence_duration_ms=100,      # 最小静音长度
            return_seconds=False  # 返回采样点而非秒数
        )

        # VAD detection complete

        # 合并分段（确保每段不超过chunk_size秒）
        segments_metadata = []
        current_start = None
        current_end = None

        for ts in speech_timestamps:
            start_sec = ts['start'] / sr
            end_sec = ts['end'] / sr

            if current_start is None:
                current_start = start_sec
                current_end = end_sec
            elif (end_sec - current_start) <= vad_config.chunk_size:
                # 可以合并
                current_end = end_sec
            else:
                # 保存当前段，开始新段
                segments_metadata.append({
                    "index": len(segments_metadata),
                    "start": current_start,
                    "end": current_end,
                    "mode": "memory"
                })
                current_start = start_sec
                current_end = end_sec

        # 保存最后一段
        if current_start is not None:
            segments_metadata.append({
                "index": len(segments_metadata),
                "start": current_start,
                "end": current_end,
                "mode": "memory"
            })

        # 如果没有检测到任何语音段，按固定时长分段
        if len(segments_metadata) == 0:
            self.logger.warning("VAD未检测到语音，使用固定时长分段")
            return self._energy_based_split(audio_array, sr, vad_config.chunk_size)

        return segments_metadata

    def _vad_pyannote(
        self,
        audio_array: np.ndarray,
        sr: int,
        vad_config: VADConfig
    ) -> List[Dict]:
        """
        Pyannote VAD分段（高精度方案，需要HF Token）

        优点：
        - 精度更高
        - 支持更复杂的语音活动检测

        注意：
        - 需要HuggingFace Token
        - 首次使用需要接受模型使用协议

        Args:
            audio_array: 音频数组
            sr: 采样率
            vad_config: VAD配置

        Returns:
            List[Dict]: 分段元数据列表

        Raises:
            ValueError: 未配置HF Token时抛出
        """
        if not vad_config.hf_token:
            raise ValueError("Pyannote VAD需要HuggingFace Token，请在设置中配置")

        self.logger.info("加载Pyannote VAD模型（需要HF Token）...")

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise RuntimeError("Pyannote未安装，请使用Silero VAD或安装pyannote-audio")

        # 初始化Pyannote VAD Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=vad_config.hf_token
        )

        # 准备输入（Pyannote需要特定格式）
        # 创建临时文件用于Pyannote处理
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_array, sr)

        try:
            # 执行VAD
            vad_result = pipeline(temp_path)

            # 合并分段
            segments_metadata = []
            current_start = None
            current_end = None

            for speech in vad_result.get_timeline().support():
                start_sec = speech.start
                end_sec = speech.end

                if current_start is None:
                    current_start = start_sec
                    current_end = end_sec
                elif (end_sec - current_start) <= vad_config.chunk_size:
                    current_end = end_sec
                else:
                    segments_metadata.append({
                        "index": len(segments_metadata),
                        "start": current_start,
                        "end": current_end,
                        "mode": "memory"
                    })
                    current_start = start_sec
                    current_end = end_sec

            # 保存最后一段
            if current_start is not None:
                segments_metadata.append({
                    "index": len(segments_metadata),
                    "start": current_start,
                    "end": current_end,
                    "mode": "memory"
                })

            return segments_metadata

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _energy_based_split(
        self,
        audio_array: np.ndarray,
        sr: int,
        chunk_size: int = 30
    ) -> List[Dict]:
        """
        基于能量的简易分段（降级方案）

        当VAD模型加载失败时使用，按固定时长分段。
        会尝试在静音处分割以避免切断语音。

        Args:
            audio_array: 音频数组
            sr: 采样率
            chunk_size: 每段最大长度（秒）

        Returns:
            List[Dict]: 分段元数据列表
        """
        self.logger.warning("使用能量检测降级分段（固定时长）")

        total_duration = len(audio_array) / sr
        segments_metadata = []
        pos = 0.0

        while pos < total_duration:
            # 计算理想结束位置
            ideal_end = min(pos + chunk_size, total_duration)

            # 尝试在静音处分割（在理想结束点前后1秒范围内寻找）
            if ideal_end < total_duration:
                search_start = max(pos, ideal_end - 1.0)
                search_end = min(total_duration, ideal_end + 1.0)

                # 计算搜索范围内的能量
                start_sample = int(search_start * sr)
                end_sample = int(search_end * sr)
                search_audio = audio_array[start_sample:end_sample]

                if len(search_audio) > 0:
                    # 计算短时能量（每100ms一个窗口）
                    window_size = int(0.1 * sr)
                    energies = []
                    for i in range(0, len(search_audio) - window_size, window_size):
                        window = search_audio[i:i + window_size]
                        energy = np.sum(window ** 2)
                        energies.append((i, energy))

                    if energies:
                        # 找到能量最低的点
                        min_energy_idx = min(energies, key=lambda x: x[1])[0]
                        actual_end = search_start + (min_energy_idx / sr)
                        # 确保分段至少有1秒
                        if actual_end - pos >= 1.0:
                            ideal_end = actual_end

            segments_metadata.append({
                "index": len(segments_metadata),
                "start": pos,
                "end": ideal_end,
                "mode": "memory"
            })
            pos = ideal_end

        self.logger.info(f"能量检测分段完成: {len(segments_metadata)}段")
        return segments_metadata

    def _extract_audio(self, input_file: str, audio_out: str) -> bool:
        """
        使用FFmpeg提取音频

        Args:
            input_file: 输入文件路径
            audio_out: 输出音频路径

        Returns:
            bool: 是否提取成功
        """
        if os.path.exists(audio_out):
            self.logger.debug(f"音频文件已存在，跳过提取: {audio_out}")
            return True

        # 使用配置中的FFmpeg命令（支持独立打包）
        ffmpeg_cmd = config.get_ffmpeg_command()
        self.logger.debug(f"使用FFmpeg: {ffmpeg_cmd}")

        cmd = [
            ffmpeg_cmd, '-y', '-i', input_file,
            '-vn',                    # 仅音频
            '-ac', '1',               # 单声道
            '-ar', '16000',           # 16kHz 采样率
            '-acodec', 'pcm_s16le',   # PCM 编码
            audio_out
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600  # 10分钟超时
            )

            if proc.returncode == 0 and os.path.exists(audio_out):
                self.logger.debug(f"音频提取成功: {audio_out}")
                return True
            else:
                error_msg = proc.stderr.decode('utf-8', errors='ignore')
                self.logger.error(f"FFmpeg执行失败: {error_msg}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg超时")
            return False
        except Exception as e:
            self.logger.error(f"音频提取失败: {e}")
            return False

    def _split_audio_to_disk(self, audio_path: str) -> List[Dict]:
        """
        硬盘分段模式（保留原有逻辑）

        使用pydub进行静音检测，生成segment_N.wav文件。
        适用于内存不足的场景。

        Args:
            audio_path: 音频文件路径

        Returns:
            List[Dict]: 分段信息列表，与内存模式格式统一
            [
                {"index": 0, "file": "segment_0.wav", "start": 0.0, "end": 30.0, "start_ms": 0, "duration_ms": 30000, "mode": "disk"},
                ...
            ]
        """
        self.logger.info("开始硬盘分段（pydub静音检测）...")

        # 使用配置中的音频处理参数
        audio_config = config.get_audio_config()
        SEGMENT_LEN_MS = audio_config['segment_length_ms']
        SILENCE_SEARCH_MS = audio_config['silence_search_ms']
        MIN_SILENCE_LEN_MS = audio_config['min_silence_len_ms']
        SILENCE_THRESH_DBFS = audio_config['silence_threshold_dbfs']

        audio = AudioSegment.from_wav(audio_path)
        length = len(audio)
        segments = []
        pos = 0
        idx = 0

        while pos < length:
            end = min(pos + SEGMENT_LEN_MS, length)

            # 智能寻找静音点（避免在句子中间分割）
            if end < length and (end - pos) > SILENCE_SEARCH_MS:
                search_start = max(pos, endSILENCE_SEARCH_MS)
                search_chunk = audio[search_start:end]

                try:
                    silences = silence.detect_silence(
                        search_chunk,
                        min_silence_len=MIN_SILENCE_LEN_MS,
                        silence_thresh=SILENCE_THRESH_DBFS
                    )

                    if silences:
                        # 使用第一个静音点
                        silence_start = silences[0][0]
                        new_end = search_start + silence_start
                        if new_end - pos > MIN_SILENCE_LEN_MS:
                            end = new_end
                except Exception as e:
                    self.logger.warning(f"silence detection failed: {e}")

            # 导出分段文件
            chunk = audio[pos:end]
            seg_file = os.path.join(os.path.dirname(audio_path), f'segment_{idx}.wav')
            chunk.export(seg_file, format='wav')

            # 统一返回格式（与内存模式一致）
            segments.append({
                'index': idx,                    # 新增：分段索引
                'file': seg_file,
                'start': pos / 1000.0,           # 新增：起始时间（秒）
                'end': end / 1000.0,             # 新增：结束时间（秒）
                'start_ms': pos,                 # 保留：兼容旧代码
                'duration_ms': end - pos,        # 保留：兼容旧代码
                'mode': 'disk'                   # 新增：模式标记
            })

            pos = end
            idx += 1

        self.logger.info(f"disk segmentation complete: {len(segments)} segments (with segment files)")
        return segments

    # 兼容性别名：保留旧方法名（指向新方法）
    def _split_audio(self, audio_path: str) -> List[Dict]:
        """兼容性别名，指向 _split_audio_to_disk()"""
        return self._split_audio_to_disk(audio_path)

    def _get_model(self, settings: JobSettings, job: Optional[JobState] = None):
        """
        获取WhisperX模型（带缓存）

        优先使用模型管理服务检查并下载模型，否则使用简单缓存

        Args:
            settings: 任务设置
            job: 任务状态对象(可选,用于更新下载进度)

        Returns:
            模型对象
        """
        # 尝试使用模型管理服务检查并下载模型
        try:
            from services.model_manager_service import get_model_manager
            model_mgr = get_model_manager()
            whisper_model_info = model_mgr.whisper_models.get(settings.model)

            if whisper_model_info:
                # 检查模型状态
                if whisper_model_info.status == "not_downloaded" or whisper_model_info.status == "incomplete":
                    self.logger.warning(f"Whisper模型未下载或不完整: {settings.model}")

                    # 获取模型大小信息
                    model_size_mb = whisper_model_info.size_mb

                    # 如果模型大小>=1GB,给出特殊提示
                    download_msg = ""
                    if model_size_mb >= 1024:
                        size_gb = model_size_mb / 1024
                        download_msg = f"当前下载模型大于1GB ({size_gb:.1f}GB),请耐心等待"
                        self.logger.info(f"{download_msg}")
                    else:
                        download_msg = f"开始下载模型 {settings.model} ({model_size_mb}MB)"

                    # 更新任务状态
                    if job:
                        job.message = download_msg

                    self.logger.info(f"自动触发下载Whisper模型: {settings.model} ({model_size_mb}MB)")

                    # 触发下载
                    success = model_mgr.download_whisper_model(settings.model)
                    if not success:
                        self.logger.warning(f"模型管理器下载失败或已在下载中,回退到whisperx")
                        raise RuntimeError("模型管理器下载失败")

                    # 等待下载完成（最多等待10分钟）
                    import time
                    max_wait_time = 600  # 10分钟
                    wait_interval = 5  # 每5秒检查一次
                    elapsed = 0

                    while elapsed < max_wait_time:
                        time.sleep(wait_interval)
                        elapsed += wait_interval

                        current_status = model_mgr.whisper_models[settings.model].status
                        progress = model_mgr.whisper_models[settings.model].download_progress

                        if current_status == "ready":
                            self.logger.info(f"Whisper模型下载完成: {settings.model}")
                            if job:
                                job.message = f"模型下载完成,准备加载"
                            break
                        elif current_status == "error":
                            self.logger.error(f"模型管理器下载失败,回退到whisperx")
                            raise RuntimeError(f"Whisper模型下载失败: {settings.model}")
                        else:
                            # 如果模型大小>=1GB,定期提醒用户耐心等待
                            if model_size_mb >= 1024 and elapsed % 30 == 0:  # 每30秒提醒一次
                                wait_msg = f"当前下载模型大于1GB,请耐心等待... {progress:.1f}% ({elapsed}s/{max_wait_time}s)"
                                self.logger.info(f"{wait_msg}")
                                if job:
                                    job.message = wait_msg
                            else:
                                wait_msg = f"等待模型下载... {progress:.1f}%"
                                self.logger.info(f"{wait_msg} ({elapsed}s/{max_wait_time}s)")
                                # 更新任务状态(每次都更新,这样用户可以看到进度变化)
                                if job:
                                    job.message = wait_msg

                    if elapsed >= max_wait_time:
                        self.logger.error(f"模型下载超时,回退到whisperx")
                        raise TimeoutError(f"Whisper模型下载超时: {settings.model}")

        except Exception as e:
            self.logger.warning(f"模型管理服务检查失败,回退到whisperx: {e}")

        # 尝试使用模型预加载管理器
        try:
            from services.model_preload_manager import get_model_manager as get_preload_manager
            model_manager = get_preload_manager()
            if model_manager:
                self.logger.debug("使用模型预加载管理器获取模型")
                if job:
                    job.message = "加载模型中"
                return model_manager.get_model(settings)
        except Exception as e:
            self.logger.debug(f"无法使用模型预加载管理器，回退到本地缓存: {e}")
            pass

        # 回退到简单缓存机制
        key = (settings.model, settings.compute_type, settings.device)
        with _model_lock:
            if key in _model_cache:
                self.logger.debug(f"命中模型缓存: {key}")
                if job:
                    job.message = "使用缓存的模型"
                return _model_cache[key]

            self.logger.info(f"加载模型: {key}")
            if job:
                job.message = f"加载模型 {settings.model}"

            # 首先尝试仅使用本地文件
            try:
                from core.config import config
                m = whisperx.load_model(
                    settings.model,
                    settings.device,
                    compute_type=settings.compute_type,
                    download_root=str(config.HF_CACHE_DIR),  # 指定缓存路径
                    local_files_only=True  # 禁止自动下载，只使用本地文件
                )
                _model_cache[key] = m
                if job:
                    job.message = "模型加载完成"
                return m
            except Exception as e:
                self.logger.warning(f"本地加载失败,允许whisperx下载: {e}")
                if job:
                    job.message = "本地模型不存在,使用whisperx下载"
                # 如果本地加载失败,允许whisperx下载
                m = whisperx.load_model(
                    settings.model,
                    settings.device,
                    compute_type=settings.compute_type,
                    download_root=str(config.HF_CACHE_DIR),  # 指定缓存路径
                    local_files_only=False  # 允许下载
                )
                _model_cache[key] = m
                if job:
                    job.message = "模型下载并加载完成"
                return m

    def _get_align_model(self, lang: str, device: str, job: Optional[JobState] = None):
        """
        获取对齐模型（带LRU缓存）

        策略:
        - 缓存命中：移到末尾（标记为最近使用）
        - 缓存已满：删除最久未使用的模型
        - 最多缓存3种语言

        Args:
            lang: 语言代码
            device: 设备 (cuda/cpu)
            job: 任务状态对象(可选,用于更新下载进度)

        Returns:
            Tuple[model, metadata]: 对齐模型和元数据
        """
        global _align_model_cache, _MAX_ALIGN_MODELS

        with _align_lock:
            # 1. 检查缓存命中
            if lang in _align_model_cache:
                # 命中：移到末尾（最近使用）
                _align_model_cache.move_to_end(lang)
                self.logger.debug(f"命中对齐模型缓存: {lang} (缓存: {list(_align_model_cache.keys())})")
                if job:
                    job.message = "使用缓存的对齐模型"
                return _align_model_cache[lang]

            # 2. 缓存未命中，检查是否需要淘汰
            if len(_align_model_cache) >= _MAX_ALIGN_MODELS:
                # 缓存已满，删除最久未使用的（队首）
                oldest_lang, (oldest_model, _) = _align_model_cache.popitem(last=False)
                self.logger.info(f"淘汰最久未用的对齐模型: {oldest_lang} (为 {lang} 腾出空间)")

                # 显式删除模型对象
                try:
                    del oldest_model
                except:
                    pass

        # 3. 加载新模型（保留原有的下载和加载逻辑）
        self.logger.debug(f"Loading alignment model: {lang}")
        if job:
            job.message = f"加载对齐模型 {lang}"

        # 尝试使用模型预加载管理器（优先从LRU缓存获取）
        try:
            from services.model_preload_manager import get_model_manager as get_preload_manager
            preload_mgr = get_preload_manager()
            if preload_mgr:
                self.logger.debug("尝试从预加载管理器获取对齐模型")
                if job:
                    job.message = "加载对齐模型"
                am, meta = preload_mgr.get_align_model(lang, device)
                # 4. 加入缓存（自动放在末尾，标记为最近使用）
                with _align_lock:
                    _align_model_cache[lang] = (am, meta)
                    self.logger.info(f"对齐模型已缓存: {lang} (当前缓存: {list(_align_model_cache.keys())})")
                return am, meta
        except Exception as e:
            self.logger.debug(f"预加载管理器获取失败，使用直接加载: {e}")

            # 检查模型是否需要下载（使用模型管理服务）
            try:
                from services.model_manager_service import get_model_manager
                model_mgr = get_model_manager()
                align_model_info = model_mgr.align_models.get(lang)

                if align_model_info and (align_model_info.status == "not_downloaded" or align_model_info.status == "incomplete"):
                    # 检查模型状态,如果未下载或不完整则触发下载
                    if align_model_info.status == "incomplete":
                        self.logger.warning(f"对齐模型不完整: {lang}")
                    else:
                        self.logger.warning(f"对齐模型未下载: {lang}")

                    # 对齐模型通常为1.2GB左右,给出大模型提示
                    download_msg = "当前下载模型大于1GB (约1.2GB),请耐心等待"
                    self.logger.info(f"{download_msg}")
                    self.logger.info(f"自动触发下载对齐模型: {lang}")

                    # 更新任务状态
                    if job:
                        job.message = download_msg

                    # 触发下载
                    success = model_mgr.download_align_model(lang)
                    if not success:
                        self.logger.warning(f"模型管理器下载失败或已在下载中,回退到whisperx")
                        raise RuntimeError("模型管理器下载失败")

                    # 等待下载完成（最多等待10分钟,对齐模型较大）
                    import time
                    max_wait_time = 600  # 10分钟
                    wait_interval = 5  # 每5秒检查一次
                    elapsed = 0

                    while elapsed < max_wait_time:
                        time.sleep(wait_interval)
                        elapsed += wait_interval

                        current_status = model_mgr.align_models[lang].status
                        progress = model_mgr.align_models[lang].download_progress

                        if current_status == "ready":
                            self.logger.info(f"对齐模型下载完成: {lang}")
                            if job:
                                job.message = "对齐模型下载完成,准备加载"
                            break
                        elif current_status == "error":
                            self.logger.error(f"模型管理器下载失败,回退到whisperx")
                            raise RuntimeError(f"对齐模型下载失败: {lang}")
                        else:
                            # 定期提醒用户耐心等待(每30秒)
                            if elapsed % 30 == 0:
                                wait_msg = f"当前下载模型大于1GB,请耐心等待... {progress:.1f}% ({elapsed}s/{max_wait_time}s)"
                                self.logger.info(f"{wait_msg}")
                                if job:
                                    job.message = wait_msg
                            else:
                                wait_msg = f"等待对齐模型下载... {progress:.1f}%"
                                self.logger.info(f"{wait_msg} ({elapsed}s/{max_wait_time}s)")
                                # 更新任务状态(每次都更新,这样用户可以看到进度变化)
                                if job:
                                    job.message = wait_msg

                    if elapsed >= max_wait_time:
                        self.logger.error(f"模型下载超时,回退到whisperx")
                        raise TimeoutError(f"对齐模型下载超时: {lang}")

            except Exception as e:
                self.logger.warning(f"模型管理服务检查失败,回退到whisperx: {e}")

            # 直接加载模型（如果已下载或下载完成）
            self.logger.debug(f"Loading alignment model: {lang}")
            if job:
                job.message = f"加载对齐模型 {lang}"

            # 首先尝试仅使用本地文件
            try:
                from core.config import config
                am, meta = whisperx.load_align_model(
                    language_code=lang,
                    device=device,
                    model_dir=str(config.HF_CACHE_DIR)  # 指定缓存路径
                )
                # 加入缓存（自动放在末尾，标记为最近使用）
                with _align_lock:
                    _align_model_cache[lang] = (am, meta)
                    self.logger.info(f"对齐模型已缓存: {lang} (当前缓存: {list(_align_model_cache.keys())})")
                if job:
                    job.message = "对齐模型加载完成"
                return am, meta
            except Exception as e:
                self.logger.warning(f"本地加载对齐模型失败,允许whisperx下载: {e}")
                if job:
                    job.message = "本地对齐模型不存在,使用whisperx下载"
                # 如果本地加载失败,允许whisperx下载
                am, meta = whisperx.load_align_model(
                    language_code=lang,
                    device=device
                )
                # 加入缓存（自动放在末尾，标记为最近使用）
                with _align_lock:
                    _align_model_cache[lang] = (am, meta)
                    self.logger.info(f"对齐模型已下载并缓存: {lang} (当前缓存: {list(_align_model_cache.keys())})")
                if job:
                    job.message = "对齐模型下载并加载完成"
                return am, meta

    def _transcribe_segment_unaligned(
        self,
        seg: Dict,
        model,
        job: JobState
    ) -> Optional[Dict]:
        """
        转录单个音频段（仅转录，不对齐）

        Args:
            seg: 段信息 {file, start_ms, duration_ms, index}
            model: Whisper模型
            job: 任务状态

        Returns:
            Dict: 未对齐的转录结果
            {
                "segment_index": 0,
                "language": "zh",
                "segments": [{"id": 0, "start": 10.5, "end": 15.2, "text": "..."}]
            }
        """
        audio = whisperx.load_audio(seg['file'])

        try:
            # 仅进行Transcription，不进行Alignment
            rs = model.transcribe(
                audio,
                batch_size=job.settings.batch_size,
                verbose=False,
                language=job.language
            )

            if not rs or 'segments' not in rs:
                return None

            # 检测语言（首次）
            if not job.language and 'language' in rs:
                job.language = rs['language']
                self.logger.info(f"🌐 检测到语言: {job.language}")

            # 时间偏移校正（针对粗略时间戳）
            start_offset = seg['start_ms'] / 1000.0
            adjusted_segments = []

            for idx, s in enumerate(rs['segments']):
                adjusted_segments.append({
                    'id': idx,
                    'start': s.get('start', 0) + start_offset,
                    'end': s.get('end', 0) + start_offset,
                    'text': s.get('text', '').strip()
                })

            return {
                'segment_index': seg.get('index', 0),  # 需要在调用时传入
                'language': rs.get('language', job.language),
                'segments': adjusted_segments
            }

        finally:
            del audio
            gc.collect()

    def _transcribe_segment_in_memory(
        self,
        audio_array: np.ndarray,
        seg_meta: Dict,
        model,
        job: JobState
    ) -> Optional[Dict]:
        """
        从内存切片转录（Zero-copy，高性能）

        内存模式下使用，直接从完整音频数组中切片，无需磁盘IO。

        Args:
            audio_array: 完整音频数组
            seg_meta: 分段元数据 {"index": 0, "start": 0.0, "end": 30.5, "mode": "memory"}
            model: Whisper模型
            job: 任务状态

        Returns:
            Dict: 未对齐的转录结果
        """
        sr = 16000
        start_sample = int(seg_meta['start'] * sr)
        end_sample = int(seg_meta['end'] * sr)

        # Zero-copy切片（numpy view，不复制数据）
        audio_slice = audio_array[start_sample:end_sample]

        try:
            # Whisper转录
            rs = model.transcribe(
                audio_slice,
                batch_size=job.settings.batch_size,
                verbose=False,
                language=job.language
            )

            if not rs or 'segments' not in rs:
                return None

            # 检测语言（首次）
            if not job.language and 'language' in rs:
                job.language = rs['language']
                self.logger.info(f"detected language: {job.language}")

            # 时间偏移校正
            start_offset = seg_meta['start']
            adjusted_segments = []

            for idx, s in enumerate(rs['segments']):
                adjusted_segments.append({
                    'id': idx,
                    'start': s.get('start', 0) + start_offset,
                    'end': s.get('end', 0) + start_offset,
                    'text': s.get('text', '').strip()
                })

            return {
                'segment_index': seg_meta['index'],
                'language': rs.get('language', job.language),
                'segments': adjusted_segments
            }

        finally:
            # 注意：audio_slice是view，不需要单独释放
            gc.collect()

    def _transcribe_segment_from_disk(
        self,
        seg: Dict,
        model,
        job: JobState
    ) -> Optional[Dict]:
        """
        从文件加载转录（硬盘模式）

        硬盘模式下使用，从segment文件加载音频进行转录。

        Args:
            seg: 分段信息 {"index": 0, "file": "segment_0.wav", "start": 0.0, "end": 30.0, "mode": "disk"}
            model: Whisper模型
            job: 任务状态

        Returns:
            Dict: 未对齐的转录结果
        """
        audio = whisperx.load_audio(seg['file'])

        try:
            rs = model.transcribe(
                audio,
                batch_size=job.settings.batch_size,
                verbose=False,
                language=job.language
            )

            if not rs or 'segments' not in rs:
                return None

            # 检测语言（首次）
            if not job.language and 'language' in rs:
                job.language = rs['language']
                self.logger.info(f"detected language: {job.language}")

            # 时间偏移校正（使用start字段，秒为单位）
            start_offset = seg.get('start', seg.get('start_ms', 0) / 1000.0)
            adjusted_segments = []

            for idx, s in enumerate(rs['segments']):
                adjusted_segments.append({
                    'id': idx,
                    'start': s.get('start', 0) + start_offset,
                    'end': s.get('end', 0) + start_offset,
                    'text': s.get('text', '').strip()
                })

            return {
                'segment_index': seg['index'],
                'language': rs.get('language', job.language),
                'segments': adjusted_segments
            }

        finally:
            del audio
            gc.collect()

    def _transcribe_segment(
        self,
        seg_meta: Dict,
        model,
        job: JobState,
        audio_array: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        统一转录入口（根据模式自动选择）

        Args:
            seg_meta: 分段元数据
            model: Whisper模型
            job: 任务状态
            audio_array: 音频数组（内存模式时必须提供）

        Returns:
            Dict: 未对齐的转录结果
        """
        mode = seg_meta.get('mode', 'disk')

        if mode == 'memory':
            if audio_array is None:
                raise ValueError("memory mode requires audio_array parameter")
            return self._transcribe_segment_in_memory(audio_array, seg_meta, model, job)
        else:
            return self._transcribe_segment_from_disk(seg_meta, model, job)

    def _check_memory_during_transcription(self, job: JobState) -> bool:
        """
        转录过程中检查内存状态

        如果内存严重不足，暂停任务并警告用户。

        Args:
            job: 任务状态对象

        Returns:
            bool: True=继续处理，False=需要暂停
        """
        mem_info = psutil.virtual_memory()
        available_mb = mem_info.available / (1024 * 1024)
        percent_used = mem_info.percent

        # 危险阈值：可用内存<500MB 或 使用率>95%
        if available_mb < 500 or percent_used > 95:
            self.logger.error(f"memory critically low! available: {available_mb:.0f}MB, usage: {percent_used}%")
            job.status = 'paused'
            job.message = f"memory insufficient (available {available_mb:.0f}MB), please close other programs"
            job.paused = True

            # 推送警告SSE
            self._push_sse_signal(job, "memory_warning",
                f"memory critically low (available {available_mb:.0f}MB), task paused")

            return False

        # 警告阈值：可用内存<1GB 或 使用率>90%
        if available_mb < 1024 or percent_used > 90:
            self.logger.warning(f"memory tight: available {available_mb:.0f}MB, usage {percent_used}%")
            # 不暂停，但记录警告

        return True

    def _align_all_results(
        self,
        unaligned_results: List[Dict],
        job: JobState,
        audio_path: str
    ) -> List[Dict]:
        """
        对所有未对齐的转录结果进行统一对齐

        Args:
            unaligned_results: 所有未对齐的转录结果
            job: 任务状态
            audio_path: 完整音频文件路径

        Returns:
            List[Dict]: 对齐后的结果
        """
        self.logger.info(f"开始统一对齐 {len(unaligned_results)} 个分段的转录结果")

        # 1. 合并所有segments
        all_segments = []
        for result in unaligned_results:
            all_segments.extend(result['segments'])

        if not all_segments:
            self.logger.warning("没有可对齐的内容")
            return []

        # 2. 加载完整音频
        audio = whisperx.load_audio(audio_path)

        try:
            # 3. 获取对齐模型
            lang = job.language or unaligned_results[0].get('language', 'zh')
            align_model, metadata = self._get_align_model(lang, job.settings.device, job)

            # 4. 执行对齐（一次性处理所有segments）
            self._update_progress(job, 'align', 0, '正在对齐时间轴...')

            aligned = whisperx.align(
                all_segments,
                align_model,
                metadata,
                audio,
                job.settings.device
            )

            self._update_progress(job, 'align', 1, '对齐完成')

            # 5. 返回对齐后的结果
            return [{
                'segments': aligned.get('segments', []),
                'word_segments': aligned.get('word_segments', [])
            }]

        finally:
            del audio
            gc.collect()

    def _push_sse_align_progress(
        self,
        job: JobState,
        current_batch: int,
        total_batches: int,
        aligned_count: int,
        total_count: int
    ):
        """
        推送对齐进度SSE事件（前端进度条实时更新）

        事件类型: "align_progress"

        Args:
            job: 任务状态对象
            current_batch: 当前批次号（1-based）
            total_batches: 总批次数
            aligned_count: 已对齐的segment数量
            total_count: 总segment数量
        """
        try:
            from services.sse_service import get_sse_manager
            sse_manager = get_sse_manager()

            channel_id = f"job:{job.job_id}"

            # 计算百分比
            batch_progress = (current_batch / total_batches) * 100 if total_batches > 0 else 0
            segment_progress = (aligned_count / total_count) * 100 if total_count > 0 else 0

            sse_manager.broadcast_sync(
                channel_id,
                "align_progress",  # 专用事件类型
                {
                    "job_id": job.job_id,
                    "phase": "align",
                    "batch": {
                        "current": current_batch,
                        "total": total_batches,
                        "progress": round(batch_progress, 2)
                    },
                    "segments": {
                        "aligned": aligned_count,
                        "total": total_count,
                        "progress": round(segment_progress, 2)
                    },
                    "message": f"aligning batch {current_batch}/{total_batches} ({aligned_count}/{total_count} segments)"
                }
            )

        except Exception as e:
            self.logger.debug(f"SSE align progress push failed (non-fatal): {e}")

    def _align_all_results_batched(
        self,
        unaligned_results: List[Dict],
        job: JobState,
        audio_source,  # Union[np.ndarray, str]
        processing_mode: ProcessingMode
    ) -> List[Dict]:
        """
        分批对齐（支持实时SSE进度推送）

        批次对齐的优势：
        1. 避免一次性对齐所有内容导致的长时间卡顿
        2. 支持前端进度条实时更新
        3. 内存使用更可控

        Args:
            unaligned_results: 所有未对齐的转录结果
            job: 任务状态对象
            audio_source: 音频来源（内存模式传数组，硬盘模式传路径）
            processing_mode: 当前处理模式

        Returns:
            List[Dict]: 对齐后的结果
        """
        self.logger.info(f"starting batched alignment: {len(unaligned_results)} segments")

        # 1. 合并所有segments
        all_segments = []
        for result in unaligned_results:
            all_segments.extend(result['segments'])

        if not all_segments:
            self.logger.warning("no segments to align")
            return []

        # 2. 加载音频（根据模式）
        if processing_mode == ProcessingMode.MEMORY:
            audio_array = audio_source  # 直接使用内存数组
            self.logger.info("align phase: reusing audio array from memory")
        else:
            # 硬盘模式：需要加载完整音频
            self.logger.info("align phase: loading complete audio from disk...")
            audio_array = whisperx.load_audio(audio_source)

        try:
            # 3. 获取对齐模型
            lang = job.language or unaligned_results[0].get('language', 'zh')
            align_model, metadata = self._get_align_model(lang, job.settings.device, job)

            # 4. 分批对齐
            BATCH_SIZE = 50  # 每批50条segment
            total_segments = len(all_segments)
            total_batches = math.ceil(total_segments / BATCH_SIZE)
            aligned_segments = []

            self.logger.info(f"alignment config: total {total_segments} segments, {BATCH_SIZE} per batch, {total_batches} batches")

            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, total_segments)
                batch = all_segments[start_idx:end_idx]

                # 计算进度
                progress = batch_idx / total_batches

                # 更新任务进度
                self._update_progress(
                    job,
                    'align',
                    progress,
                    f'aligning batch {batch_idx + 1}/{total_batches}'
                )

                # 推送对齐进度SSE（专用事件）
                self._push_sse_align_progress(
                    job,
                    batch_idx + 1,
                    total_batches,
                    len(aligned_segments),
                    total_segments
                )

                # 执行对齐
                try:
                    aligned_batch = whisperx.align(
                        batch,
                        align_model,
                        metadata,
                        audio_array,
                        job.settings.device
                    )
                    aligned_segments.extend(aligned_batch.get('segments', []))
                    self.logger.debug(f"batch {batch_idx + 1}/{total_batches} completed")

                except Exception as e:
                    self.logger.error(f"batch {batch_idx + 1} alignment failed: {e}")
                    # 继续处理其他批次，不中断整体流程
                    continue

            # 5. 完成
            self._update_progress(job, 'align', 1, 'alignment complete')
            self._push_sse_align_progress(job, total_batches, total_batches, total_segments, total_segments)

            self.logger.info(f"batched alignment complete: {len(aligned_segments)} segments")

            return [{
                'segments': aligned_segments,
                'word_segments': []
            }]

        finally:
            # 如果是硬盘模式，释放加载的音频
            if processing_mode == ProcessingMode.DISK:
                del audio_array
                gc.collect()

    def _format_ts(self, sec: float) -> str:
        """
        格式化时间戳为SRT格式

        Args:
            sec: 秒数

        Returns:
            str: SRT时间戳 (HH:MM:SS,mmm)
        """
        if sec < 0:
            sec = 0

        ms = int(round(sec * 1000))
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000

        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def _generate_srt(self, results: List[Dict], path: str, word_level: bool):
        """
        生成SRT字幕文件

        Args:
            results: 转录结果列表
            path: 输出文件路径
            word_level: 是否使用词级时间戳
        """
        lines = []
        n = 1  # 字幕序号

        for r in results:
            if not r:
                continue

            entries = []

            # 词级时间戳模式
            if word_level and r.get('word_segments'):
                for w in r['word_segments']:
                    if w.get('start') is not None and w.get('end') is not None:
                        txt = (w.get('word') or '').strip()
                        if txt:
                            entries.append({
                                'start': w['start'],
                                'end': w['end'],
                                'text': txt
                            })

            # 句子级时间戳模式（默认）
            elif r.get('segments'):
                for s in r['segments']:
                    if s.get('start') is not None and s.get('end') is not None:
                        txt = (s.get('text') or '').strip()
                        if txt:
                            entries.append({
                                'start': s['start'],
                                'end': s['end'],
                                'text': txt
                            })

            # 写入SRT格式
            for e in entries:
                if e['end'] <= e['start']:
                    continue  # 跳过无效时间戳

                lines.append(str(n))  # 序号
                lines.append(
                    f"{self._format_ts(e['start'])} --> {self._format_ts(e['end'])}"
                )  # 时间戳
                lines.append(e['text'])  # 字幕文本
                lines.append("")  # 空行
                n += 1

        # 写入文件
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"SRT文件已生成: {path}, 共{n-1}条字幕")

    def clear_model_cache(self):
        """
        清空模型缓存（供队列服务调用）

        策略:
        - 总是清理 Whisper 模型（显存占用大，1-3GB）
        - 保留对齐模型的 LRU 缓存（占用小，每个~200MB）
        """
        global _model_cache, _align_model_cache

        # 1. 总是清理 Whisper 模型
        with _model_lock:
            for key in list(_model_cache.keys()):
                try:
                    del _model_cache[key]
                except:
                    pass
            _model_cache.clear()
            self.logger.info("Whisper模型缓存已清空")

        # 2. 保留对齐模型（记录当前缓存状态）
        with _align_lock:
            cached_langs = list(_align_model_cache.keys())
            if cached_langs:
                self.logger.debug(f"保留对齐模型缓存 (LRU): {cached_langs}")
            else:
                self.logger.debug("对齐模型缓存为空")


# 单例处理器
_service_instance: Optional[TranscriptionService] = None


def get_transcription_service(root: str) -> TranscriptionService:
    """获取转录服务实例（单例模式）"""
    global _service_instance
    if _service_instance is None:
        _service_instance = TranscriptionService(root)
    return _service_instance