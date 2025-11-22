"""
转录处理服务
整合了processor.py和原transcription_service.py的所有功能
"""
import os, subprocess, uuid, threading, json, math, gc, logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from pydub import AudioSegment, silence
import whisperx
import torch
import shutil

from models.job_models import JobSettings, JobState
from models.hardware_models import HardwareInfo, OptimizationConfig
from services.hardware_service import get_hardware_detector, get_hardware_optimizer
from services.cpu_affinity_service import CPUAffinityManager, CPUAffinityConfig
from services.job_index_service import get_job_index_service
from core.config import config  # 导入统一配置

# 全局模型缓存 (按 (model, compute_type, device) 键)
_model_cache: Dict[Tuple[str, str, str], object] = {}
_align_model_cache: Dict[str, Tuple[object, object]] = {}
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

        # 记录CPU信息
        sys_info = self.cpu_manager.get_system_info()
        if sys_info.get('supported', False):
            self.logger.info(
                f"💻 CPU信息: {sys_info['logical_cores']}个逻辑核心, "
                f"{sys_info.get('physical_cores', '?')}个物理核心, "
                f"平台: {sys_info.get('platform', '?')}"
            )
        else:
            self.logger.warning("⚠️ CPU亲和性功能不可用")

        # 执行硬件检测
        self._detect_hardware()

    def _detect_hardware(self):
        """执行硬件检测并生成优化配置"""
        try:
            self.logger.info("开始硬件检测...")
            self._hardware_info = self.hardware_detector.detect()
            self._optimization_config = self.hardware_optimizer.get_optimization_config(self._hardware_info)
            
            # 记录检测结果
            hw = self._hardware_info
            opt = self._optimization_config
            self.logger.info(f"硬件检测完成 - GPU: {'✓' if hw.cuda_available else '✗'}, "
                           f"CPU: {hw.cpu_cores}核/{hw.cpu_threads}线程, "
                           f"内存: {hw.memory_total_mb}MB, "
                           f"优化配置: batch={opt.batch_size}, device={opt.recommended_device}")
        except Exception as e:
            self.logger.error(f"硬件检测失败: {e}")
    
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

        self.logger.info(f"✅ 任务已创建: {job_id} - {filename}")
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
            return self.jobs.get(job_id)

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

            self.logger.info(f"✅ 从检查点恢复任务: {job_id}")
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
        启动转录任务（支持从paused状态恢复）

        Args:
            job_id: 任务ID
        """
        job = self.get_job(job_id)
        if not job or job.status not in ("uploaded", "failed", "paused"):
            self.logger.warning(f"任务无法启动: {job_id}, 状态: {job.status if job else 'not found'}")
            return

        job.canceled = False
        job.paused = False  # 清除暂停标志
        job.error = None
        job.status = "processing"
        job.message = "开始处理" if job.status != "paused" else "恢复处理"

        # 在独立线程中执行转录
        threading.Thread(
            target=self._run_pipeline,
            args=(job,),
            daemon=True,
            name=f"Transcription-{job_id[:8]}"
        ).start()

        self.logger.info(f"🚀 任务已启动: {job_id}")

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
                    self.logger.info(f"🗑️ 已删除任务数据: {job_id}")
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

    def _save_checkpoint(self, job_dir: Path, data: dict):
        """
        原子性保存检查点
        使用"写临时文件 -> 重命名"策略，确保文件要么完整写入，要么保持原样

        Args:
            job_dir: 任务目录
            data: 检查点数据
        """
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
            processed_results = []
            current_segments = []

            if checkpoint:
                self.logger.info(f"🔄 发现检查点，从 {checkpoint.get('phase', 'unknown')} 阶段恢复")
                # 恢复数据到内存
                processed_indices = set(checkpoint.get('processed_indices', []))
                processed_results = checkpoint.get('results', [])
                current_segments = checkpoint.get('segments', [])
                # 恢复任务基本信息
                job.total = checkpoint.get('total_segments', 0)
                job.processed = len(processed_indices)
                self.logger.info(f"📊 已处理 {job.processed}/{job.total} 段")

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
                self.logger.info("✅ 跳过音频提取，使用已有文件")

            if job.canceled:
                raise RuntimeError('任务已取消')

            # ==========================================
            # 3. 阶段2: 智能分段
            # ==========================================
            # 如果检查点里没有分段信息，说明上次没跑到分段完成
            if not current_segments:
                self._update_progress(job, 'split', 0, '音频分段中')
                current_segments = self._split_audio(str(audio_path))
                if job.canceled:
                    raise RuntimeError('任务已取消')

                job.segments = current_segments
                job.total = len(current_segments)
                self._update_progress(job, 'split', 1, f'分段完成 共{job.total}段')

                # 【关键埋点1】分段完成后立即保存
                checkpoint_data = {
                    "job_id": job.job_id,
                    "phase": "split",
                    "total_segments": job.total,
                    "processed_indices": [],
                    "segments": current_segments,  # 保存分段结果
                    "results": []
                }
                self._save_checkpoint(job_dir, checkpoint_data)
                self.logger.info("💾 检查点已保存: 分段完成")
            else:
                self.logger.info(f"✅ 跳过分段，使用检查点数据（共{len(current_segments)}段）")
                job.segments = current_segments  # 恢复到 job 对象
                job.total = len(current_segments)

            # ==========================================
            # 4. 阶段3: 转录处理（核心循环）
            # ==========================================
            self._update_progress(job, 'transcribe', 0, '加载模型中')
            if job.canceled:
                raise RuntimeError('任务已取消')

            model = self._get_model(job.settings, job)
            align_cache = {}

            # 过滤出需要处理的段
            todo_segments = [
                seg for i, seg in enumerate(current_segments)
                if i not in processed_indices
            ]

            self.logger.info(f"📝 剩余 {len(todo_segments)}/{len(current_segments)} 段需要转录")

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

                ratio = len(processed_indices) / max(1, len(current_segments))
                self._update_progress(
                    job,
                    'transcribe',
                    ratio,
                    f'转录 {len(processed_indices)+1}/{len(current_segments)}'
                )

                seg_result = self._transcribe_segment(seg, model, job, align_cache)

                # --- 更新内存状态 ---
                if seg_result:
                    processed_results.append(seg_result)
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

                # 【关键埋点2】每处理一段保存一次
                # 单机版每段保存开销很小，建议直接每段保存，体验最好
                checkpoint_data = {
                    "job_id": job.job_id,
                    "phase": "transcribe",
                    "total_segments": len(current_segments),
                    "processed_indices": list(processed_indices),  # set转list
                    "segments": current_segments,
                    "results": processed_results
                }
                self._save_checkpoint(job_dir, checkpoint_data)
                self.logger.debug(f"💾 检查点已保存: {len(processed_indices)}/{len(current_segments)}")

            self._update_progress(job, 'transcribe', 1, '转录完成 生成字幕中')
            if job.canceled:
                raise RuntimeError('任务已取消')

            # ==========================================
            # 5. 阶段4: 生成SRT
            # ==========================================
            base_name = os.path.splitext(job.filename)[0]
            srt_path = job_dir / f'{base_name}.srt'
            self._update_progress(job, 'srt', 0, '写入 SRT...')
            self._generate_srt(
                processed_results,
                str(srt_path),
                job.settings.word_timestamps
            )
            self._update_progress(job, 'srt', 1, '处理完成')

            job.srt_path = str(srt_path)

            # 【清理】任务成功完成后，删除 checkpoint
            try:
                checkpoint_file = job_dir / "checkpoint.json"
                checkpoint_file.unlink(missing_ok=True)
                self.logger.info("🧹 检查点已清理")
            except Exception as e:
                self.logger.warning(f"清理检查点失败: {e}")

            if job.canceled:
                job.status = 'canceled'
                job.message = '已取消'
            else:
                job.status = 'finished'
                job.message = '完成'
                self.logger.info(f"✅ 任务完成: {job.job_id}")

        except Exception as e:
            if job.canceled and '取消' in str(e):
                job.status = 'canceled'
                job.message = '已取消'
                self.logger.info(f"🛑 任务已取消: {job.job_id}")
            elif job.paused and '暂停' in str(e):
                job.status = 'paused'
                job.message = '已暂停'
                self.logger.info(f"⏸️ 任务已暂停: {job.job_id}")
            else:
                job.status = 'failed'
                job.message = f'失败: {e}'
                job.error = str(e)
                self.logger.error(f"❌ 任务失败: {job.job_id} - {e}", exc_info=True)

        finally:
            # 恢复CPU亲和性设置
            if cpu_applied:
                restored = self.cpu_manager.restore_cpu_affinity()
                if restored:
                    self.logger.info(f"🔄 任务 {job.job_id} 已恢复CPU亲和性设置")

            # 释放内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ========== 核心处理方法 ==========

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
                self.logger.debug(f"✅ 音频提取成功: {audio_out}")
                return True
            else:
                error_msg = proc.stderr.decode('utf-8', errors='ignore')
                self.logger.error(f"❌ FFmpeg执行失败: {error_msg}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("❌ FFmpeg超时")
            return False
        except Exception as e:
            self.logger.error(f"❌ 音频提取失败: {e}")
            return False

    def _split_audio(self, audio_path: str) -> List[Dict]:
        """
        智能分段音频（基于静音检测）

        Args:
            audio_path: 音频文件路径

        Returns:
            List[Dict]: 段信息列表，每项包含 file 和 start_ms
        """
        self.logger.debug(f"开始音频分段: {audio_path}")

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
                search_start = max(pos, end - SILENCE_SEARCH_MS)
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
                    self.logger.warning(f"静音检测失败: {e}")

            # 导出分段
            chunk = audio[pos:end]
            seg_file = os.path.join(os.path.dirname(audio_path), f'segment_{idx}.wav')
            chunk.export(seg_file, format='wav')

            segments.append({
                'file': seg_file,
                'start_ms': pos,
                'duration_ms': end - pos
            })

            pos = end
            idx += 1

        self.logger.debug(f"✅ 音频分段完成: 共{len(segments)}段")
        return segments

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
                    self.logger.warning(f"⚠️ Whisper模型未下载或不完整: {settings.model}")

                    # 获取模型大小信息
                    model_size_mb = whisper_model_info.size_mb

                    # 如果模型大小>=1GB,给出特殊提示
                    download_msg = ""
                    if model_size_mb >= 1024:
                        size_gb = model_size_mb / 1024
                        download_msg = f"当前下载模型大于1GB ({size_gb:.1f}GB),请耐心等待"
                        self.logger.info(f"📦 {download_msg}")
                    else:
                        download_msg = f"开始下载模型 {settings.model} ({model_size_mb}MB)"

                    # 更新任务状态
                    if job:
                        job.message = download_msg

                    self.logger.info(f"🚀 自动触发下载Whisper模型: {settings.model} ({model_size_mb}MB)")

                    # 触发下载
                    success = model_mgr.download_whisper_model(settings.model)
                    if not success:
                        self.logger.warning(f"⚠️ 模型管理器下载失败或已在下载中,回退到whisperx")
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
                            self.logger.info(f"✅ Whisper模型下载完成: {settings.model}")
                            if job:
                                job.message = f"模型下载完成,准备加载"
                            break
                        elif current_status == "error":
                            self.logger.error(f"❌ 模型管理器下载失败,回退到whisperx")
                            raise RuntimeError(f"Whisper模型下载失败: {settings.model}")
                        else:
                            # 如果模型大小>=1GB,定期提醒用户耐心等待
                            if model_size_mb >= 1024 and elapsed % 30 == 0:  # 每30秒提醒一次
                                wait_msg = f"当前下载模型大于1GB,请耐心等待... {progress:.1f}% ({elapsed}s/{max_wait_time}s)"
                                self.logger.info(f"⏳ {wait_msg}")
                                if job:
                                    job.message = wait_msg
                            else:
                                wait_msg = f"等待模型下载... {progress:.1f}%"
                                self.logger.info(f"⏳ {wait_msg} ({elapsed}s/{max_wait_time}s)")
                                # 更新任务状态(每次都更新,这样用户可以看到进度变化)
                                if job:
                                    job.message = wait_msg

                    if elapsed >= max_wait_time:
                        self.logger.error(f"❌ 模型下载超时,回退到whisperx")
                        raise TimeoutError(f"Whisper模型下载超时: {settings.model}")

        except Exception as e:
            self.logger.warning(f"⚠️ 模型管理服务检查失败,回退到whisperx: {e}")

        # 尝试使用模型预加载管理器
        try:
            from services.model_preload_manager import get_model_manager as get_preload_manager
            model_manager = get_preload_manager()
            if model_manager:
                self.logger.debug("✅ 使用模型预加载管理器获取模型")
                if job:
                    job.message = "加载模型中"
                return model_manager.get_model(settings)
        except Exception as e:
            self.logger.debug(f"⚠️ 无法使用模型预加载管理器，回退到本地缓存: {e}")
            pass

        # 回退到简单缓存机制
        key = (settings.model, settings.compute_type, settings.device)
        with _model_lock:
            if key in _model_cache:
                self.logger.debug(f"✅ 命中模型缓存: {key}")
                if job:
                    job.message = "使用缓存的模型"
                return _model_cache[key]

            self.logger.info(f"🔍 加载模型: {key}")
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
                self.logger.warning(f"⚠️ 本地加载失败,允许whisperx下载: {e}")
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
        获取对齐模型（带缓存）

        集成模型管理器：如果模型不存在或不完整，会自动触发下载并等待完成

        Args:
            lang: 语言代码
            device: 设备 (cuda/cpu)
            job: 任务状态对象(可选,用于更新下载进度)

        Returns:
            Tuple[model, metadata]: 对齐模型和元数据
        """
        with _align_lock:
            # 检查本地缓存
            if lang in _align_model_cache:
                self.logger.debug(f"✅ 命中对齐模型缓存: {lang}")
                if job:
                    job.message = "使用缓存的对齐模型"
                return _align_model_cache[lang]

            # 尝试使用模型预加载管理器（优先从LRU缓存获取）
            try:
                from services.model_preload_manager import get_model_manager as get_preload_manager
                preload_mgr = get_preload_manager()
                if preload_mgr:
                    self.logger.debug("✅ 尝试从预加载管理器获取对齐模型")
                    if job:
                        job.message = "加载对齐模型"
                    am, meta = preload_mgr.get_align_model(lang, device)
                    _align_model_cache[lang] = (am, meta)
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
                        self.logger.warning(f"⚠️ 对齐模型不完整: {lang}")
                    else:
                        self.logger.warning(f"⚠️ 对齐模型未下载: {lang}")

                    # 对齐模型通常为1.2GB左右,给出大模型提示
                    download_msg = "当前下载模型大于1GB (约1.2GB),请耐心等待"
                    self.logger.info(f"📦 {download_msg}")
                    self.logger.info(f"🚀 自动触发下载对齐模型: {lang}")

                    # 更新任务状态
                    if job:
                        job.message = download_msg

                    # 触发下载
                    success = model_mgr.download_align_model(lang)
                    if not success:
                        self.logger.warning(f"⚠️ 模型管理器下载失败或已在下载中,回退到whisperx")
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
                            self.logger.info(f"✅ 对齐模型下载完成: {lang}")
                            if job:
                                job.message = "对齐模型下载完成,准备加载"
                            break
                        elif current_status == "error":
                            self.logger.error(f"❌ 模型管理器下载失败,回退到whisperx")
                            raise RuntimeError(f"对齐模型下载失败: {lang}")
                        else:
                            # 定期提醒用户耐心等待(每30秒)
                            if elapsed % 30 == 0:
                                wait_msg = f"当前下载模型大于1GB,请耐心等待... {progress:.1f}% ({elapsed}s/{max_wait_time}s)"
                                self.logger.info(f"⏳ {wait_msg}")
                                if job:
                                    job.message = wait_msg
                            else:
                                wait_msg = f"等待对齐模型下载... {progress:.1f}%"
                                self.logger.info(f"⏳ {wait_msg} ({elapsed}s/{max_wait_time}s)")
                                # 更新任务状态(每次都更新,这样用户可以看到进度变化)
                                if job:
                                    job.message = wait_msg

                    if elapsed >= max_wait_time:
                        self.logger.error(f"❌ 模型下载超时,回退到whisperx")
                        raise TimeoutError(f"对齐模型下载超时: {lang}")

            except Exception as e:
                self.logger.warning(f"⚠️ 模型管理服务检查失败,回退到whisperx: {e}")

            # 直接加载模型（如果已下载或下载完成）
            self.logger.info(f"🔍 加载对齐模型: {lang}")
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
                _align_model_cache[lang] = (am, meta)
                if job:
                    job.message = "对齐模型加载完成"
                return am, meta
            except Exception as e:
                self.logger.warning(f"⚠️ 本地加载对齐模型失败,允许whisperx下载: {e}")
                if job:
                    job.message = "本地对齐模型不存在,使用whisperx下载"
                # 如果本地加载失败,允许whisperx下载
                am, meta = whisperx.load_align_model(
                    language_code=lang,
                    device=device
                )
                _align_model_cache[lang] = (am, meta)
                if job:
                    job.message = "对齐模型下载并加载完成"
                return am, meta

    def _transcribe_segment(
        self,
        seg: Dict,
        model,
        job: JobState,
        align_cache: Dict
    ):
        """
        转录单个音频段

        Args:
            seg: 段信息 {file, start_ms, duration_ms}
            model: Whisper模型
            job: 任务状态
            align_cache: 对齐模型缓存

        Returns:
            Dict: 转录结果（包含segments和word_segments）
        """
        audio = whisperx.load_audio(seg['file'])

        try:
            # Whisper转录
            rs = model.transcribe(
                audio,
                batch_size=job.settings.batch_size,
                verbose=False,
                language=job.language
            )

            if not rs or 'segments' not in rs:
                return None

            # 检测语言
            if not job.language and 'language' in rs:
                job.language = rs['language']
                self.logger.info(f"🌐 检测到语言: {job.language}")

            lang = job.language or rs.get('language')

            # 加载对齐模型
            if lang not in align_cache:
                am, meta = self._get_align_model(lang, job.settings.device, job)
                align_cache[lang] = (am, meta)

            am, meta = align_cache[lang]

            # 词级对齐
            aligned = whisperx.align(
                rs['segments'],
                am,
                meta,
                audio,
                job.settings.device
            )

            # 时间偏移校正（重要！）
            start_offset = seg['start_ms'] / 1000.0
            final = {'segments': []}

            if 'segments' in aligned:
                for s in aligned['segments']:
                    if 'start' in s:
                        s['start'] += start_offset
                    if 'end' in s:
                        s['end'] += start_offset
                    final['segments'].append(s)

            if 'word_segments' in aligned:
                final['word_segments'] = []
                for w in aligned['word_segments']:
                    if 'start' in w:
                        w['start'] += start_offset
                    if 'end' in w:
                        w['end'] += start_offset
                    final['word_segments'].append(w)

            return final

        finally:
            del audio
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

        self.logger.info(f"✅ SRT文件已生成: {path}, 共{n-1}条字幕")


# 单例处理器
_service_instance: Optional[TranscriptionService] = None


def get_transcription_service(root: str) -> TranscriptionService:
    """获取转录服务实例（单例模式）"""
    global _service_instance
    if _service_instance is None:
        _service_instance = TranscriptionService(root)
    return _service_instance