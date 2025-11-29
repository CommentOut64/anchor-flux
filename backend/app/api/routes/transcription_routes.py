"""
转录任务相关API路由
"""
import os
import uuid
import shutil
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, Body
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json

from models.job_models import JobSettings, JobState
from services.transcription_service import TranscriptionService
from services.file_service import FileManagementService
from services.sse_service import get_sse_manager
from services.job_queue_service import get_queue_service  # 新增导入


class TranscribeSettings(BaseModel):
    """转录设置请求模型"""
    model: str = "medium"
    compute_type: str = "float16"
    device: str = "cuda"
    batch_size: int = 16
    word_timestamps: bool = False


class UploadResponse(BaseModel):
    """上传响应模型"""
    job_id: str
    filename: str
    original_name: str
    message: str


def create_transcription_router(
    transcription_service: TranscriptionService,
    file_service: FileManagementService,
    output_dir: str
):
    """创建转录任务路由"""

    # 创建路由器实例
    router = APIRouter(prefix="/api", tags=["transcription"])

    # 获取SSE管理器
    sse_manager = get_sse_manager()

    @router.get("/stream/{job_id}")
    async def stream_job_progress(job_id: str, request: Request):
        """
        SSE流式端点 - 实时推送转录任务进度

        频道ID格式: job:{job_id}
        事件类型:
        - progress: 进度更新 (包含 percent, phase, message, status等)
        - signal: 关键节点信号 (job_complete, job_failed, job_canceled, job_paused)
        - ping: 心跳
        """
        # 验证任务是否存在
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        channel_id = f"job:{job_id}"

        # 定义初始状态回调 - 连接时立即发送当前状态
        def get_initial_state():
            current_job = transcription_service.get_job(job_id)
            if current_job:
                return {
                    "job_id": current_job.job_id,
                    "phase": current_job.phase,
                    "percent": current_job.progress,
                    "message": current_job.message,
                    "status": current_job.status,
                    "processed": current_job.processed,
                    "total": current_job.total,
                    "language": current_job.language or ""
                }
            return None

        # 订阅SSE流
        return StreamingResponse(
            sse_manager.subscribe(channel_id, request, initial_state_callback=get_initial_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    @router.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """上传文件并自动创建转录任务（V2.2: 加入队列）"""
        try:
            # 验证文件类型
            if not file_service.is_supported_file(file.filename):
                raise HTTPException(status_code=400, detail="不支持的文件格式")

            # 保存用户原始文件路径信息
            original_filename = file.filename

            # 将文件保存到input目录
            input_path = file_service.get_input_file_path(original_filename)

            # 如果同名文件已存在，添加时间戳
            counter = 1
            base_name, ext = os.path.splitext(original_filename)
            while os.path.exists(input_path):
                new_filename = f"{base_name}_{counter}{ext}"
                input_path = file_service.get_input_file_path(new_filename)
                original_filename = new_filename
                counter += 1

            # 保存文件
            with open(input_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # 创建任务
            job_id = uuid.uuid4().hex
            settings = JobSettings()
            job = transcription_service.create_job(original_filename, input_path, settings, job_id=job_id)

            # 🔥 新增: 加入队列（而非直接启动）
            queue_service = get_queue_service()
            queue_service.add_job(job)

            return {
                "job_id": job_id,
                "filename": original_filename,
                "original_name": file.filename,
                "message": "文件上传成功，已加入转录队列",
                "queue_position": len(queue_service.queue)  # 新增: 队列位置
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"上传文件失败: {str(e)}")

    @router.post("/create-job")
    async def create_job(filename: str = Form(...)):
        """为指定文件创建转录任务（本地input模式）"""
        try:
            input_path = file_service.get_input_file_path(filename)
            if not os.path.exists(input_path):
                raise HTTPException(status_code=404, detail="文件不存在")

            if not file_service.is_supported_file(filename):
                raise HTTPException(status_code=400, detail="不支持的文件格式")

            job_id = uuid.uuid4().hex
            settings = JobSettings()
            transcription_service.create_job(filename, input_path, settings, job_id=job_id)

            return {"job_id": job_id, "filename": filename}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")

    @router.post("/create-jobs-batch")
    async def create_jobs_batch(filenames: list = Body(..., embed=True)):
        """
        批量创建转录任务（从 input 目录选择多个文件）

        Args:
            filenames: 文件名列表

        Returns:
            {
                "success": true,
                "jobs": [{job_id, filename, queue_position}, ...],
                "failed": [{filename, error}, ...],
                "total": int,
                "succeeded": int,
                "failed_count": int
            }
        """
        try:
            queue_service = get_queue_service()
            jobs = []
            failed = []

            for filename in filenames:
                try:
                    # 验证文件存在
                    input_path = file_service.get_input_file_path(filename)
                    if not os.path.exists(input_path):
                        failed.append({"filename": filename, "error": "文件不存在"})
                        continue

                    # 验证文件格式
                    if not file_service.is_supported_file(filename):
                        failed.append({"filename": filename, "error": "不支持的文件格式"})
                        continue

                    # 创建任务
                    job_id = uuid.uuid4().hex
                    settings = JobSettings()
                    job = transcription_service.create_job(filename, input_path, settings, job_id=job_id)

                    # 加入队列
                    queue_service.add_job(job)

                    jobs.append({
                        "job_id": job_id,
                        "filename": filename,
                        "queue_position": len(queue_service.queue)
                    })

                except Exception as e:
                    failed.append({"filename": filename, "error": str(e)})

            return {
                "success": True,
                "jobs": jobs,
                "failed": failed,
                "total": len(filenames),
                "succeeded": len(jobs),
                "failed_count": len(failed)
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"批量创建任务失败: {str(e)}")

    @router.post("/start")
    async def start_job(job_id: str = Form(...), settings: str = Form(...)):
        """启动转录任务（V2.2: 加入队列而非直接启动）"""
        try:
            from pathlib import Path

            settings_obj = TranscribeSettings(**json.loads(settings))

            # 获取队列服务
            queue_service = get_queue_service()
            job = queue_service.get_job(job_id)

            if not job:
                # 如果队列服务中没有，尝试从transcription_service获取
                job = transcription_service.get_job(job_id)

            if not job:
                raise HTTPException(status_code=404, detail="无效 job_id")

            # 检查是否有checkpoint（断点续传场景）
            job_dir = Path(job.dir) if job.dir else None
            checkpoint_path = job_dir / "checkpoint.json" if job_dir else None

            if checkpoint_path and checkpoint_path.exists():
                # 有checkpoint，需要校验参数并强制覆盖禁止修改的参数
                try:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)

                    original_settings = checkpoint_data.get("original_settings", {})

                    if original_settings:
                        # 强制覆盖禁止修改的参数
                        # 1. word_timestamps - 禁止修改
                        if "word_timestamps" in original_settings:
                            settings_obj.word_timestamps = original_settings["word_timestamps"]

                        # 注意：device和model虽然会警告，但仍允许用户修改
                        # 前端应该在调用此接口前显示警告并获得用户确认
                except Exception as e:
                    # 如果读取checkpoint失败，记录日志但继续
                    print(f"读取checkpoint设置失败: {e}")

            # 应用设置
            job.settings = JobSettings(**settings_obj.dict())

            # 🔥 关键改动: 如果任务不在队列中，加入队列
            with queue_service.lock:
                if job.status == "paused" or job.status == "failed":
                    # 恢复任务：重新加入队列
                    job.canceled = False
                    job.paused = False
                    job.error = None
                    queue_service.queue.append(job_id)
                    job.status = "queued"
                    job.message = f"已加入队列 (位置: {len(queue_service.queue)})"
                    # 确保任务在jobs字典中
                    queue_service.jobs[job_id] = job
                elif job.status == "uploaded" or job.status == "created":
                    # 新任务：加入队列
                    queue_service.queue.append(job_id)
                    job.status = "queued"
                    job.message = f"已加入队列 (位置: {len(queue_service.queue)})"
                    # 确保任务在jobs字典中
                    queue_service.jobs[job_id] = job
                elif job.status == "queued":
                    # 任务已在队列中
                    queue_position = list(queue_service.queue).index(job_id) + 1 if job_id in queue_service.queue else -1
                    job.message = f"已在队列中 (位置: {queue_position})"

            # 保存队列状态并推送 SSE 通知（修复：之前缺少这一步导致前端收不到状态更新）
            queue_service._save_state()
            queue_service._notify_queue_change()
            queue_service._notify_job_status(job_id, job.status)

            return {
                "job_id": job_id,
                "started": True,
                "queue_position": len(queue_service.queue)
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"启动任务失败: {str(e)}")

    @router.post("/cancel/{job_id}")
    async def cancel_job(job_id: str, delete_data: bool = False):
        """取消转录任务（V2.2: 使用队列服务）"""
        queue_service = get_queue_service()
        ok = queue_service.cancel_job(job_id, delete_data=delete_data)
        if not ok:
            raise HTTPException(status_code=404, detail="任务未找到")
        return {"job_id": job_id, "canceled": ok, "data_deleted": delete_data}

    @router.post("/pause/{job_id}")
    async def pause_job(job_id: str):
        """暂停转录任务（V2.2: 使用队列服务）"""
        queue_service = get_queue_service()
        ok = queue_service.pause_job(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail="任务未找到")
        return {"job_id": job_id, "paused": ok}

    @router.post("/resume/{job_id}")
    async def resume_job(job_id: str):
        """
        恢复暂停的任务（重新加入队列）

        与 /restore-job 不同：
        - /resume: 恢复暂停的任务，重新加入队列尾部，状态变为 queued
        - /restore-job: 从 checkpoint 断点续传
        """
        queue_service = get_queue_service()
        ok = queue_service.resume_job(job_id)
        if not ok:
            raise HTTPException(status_code=400, detail="无法恢复任务（任务未暂停或不存在）")

        job = queue_service.get_job(job_id)
        queue_position = 0
        if job_id in queue_service.queue:
            queue_position = list(queue_service.queue).index(job_id) + 1

        return {
            "job_id": job_id,
            "resumed": True,
            "status": job.status if job else "queued",
            "queue_position": queue_position
        }

    @router.post("/prioritize/{job_id}")
    async def prioritize_job(job_id: str, mode: Optional[str] = None):
        """
        将任务移到队列头部（插队）

        Args:
            job_id: 任务ID
            mode: 插队模式
                - "gentle": 温和插队，放到队列头部，等当前任务完成后执行
                - "force": 强制插队，暂停当前任务A -> 执行B -> B完成后自动恢复A
                - None: 使用默认模式（可通过 /api/queue-settings 配置）
        """
        queue_service = get_queue_service()
        result = queue_service.prioritize_job(job_id, mode=mode)

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "无法优先此任务")
            )

        return {
            "job_id": job_id,
            "prioritized": True,
            "mode": result.get("mode"),
            "interrupted_job_id": result.get("interrupted_job_id"),
            "queue_position": 1
        }

    @router.get("/queue-settings")
    async def get_queue_settings():
        """
        获取队列设置

        返回:
            - default_prioritize_mode: 默认插队模式 ("gentle" 或 "force")
        """
        queue_service = get_queue_service()
        return queue_service.get_settings()

    @router.post("/queue-settings")
    async def update_queue_settings(
        default_prioritize_mode: Optional[str] = Body(None, embed=True)
    ):
        """
        更新队列设置

        Args:
            default_prioritize_mode: 默认插队模式
                - "gentle": 温和插队（默认）
                - "force": 强制插队
        """
        queue_service = get_queue_service()
        try:
            settings = queue_service.update_settings(
                default_prioritize_mode=default_prioritize_mode
            )
            return {
                "success": True,
                "settings": settings
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/reorder-queue")
    async def reorder_queue(job_ids: list = Body(..., embed=True)):
        """
        重新排序队列

        Args:
            job_ids: 按新顺序排列的任务ID列表
        """
        queue_service = get_queue_service()
        ok = queue_service.reorder_queue(job_ids)

        if not ok:
            raise HTTPException(status_code=400, detail="重排队列失败（任务ID不匹配）")

        return {
            "reordered": True,
            "queue": job_ids
        }

    @router.get("/queue-status")
    async def get_queue_status():
        """获取队列状态摘要"""
        queue_service = get_queue_service()
        return queue_service.get_queue_status()

    @router.get("/events/global")
    async def stream_global_events(request: Request):
        """
        全局SSE流 - 推送所有任务的状态变化 (V3.0)

        事件类型:
        - initial_state: 连接时的初始状态
        - queue_update: 队列顺序变化
        - job_status: 任务状态变化
        - job_progress: 任务进度更新

        注意:
        - initial_state返回所有任务（处理中 + 已完成）
        - 避免客户端连接时漏掉完成任务的实时更新
        """
        queue_service = get_queue_service()

        def get_initial_state():
            """
            返回所有任务列表（第二阶段修复：实时更新）
            包含活跃任务 + 历史完成任务
            """
            from services.job_index_service import get_job_index_service
            from core.config import config
            from pathlib import Path
            import json as json_module

            jobs_summary = []

            # 1. 添加活跃任务（从队列中）
            with queue_service.lock:
                for jid, job in queue_service.jobs.items():
                    jobs_summary.append({
                        "id": jid,
                        "status": job.status,
                        "progress": job.progress,
                        "filename": job.filename,
                        "title": job.title if hasattr(job, 'title') else "",  # 用户自定义名称
                        "message": job.message,
                        "created_time": job.createdAt if hasattr(job, 'createdAt') else None,
                        "phase": job.phase if hasattr(job, 'phase') else 'unknown'
                    })

                queue_list = list(queue_service.queue)
                running_id = queue_service.running_job_id
                interrupted_id = queue_service.interrupted_job_id

            # 2. 添加历史完成任务（从 jobs 目录）
            try:
                jobs_root = Path(config.JOBS_DIR)
                job_index = get_job_index_service(config.JOBS_DIR)
                active_job_ids = set(jid for jid, _ in queue_service.jobs.items())

                for job_dir in jobs_root.iterdir():
                    if not job_dir.is_dir():
                        continue

                    job_id = job_dir.name
                    if job_id in active_job_ids:
                        # 已在活跃任务中，跳过
                        continue

                    # 尝试找到文件名
                    filename = "未知文件"
                    file_path = job_index.get_file_path(job_id)
                    if file_path:
                        filename = os.path.basename(file_path)
                    else:
                        # 从目录中找视频文件
                        for ext in ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mp3', '.wav', '.m4a']:
                            matches = list(job_dir.glob(f"*{ext}"))
                            if matches:
                                filename = matches[0].name
                                break

                    # 检查是否完成
                    srt_files = list(job_dir.glob("*.srt"))
                    is_finished = len(srt_files) > 0

                    # 获取创建时间
                    try:
                        stat = job_dir.stat()
                        created_time = int(stat.st_ctime * 1000)
                    except:
                        created_time = None

                    # 尝试从 checkpoint 获取进度
                    progress = 100 if is_finished else 0
                    phase = 'editing' if is_finished else 'transcribing'
                    status = 'finished' if is_finished else 'processing'

                    checkpoint_path = job_dir / "checkpoint.json"
                    if checkpoint_path.exists():
                        try:
                            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                                checkpoint_data = json_module.load(f)
                                total_segments = checkpoint_data.get('total_segments', 0)
                                processed_indices = checkpoint_data.get('processed_indices', [])
                                if total_segments > 0:
                                    progress = (len(processed_indices) / total_segments) * 100
                                phase = checkpoint_data.get('phase', 'transcribing')
                        except:
                            pass

                    # 尝试从 state.json 读取 title
                    title = ""
                    state_file = job_dir / "state.json"
                    if state_file.exists():
                        try:
                            with open(state_file, 'r', encoding='utf-8') as f:
                                state_data = json_module.load(f)
                                title = state_data.get('title', '')
                        except:
                            pass

                    jobs_summary.append({
                        "id": job_id,
                        "status": status,
                        "progress": min(progress, 100),
                        "filename": filename,
                        "title": title,  # 用户自定义名称
                        "message": "已完成" if is_finished else "处理中",
                        "created_time": created_time,
                        "phase": phase
                    })

            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"加载历史任务失败: {e}")

            return {
                "queue": queue_list,
                "running": running_id,
                "interrupted": interrupted_id,
                "jobs": jobs_summary
            }

        # 订阅SSE流，频道名为 "global"
        return StreamingResponse(
            sse_manager.subscribe("global", request, initial_state_callback=get_initial_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    @router.get("/sync-tasks")
    async def sync_tasks():
        """
        同步所有任务（第一阶段修复：数据同步）

        返回所有任务列表（处理中 + 已完成），前端用此接口同步后端实际存在的任务
        此接口为真实源，用于修复幽灵任务问题
        """
        from services.job_index_service import get_job_index_service
        from core.config import config
        from pathlib import Path
        import json as json_module

        queue_service = get_queue_service()
        job_index = get_job_index_service(config.JOBS_DIR)
        jobs_root = Path(config.JOBS_DIR)

        # 清理无效映射（任务或文件不存在的映射）
        job_index.cleanup_invalid_mappings()

        # 收集所有任务
        all_tasks = {}  # 使用 dict 避免重复，key 为 job_id

        # 1. 队列中的任务（处理中或等待中）- 优先级最高
        with queue_service.lock:
            for job_id, job in queue_service.jobs.items():
                all_tasks[job_id] = {
                    "id": job.job_id,
                    "filename": job.filename,
                    "title": job.title if hasattr(job, 'title') else "",  # 用户自定义名称
                    "status": job.status,
                    "progress": job.progress,
                    "message": job.message,
                    "created_time": job.createdAt if hasattr(job, 'createdAt') else None,
                    "phase": job.phase if hasattr(job, 'phase') else 'unknown'
                }

        # 2. 扫描 jobs 目录中的所有任务（包括已完成的）
        try:
            for job_dir in jobs_root.iterdir():
                if not job_dir.is_dir():
                    continue

                job_id = job_dir.name
                if job_id in all_tasks:
                    # 已在队列中，跳过
                    continue

                # 尝试找到文件名
                filename = "未知文件"

                # 1. 从 job_index 查找
                file_path = job_index.get_file_path(job_id)
                if file_path:
                    filename = os.path.basename(file_path)
                else:
                    # 2. 从目录中找视频文件
                    for ext in ['.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.mp3', '.wav', '.m4a']:
                        matches = list(job_dir.glob(f"*{ext}"))
                        if matches:
                            filename = matches[0].name
                            break

                # 判断任务是否完成
                srt_files = list(job_dir.glob("*.srt"))
                is_finished = len(srt_files) > 0

                # 获取创建时间
                try:
                    stat = job_dir.stat()
                    created_time = int(stat.st_ctime * 1000)
                except:
                    created_time = None

                # 尝试从 checkpoint 获取进度信息
                checkpoint_path = job_dir / "checkpoint.json"
                progress = 100 if is_finished else 0
                phase = 'editing' if is_finished else 'transcribing'
                status = 'finished' if is_finished else 'processing'

                if checkpoint_path.exists():
                    try:
                        with open(checkpoint_path, 'r', encoding='utf-8') as f:
                            checkpoint_data = json_module.load(f)
                            total_segments = checkpoint_data.get('total_segments', 0)
                            processed_indices = checkpoint_data.get('processed_indices', [])
                            if total_segments > 0:
                                progress = (len(processed_indices) / total_segments) * 100
                            phase = checkpoint_data.get('phase', 'transcribing')
                    except:
                        pass

                # 尝试从 state.json 读取 title
                title = ""
                state_file = job_dir / "state.json"
                if state_file.exists():
                    try:
                        with open(state_file, 'r', encoding='utf-8') as f:
                            state_data = json_module.load(f)
                            title = state_data.get('title', '')
                    except:
                        pass

                all_tasks[job_id] = {
                    "id": job_id,
                    "filename": filename,
                    "title": title,  # 用户自定义名称
                    "status": status,
                    "progress": min(progress, 100),  # 确保不超过100
                    "message": "已完成" if is_finished else "处理中",
                    "created_time": created_time,
                    "phase": phase
                }

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"扫描 jobs 目录失败: {e}")

        return {
            "success": True,
            "tasks": list(all_tasks.values()),
            "count": len(all_tasks),
            "timestamp": int(time.time() * 1000)
        }

    @router.get("/incomplete-jobs")
    async def get_incomplete_jobs():
        """获取所有未完成的任务"""
        jobs = transcription_service.scan_incomplete_jobs()
        return {"jobs": jobs, "count": len(jobs)}

    @router.post("/restore-job/{job_id}")
    async def restore_job(job_id: str):
        """从检查点恢复任务"""
        job = transcription_service.restore_job_from_checkpoint(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="无法恢复任务，检查点不存在或已损坏")

        return job.to_dict()

    @router.get("/status/{job_id}")
    async def get_job_status(job_id: str, include_media: bool = True):
        """
        获取任务状态（V2.3: 包含队列位置和媒体状态）

        Args:
            job_id: 任务ID
            include_media: 是否包含媒体状态信息（默认True）
        """
        queue_service = get_queue_service()
        job = queue_service.get_job(job_id)
        if not job:
            # 如果队列服务中没有，尝试从transcription_service获取
            job = transcription_service.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="任务未找到")

        # 返回状态（新增queue_position字段）
        result = job.to_dict()

        # 计算队列位置
        with queue_service.lock:
            if job_id in queue_service.queue:
                result["queue_position"] = list(queue_service.queue).index(job_id) + 1
            elif job_id == queue_service.running_job_id:
                result["queue_position"] = 0  # 0表示正在执行
            else:
                result["queue_position"] = -1  # -1表示不在队列中

        # 添加媒体状态信息（用于编辑器）
        if include_media and job.status == "finished" and job.dir:
            job.update_media_status(job.dir)
            if job.media_status:
                result["media_status"] = {
                    "video_exists": job.media_status.video_exists,
                    "video_format": job.media_status.video_format,
                    "needs_proxy": job.media_status.needs_proxy,
                    "proxy_exists": job.media_status.proxy_exists,
                    "audio_exists": job.media_status.audio_exists,
                    "peaks_ready": job.media_status.peaks_ready,
                    "thumbnails_ready": job.media_status.thumbnails_ready,
                    "srt_exists": job.media_status.srt_exists,
                    # 便捷的URL字段
                    "video_url": f"/api/media/{job_id}/video" if job.media_status.video_exists or job.media_status.proxy_exists else None,
                    "audio_url": f"/api/media/{job_id}/audio" if job.media_status.audio_exists else None,
                    "peaks_url": f"/api/media/{job_id}/peaks" if job.media_status.audio_exists else None,
                    "thumbnails_url": f"/api/media/{job_id}/thumbnails" if job.media_status.video_exists else None,
                    "srt_url": f"/api/media/{job_id}/srt" if job.media_status.srt_exists else None
                }

        return result

    @router.get("/download/{job_id}")
    async def download_result(job_id: str, copy_to_source: bool = False):
        """下载转录结果"""
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")
        
        if not job.srt_path or not os.path.exists(job.srt_path):
            raise HTTPException(status_code=404, detail="字幕文件未生成")
        
        filename = os.path.basename(job.srt_path)
        
        # 如果需要复制到源文件目录
        if copy_to_source and job.input_path:
            source_dir = os.path.dirname(job.input_path)
            source_srt_path = os.path.join(source_dir, filename)
            
            try:
                shutil.copy2(job.srt_path, source_srt_path)
                print(f"SRT文件已复制到源目录: {source_srt_path}")
            except Exception as e:
                print(f"复制到源目录失败: {e}")
        
        # 同时复制到输出目录
        output_path = os.path.join(output_dir, filename)
        try:
            if not os.path.exists(output_path):
                shutil.copy2(job.srt_path, output_path)
            
            return FileResponse(
                path=output_path, 
                filename=filename, 
                media_type='text/plain; charset=utf-8'
            )
        except Exception as e:
            # 如果复制失败，直接返回原文件
            return FileResponse(
                path=job.srt_path, 
                filename=filename, 
                media_type='text/plain; charset=utf-8'
            )

    @router.post("/copy-result/{job_id}")
    async def copy_result_to_source(job_id: str):
        """将转录结果复制到源文件目录"""
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        if not job.srt_path or not os.path.exists(job.srt_path):
            raise HTTPException(status_code=404, detail="字幕文件未生成")

        try:
            # 获取原始文件目录
            if job.input_path:
                source_dir = os.path.dirname(job.input_path)
            else:
                # 如果没有input_path，使用input目录
                source_dir = file_service.input_dir

            # 生成目标路径
            srt_filename = os.path.basename(job.srt_path)
            target_path = os.path.join(source_dir, srt_filename)

            # 复制文件
            shutil.copy2(job.srt_path, target_path)

            return {
                "success": True,
                "message": f"字幕文件已复制到: {target_path}",
                "target_path": target_path
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"复制文件失败: {str(e)}")

    @router.get("/check-resume/{job_id}")
    async def check_resume(job_id: str):
        """检查任务是否可以断点续传"""
        from pathlib import Path

        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        job_dir = Path(job.dir)
        checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return {
                "can_resume": False,
                "message": "无检查点"
            }

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            total_segments = data.get('total_segments', 0)
            processed_indices = data.get('processed_indices', [])
            processed_count = len(processed_indices)

            if total_segments > 0:
                progress = (processed_count / total_segments) * 100
            else:
                progress = 0

            return {
                "can_resume": True,
                "progress": round(progress, 2),
                "processed_segments": processed_count,
                "total_segments": total_segments,
                "phase": data.get('phase', 'unknown'),
                "message": f"检测到上次进度 ({progress:.1f}%)，可从断点继续"
            }
        except Exception as e:
            return {
                "can_resume": False,
                "message": f"检查点文件损坏: {str(e)}"
            }

    @router.get("/checkpoint-settings/{job_id}")
    async def get_checkpoint_settings(job_id: str):
        """获取checkpoint中保存的原始设置（用于参数校验）"""
        from pathlib import Path

        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        job_dir = Path(job.dir)
        checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return {"has_checkpoint": False}

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return {
                "has_checkpoint": True,
                "original_settings": data.get("original_settings", {}),
                "progress": {
                    "phase": data.get("phase"),
                    "processed": len(data.get("processed_indices", [])),
                    "total": data.get("total_segments", 0)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取检查点失败: {str(e)}")

    @router.get("/transcription-text/{job_id}")
    async def get_transcription_text(job_id: str):
        """
        从checkpoint中提取已完成的转录文字（未对齐版本）

        用于SSE断线重连后，前端可以调用此API获取当前已转录的所有文字

        返回格式：
        {
            "job_id": "...",
            "has_checkpoint": true,
            "language": "zh",
            "segments": [
                {"id": 0, "start": 10.5, "end": 15.2, "text": "第一句话"},
                {"id": 1, "start": 15.2, "end": 20.0, "text": "第二句话"}
            ],
            "progress": {
                "processed": 50,
                "total": 100,
                "percentage": 50.0
            }
        }
        """
        from pathlib import Path

        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        job_dir = Path(job.dir)
        checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return {
                "job_id": job_id,
                "has_checkpoint": False,
                "message": "没有检查点数据"
            }

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取未对齐结果
            unaligned_results = data.get("unaligned_results", [])

            # 合并所有segments
            all_segments = []
            detected_language = None
            for result in unaligned_results:
                if not detected_language and 'language' in result:
                    detected_language = result['language']
                all_segments.extend(result.get('segments', []))

            # 按时间排序
            all_segments.sort(key=lambda x: x.get('start', 0))

            # 重新编号
            for idx, seg in enumerate(all_segments):
                seg['id'] = idx

            return {
                "job_id": job_id,
                "has_checkpoint": True,
                "language": detected_language or job.language or "unknown",
                "segments": all_segments,
                "progress": {
                    "processed": len(data.get("processed_indices", [])),
                    "total": data.get("total_segments", 0),
                    "percentage": round(
                        len(data.get("processed_indices", [])) / max(1, data.get("total_segments", 1)) * 100,
                        2
                    )
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取转录文字失败: {str(e)}")

    @router.post("/validate-resume-settings")
    async def validate_resume_settings(
        job_id: str = Form(...),
        new_settings: str = Form(...)
    ):
        """
        校验恢复任务时的参数修改

        返回：
        - valid: bool - 是否可以使用新参数
        - warnings: list - 警告信息
        - errors: list - 错误信息（禁止修改的参数）
        - force_original: dict - 必须强制使用的原始参数
        """
        from pathlib import Path

        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        job_dir = Path(job.dir)
        checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return {
                "valid": True,
                "warnings": [],
                "errors": [],
                "force_original": {},
                "message": "无检查点，可以使用任意参数"
            }

        try:
            # 加载checkpoint
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            original_settings = checkpoint_data.get("original_settings", {})
            if not original_settings:
                return {
                    "valid": True,
                    "warnings": [],
                    "errors": [],
                    "force_original": {},
                    "message": "旧版checkpoint格式，建议使用默认参数"
                }

            # 解析新设置
            new_settings_obj = json.loads(new_settings)

            warnings = []
            errors = []
            force_original = {}

            # 检查禁止修改的参数
            # 1. word_timestamps - 禁止修改
            if "word_timestamps" in original_settings:
                if new_settings_obj.get("word_timestamps") != original_settings["word_timestamps"]:
                    errors.append({
                        "param": "word_timestamps",
                        "reason": "修改此参数会导致前后SRT格式不一致",
                        "impact": "严重",
                        "original": original_settings["word_timestamps"],
                        "new": new_settings_obj.get("word_timestamps")
                    })
                    force_original["word_timestamps"] = original_settings["word_timestamps"]

            # 2. device - 建议不修改（中等影响）
            if "device" in original_settings:
                if new_settings_obj.get("device") != original_settings["device"]:
                    warnings.append({
                        "param": "device",
                        "level": "medium",
                        "reason": "不同设备的精度可能有细微差异",
                        "impact": "中等",
                        "original": original_settings["device"],
                        "new": new_settings_obj.get("device"),
                        "suggestion": "建议保持原设备设置"
                    })

            # 3. model - 允许但需严重警告
            if "model" in original_settings:
                if new_settings_obj.get("model") != original_settings["model"]:
                    warnings.append({
                        "param": "model",
                        "level": "high",
                        "reason": "不同模型的输出格式和质量可能不同，混用会导致前后字幕质量不一致",
                        "impact": "高",
                        "original": original_settings["model"],
                        "new": new_settings_obj.get("model"),
                        "suggestion": "仅在确认用错模型时才修改"
                    })

            # compute_type 和 batch_size 可以自由修改，不需要警告

            return {
                "valid": len(errors) == 0,
                "warnings": warnings,
                "errors": errors,
                "force_original": force_original,
                "message": "参数校验完成" if len(errors) == 0 else "检测到不兼容的参数修改"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"参数校验失败: {str(e)}")

    @router.post("/rename-job/{job_id}")
    async def rename_job(job_id: str, title: str = Body(..., embed=True)):
        """
        重命名任务

        Args:
            job_id: 任务ID
            title: 新的任务名称（为空时恢复使用 filename）

        Returns:
            {
                "success": bool,
                "job_id": str,
                "title": str,
                "message": str
            }
        """
        try:
            # 从队列服务或转录服务获取任务
            queue_service = get_queue_service()
            job = queue_service.get_job(job_id)

            if not job:
                # 如果队列服务中没有，尝试从 jobs 目录恢复
                job = transcription_service.get_job(job_id)

            if not job:
                raise HTTPException(status_code=404, detail="任务未找到")

            # 更新 title 字段
            job.title = title.strip() if title else ""

            # 保存任务状态到文件
            if job.dir:
                from pathlib import Path
                job_dir = Path(job.dir)
                state_file = job_dir / "state.json"

                try:
                    state_data = job.to_dict()
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"保存任务状态失败: {e}")

            # 通知 SSE 订阅者任务信息已更新
            sse_manager.broadcast_sync("global", "job_renamed", {
                "job_id": job_id,
                "title": job.title,
                "filename": job.filename
            })

            return {
                "success": True,
                "job_id": job_id,
                "title": job.title,
                "message": "任务重命名成功"
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"重命名任务失败: {str(e)}")

    return router