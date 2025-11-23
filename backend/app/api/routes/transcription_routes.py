"""
转录任务相关API路由
"""
import os
import uuid
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import json

from models.job_models import JobSettings, JobState
from services.transcription_service import TranscriptionService
from services.file_service import FileManagementService
from services.sse_service import get_sse_manager

router = APIRouter(prefix="/api", tags=["transcription"])


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
        """上传文件并自动创建转录任务"""
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
            
            # 创建转录任务
            job_id = uuid.uuid4().hex
            settings = JobSettings()
            transcription_service.create_job(original_filename, input_path, settings, job_id=job_id)
            
            return {
                "job_id": job_id, 
                "filename": original_filename,
                "original_name": file.filename,
                "message": "文件上传成功，转录任务已创建"
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

    @router.post("/start")
    async def start_job(job_id: str = Form(...), settings: str = Form(...)):
        """启动转录任务（支持断点续传参数校验）"""
        try:
            from pathlib import Path

            settings_obj = TranscribeSettings(**json.loads(settings))
            job = transcription_service.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="无效 job_id")

            # 检查是否有checkpoint（断点续传场景）
            job_dir = Path(job.dir)
            checkpoint_path = job_dir / "checkpoint.json"

            if checkpoint_path.exists():
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
            transcription_service.start_job(job_id)
            return {"job_id": job_id, "started": True}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"启动任务失败: {str(e)}")

    @router.post("/cancel/{job_id}")
    async def cancel_job(job_id: str, delete_data: bool = False):
        """取消转录任务"""
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        ok = transcription_service.cancel_job(job_id, delete_data=delete_data)
        return {"job_id": job_id, "canceled": ok, "data_deleted": delete_data}

    @router.post("/pause/{job_id}")
    async def pause_job(job_id: str):
        """暂停转录任务（保存断点）"""
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        ok = transcription_service.pause_job(job_id)
        return {"job_id": job_id, "paused": ok}

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
    async def get_job_status(job_id: str):
        """获取任务状态"""
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")
        return job.to_dict()

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

    return router