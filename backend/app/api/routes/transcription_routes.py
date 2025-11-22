"""
转录任务相关API路由
"""
import os
import uuid
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json

from models.job_models import JobSettings, JobState
from services.transcription_service import TranscriptionService
from services.file_service import FileManagementService

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
        """启动转录任务"""
        try:
            settings_obj = TranscribeSettings(**json.loads(settings))
            job = transcription_service.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="无效 job_id")
            
            # 覆盖设置
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

    return router