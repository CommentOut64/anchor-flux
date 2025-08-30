import os
import sys
import uuid
import shutil
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import Optional, List
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor import JobSettings, CPUAffinityConfig, get_processor

app = FastAPI(title="Video To SRT API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output") 
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

print(f"DEBUG: BASE_DIR = {BASE_DIR}")
print(f"DEBUG: INPUT_DIR = {INPUT_DIR}")
print(f"DEBUG: INPUT_DIR exists = {os.path.exists(INPUT_DIR)}")

# 确保目录存在
for dir_path in [INPUT_DIR, OUTPUT_DIR, JOBS_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

proc = get_processor(JOBS_DIR)

class TranscribeSettings(BaseModel):
    model: str = "medium"
    compute_type: str = "float16"
    device: str = "cuda"
    batch_size: int = 16
    word_timestamps: bool = False
    # CPU亲和性配置
    cpu_affinity_enabled: bool = True
    cpu_affinity_strategy: str = "auto"  # "auto", "half", "custom"
    cpu_affinity_custom_cores: Optional[List[int]] = None
    cpu_affinity_exclude_cores: Optional[List[int]] = None

class UploadResponse(BaseModel):
    job_id: str
    filename: str
    original_name: str
    message: str

class FileInfo(BaseModel):
    name: str
    size: int
    modified: str
    path: str

def get_file_size_str(size_bytes):
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def is_video_or_audio_file(filename):
    """检查是否为支持的视频或音频文件"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    ext = os.path.splitext(filename.lower())[1]
    return ext in video_extensions or ext in audio_extensions

@app.get("/api/files")
async def list_files():
    """获取输入目录中的所有媒体文件"""
    try:
        files = []
        if os.path.exists(INPUT_DIR):
            for filename in os.listdir(INPUT_DIR):
                file_path = os.path.join(INPUT_DIR, filename)
                if os.path.isfile(file_path) and is_video_or_audio_file(filename):
                    stat = os.stat(file_path)
                    files.append(FileInfo(
                        name=filename,
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        path=file_path
                    ))
        
        # 按修改时间倒序排列
        files.sort(key=lambda x: x.modified, reverse=True)
        return {"files": files, "input_dir": INPUT_DIR}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    """删除input目录中的文件"""
    try:
        file_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        os.remove(file_path)
        return {"success": True, "message": f"文件 {filename} 已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件并自动创建转录任务"""
    try:
        # 验证文件类型
        if not is_video_or_audio_file(file.filename):
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        # 保存用户原始文件路径信息
        original_filename = file.filename
        
        # 将文件保存到input目录
        input_path = os.path.join(INPUT_DIR, original_filename)
        
        # 如果同名文件已存在，添加时间戳
        counter = 1
        base_name, ext = os.path.splitext(original_filename)
        while os.path.exists(input_path):
            new_filename = f"{base_name}_{counter}{ext}"
            input_path = os.path.join(INPUT_DIR, new_filename)
            original_filename = new_filename
            counter += 1
        
        # 保存文件
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 创建转录任务
        job_id = uuid.uuid4().hex
        settings = JobSettings()
        proc.create_job(original_filename, input_path, settings, job_id=job_id)
        
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

@app.post("/api/create-job")
async def create_job(filename: str = Form(...)):
    """为指定文件创建转录任务（保留兼容性）"""
    try:
        input_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        if not is_video_or_audio_file(filename):
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        job_id = uuid.uuid4().hex
        settings = JobSettings()
        proc.create_job(filename, input_path, settings, job_id=job_id)
        
        return {"job_id": job_id, "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")

@app.post("/api/start")
async def start(job_id: str = Form(...), settings: str = Form(...)):
    settings_obj = TranscribeSettings(**json.loads(settings))
    job = proc.get_job(job_id)
    if not job:
        return {"error": "无效 job_id"}
    
    # 创建CPU亲和性配置
    cpu_config = CPUAffinityConfig(
        enabled=settings_obj.cpu_affinity_enabled,
        strategy=settings_obj.cpu_affinity_strategy,
        custom_cores=settings_obj.cpu_affinity_custom_cores,
        exclude_cores=settings_obj.cpu_affinity_exclude_cores
    )
    
    # 覆盖设置
    job.settings = JobSettings(
        model=settings_obj.model,
        compute_type=settings_obj.compute_type,
        device=settings_obj.device,
        batch_size=settings_obj.batch_size,
        word_timestamps=settings_obj.word_timestamps,
        cpu_affinity=cpu_config
    )
    
    proc.start_job(job_id)
    return {"job_id": job_id, "started": True}

@app.post("/api/cancel/{job_id}")
async def cancel(job_id: str):
    job = proc.get_job(job_id)
    if not job:
        return {"error": "未找到"}
    ok = proc.cancel_job(job_id)
    return {"job_id": job_id, "canceled": ok}

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = proc.get_job(job_id)
    if not job:
        return {"error": "未找到"}
    return job.to_dict()

@app.get("/api/download/{job_id}")
async def download(job_id: str, copy_to_source: bool = False):
    job = proc.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    if job.srt_path and os.path.exists(job.srt_path):
        filename = os.path.basename(job.srt_path)
        
        # 如果需要复制到源文件目录
        if copy_to_source:
            # 获取原始文件路径的目录
            source_dir = os.path.dirname(job.input_path)
            source_srt_path = os.path.join(source_dir, filename)
            
            try:
                # 复制SRT文件到源文件目录
                shutil.copy2(job.srt_path, source_srt_path)
                print(f"SRT文件已复制到源目录: {source_srt_path}")
            except Exception as e:
                print(f"复制到源目录失败: {e}")
        
        # 同时复制到输出目录
        output_path = os.path.join(OUTPUT_DIR, filename)
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
    
    raise HTTPException(status_code=404, detail="字幕文件未生成")

@app.post("/api/copy-result/{job_id}")
async def copy_result_to_source(job_id: str):
    """将转录结果复制到源文件目录"""
    job = proc.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    if not job.srt_path or not os.path.exists(job.srt_path):
        raise HTTPException(status_code=404, detail="字幕文件未生成")
    
    try:
        # 获取原始文件目录
        if hasattr(job, 'original_path') and job.original_path:
            source_dir = os.path.dirname(job.original_path)
        else:
            # 如果没有original_path，使用input_path的同级目录
            source_dir = os.path.dirname(job.input_path)
        
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

@app.get("/api/ping")
async def ping():
    return {"pong": True}

@app.get("/api/cpu-info")
async def get_cpu_info():
    """获取系统CPU信息和亲和性支持状态"""
    try:
        cpu_info = proc.cpu_manager.get_system_info()
        return {
            "success": True,
            "cpu_info": cpu_info,
            "available_strategies": ["auto", "half", "custom"]
        }
    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "cpu_info": {"supported": False}
        }

@app.get("/api/hardware/basic")
async def get_hardware_basic():
    """获取核心硬件信息"""
    try:
        # 创建临时的硬件检测服务以获取信息
        from services.hardware_service import get_hardware_detector
        detector = get_hardware_detector()
        hardware_info = detector.detect()
        
        return {
            "success": True,
            "hardware": hardware_info.to_dict(),
            "message": "硬件信息获取成功"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"获取硬件信息失败: {str(e)}"
        }

@app.get("/api/hardware/optimize")
async def get_hardware_optimization():
    """获取基于硬件的优化配置"""
    try:
        from services.hardware_service import get_hardware_detector, get_hardware_optimizer
        detector = get_hardware_detector()
        optimizer = get_hardware_optimizer()
        
        hardware_info = detector.detect()
        optimization_config = optimizer.get_optimization_config(hardware_info)
        
        return {
            "success": True,
            "optimization": optimization_config.to_dict(),
            "message": "优化配置获取成功"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"获取优化配置失败: {str(e)}"
        }

@app.get("/api/hardware/status")
async def get_hardware_status():
    """获取完整的硬件状态和优化信息"""
    try:
        from services.hardware_service import get_hardware_detector, get_hardware_optimizer
        detector = get_hardware_detector()
        optimizer = get_hardware_optimizer()
        
        hardware_info = detector.detect()
        optimization_config = optimizer.get_optimization_config(hardware_info)
        
        return {
            "success": True,
            "hardware": hardware_info.to_dict(),
            "optimization": optimization_config.to_dict(),
            "message": "硬件状态获取成功"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"获取硬件状态失败: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    # 直接传入 app，关闭 reload，确保使用当前文件内定义的应用实例
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False,
                limit_max_requests=1000, limit_concurrency=50)
