import os
import sys
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import Optional

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor import JobSettings, get_processor

app = FastAPI(title="Video To SRT API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_DIR = os.path.join(os.getcwd(), "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)
proc = get_processor(JOBS_DIR)

class TranscribeSettings(BaseModel):
    model: str = "medium"
    compute_type: str = "float16"
    device: str = "cuda"
    batch_size: int = 16
    word_timestamps: bool = False

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    print(f"收到上传请求: {file.filename}")
    
    # 检查文件大小 (可选)
    # if file.size and file.size > 1024 * 1024 * 1024:  # 1GB 限制
    #     return {"error": "文件过大，请上传小于1GB的文件"}
    
    job_id = uuid.uuid4().hex
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    input_path = os.path.join(job_dir, file.filename)
    print(f"保存文件到: {input_path}")
    
    # 使用异步流式写入，支持大文件
    try:
        with open(input_path, 'wb') as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
        
        print(f"文件保存完成，文件大小: {os.path.getsize(input_path)} bytes")
        print(f"创建任务: {job_id}")
        
        settings = JobSettings()
        proc.create_job(file.filename, input_path, settings, job_id=job_id)
        return {"job_id": job_id, "filename": file.filename}
        
    except Exception as e:
        print(f"文件上传失败: {e}")
        # 清理失败的文件
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(job_dir):
            os.rmdir(job_dir)
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

@app.post("/api/start")
async def start(job_id: str = Form(...), settings: str = Form(...)):
    settings_obj = TranscribeSettings(**json.loads(settings))
    job = proc.get_job(job_id)
    if not job:
        return {"error": "无效 job_id"}
    # 覆盖设置
    job.settings = JobSettings(**settings_obj.dict())
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
async def download(job_id: str):
    job = proc.get_job(job_id)
    if not job:
        return {"error": "未找到"}
    if job.srt_path and os.path.exists(job.srt_path):
        return FileResponse(path=job.srt_path, filename=os.path.basename(job.srt_path), media_type='text/plain')
    return {"error": "结果未生成"}

@app.get("/api/ping")
async def ping():
    return {"pong": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, 
                limit_max_requests=1000, limit_concurrency=50)
