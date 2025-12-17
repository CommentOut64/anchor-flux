import os
import sys
import uuid
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import List
from datetime import datetime

app = FastAPI(title="Video To SRT API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 目录配置
BASE_DIR = "F:\\video_to_srt_gpu"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")

print(f"Using directories:")
print(f"  INPUT_DIR: {INPUT_DIR}")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  JOBS_DIR: {JOBS_DIR}")

class FileInfo(BaseModel):
    name: str
    size: int
    modified: str
    path: str

def is_video_or_audio_file(filename):
    """检查是否为支持的视频或音频文件"""
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    ext = os.path.splitext(filename.lower())[1]
    return ext in video_extensions or ext in audio_extensions

@app.get("/api/files")
async def list_files():
    """获取输入目录中的所有媒体文件"""
    print(f"Listing files from: {INPUT_DIR}")
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
        files.sort(key=lambda x: x.modified, reverse=True)
        print(f"Found {len(files)} files")
        return {"files": files}
    except Exception as e:
        print(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

@app.post("/api/create-job")  
async def create_job(filename: str = Form(...)):
    """为指定文件创建转录任务"""
    try:
        input_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        if not is_video_or_audio_file(filename):
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        job_id = uuid.uuid4().hex
        print(f"Created job {job_id} for file {filename}")
        return {"job_id": job_id, "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")

@app.get("/api/ping")
async def ping():
    return {"pong": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)