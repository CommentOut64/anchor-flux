import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Debug API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 简化的目录配置
BASE_DIR = "F:\\video_to_srt_gpu"  # 硬编码路径
INPUT_DIR = os.path.join(BASE_DIR, "input")

print(f"BASE_DIR: {BASE_DIR}")
print(f"INPUT_DIR: {INPUT_DIR}")
print(f"INPUT_DIR exists: {os.path.exists(INPUT_DIR)}")

@app.get("/api/files")
async def list_files():
    """获取输入目录中的所有媒体文件"""
    files = []
    if os.path.exists(INPUT_DIR):
        for filename in os.listdir(INPUT_DIR):
            files.append({"name": filename, "size": 0})
    return {"files": files, "input_dir": INPUT_DIR}

@app.get("/api/ping")
async def ping():
    return {"pong": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("debug_main:app", host="127.0.0.1", port=8001, reload=True)