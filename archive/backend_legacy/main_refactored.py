"""
Video To SRT - 主应用程序
重构版本 - 更好的代码组织和模块分离
"""
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services import get_transcription_service, FileManagementService
from api.routes import create_file_router, create_transcription_router
from api.routes.hardware_routes import create_hardware_router

# FastAPI应用实例
app = FastAPI(title="Video To SRT API", version="0.4.0")

# CORS中间件
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

# 初始化服务
transcription_service = get_transcription_service(JOBS_DIR)
file_service = FileManagementService(INPUT_DIR, OUTPUT_DIR)

# 创建路由
file_router = create_file_router(file_service)
transcription_router = create_transcription_router(transcription_service, file_service, OUTPUT_DIR)
hardware_router = create_hardware_router(transcription_service)

# 注册路由
app.include_router(file_router)
app.include_router(transcription_router)
app.include_router(hardware_router)


@app.get("/api/ping")
async def ping():
    """健康检查接口"""
    return {"pong": True, "version": "0.4.0", "status": "running"}


if __name__ == "__main__":
    import uvicorn
    # 直接传入 app，关闭 reload，确保使用当前文件内定义的应用实例
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False,
                limit_max_requests=1000, limit_concurrency=50)