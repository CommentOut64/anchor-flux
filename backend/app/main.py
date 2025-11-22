import os
import sys
import uuid
import shutil
import logging
import asyncio
import time
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from typing import Optional, List
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心配置和日志
from core.config import config
from core.logging import setup_logging

# 导入新的转录服务（替换processor）
from services.transcription_service import get_transcription_service
from models.job_models import JobSettings
from services.cpu_affinity_service import CPUAffinityConfig
from services.model_preload_manager import (
    PreloadConfig,
    get_model_manager,
    initialize_model_manager,
    preload_default_models,
    get_preload_status,
    get_cache_status
)
from config.model_config import ModelPreloadConfig

# 导入API路由
from api.routes import model_routes
from api.routes.transcription_routes import create_transcription_router
from services.file_service import FileManagementService

# 导入FFmpeg管理器
from services.ffmpeg_manager import get_ffmpeg_manager

# 配置日志（在其他初始化之前）
logger = setup_logging()

app = FastAPI(title="Video To SRT API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(model_routes.router)

@app.on_event("startup")
async def startup_event():
    """应用启动事件 - 初始化模型管理器和FFmpeg检测"""
    try:
        logger.info("="  * 60)
        logger.info("服务启动中...")
        logger.info("=" * 60)

        # 1. 设置SSE事件循环引用（必须在模型管理器初始化之前！）
        logger.info("步骤 1/3: 设置SSE事件循环...")
        try:
            from api.routes.model_routes import set_event_loop
            if set_event_loop():
                logger.info("✅ SSE事件循环已设置")
            else:
                logger.warning("⚠️ SSE事件循环设置失败，将使用备用机制")
        except Exception as e:
            logger.warning(f"设置SSE事件循环异常: {e}")

        # 2. FFmpeg检测和自动下载
        logger.info("步骤 2/3: 检测FFmpeg...")
        ffmpeg_mgr = get_ffmpeg_manager()
        try:
            ffmpeg_path = ffmpeg_mgr.ensure_ffmpeg()
            logger.info(f"FFmpeg检测完成: {ffmpeg_path}")
        except RuntimeError as e:
            # FFmpeg不可用但不阻止启动，只是记录警告
            logger.warning(f"FFmpeg检测失败: {e}")
            logger.warning("转录功能可能无法使用，请手动安装FFmpeg")

        # 3. 初始化模型管理器（此时事件循环已设置，后台验证可以正常推送SSE）
        logger.info("步骤 3/3: 初始化模型管理器...")
        model_manager = initialize_model_manager(preload_config)
        logger.info("模型管理器初始化成功")

        # 不在启动时预加载模型，等待前端就绪后通过API调用
        logger.info("后端服务已就绪，等待前端启动后进行模型预加载")

        logger.info("=" * 60)
        logger.info("服务启动完成")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"启动初始化失败: {str(e)}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件 - 清理资源"""
    try:
        model_manager = get_model_manager()
        if model_manager:
            model_manager.clear_cache()
            logger.info("已清理模型缓存")
    except Exception as e:
        logger.error(f"清理资源失败: {str(e)}")

# 使用统一配置中的目录
INPUT_DIR = str(config.INPUT_DIR)
OUTPUT_DIR = str(config.OUTPUT_DIR)
JOBS_DIR = str(config.JOBS_DIR)
TEMP_DIR = str(config.TEMP_DIR)

# 初始化转录服务
transcription_service = get_transcription_service(JOBS_DIR)

# 初始化文件管理服务
file_service = FileManagementService(INPUT_DIR, OUTPUT_DIR)

# 注册API路由
app.include_router(model_routes.router)
# 注册转录路由（包含暂停、恢复等新功能）
transcription_router = create_transcription_router(transcription_service, file_service, OUTPUT_DIR)
app.include_router(transcription_router)

# 初始化模型预加载管理器
preload_config = ModelPreloadConfig.get_preload_config()

# 打印配置信息
ModelPreloadConfig.print_config()

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
    """获取输入目录中的所有媒体文件（含断点信息）"""
    try:
        files = []
        if os.path.exists(INPUT_DIR):
            for filename in os.listdir(INPUT_DIR):
                file_path = os.path.join(INPUT_DIR, filename)
                if os.path.isfile(file_path) and is_video_or_audio_file(filename):
                    stat = os.stat(file_path)

                    # 检查是否有断点
                    checkpoint_info = transcription_service.check_file_checkpoint(file_path)

                    file_info = FileInfo(
                        name=filename,
                        size=stat.st_size,
                        modified=datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        path=file_path
                    )

                    # 添加断点信息
                    file_dict = file_info.dict()
                    if checkpoint_info:
                        file_dict['checkpoint'] = checkpoint_info
                    else:
                        file_dict['checkpoint'] = None

                    files.append(file_dict)

        # 按修改时间倒序排列
        files.sort(key=lambda x: x['modified'], reverse=True)
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
        transcription_service.create_job(filename, input_path, settings, job_id=job_id)
        
        return {"job_id": job_id, "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")

@app.post("/api/start")
async def start(job_id: str = Form(...), settings: str = Form(...)):
    try:
        settings_obj = TranscribeSettings(**json.loads(settings))
        job = transcription_service.get_job(job_id)
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
        
        transcription_service.start_job(job_id)
        return {"job_id": job_id, "started": True}
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {str(e)}, 原始数据: {settings}")
        raise HTTPException(status_code=400, detail=f"设置参数 JSON 格式无效: {str(e)}")
    except Exception as e:
        logger.error(f"启动任务失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"启动任务失败: {str(e)}")

# 取消任务端点已移至 transcription_routes.py，避免路由冲突

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = transcription_service.get_job(job_id)
    if not job:
        return {"error": "未找到"}
    return job.to_dict()

@app.get("/api/download/{job_id}")
async def download(job_id: str, copy_to_source: bool = False):
    job = transcription_service.get_job(job_id)
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
    job = transcription_service.get_job(job_id)
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
        cpu_info = transcription_service.cpu_manager.get_system_info()
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

# 模型管理API端点
@app.get("/api/models/preload/status")
async def get_models_preload_status():
    """获取模型预加载状态"""
    try:
        status = get_preload_status()
        return {
            "success": True,
            "data": status,
            "message": "获取预加载状态成功"
        }
    except Exception as e:
        logger.error(f"获取预加载状态失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"获取预加载状态失败: {str(e)}"
        }

@app.get("/api/models/cache/status")
async def get_models_cache_status():
    """获取模型缓存状态"""
    try:
        status = get_cache_status()
        return {
            "success": True,
            "data": status,
            "message": "获取缓存状态成功"
        }
    except Exception as e:
        logger.error(f"获取缓存状态失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"获取缓存状态失败: {str(e)}"
        }

@app.post("/api/models/preload/start")
async def start_models_preload():
    """手动启动模型预加载 - 简化版本，实现真正的幂等性"""
    try:
        logger.info("🚀 收到模型预加载请求")

        # 检查模型管理器
        model_manager = get_model_manager()
        if not model_manager:
            logger.error("❌ 模型管理器未初始化")
            return {"success": False, "message": "模型管理器未初始化"}

        # 直接调用模型管理器的预加载方法 - 它已经实现了幂等性
        result = await model_manager.preload_models()
        
        if result["success"]:
            logger.info(f"✅ 模型预加载成功: {result.get('loaded_models', 0)}/{result.get('total_models', 0)} 个模型")
            return {
                "success": True,
                "message": "预加载已启动",
                "loaded_models": result.get("loaded_models", 0),
                "total_models": result.get("total_models", 0)
            }
        else:
            logger.warning(f"⚠️ 模型预加载未成功: {result.get('message', 'Unknown error')}")
            return {
                "success": False,
                "message": result.get("message", "预加载失败"),
                "failed_attempts": result.get("failed_attempts", 0)
            }

    except Exception as e:
        logger.error(f"❌ 模型预加载异常: {str(e)}", exc_info=True)
        return {"success": False, "message": f"启动预加载失败: {str(e)}"}

@app.post("/api/models/cache/clear")
async def clear_models_cache():
    """清空模型缓存 - 简化版本，立即同步状态"""
    try:
        from services.model_preload_manager import get_model_manager
        model_manager = get_model_manager()
        
        if model_manager:
            model_manager.clear_cache()
            logger.info("✅ 手动清空模型缓存成功")
            return {
                "success": True,
                "message": "模型缓存已清空",
                "cache_version": model_manager.get_preload_status().get("cache_version", 0)
            }
        else:
            return {
                "success": False,
                "message": "模型管理器未初始化"
            }
            
    except Exception as e:
        logger.error(f"❌ 清空模型缓存失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"清空缓存失败: {str(e)}"
        }

@app.post("/api/models/preload/reset")
async def reset_preload_attempts():
    """重置预加载失败计数"""
    try:
        from services.model_preload_manager import get_model_manager
        model_manager = get_model_manager()

        if model_manager:
            model_manager.reset_preload_attempts()
            logger.info("手动重置预加载失败计数成功")
            return {
                "success": True,
                "message": "预加载失败计数已重置"
            }
        else:
            return {
                "success": False,
                "message": "模型管理器未初始化"
            }
    except Exception as e:
        logger.error(f"❌ 重置预加载失败计数失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"重置失败: {str(e)}"
        }

# ========== 默认预加载模型配置API ==========

@app.get("/api/models/preload/config")
async def get_default_preload_config():
    """获取默认预加载模型配置"""
    try:
        from services.user_config_service import get_user_config_service
        from services.model_manager_service import get_model_manager

        user_config = get_user_config_service()
        model_manager = get_model_manager()

        # 获取用户选择的模型
        user_selected = user_config.get_default_preload_model()

        # 获取所有ready的模型
        ready_models = model_manager.get_ready_whisper_models() if model_manager else []

        # 获取体积最大的ready模型
        largest_model = model_manager.get_largest_ready_model() if model_manager else None

        # 确定实际会使用的模型
        actual_model = user_selected if user_selected and user_selected in ready_models else largest_model

        return {
            "success": True,
            "data": {
                "user_selected": user_selected,  # 用户选择的模型
                "largest_model": largest_model,  # 体积最大的ready模型
                "actual_model": actual_model,    # 实际会使用的模型
                "ready_models": ready_models     # 所有ready的模型列表
            }
        }
    except Exception as e:
        logger.error(f"❌ 获取默认预加载配置失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"获取配置失败: {str(e)}"
        }

@app.post("/api/models/preload/config")
async def set_default_preload_model(request: dict):
    """设置默认预加载模型"""
    try:
        from services.user_config_service import get_user_config_service

        model_id = request.get("model_id")
        user_config = get_user_config_service()

        success = user_config.set_default_preload_model(model_id)

        if success:
            logger.info(f"✅ 设置默认预加载模型: {model_id}")
            return {
                "success": True,
                "message": f"默认预加载模型已设置为: {model_id or '自动选择'}"
            }
        else:
            return {
                "success": False,
                "message": "设置失败"
            }
    except Exception as e:
        logger.error(f"❌ 设置默认预加载模型失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"设置失败: {str(e)}"
        }

# ========== 模型加载/卸载API ==========

@app.post("/api/models/cache/unload")
async def unload_model(request: dict):
    """卸载指定模型"""
    try:
        from services.model_preload_manager import get_model_manager as get_preload_manager

        model_id = request.get("model_id")
        device = request.get("device", "cuda")
        compute_type = request.get("compute_type", "float16")

        if not model_id:
            return {
                "success": False,
                "message": "缺少model_id参数"
            }

        preload_manager = get_preload_manager()
        if not preload_manager:
            return {
                "success": False,
                "message": "模型管理器未初始化"
            }

        preload_manager.evict_model(model_id, device, compute_type)
        logger.info(f"✅ 卸载模型: {model_id}")

        return {
            "success": True,
            "message": f"模型 {model_id} 已卸载"
        }
    except Exception as e:
        logger.error(f"❌ 卸载模型失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"卸载失败: {str(e)}"
        }

@app.post("/api/models/preload/load-specific")
async def load_specific_model(request: dict):
    """加载指定模型"""
    try:
        from services.model_preload_manager import get_model_manager as get_preload_manager, PreloadConfig
        from models.job_models import JobSettings
        import torch

        model_id = request.get("model_id")

        if not model_id:
            return {
                "success": False,
                "message": "缺少model_id参数"
            }

        preload_manager = get_preload_manager()
        if not preload_manager:
            return {
                "success": False,
                "message": "模型管理器未初始化"
            }

        # 检查模型状态
        from services.model_manager_service import get_model_manager
        model_mgr = get_model_manager()
        status, local_path, detail = model_mgr._check_whisper_model_exists(model_id)

        if status != "ready":
            return {
                "success": False,
                "message": f"模型未就绪: {status}"
            }

        # 准备加载参数
        device = "cuda" if torch.cuda.is_available() else "cpu"
        settings = JobSettings(
            model=model_id,
            compute_type="float16",
            device=device
        )

        # 加载模型
        logger.info(f"🔄 开始加载模型: {model_id}")
        model = await asyncio.get_event_loop().run_in_executor(
            None,
            preload_manager.get_model,
            settings
        )

        if model:
            logger.info(f"✅ 模型加载成功: {model_id}")
            return {
                "success": True,
                "message": f"模型 {model_id} 加载成功"
            }
        else:
            return {
                "success": False,
                "message": "模型加载失败"
            }
    except Exception as e:
        logger.error(f"❌ 加载模型失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"加载失败: {str(e)}"
        }

@app.post("/api/shutdown")
async def shutdown_server():
    """优雅关闭服务器"""
    try:
        logger.info("收到关闭服务器请求")

        # 清理资源
        from services.model_preload_manager import get_model_manager
        model_manager = get_model_manager()
        if model_manager:
            model_manager.clear_cache()
            logger.info("已清理模型缓存")
        
        # 返回成功响应
        response = {
            "success": True,
            "message": "服务器正在优雅关闭"
        }
        
        # 异步关闭服务器
        import asyncio
        import os
        asyncio.create_task(delayed_shutdown())
        
        return response
        
    except Exception as e:
        logger.error(f"关闭服务器失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"关闭服务器失败: {str(e)}"
        }

async def delayed_shutdown():
    """延迟关闭服务器，给响应时间返回"""
    await asyncio.sleep(1)  # 等待1秒让响应返回
    logger.info("服务器即将关闭...")
    import os
    os._exit(0)

if __name__ == "__main__":
    import uvicorn
    # 直接传入 app，关闭 reload，确保使用当前文件内定义的应用实例
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False,
                limit_max_requests=1000, limit_concurrency=50)
