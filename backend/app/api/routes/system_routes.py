"""
系统管理API路由
包含客户端心跳、系统关闭等功能
"""

import asyncio
import logging
import subprocess
import os
import gc
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()


# ========== 请求/响应模型 ==========

class HeartbeatRequest(BaseModel):
    """心跳请求"""
    client_id: str


class RegisterRequest(BaseModel):
    """客户端注册请求"""
    client_id: str
    user_agent: Optional[str] = None


class UnregisterRequest(BaseModel):
    """客户端注销请求"""
    client_id: str


class ShutdownRequest(BaseModel):
    """系统关闭请求"""
    cleanup_temp: bool = False
    force: bool = False


# ========== 客户端心跳管理 ==========

@router.post("/api/system/heartbeat")
async def heartbeat(req: HeartbeatRequest):
    """客户端心跳"""
    from services.client_registry import get_client_registry

    client_registry = get_client_registry()
    success = client_registry.heartbeat(req.client_id)

    if not success:
        # 客户端未注册，自动注册
        client_registry.register(req.client_id)
        logger.debug(f"客户端自动注册: {req.client_id}")

    return {
        "success": True,
        "active_clients": client_registry.get_active_count()
    }


@router.post("/api/system/register")
async def register_client(req: RegisterRequest):
    """注册新客户端"""
    from services.client_registry import get_client_registry

    client_registry = get_client_registry()
    client_registry.register(req.client_id, req.user_agent)

    return {
        "success": True,
        "client_id": req.client_id
    }


@router.post("/api/system/unregister")
async def unregister_client(req: UnregisterRequest):
    """注销客户端（页面关闭时调用）"""
    from services.client_registry import get_client_registry

    client_registry = get_client_registry()
    client_registry.unregister(req.client_id)

    return {"success": True}


@router.get("/api/system/has-active-clients")
async def has_active_clients():
    """检查是否有活跃的浏览器标签页"""
    from services.client_registry import get_client_registry

    client_registry = get_client_registry()
    return {
        "has_active": client_registry.has_active_clients(),
        "count": client_registry.get_active_count()
    }


# ========== 系统关闭 ==========

@router.post("/api/system/shutdown")
async def shutdown_system(req: ShutdownRequest):
    """安全关闭系统"""
    cleanup_report = {}

    try:
        logger.info("="  * 60)
        logger.info("收到系统关闭请求")
        logger.info("="  * 60)

        # Phase 1: 清理 GPU 资源
        logger.info("Phase 1: 清理GPU资源...")

        # 1.1 卸载所有模型
        try:
            from services.model_preload_manager import get_model_manager
            model_manager = get_model_manager()
            if model_manager:
                model_manager.clear_cache()
                cleanup_report["models_unloaded"] = True
                logger.info("GPU模型已卸载")
        except Exception as e:
            logger.warning(f"卸载模型失败: {e}")
            cleanup_report["models_unloaded"] = False

        # 1.2 清理 GPU 缓存
        try:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU缓存已清理")
                    cleanup_report["gpu_cache_cleared"] = True
            except ImportError:
                cleanup_report["gpu_cache_cleared"] = False
        except Exception as e:
            logger.warning(f"清理GPU缓存失败: {e}")
            cleanup_report["gpu_cache_cleared"] = False

        # 1.3 终止 FFmpeg 子进程
        try:
            from services.ffmpeg_manager import get_ffmpeg_manager
            ffmpeg_mgr = get_ffmpeg_manager()
            # 这里假设有 kill_all 方法，如果没有则跳过
            if hasattr(ffmpeg_mgr, 'kill_all_subprocesses'):
                killed = ffmpeg_mgr.kill_all_subprocesses()
                cleanup_report["subprocesses_killed"] = killed
                logger.info(f"FFmpeg子进程已终止: {killed}个")
            else:
                cleanup_report["subprocesses_killed"] = 0
        except Exception as e:
            logger.warning(f"终止子进程失败: {e}")
            cleanup_report["subprocesses_killed"] = 0

        # 1.4 清理临时文件（可选）
        if req.cleanup_temp:
            try:
                from core.config import config
                import shutil
                if config.TEMP_DIR.exists():
                    shutil.rmtree(config.TEMP_DIR, ignore_errors=True)
                    config.TEMP_DIR.mkdir(exist_ok=True)
                    cleanup_report["temp_files_cleaned"] = True
                    logger.info("临时文件已清理")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {e}")
                cleanup_report["temp_files_cleaned"] = False

        # 发送成功响应
        response = {
            "success": True,
            "message": "系统正在关闭...",
            "cleanup_report": cleanup_report
        }

        logger.info("资源清理完成，准备关闭进程")

    except Exception as e:
        logger.error(f"关闭系统失败: {str(e)}", exc_info=True)
        response = {
            "success": False,
            "message": f"关闭系统失败: {str(e)}"
        }

    # Phase 2: 异步执行进程终止（响应发送后执行）
    asyncio.create_task(_terminate_processes())

    return response


async def _terminate_processes():
    """终止所有相关进程"""
    await asyncio.sleep(0.5)  # 等待响应发送完成

    logger.info("="  * 60)
    logger.info("Phase 2: 终止所有进程")
    logger.info("="  * 60)

    try:
        # 关闭 Vite 进程（按窗口标题）
        logger.info("关闭前端进程...")
        subprocess.run(
            'taskkill /F /FI "WINDOWTITLE eq 前端服务*"',
            shell=True,
            capture_output=True,
            timeout=5
        )

        # 关闭 cmd 窗口
        logger.info("关闭命令行窗口...")
        subprocess.run(
            'taskkill /F /FI "WINDOWTITLE eq 后端服务*"',
            shell=True,
            capture_output=True,
            timeout=5
        )
        subprocess.run(
            'taskkill /F /FI "WINDOWTITLE eq 前端服务*"',
            shell=True,
            capture_output=True,
            timeout=5
        )

    except Exception as e:
        logger.error(f"终止进程失败: {e}")

    # 自我终止
    logger.info("后端进程即将退出...")
    logger.info("="  * 60)
    os._exit(0)
