"""
模型管理API路由
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import List, AsyncGenerator
import asyncio
import json
import time
import logging

from app.models.model_models import ModelInfo
from app.services.model_manager_service import get_model_manager
from app.services.sse_service import get_sse_manager  # 导入统一SSE管理器

router = APIRouter(prefix="/api/models", tags=["models"])
logger = logging.getLogger(__name__)

# 获取模型管理器实例
model_manager = get_model_manager()


@router.get("/whisper", response_model=List[dict])
async def list_whisper_models():
    """
    列出所有Whisper模型

    Returns:
        List[dict]: 模型信息列表
    """
    try:
        models = model_manager.list_whisper_models()
        return [model.to_dict() for model in models]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.post("/whisper/{model_id}/download")
async def download_whisper_model(model_id: str):
    """
    下载指定的Whisper模型

    Args:
        model_id: 模型ID (tiny, base, small, medium, large-v2, large-v3)

    Returns:
        dict: 操作结果
    """
    try:
        success = model_manager.download_whisper_model(model_id)
        if not success:
            raise HTTPException(status_code=400, detail="模型不存在或正在下载中")
        return {"success": True, "message": f"开始下载模型 {model_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")


@router.delete("/whisper/{model_id}")
async def delete_whisper_model(model_id: str):
    """
    删除指定的Whisper模型

    Args:
        model_id: 模型ID

    Returns:
        dict: 操作结果
    """
    try:
        success = model_manager.delete_whisper_model(model_id)
        if not success:
            raise HTTPException(status_code=400, detail="删除失败：模型不存在或未下载")
        return {"success": True, "message": f"已删除模型 {model_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/progress")
async def get_download_progress():
    """
    获取所有下载进度（轮询模式，建议使用SSE端点）

    Returns:
        dict: 下载进度信息
    """
    try:
        return model_manager.get_download_progress()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取进度失败: {str(e)}")


# ========== SSE相关 - 使用统一管理器 ==========

# 获取统一SSE管理器实例
sse_manager = get_sse_manager()

def progress_callback(model_type: str, model_id: str, progress: float, status: str, message: str = ""):
    """
    进度回调函数，推送到SSE频道 "models"（线程安全）

    使用统一SSE管理器，替代原来的 sse_queues 方案
    """
    # 确定事件类型（根据状态）
    if status == 'ready':
        event_type = 'model_complete'
    elif status == 'error':
        event_type = 'model_error'
    elif status == 'incomplete':
        event_type = 'model_incomplete'
    else:
        event_type = 'model_progress'

    # 构造数据（保持与前端兼容的格式）
    event_data = {
        "type": model_type,
        "model_id": model_id,
        "progress": progress,
        "status": status,
        "message": message,
        "timestamp": time.time()
    }

    # 使用统一管理器的线程安全广播
    try:
        sse_manager.broadcast_sync("models", event_type, event_data)
    except Exception as e:
        logger.debug(f"SSE推送失败（非致命）: {e}")

# 注册进度回调
model_manager.register_progress_callback(progress_callback)


@router.get("/events/progress")
async def stream_all_progress(request: Request):
    """
    SSE端点：实时推送所有模型下载进度（使用统一SSE管理器）

    频道: "models"

    事件类型：
    - initial_state: 初始状态（所有模型）
    - model_progress: 下载进度更新
    - model_complete: 下载完成
    - model_error: 下载失败
    - model_incomplete: 模型不完整
    - ping: 心跳

    Returns:
        StreamingResponse: SSE事件流
    """

    # 定义初始状态回调 - 返回所有模型的当前状态
    def get_initial_state():
        return {
            "whisper": {
                m.model_id: {
                    "status": m.status,
                    "progress": m.download_progress
                }
                for m in model_manager.list_whisper_models()
            }
        }

    # 使用统一SSE管理器订阅 "models" 频道
    return StreamingResponse(
        sse_manager.subscribe("models", request, initial_state_callback=get_initial_state),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/events/{model_type}/{model_id}")
async def stream_single_progress(model_type: str, model_id: str):
    """
    SSE端点：推送单个模型的下载进度

    ⚠️ 已过时：建议使用 /events/progress 并在前端过滤，更高效

    Args:
        model_type: 模型类型 (whisper)
        model_id: 模型ID

    Returns:
        StreamingResponse: SSE事件流
    """
    # 验证参数
    if model_type != "whisper":
        raise HTTPException(status_code=400, detail="model_type 必须是 'whisper'")

    # 验证模型存在
    models = {m.model_id: m for m in model_manager.list_whisper_models()}

    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_type}/{model_id}")

    async def event_generator() -> AsyncGenerator[str, None]:
        """单模型SSE事件生成器"""
        last_state = {}
        heartbeat_counter = 0

        try:
            logger.info(f"单模型SSE连接已建立: {model_type}/{model_id}")

            while True:
                # 获取当前模型状态
                current_models = {m.model_id: m for m in model_manager.list_whisper_models()}

                if model_id not in current_models:
                    break

                model = current_models[model_id]
                current_state = {
                    "status": model.status,
                    "progress": model.download_progress
                }

                # 检测变化
                if current_state != last_state:
                    event_data = {
                        "type": model_type,
                        "model_id": model_id,
                        "status": current_state["status"],
                        "progress": current_state["progress"]
                    }

                    # 确定事件类型
                    if current_state["status"] == "ready" and last_state.get("status") != "ready":
                        event_name = "model_complete"
                    elif current_state["status"] == "error":
                        event_name = "model_error"
                        event_data["error"] = "下载失败"
                    else:
                        event_name = "model_progress"

                    # 推送事件
                    yield f"event: {event_name}\n"
                    yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

                last_state = current_state

                # 每30秒发送心跳
                heartbeat_counter += 1
                if heartbeat_counter >= 30:
                    yield f"event: heartbeat\n"
                    yield f"data: {json.dumps({'timestamp': int(time.time())})}\n\n"
                    heartbeat_counter = 0

                # 每秒检查一次
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"单模型SSE连接已关闭: {model_type}/{model_id}")
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
