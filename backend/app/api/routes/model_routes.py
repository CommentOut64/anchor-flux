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

from models.model_models import ModelInfo, AlignModelInfo
from services.model_manager_service import get_model_manager

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


@router.get("/align", response_model=List[dict])
async def list_align_models():
    """
    列出所有对齐模型

    Returns:
        List[dict]: 对齐模型信息列表
    """
    try:
        models = model_manager.list_align_models()
        return [model.to_dict() for model in models]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对齐模型列表失败: {str(e)}")


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


@router.post("/align/{language}/download")
async def download_align_model(language: str):
    """
    下载指定语言的对齐模型

    Args:
        language: 语言代码 (zh, en, ja, ko, etc.)

    Returns:
        dict: 操作结果
    """
    try:
        success = model_manager.download_align_model(language)
        if not success:
            raise HTTPException(status_code=400, detail="语言不支持或正在下载中")
        return {"success": True, "message": f"开始下载 {language} 对齐模型"}
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


@router.delete("/align/{language}")
async def delete_align_model(language: str):
    """
    删除指定的对齐模型

    Args:
        language: 语言代码

    Returns:
        dict: 操作结果
    """
    try:
        success = model_manager.delete_align_model(language)
        if not success:
            raise HTTPException(status_code=400, detail="删除失败：模型不存在或未下载")
        return {"success": True, "message": f"已删除对齐模型 {language}"}
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


# ========== SSE 实时进度推送端点 ==========

# 存储活动的 SSE 连接队列（改用 asyncio.Queue）
# 增大队列容量，避免频繁更新时队列溢出
sse_queues: List[asyncio.Queue] = []

# 获取当前事件循环的引用（在应用启动时设置）
_event_loop = None

def set_event_loop():
    """设置事件循环引用（在应用启动时调用）"""
    global _event_loop
    try:
        _event_loop = asyncio.get_running_loop()
        logger.info("✅ SSE事件循环已设置")
        return True
    except RuntimeError as e:
        logger.warning(f"⚠️ 无法获取事件循环: {e}")
        return False

def get_event_loop():
    """获取事件循环引用（线程安全，支持延迟获取）"""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        try:
            _event_loop = asyncio.get_running_loop()
            logger.debug("🔄 延迟获取事件循环成功")
        except RuntimeError:
            logger.debug("⚠️ 当前线程没有运行中的事件循环")
            return None
    return _event_loop

def progress_callback(model_type: str, model_id: str, progress: float, status: str, message: str = ""):
    """
    进度回调函数，用于推送到所有 SSE 连接（线程安全，优化队列溢出处理）

    关键优化：
    1. 检查队列容量，避免 QueueFull 异常
    2. 队列满时跳过更新，而不是抛出异常
    3. 降低日志级别，避免日志洪水
    """
    event_data = {
        "type": model_type,
        "model_id": model_id,
        "progress": progress,
        "status": status,
        "message": message,
        "timestamp": time.time()
    }

    # 获取事件循环
    loop = get_event_loop()
    if loop is None:
        # 只在第一次失败时警告，避免日志洪水
        if not hasattr(progress_callback, '_warned'):
            logger.warning(f"⚠️ 事件循环未设置，无法推送SSE消息")
            progress_callback._warned = True
        return

    # 使用 call_soon_threadsafe 从下载线程向主事件循环注入消息
    success_count = 0
    skipped_count = 0

    for q in sse_queues[:]:  # 使用切片复制列表
        try:
            # 关键优化：检查队列容量，避免 QueueFull 异常
            current_size = q.qsize()
            max_size = q.maxsize

            # 如果队列使用率超过90%，跳过此次更新（保留紧急容量）
            if current_size >= max_size * 0.9:
                skipped_count += 1
                # 只记录 debug 级别，避免日志洪水
                if skipped_count == 1:  # 只记录第一次跳过
                    logger.debug(f"队列接近满({current_size}/{max_size})，跳过更新: {model_type}/{model_id}")
                continue

            # 线程安全地将消息放入队列
            loop.call_soon_threadsafe(q.put_nowait, event_data)
            success_count += 1

        except Exception as e:
            # 降低日志级别，避免日志洪水
            logger.debug(f"推送SSE消息失败: {e}")

    # 只在成功推送或有跳过时记录 debug 日志
    if success_count > 0:
        logger.debug(f"SSE消息已推送到 {success_count} 个连接: {model_type}/{model_id} - {status} ({progress:.1f}%)")
    if skipped_count > 0 and skipped_count % 10 == 0:  # 每10次跳过才记录一次
        logger.debug(f"已跳过 {skipped_count} 次更新（队列繁忙）")

# 注册进度回调
model_manager.register_progress_callback(progress_callback)

@router.get("/events/progress")
async def stream_all_progress(request: Request):
    """
    SSE端点：实时推送所有模型下载进度（改进版 - 非阻塞 asyncio.Queue）

    事件类型：
    - model_progress: 下载进度更新
    - model_complete: 下载完成
    - model_error: 下载失败
    - model_incomplete: 模型不完整
    - heartbeat: 心跳（每15秒）

    Returns:
        StreamingResponse: SSE事件流
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """SSE事件生成器（非阻塞实现，优化队列大小）"""
        # 创建此连接的专用 asyncio.Queue（增大容量到1000，避免频繁溢出）
        event_queue = asyncio.Queue(maxsize=1000)
        sse_queues.append(event_queue)

        heartbeat_counter = 0

        try:
            logger.info(f"✅ SSE连接已建立（当前连接数: {len(sse_queues)}）")

            # 首次发送所有模型状态
            initial_state = {
                "whisper": {
                    m.model_id: {
                        "status": m.status,
                        "progress": m.download_progress
                    }
                    for m in model_manager.list_whisper_models()
                },
                "align": {
                    m.language: {
                        "status": m.status,
                        "progress": m.download_progress
                    }
                    for m in model_manager.list_align_models()
                }
            }

            yield f"event: initial_state\ndata: {json.dumps(initial_state)}\n\n"

            while True:
                # 检查客户端是否断开
                if await request.is_disconnected():
                    logger.info("⚠️ 客户端已断开连接")
                    break

                try:
                    # 非阻塞等待消息，超时后发送心跳
                    event_data = await asyncio.wait_for(event_queue.get(), timeout=15)

                    # 根据状态发送不同类型的事件
                    if event_data['status'] == 'ready':
                        event_type = 'model_complete'
                    elif event_data['status'] == 'error':
                        event_type = 'model_error'
                    elif event_data['status'] == 'incomplete':
                        event_type = 'model_incomplete'
                    else:
                        event_type = 'model_progress'

                    yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"

                except asyncio.TimeoutError:
                    # 超时，发送心跳
                    heartbeat_counter += 1
                    yield f"event: heartbeat\ndata: {json.dumps({'count': heartbeat_counter})}\n\n"

        except asyncio.CancelledError:
            logger.info("⚠️ SSE连接被客户端取消")
        except Exception as e:
            logger.error(f"❌ SSE错误: {e}")
        finally:
            # 清理：移除此连接的队列
            try:
                sse_queues.remove(event_queue)
                logger.info(f"🔌 SSE连接已断开（剩余连接数: {len(sse_queues)}）")
            except ValueError:
                pass

    return StreamingResponse(
        event_generator(),
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

    Args:
        model_type: 模型类型 (whisper 或 align)
        model_id: 模型ID或语言代码

    Returns:
        StreamingResponse: SSE事件流
    """
    # 验证参数
    if model_type not in ["whisper", "align"]:
        raise HTTPException(status_code=400, detail="model_type 必须是 'whisper' 或 'align'")

    # 验证模型存在
    if model_type == "whisper":
        models = {m.model_id: m for m in model_manager.list_whisper_models()}
    else:
        models = {m.language: m for m in model_manager.list_align_models()}

    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"模型不存在: {model_type}/{model_id}")

    async def event_generator() -> AsyncGenerator[str, None]:
        """单模型SSE事件生成器"""
        last_state = {}
        heartbeat_counter = 0

        try:
            logger.info(f"✅ 单模型SSE连接已建立: {model_type}/{model_id}")

            while True:
                # 获取当前模型状态
                if model_type == "whisper":
                    current_models = {m.model_id: m for m in model_manager.list_whisper_models()}
                else:
                    current_models = {m.language: m for m in model_manager.list_align_models()}

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
            logger.info(f"🔌 单模型SSE连接已关闭: {model_type}/{model_id}")
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
