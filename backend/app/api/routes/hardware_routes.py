"""
硬件检测相关API路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from models.hardware_models import HardwareInfo, OptimizationConfig
from services.transcription_service import TranscriptionService

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


class HardwareResponse(BaseModel):
    """硬件信息响应模型"""
    success: bool
    hardware: Optional[Dict[str, Any]] = None
    optimization: Optional[Dict[str, Any]] = None
    message: str = ""


class OptimizationResponse(BaseModel):
    """优化配置响应模型"""
    success: bool
    optimization: Optional[Dict[str, Any]] = None
    message: str = ""


def create_hardware_router(transcription_service: TranscriptionService):
    """创建硬件检测路由"""
    
    @router.get("/basic", response_model=HardwareResponse)
    async def get_hardware_info():
        """获取核心硬件信息"""
        try:
            hardware_info = transcription_service.get_hardware_info()
            
            if not hardware_info:
                return HardwareResponse(
                    success=False,
                    message="硬件信息不可用，可能检测失败"
                )
            
            return HardwareResponse(
                success=True,
                hardware=hardware_info.to_dict(),
                message="硬件信息获取成功"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取硬件信息失败: {str(e)}")
    
    @router.get("/optimize", response_model=OptimizationResponse)
    async def get_optimization_config():
        """获取基于硬件的优化配置"""
        try:
            optimization_config = transcription_service.get_optimization_config()
            
            if not optimization_config:
                return OptimizationResponse(
                    success=False,
                    message="优化配置不可用，可能硬件检测失败"
                )
            
            return OptimizationResponse(
                success=True,
                optimization=optimization_config.to_dict(),
                message="优化配置获取成功"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取优化配置失败: {str(e)}")
    
    @router.get("/status", response_model=HardwareResponse)
    async def get_hardware_status():
        """获取完整的硬件状态和优化信息"""
        try:
            hardware_info = transcription_service.get_hardware_info()
            optimization_config = transcription_service.get_optimization_config()
            
            if not hardware_info or not optimization_config:
                return HardwareResponse(
                    success=False,
                    message="硬件检测不完整，部分信息不可用"
                )
            
            return HardwareResponse(
                success=True,
                hardware=hardware_info.to_dict(),
                optimization=optimization_config.to_dict(),
                message="硬件状态获取成功"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"获取硬件状态失败: {str(e)}")
    
    @router.post("/redetect")
    async def redetect_hardware():
        """重新检测硬件"""
        try:
            # 重新执行硬件检测
            transcription_service._detect_hardware()
            
            hardware_info = transcription_service.get_hardware_info()
            optimization_config = transcription_service.get_optimization_config()
            
            if not hardware_info or not optimization_config:
                return HardwareResponse(
                    success=False,
                    message="重新检测失败，无法获取硬件信息"
                )
            
            return HardwareResponse(
                success=True,
                hardware=hardware_info.to_dict(),
                optimization=optimization_config.to_dict(),
                message="硬件重新检测完成"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"重新检测硬件失败: {str(e)}")
    
    return router