"""
Demucs分级人声分离配置API路由
"""
import json
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any


class DemucsUserSettings(BaseModel):
    """Demucs用户配置模型"""
    # 基础配置
    enabled: bool = True
    mode: str = "auto"  # auto/always/never
    quality_preset: str = "balanced"  # fast/balanced/quality

    # 高级配置（可选）
    weak_model: Optional[str] = None
    strong_model: Optional[str] = None
    fallback_model: Optional[str] = None
    auto_escalation: Optional[bool] = None
    max_escalations: Optional[int] = None
    bgm_light_threshold: Optional[float] = None
    bgm_heavy_threshold: Optional[float] = None
    on_break: Optional[str] = None  # continue/fallback/fail/pause


def create_demucs_router():
    """创建Demucs配置路由"""

    router = APIRouter(prefix="/api/demucs", tags=["demucs"])

    # 获取配置文件路径
    config_path = Path(__file__).parent.parent.parent / "config" / "demucs_tiers.json"

    @router.get("/config")
    async def get_demucs_config():
        """
        获取Demucs配置

        返回:
            - presets: 质量预设配置
            - models: 可用模型信息
            - defaults: 默认参数值
        """
        try:
            if not config_path.exists():
                raise HTTPException(status_code=500, detail="配置文件不存在")

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            return {
                "version": config_data.get("version", "1.0"),
                "description": config_data.get("description", ""),
                "presets": config_data.get("presets", {}),
                "models": config_data.get("models", {}),
                "defaults": config_data.get("defaults", {})
            }
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="配置文件格式错误")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取配置失败: {str(e)}")

    @router.get("/presets")
    async def get_presets():
        """
        获取质量预设列表

        返回:
            {
                "fast": {...},
                "balanced": {...},
                "quality": {...}
            }
        """
        try:
            if not config_path.exists():
                raise HTTPException(status_code=500, detail="配置文件不存在")

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            return config_data.get("presets", {})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取预设失败: {str(e)}")

    @router.get("/models")
    async def get_models():
        """
        获取可用模型信息

        返回:
            {
                "htdemucs": {
                    "description": "...",
                    "size_mb": 80,
                    "quality_score": 3,
                    "speed_score": 5
                },
                ...
            }
        """
        try:
            if not config_path.exists():
                raise HTTPException(status_code=500, detail="配置文件不存在")

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            return config_data.get("models", {})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取模型信息失败: {str(e)}")

    @router.get("/defaults")
    async def get_defaults():
        """
        获取默认配置参数

        返回:
            {
                "bgm_light_threshold": 0.02,
                "bgm_heavy_threshold": 0.15,
                "consecutive_threshold": 3,
                "ratio_threshold": 0.2,
                "max_escalations": 1
            }
        """
        try:
            if not config_path.exists():
                raise HTTPException(status_code=500, detail="配置文件不存在")

            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            return config_data.get("defaults", {})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取默认配置失败: {str(e)}")

    return router
