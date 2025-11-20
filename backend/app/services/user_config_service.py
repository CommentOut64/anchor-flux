"""
用户配置管理服务
管理用户的个性化配置（默认预加载模型等）
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import threading

from core.config import config

logger = logging.getLogger(__name__)


class UserConfigService:
    """用户配置管理器"""

    def __init__(self):
        # 配置文件路径（保存在项目根目录）
        self.config_file = config.BASE_DIR / "user_config.json"
        self._lock = threading.RLock()
        self._config_cache: Optional[Dict[str, Any]] = None

        # 确保配置文件存在
        self._ensure_config_file()

    def _ensure_config_file(self):
        """确保配置文件存在"""
        if not self.config_file.exists():
            default_config = {
                "default_preload_model": None,  # 用户选择的默认预加载模型
                "version": "1.0"
            }
            self._save_config(default_config)
            logger.info(f"✅ 创建默认用户配置文件: {self.config_file}")

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with self._lock:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ 加载用户配置失败: {e}")
                return {
                    "default_preload_model": None,
                    "version": "1.0"
                }

    def _save_config(self, config_data: Dict[str, Any]):
        """保存配置文件"""
        with self._lock:
            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                # 清除缓存
                self._config_cache = None
                logger.debug(f"✅ 用户配置已保存: {self.config_file}")
            except Exception as e:
                logger.error(f"❌ 保存用户配置失败: {e}")

    def get_default_preload_model(self) -> Optional[str]:
        """
        获取用户选择的默认预加载模型

        Returns:
            Optional[str]: 模型ID，如果用户未选择则返回None
        """
        config_data = self._load_config()
        model_id = config_data.get("default_preload_model")
        logger.debug(f"📖 读取默认预加载模型: {model_id}")
        return model_id

    def set_default_preload_model(self, model_id: Optional[str]) -> bool:
        """
        设置默认预加载模型

        Args:
            model_id: 模型ID，None表示清除用户选择

        Returns:
            bool: 是否设置成功
        """
        try:
            config_data = self._load_config()
            config_data["default_preload_model"] = model_id
            self._save_config(config_data)
            logger.info(f"✅ 设置默认预加载模型: {model_id}")
            return True
        except Exception as e:
            logger.error(f"❌ 设置默认预加载模型失败: {e}")
            return False

    def get_all_config(self) -> Dict[str, Any]:
        """获取所有用户配置"""
        return self._load_config()

    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        更新用户配置

        Args:
            updates: 要更新的配置项

        Returns:
            bool: 是否更新成功
        """
        try:
            config_data = self._load_config()
            config_data.update(updates)
            self._save_config(config_data)
            logger.info(f"✅ 更新用户配置: {list(updates.keys())}")
            return True
        except Exception as e:
            logger.error(f"❌ 更新用户配置失败: {e}")
            return False


# 全局单例
_user_config_service: Optional[UserConfigService] = None


def get_user_config_service() -> UserConfigService:
    """获取用户配置服务实例"""
    global _user_config_service
    if _user_config_service is None:
        _user_config_service = UserConfigService()
        logger.info("🏗️ 用户配置服务已初始化")
    return _user_config_service
