"""
CheckpointManager - 断点续传管理器

Phase 4 实现 - 2025-12-11

核心职责：
1. 保存和加载检查点（checkpoint.json）
2. 原子性写入（临时文件 + 重命名）
3. 检查点验证和恢复
4. 检查点清理

从 transcription_service.py 拆分出来，专注于断点续传功能。
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CheckpointData:
    """检查点数据结构"""
    job_id: str
    phase: str
    total_chunks: int
    processed_chunks: int
    processed_indices: list
    language: Optional[str] = None
    original_settings: Optional[Dict[str, Any]] = None
    # 可扩展字段
    extra_data: Optional[Dict[str, Any]] = None


class CheckpointManager:
    """
    断点续传管理器

    负责检查点的保存、加载、验证和清理。
    使用原子性写入策略确保数据完整性。
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化断点续传管理器

        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)

    def save_checkpoint(
        self,
        job_dir: Path,
        data: Dict[str, Any],
        original_settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        原子性保存检查点

        使用"写临时文件 -> 重命名"策略，确保文件要么完整写入，要么保持原样。

        Args:
            job_dir: 任务目录
            data: 检查点数据
            original_settings: 原始设置（用于校验参数兼容性）

        Returns:
            bool: 保存是否成功
        """
        # 添加原始设置到 checkpoint（用于校验参数兼容性）
        if original_settings:
            data["original_settings"] = original_settings

        checkpoint_path = job_dir / "checkpoint.json"
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            # 1. 写入临时文件
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 2. 原子替换（Windows/Linux/macOS 均支持）
            # 如果程序在这里崩溃，checkpoint.json 依然是旧版本，不会损坏
            os.replace(temp_path, checkpoint_path)

            self.logger.debug(f"检查点已保存: {checkpoint_path}")
            return True

        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}", exc_info=True)
            # 清理临时文件
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            return False

    def load_checkpoint(self, job_dir: Path) -> Optional[Dict[str, Any]]:
        """
        加载检查点

        Args:
            job_dir: 任务目录

        Returns:
            Optional[Dict[str, Any]]: 检查点数据，不存在或损坏则返回 None
        """
        checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            self.logger.debug(f"检查点不存在: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.debug(f"检查点已加载: {checkpoint_path}")
                return data

        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"检查点文件损坏，将重新开始任务: {checkpoint_path} - {e}")
            return None

    def checkpoint_exists(self, job_dir: Path) -> bool:
        """
        检查检查点是否存在

        Args:
            job_dir: 任务目录

        Returns:
            bool: 检查点是否存在
        """
        checkpoint_path = job_dir / "checkpoint.json"
        return checkpoint_path.exists()

    def delete_checkpoint(self, job_dir: Path) -> bool:
        """
        删除检查点

        Args:
            job_dir: 任务目录

        Returns:
            bool: 删除是否成功
        """
        checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return True

        try:
            checkpoint_path.unlink()
            self.logger.debug(f"检查点已删除: {checkpoint_path}")
            return True

        except Exception as e:
            self.logger.error(f"删除检查点失败: {e}", exc_info=True)
            return False

    def validate_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        current_settings: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        验证检查点数据的完整性和兼容性

        Args:
            checkpoint_data: 检查点数据
            current_settings: 当前设置（用于兼容性检查）

        Returns:
            tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        # 检查必需字段
        required_fields = ['job_id', 'phase', 'total_chunks', 'processed_chunks']
        for field in required_fields:
            if field not in checkpoint_data:
                return False, f"缺少必需字段: {field}"

        # 检查数据类型
        if not isinstance(checkpoint_data['total_chunks'], int):
            return False, "total_chunks 必须是整数"

        if not isinstance(checkpoint_data['processed_chunks'], int):
            return False, "processed_chunks 必须是整数"

        # 检查数据合理性
        if checkpoint_data['processed_chunks'] > checkpoint_data['total_chunks']:
            return False, "processed_chunks 不能大于 total_chunks"

        # 检查设置兼容性（如果提供了当前设置）
        if current_settings and 'original_settings' in checkpoint_data:
            original = checkpoint_data['original_settings']
            # 检查关键设置是否一致
            critical_settings = ['model', 'device', 'word_timestamps']
            for setting in critical_settings:
                if setting in original and setting in current_settings:
                    if original[setting] != current_settings[setting]:
                        return False, f"设置不兼容: {setting} 已更改"

        return True, None

    def get_checkpoint_info(self, job_dir: Path) -> Optional[Dict[str, Any]]:
        """
        获取检查点的摘要信息（不加载完整数据）

        Args:
            job_dir: 任务目录

        Returns:
            Optional[Dict[str, Any]]: 检查点摘要信息
        """
        checkpoint_data = self.load_checkpoint(job_dir)

        if not checkpoint_data:
            return None

        return {
            'job_id': checkpoint_data.get('job_id'),
            'phase': checkpoint_data.get('phase'),
            'total_chunks': checkpoint_data.get('total_chunks', 0),
            'processed_chunks': checkpoint_data.get('processed_chunks', 0),
            'progress': (
                checkpoint_data.get('processed_chunks', 0) /
                checkpoint_data.get('total_chunks', 1) * 100
                if checkpoint_data.get('total_chunks', 0) > 0 else 0
            ),
            'language': checkpoint_data.get('language'),
            'has_original_settings': 'original_settings' in checkpoint_data
        }

    def merge_checkpoint_data(
        self,
        old_data: Dict[str, Any],
        new_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        合并检查点数据（用于增量更新）

        Args:
            old_data: 旧的检查点数据
            new_data: 新的检查点数据

        Returns:
            Dict[str, Any]: 合并后的检查点数据
        """
        merged = old_data.copy()
        merged.update(new_data)

        # 合并 processed_indices（去重并排序）
        if 'processed_indices' in old_data and 'processed_indices' in new_data:
            old_indices = set(old_data['processed_indices'])
            new_indices = set(new_data['processed_indices'])
            merged['processed_indices'] = sorted(old_indices | new_indices)

        return merged


# 便捷函数
def get_checkpoint_manager(logger: Optional[logging.Logger] = None) -> CheckpointManager:
    """
    获取 CheckpointManager 实例

    Args:
        logger: 日志记录器

    Returns:
        CheckpointManager 实例
    """
    return CheckpointManager(logger=logger)
