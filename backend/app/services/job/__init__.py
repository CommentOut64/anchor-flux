"""
任务管理模块

Phase 4 实现 - 2025-12-11

从 transcription_service.py 拆分出来的任务管理和断点续传功能。

包含:
- JobManager: 任务管理器（创建、查询、更新、删除）
- CheckpointManager: 断点续传管理器（保存、加载、验证）
- job_state: 任务状态定义
"""

from .job_manager import JobManager, get_job_manager
from .checkpoint_manager import CheckpointManager, get_checkpoint_manager, CheckpointData

__all__ = [
    'JobManager',
    'get_job_manager',
    'CheckpointManager',
    'get_checkpoint_manager',
    'CheckpointData',
]
