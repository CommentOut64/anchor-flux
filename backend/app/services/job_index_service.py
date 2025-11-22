"""
任务索引服务 - 维护文件路径到任务ID的映射
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict
import logging


class JobIndexService:
    """任务索引服务，维护文件路径到任务ID的映射"""

    def __init__(self, jobs_root: str):
        """
        初始化任务索引服务

        Args:
            jobs_root: 任务根目录
        """
        self.jobs_root = Path(jobs_root)
        self.index_file = self.jobs_root / "job_index.json"
        self.logger = logging.getLogger(__name__)
        self._ensure_index_file()

    def _ensure_index_file(self):
        """确保索引文件存在"""
        if not self.index_file.exists():
            self._save_index({})

    def _load_index(self) -> Dict[str, str]:
        """加载索引文件"""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"加载索引文件失败: {e}")
            return {}

    def _save_index(self, index: Dict[str, str]):
        """保存索引文件"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存索引文件失败: {e}")

    def add_mapping(self, file_path: str, job_id: str):
        """
        添加文件路径到任务ID的映射

        Args:
            file_path: 文件路径
            job_id: 任务ID
        """
        # 规范化路径
        normalized_path = os.path.normpath(file_path)

        index = self._load_index()
        index[normalized_path] = job_id
        self._save_index(index)
        self.logger.debug(f"添加映射: {normalized_path} -> {job_id}")

    def remove_mapping(self, file_path: str):
        """
        移除文件路径的映射

        Args:
            file_path: 文件路径
        """
        normalized_path = os.path.normpath(file_path)

        index = self._load_index()
        if normalized_path in index:
            del index[normalized_path]
            self._save_index(index)
            self.logger.debug(f"移除映射: {normalized_path}")

    def get_job_id(self, file_path: str) -> Optional[str]:
        """
        获取文件路径对应的任务ID

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 任务ID，不存在则返回None
        """
        normalized_path = os.path.normpath(file_path)
        index = self._load_index()
        return index.get(normalized_path)

    def get_file_path(self, job_id: str) -> Optional[str]:
        """
        获取任务ID对应的文件路径

        Args:
            job_id: 任务ID

        Returns:
            Optional[str]: 文件路径，不存在则返回None
        """
        index = self._load_index()
        for file_path, jid in index.items():
            if jid == job_id:
                return file_path
        return None

    def cleanup_invalid_mappings(self):
        """
        清理无效的映射（文件不存在或任务不存在）
        """
        index = self._load_index()
        valid_mappings = {}

        for file_path, job_id in index.items():
            # 检查文件是否存在
            if os.path.exists(file_path):
                # 检查任务目录是否存在
                job_dir = self.jobs_root / job_id
                if job_dir.exists():
                    valid_mappings[file_path] = job_id
                else:
                    self.logger.debug(f"清理映射（任务不存在）: {file_path} -> {job_id}")
            else:
                self.logger.debug(f"清理映射（文件不存在）: {file_path} -> {job_id}")

        self._save_index(valid_mappings)
        removed_count = len(index) - len(valid_mappings)
        if removed_count > 0:
            self.logger.info(f"清理了 {removed_count} 个无效映射")

    def get_all_mappings(self) -> Dict[str, str]:
        """获取所有映射"""
        return self._load_index()


# 单例实例
_job_index_service: Optional[JobIndexService] = None


def get_job_index_service(jobs_root: str) -> JobIndexService:
    """获取任务索引服务实例（单例模式）"""
    global _job_index_service
    if _job_index_service is None:
        _job_index_service = JobIndexService(jobs_root)
    return _job_index_service
