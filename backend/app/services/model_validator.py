"""
模型完整性验证工具
负责验证模型文件的完整性
"""

from pathlib import Path
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ModelValidator:
    """模型完整性验证器"""
    
    # Whisper 模型必需的文件
    WHISPER_REQUIRED_FILES = [
        "model.bin",
        "config.json",
        "vocabulary.txt",
        "tokenizer.json"
    ]
    
    # 对齐模型必需的文件
    ALIGN_REQUIRED_FILES = [
        "pytorch_model.bin",
        "config.json",
        "vocab.json"
    ]
    
    @staticmethod
    def validate_whisper_model(model_path: Path) -> Tuple[bool, List[str], str]:
        """
        验证 Whisper 模型完整性
        
        Args:
            model_path: 模型目录路径
            
        Returns:
            Tuple[bool, List[str], str]: (是否完整, 缺失的文件列表, 详细信息)
        """
        if not model_path.exists():
            return False, [], f"模型目录不存在: {model_path}"
        
        missing_files = []
        file_info = []
        
        for file_name in ModelValidator.WHISPER_REQUIRED_FILES:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
                file_info.append(f"  ✗ {file_name}: 缺失")
            else:
                size = file_path.stat().st_size
                if size == 0:
                    missing_files.append(file_name)
                    file_info.append(f"  ✗ {file_name}: 0 字节（损坏）")
                else:
                    file_info.append(f"  ✓ {file_name}: {size:,} 字节")
        
        is_complete = len(missing_files) == 0
        detail = "\n".join(file_info)
        
        return is_complete, missing_files, detail
    
    @staticmethod
    def validate_align_model(model_path: Path) -> Tuple[bool, List[str], str]:
        """
        验证对齐模型完整性
        
        Args:
            model_path: 模型目录路径
            
        Returns:
            Tuple[bool, List[str], str]: (是否完整, 缺失的文件列表, 详细信息)
        """
        if not model_path.exists():
            return False, [], f"模型目录不存在: {model_path}"
        
        missing_files = []
        file_info = []
        
        for file_name in ModelValidator.ALIGN_REQUIRED_FILES:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
                file_info.append(f"  ✗ {file_name}: 缺失")
            else:
                size = file_path.stat().st_size
                if size == 0:
                    missing_files.append(file_name)
                    file_info.append(f"  ✗ {file_name}: 0 字节（损坏）")
                else:
                    file_info.append(f"  ✓ {file_name}: {size:,} 字节")
        
        is_complete = len(missing_files) == 0
        detail = "\n".join(file_info)
        
        return is_complete, missing_files, detail
    
    @staticmethod
    def find_model_snapshots(hub_dir: Path, model_name: str) -> List[Path]:
        """
        查找模型的所有快照目录
        
        Args:
            hub_dir: HuggingFace hub 缓存目录
            model_name: 模型名称（如 "models--Systran--faster-whisper-medium"）
            
        Returns:
            List[Path]: 快照目录列表
        """
        model_dir = hub_dir / model_name
        if not model_dir.exists():
            return []
        
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return []
        
        # 返回所有快照目录
        return [d for d in snapshots_dir.iterdir() if d.is_dir()]
