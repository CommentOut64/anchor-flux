"""
初始化服务包

注意：为了支持独立模块导入，这里不自动导入所有服务。
需要使用时请显式导入，例如：
    from app.services.transcription_service import TranscriptionService
    from app.services.monitoring.hardware_monitor import HardwareMonitor
"""

# 延迟导入，避免在包初始化时触发依赖检查
def __getattr__(name):
    """延迟导入支持"""
    if name == "TranscriptionService":
        from .transcription_service import TranscriptionService
        return TranscriptionService
    elif name == "get_transcription_service":
        from .transcription_service import get_transcription_service
        return get_transcription_service
    elif name == "FileManagementService":
        from .file_service import FileManagementService
        return FileManagementService
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'TranscriptionService',
    'get_transcription_service',
    'FileManagementService',
    # Phase 1 新增服务
    'monitoring',
    'inference',
    'alignment',
    'audio',
    'streaming',
    'job',
    'llm'
]