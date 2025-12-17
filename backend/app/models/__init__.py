"""
初始化模型包

注意：为了支持独立模块导入，这里不自动导入所有模块。
需要使用时请显式导入，例如：
    from app.models.job_models import JobSettings, JobState
    from app.models.confidence_models import AlignedWord, AlignedSubtitle
    from app.models.preset_models import PresetConfig
"""

# 延迟导入，避免在包初始化时触发依赖检查
def __getattr__(name):
    """延迟导入支持"""
    if name == "JobSettings" or name == "JobState":
        from .job_models import JobSettings, JobState
        return JobSettings if name == "JobSettings" else JobState
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'JobSettings',
    'JobState',
    # Phase 1 新增模型
    'confidence_models',
    'preset_models'
]