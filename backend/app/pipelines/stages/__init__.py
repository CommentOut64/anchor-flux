"""
Stages 包 - 流水线阶段

包含:
- SpectralTriageStage: 频谱分诊阶段
- SeparationStage: 人声分离阶段
"""

from app.pipelines.stages.spectral_triage_stage import SpectralTriageStage
from app.pipelines.stages.separation_stage import SeparationStage

__all__ = [
    "SpectralTriageStage",
    "SeparationStage",
]
