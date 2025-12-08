"""
任务相关的数据模型定义
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, TYPE_CHECKING
import torch

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from app.services.cpu_affinity_service import CPUAffinityConfig


@dataclass
class DemucsSettings:
    """Demucs人声分离配置（用户可配置）"""
    # === 基础开关 ===
    enabled: bool = True                        # 是否启用Demucs
    mode: str = "auto"                          # 模式: auto/always/never/on_demand

    # === 分级模型配置 ===
    weak_model: str = "htdemucs"                # 弱BGM使用的模型（速度优先）
    strong_model: str = "htdemucs"              # 强BGM使用的模型（质量优先）
    fallback_model: str = "htdemucs"            # 兜底模型（熔断升级后使用）
    auto_escalation: bool = True                # 是否允许自动升级模型
    max_escalations: int = 1                    # 最大升级次数

    # === BGM检测阈值 ===
    bgm_light_threshold: float = 0.02           # 轻微BGM阈值
    bgm_heavy_threshold: float = 0.15           # 强BGM阈值

    # === 质量评估阈值 ===
    retry_threshold_logprob: float = -0.8       # 重试阈值（avg_logprob）
    retry_threshold_no_speech: float = 0.6      # 重试阈值（no_speech_prob）

    # === 熔断配置 ===
    circuit_breaker_enabled: bool = True        # 是否启用熔断机制
    consecutive_threshold: int = 3              # 连续重试触发熔断的阈值
    ratio_threshold: float = 0.2                # 总重试比例触发熔断的阈值（20%）

    # === 熔断处理策略 ===
    on_break: str = "continue"                  # 熔断后处理策略: continue/fallback/fail/pause
    mark_problem_segments: bool = True          # 是否在结果中标记问题段落
    problem_segment_suffix: str = "[?]"         # 问题段落的标记后缀

    # === 质量预设（简化配置入口） ===
    quality_preset: str = "balanced"            # 质量预设: fast/balanced/quality


@dataclass
class SenseVoiceSettings:
    """SenseVoice 转录引擎配置"""
    # === 预设方案 ===
    preset_id: str = "default"                  # 预设ID: default/preset1-5/custom

    # === 增强模式 ===
    enhancement: str = "off"                    # off/smart_patch/deep_listen
    # - off: 仅 SenseVoice
    # - smart_patch: 低置信度自动 Whisper 补刀
    # - deep_listen: Whisper 全文转录

    # === 校对模式 ===
    proofread: str = "off"                      # off/sparse/full
    # - off: 不校对
    # - sparse: 仅校对低置信度和疑问片段
    # - full: 滑动窗口全量校对润色

    # === 翻译模式 ===
    translate: str = "off"                      # off/full/partial
    target_language: str = "en"                 # 翻译目标语言

    # === 阈值配置 ===
    confidence_threshold: float = 0.6           # 低置信度阈值
    whisper_patch_threshold: float = 0.5        # 触发 Whisper 补刀的阈值


@dataclass
class JobSettings:
    """转录任务设置"""
    # === 转录引擎选择 ===
    engine: str = "sensevoice"                  # 转录引擎: whisper/sensevoice（默认 SenseVoice）

    model: str = "medium"
    compute_type: str = "float16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    word_timestamps: bool = False
    cpu_affinity: Optional["CPUAffinityConfig"] = None  # 使用字符串形式的类型注解
    demucs: DemucsSettings = field(default_factory=DemucsSettings)  # Demucs配置

    # === SenseVoice 配置 ===
    sensevoice: SenseVoiceSettings = field(default_factory=SenseVoiceSettings)


@dataclass
class MediaStatus:
    """媒体资源状态（用于编辑器）"""
    video_exists: bool = False          # 视频文件是否存在
    video_format: Optional[str] = None  # 视频格式（.mp4, .mkv等）
    needs_proxy: bool = False           # 是否需要Proxy转码
    proxy_exists: bool = False          # Proxy视频是否已生成
    audio_exists: bool = False          # 音频文件是否存在
    peaks_ready: bool = False           # 波形峰值数据是否就绪
    thumbnails_ready: bool = False      # 缩略图是否就绪
    srt_exists: bool = False            # SRT文件是否存在


@dataclass
class JobState:
    """转录任务状态"""
    job_id: str
    filename: str
    dir: str
    input_path: str = ""  # 添加原始输入路径记录
    settings: JobSettings = field(default_factory=JobSettings)
    status: str = "queued"  # queued, processing, finished, failed, canceled, paused
    phase: str = "pending"  # extract, split, transcribe, srt
    progress: float = 0.0
    phase_percent: float = 0.0  # 当前阶段内进度 (0-100)
    message: str = "等待开始"
    error: Optional[str] = None
    segments: List[Dict] = field(default_factory=list)
    processed: int = 0
    total: int = 0
    language: Optional[str] = None
    srt_path: Optional[str] = None
    canceled: bool = False
    paused: bool = False  # 暂停标志
    title: str = ""  # 用户自定义的任务名称，为空时使用 filename

    # 媒体状态（用于编辑器，转录完成后更新）
    media_status: Optional[MediaStatus] = None

    def to_dict(self):
        """转换为字典格式，用于API响应"""
        d = asdict(self)
        d.pop('segments', None)  # 不透出内部详情
        return d

    def to_meta_dict(self) -> dict:
        """
        转换为元信息字典格式，用于持久化到 job_meta.json
        只保存恢复任务所需的核心信息，不包含 segments 等大数据
        """
        import time
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "title": self.title,
            "dir": self.dir,
            "input_path": self.input_path,
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "phase_percent": self.phase_percent,
            "message": self.message,
            "error": self.error,
            "processed": self.processed,
            "total": self.total,
            "language": self.language,
            "srt_path": self.srt_path,
            "canceled": self.canceled,
            "paused": self.paused,
            "settings": {
                "engine": self.settings.engine,
                "model": self.settings.model,
                "compute_type": self.settings.compute_type,
                "device": self.settings.device,
                "batch_size": self.settings.batch_size,
                "word_timestamps": self.settings.word_timestamps,
                "demucs": {
                    "enabled": self.settings.demucs.enabled,
                    "mode": self.settings.demucs.mode,
                    # 分级模型配置
                    "weak_model": self.settings.demucs.weak_model,
                    "strong_model": self.settings.demucs.strong_model,
                    "fallback_model": self.settings.demucs.fallback_model,
                    "auto_escalation": self.settings.demucs.auto_escalation,
                    "max_escalations": self.settings.demucs.max_escalations,
                    # BGM检测阈值
                    "bgm_light_threshold": self.settings.demucs.bgm_light_threshold,
                    "bgm_heavy_threshold": self.settings.demucs.bgm_heavy_threshold,
                    # 质量评估阈值
                    "retry_threshold_logprob": self.settings.demucs.retry_threshold_logprob,
                    "retry_threshold_no_speech": self.settings.demucs.retry_threshold_no_speech,
                    # 熔断配置
                    "circuit_breaker_enabled": self.settings.demucs.circuit_breaker_enabled,
                    "consecutive_threshold": self.settings.demucs.consecutive_threshold,
                    "ratio_threshold": self.settings.demucs.ratio_threshold,
                    # 熔断处理策略
                    "on_break": self.settings.demucs.on_break,
                    "mark_problem_segments": self.settings.demucs.mark_problem_segments,
                    "problem_segment_suffix": self.settings.demucs.problem_segment_suffix,
                    # 质量预设
                    "quality_preset": self.settings.demucs.quality_preset,
                },
                "sensevoice": {
                    "preset_id": self.settings.sensevoice.preset_id,
                    "enhancement": self.settings.sensevoice.enhancement,
                    "proofread": self.settings.sensevoice.proofread,
                    "translate": self.settings.sensevoice.translate,
                    "target_language": self.settings.sensevoice.target_language,
                    "confidence_threshold": self.settings.sensevoice.confidence_threshold,
                    "whisper_patch_threshold": self.settings.sensevoice.whisper_patch_threshold,
                }
            },
            "updated_at": time.time()
        }

    @classmethod
    def from_meta_dict(cls, data: dict) -> "JobState":
        """
        从元信息字典恢复 JobState 对象

        Args:
            data: job_meta.json 中的数据

        Returns:
            JobState: 恢复的任务状态对象
        """
        settings_data = data.get("settings", {})

        # 处理 Demucs 配置（向后兼容：如果没有则使用默认值）
        demucs_data = settings_data.get("demucs", {})
        demucs_settings = DemucsSettings(
            # 基础配置
            enabled=demucs_data.get("enabled", True),
            mode=demucs_data.get("mode", "auto"),
            # 分级模型配置
            weak_model=demucs_data.get("weak_model", "htdemucs"),
            strong_model=demucs_data.get("strong_model", "htdemucs"),
            fallback_model=demucs_data.get("fallback_model", "htdemucs"),
            auto_escalation=demucs_data.get("auto_escalation", True),
            max_escalations=demucs_data.get("max_escalations", 1),
            # BGM检测阈值
            bgm_light_threshold=demucs_data.get("bgm_light_threshold", 0.02),
            bgm_heavy_threshold=demucs_data.get("bgm_heavy_threshold", 0.15),
            # 质量评估阈值
            retry_threshold_logprob=demucs_data.get("retry_threshold_logprob", -0.8),
            retry_threshold_no_speech=demucs_data.get("retry_threshold_no_speech", 0.6),
            # 熔断配置
            circuit_breaker_enabled=demucs_data.get("circuit_breaker_enabled", True),
            consecutive_threshold=demucs_data.get("consecutive_threshold", 3),
            ratio_threshold=demucs_data.get("ratio_threshold", 0.2),
            # 熔断处理策略
            on_break=demucs_data.get("on_break", "continue"),
            mark_problem_segments=demucs_data.get("mark_problem_segments", True),
            problem_segment_suffix=demucs_data.get("problem_segment_suffix", "[?]"),
            # 质量预设
            quality_preset=demucs_data.get("quality_preset", "balanced"),
        )

        # 处理 SenseVoice 配置（向后兼容）
        sensevoice_data = settings_data.get("sensevoice", {})
        sensevoice_settings = SenseVoiceSettings(
            preset_id=sensevoice_data.get("preset_id", "default"),
            enhancement=sensevoice_data.get("enhancement", "off"),
            proofread=sensevoice_data.get("proofread", "off"),
            translate=sensevoice_data.get("translate", "off"),
            target_language=sensevoice_data.get("target_language", "en"),
            confidence_threshold=sensevoice_data.get("confidence_threshold", 0.6),
            whisper_patch_threshold=sensevoice_data.get("whisper_patch_threshold", 0.5),
        )

        settings = JobSettings(
            engine=settings_data.get("engine", "whisper"),
            model=settings_data.get("model", "medium"),
            compute_type=settings_data.get("compute_type", "float16"),
            device=settings_data.get("device", "cuda"),
            batch_size=settings_data.get("batch_size", 16),
            word_timestamps=settings_data.get("word_timestamps", False),
            demucs=demucs_settings,
            sensevoice=sensevoice_settings,
        )

        return cls(
            job_id=data["job_id"],
            filename=data.get("filename", "unknown"),
            title=data.get("title", ""),
            dir=data.get("dir", ""),
            input_path=data.get("input_path", ""),
            settings=settings,
            status=data.get("status", "queued"),
            phase=data.get("phase", "pending"),
            progress=data.get("progress", 0.0),
            phase_percent=data.get("phase_percent", 0.0),
            message=data.get("message", ""),
            error=data.get("error"),
            processed=data.get("processed", 0),
            total=data.get("total", 0),
            language=data.get("language"),
            srt_path=data.get("srt_path"),
            canceled=data.get("canceled", False),
            paused=data.get("paused", False),
        )

    def update_media_status(self, job_dir: str):
        """
        更新媒体状态（检查各类资源文件是否就绪）

        Args:
            job_dir: 任务目录路径
        """
        from pathlib import Path

        job_path = Path(job_dir)
        if not job_path.exists():
            return

        # 需要转码的格式
        need_transcode_formats = {'.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v'}

        # 查找视频文件
        video_file = None
        video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv', '.m4v']
        for file in job_path.iterdir():
            if file.is_file() and file.suffix.lower() in video_exts:
                video_file = file
                break

        # 检查各项资源
        audio_file = job_path / "audio.wav"
        proxy_file = job_path / "proxy.mp4"
        peaks_file = job_path / "peaks_2000.json"
        thumbnails_file = job_path / "thumbnails_10.json"

        # 查找SRT文件
        srt_exists = False
        for file in job_path.iterdir():
            if file.suffix.lower() == '.srt':
                srt_exists = True
                break

        # 更新媒体状态
        self.media_status = MediaStatus(
            video_exists=video_file is not None,
            video_format=video_file.suffix if video_file else None,
            needs_proxy=video_file is not None and video_file.suffix.lower() in need_transcode_formats,
            proxy_exists=proxy_file.exists(),
            audio_exists=audio_file.exists(),
            peaks_ready=peaks_file.exists(),
            thumbnails_ready=thumbnails_file.exists(),
            srt_exists=srt_exists
        )