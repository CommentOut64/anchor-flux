"""
任务相关的数据模型定义
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, TYPE_CHECKING
import torch

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from services.cpu_affinity_service import CPUAffinityConfig


@dataclass
class JobSettings:
    """转录任务设置"""
    model: str = "medium"
    compute_type: str = "float16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    word_timestamps: bool = False
    cpu_affinity: Optional["CPUAffinityConfig"] = None  # 使用字符串形式的类型注解


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
                "model": self.settings.model,
                "compute_type": self.settings.compute_type,
                "device": self.settings.device,
                "batch_size": self.settings.batch_size,
                "word_timestamps": self.settings.word_timestamps,
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
        settings = JobSettings(
            model=settings_data.get("model", "medium"),
            compute_type=settings_data.get("compute_type", "float16"),
            device=settings_data.get("device", "cuda"),
            batch_size=settings_data.get("batch_size", 16),
            word_timestamps=settings_data.get("word_timestamps", False),
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