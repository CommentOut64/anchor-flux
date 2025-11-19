"""
统一配置管理
严格遵守独立打包原则：
1. 杜绝硬编码绝对路径
2. 杜绝依赖系统环境变量
3. 强制接管模型下载路径
"""

import os
from pathlib import Path
from typing import Optional


class ProjectConfig:
    """项目配置类"""

    def __init__(self):
        # ========== 路径配置（基于项目根目录） ==========
        # 获取项目根目录（从当前文件位置向上三级）
        # backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> project_root
        self.BASE_DIR = Path(__file__).parent.parent.parent.parent.resolve()

        # 输入输出目录
        self.INPUT_DIR = self.BASE_DIR / "input"
        self.OUTPUT_DIR = self.BASE_DIR / "output"
        self.JOBS_DIR = self.BASE_DIR / "jobs"
        self.TEMP_DIR = self.BASE_DIR / "temp"

        # FFmpeg路径（优先使用项目内的，支持独立打包）
        self.FFMPEG_DIR = self.BASE_DIR / "ffmpeg" / "bin"
        self.FFMPEG_EXE = self.FFMPEG_DIR / "ffmpeg.exe"

        # 模型缓存目录（强制接管，不使用默认的用户目录）
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.HF_CACHE_DIR = self.MODELS_DIR / "huggingface"
        self.TORCH_CACHE_DIR = self.MODELS_DIR / "torch"

        # 设置环境变量，强制模型下载到项目目录
        os.environ['HF_HOME'] = str(self.HF_CACHE_DIR)
        os.environ['TORCH_HOME'] = str(self.TORCH_CACHE_DIR)
        os.environ['TRANSFORMERS_CACHE'] = str(self.HF_CACHE_DIR / "transformers")
        os.environ['HF_HUB_CACHE'] = str(self.HF_CACHE_DIR / "hub")
        
        # HuggingFace 镜像源配置（解决国内访问问题）
        # 默认启用镜像源，可通过环境变量 USE_HF_MIRROR=false 禁用
        use_mirror = os.getenv('USE_HF_MIRROR', 'true').lower() == 'true'

        if use_mirror:
            # 使用国内镜像源（HF-Mirror 公益镜像）
            self.HF_ENDPOINT = 'https://hf-mirror.com'
            os.environ['HF_ENDPOINT'] = self.HF_ENDPOINT
            print(f"🌐 HuggingFace 镜像源: {self.HF_ENDPOINT}")
            print("💡 提示：如需使用官方源，请设置环境变量 USE_HF_MIRROR=false")
        else:
            # 使用官方源
            self.HF_ENDPOINT = 'https://huggingface.co'
            if 'HF_ENDPOINT' in os.environ:
                del os.environ['HF_ENDPOINT']
            print(f"🌐 使用 HuggingFace 官方源: {self.HF_ENDPOINT}")
            print("💡 提示：如遇访问问题，可设置环境变量 USE_HF_MIRROR=true 使用镜像源")

        # 确保目录存在
        for dir_path in [
            self.INPUT_DIR,
            self.OUTPUT_DIR,
            self.JOBS_DIR,
            self.TEMP_DIR,
            self.MODELS_DIR,
            self.HF_CACHE_DIR,
            self.TORCH_CACHE_DIR,
            self.FFMPEG_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # ========== 音频处理配置 ==========
        self.SEGMENT_LENGTH_MS = 60_000      # 60秒
        self.SILENCE_SEARCH_MS = 2_000       # 2秒
        self.MIN_SILENCE_LEN_MS = 300        # 300毫秒
        self.SILENCE_THRESHOLD_DBFS = -40    # -40dB

        # ========== 进度权重配置 ==========
        self.PHASE_WEIGHTS = {
            "extract": 5,      # 音频提取占5%
            "split": 10,       # 音频分段占10%
            "transcribe": 80,  # 转录处理占80%（主要耗时）
            "srt": 5           # SRT生成占5%
        }
        self.TOTAL_WEIGHT = sum(self.PHASE_WEIGHTS.values())

        # ========== 模型配置 ==========
        self.DEFAULT_MODEL = "medium"
        self.DEFAULT_DEVICE = "cuda"  # 自动检测会覆盖
        self.DEFAULT_COMPUTE_TYPE = "float16"
        self.DEFAULT_BATCH_SIZE = 16
        self.MAX_CACHE_SIZE = 3              # 最多缓存3个模型
        self.MEMORY_THRESHOLD = 0.8          # 内存使用阈值

        # ========== 服务器配置 ==========
        self.API_HOST = "127.0.0.1"
        self.API_PORT = 8000
        self.API_RELOAD = False

        # ========== CPU亲和性配置 ==========
        self.CPU_AFFINITY_ENABLED = True
        self.CPU_AFFINITY_STRATEGY = "auto"  # auto/half/custom

        # ========== 日志配置 ==========
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_DIR = self.BASE_DIR / "logs"
        self.LOG_FILE = self.LOG_DIR / "app.log"
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def get_ffmpeg_command(self) -> str:
        """
        获取FFmpeg命令
        优先使用项目内的FFmpeg，支持独立打包

        Returns:
            str: FFmpeg可执行文件路径
        """
        if self.FFMPEG_EXE.exists():
            # 使用项目内的FFmpeg
            return str(self.FFMPEG_EXE)
        else:
            # 回退到系统命令
            return "ffmpeg"

    def get_audio_config(self) -> dict:
        """获取音频处理配置"""
        return {
            "segment_length_ms": self.SEGMENT_LENGTH_MS,
            "silence_search_ms": self.SILENCE_SEARCH_MS,
            "min_silence_len_ms": self.MIN_SILENCE_LEN_MS,
            "silence_threshold_dbfs": self.SILENCE_THRESHOLD_DBFS
        }

    def get_phase_weights(self) -> dict:
        """获取进度权重配置"""
        return {
            "weights": self.PHASE_WEIGHTS,
            "total": self.TOTAL_WEIGHT
        }

    def get_model_config(self) -> dict:
        """获取模型配置"""
        return {
            "default_model": self.DEFAULT_MODEL,
            "default_device": self.DEFAULT_DEVICE,
            "default_compute_type": self.DEFAULT_COMPUTE_TYPE,
            "default_batch_size": self.DEFAULT_BATCH_SIZE,
            "max_cache_size": self.MAX_CACHE_SIZE,
            "memory_threshold": self.MEMORY_THRESHOLD
        }


# 全局配置实例
config = ProjectConfig()

# 打印配置信息（启动时显示）
# 注意：避免在模块导入时使用emoji，以防编码问题
try:
    import sys
    # 在Windows上设置UTF-8输出
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    print(f"""
项目配置已加载
项目根目录: {config.BASE_DIR}
输入目录: {config.INPUT_DIR}
输出目录: {config.OUTPUT_DIR}
FFmpeg: {config.get_ffmpeg_command()}
模型缓存: {config.MODELS_DIR}
日志目录: {config.LOG_DIR}
""")
except Exception:
    # 如果打印失败，静默忽略
    pass
