"""
FFmpeg管理服务
- 检测项目目录中的FFmpeg
- 自动下载FFmpeg（如果缺失）
- 多次下载失败后回退到系统FFmpeg
"""

import os
import sys
import logging
import subprocess
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import platform

from core.config import config


class FFmpegManager:
    """FFmpeg管理器"""

    # FFmpeg下载链接（Windows平台）
    FFMPEG_DOWNLOAD_URLS = [
        # 主要源：GitHub Releases
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        # 备用源：gyan.dev
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    ]

    MAX_DOWNLOAD_ATTEMPTS = 3  # 最大下载尝试次数
    DOWNLOAD_TIMEOUT = 600  # 下载超时时间（秒）

    def __init__(self):
        """初始化FFmpeg管理器"""
        self.logger = logging.getLogger(__name__)
        self.ffmpeg_dir = config.FFMPEG_DIR
        self.ffmpeg_exe = config.FFMPEG_EXE
        # 也记录预期的 ffprobe 路径
        self.ffprobe_exe = self.ffmpeg_dir / "ffprobe.exe"
        self.download_attempts = 0

    def check_ffmpeg(self) -> Tuple[bool, str]:
        """
        检查FFmpeg是否可用

        Returns:
            Tuple[bool, str]: (是否可用, FFmpeg路径或错误信息)
        """
        # 首先检查项目目录中的FFmpeg
        if self.ffmpeg_exe.exists():
            # 如果项目内同时包含 ffprobe.exe，则优先验证两者
            if self.ffprobe_exe.exists():
                ffmpeg_ok = self._test_ffmpeg(str(self.ffmpeg_exe))
                ffprobe_ok = self._test_ffmpeg(str(self.ffprobe_exe))
                if ffmpeg_ok and ffprobe_ok:
                    self.logger.info(f"发现项目内FFmpeg与ffprobe: {self.ffmpeg_exe}, {self.ffprobe_exe}")
                    return True, str(self.ffmpeg_exe)
                else:
                    self.logger.warning(f"⚠️ 项目内FFmpeg或ffprobe不可用: ffmpeg_ok={ffmpeg_ok}, ffprobe_ok={ffprobe_ok}")
                    # 继续尝试系统环境或自动下载
            else:
                self.logger.warning(f"⚠️ 项目内缺少 ffprobe.exe，期待位置: {self.ffprobe_exe}")
                # 虽然项目内存在 ffmpeg.exe，但缺少 ffprobe，继续尝试系统路径或下载

        # 检查系统环境变量中的FFmpeg和ffprobe
        sys_ffmpeg_ok = self._test_ffmpeg("ffmpeg")
        sys_ffprobe_ok = self._test_ffmpeg("ffprobe")
        if sys_ffmpeg_ok and sys_ffprobe_ok:
            self.logger.info("发现系统FFmpeg与ffprobe")
            return True, "ffmpeg (系统环境变量)"
        if sys_ffmpeg_ok and not sys_ffprobe_ok:
            self.logger.warning("系统已安装 ffmpeg，但未检测到 ffprobe（或不可执行）")
        if not sys_ffmpeg_ok and sys_ffprobe_ok:
            self.logger.warning("检测到系统 ffprobe，但未检测到 ffmpeg")

        # FFmpeg不可用
        self.logger.warning("❌ 未找到可用的FFmpeg")
        return False, "FFmpeg未安装"

    def _test_ffmpeg(self, ffmpeg_path: str) -> bool:
        """
        测试FFmpeg是否可用

        Args:
            ffmpeg_path: FFmpeg可执行文件路径

        Returns:
            bool: 是否可用
        """
        try:
            result = subprocess.run(
                [ffmpeg_path, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.debug(f"FFmpeg测试失败: {ffmpeg_path} - {e}")
            return False

    def download_ffmpeg(self) -> bool:
        """
        下载FFmpeg到项目目录

        Returns:
            bool: 是否下载成功
        """
        if self.download_attempts >= self.MAX_DOWNLOAD_ATTEMPTS:
            self.logger.error(f"❌ 已达到最大下载尝试次数 ({self.MAX_DOWNLOAD_ATTEMPTS})")
            return False

        self.download_attempts += 1
        self.logger.info(f" 开始下载FFmpeg (尝试 {self.download_attempts}/{self.MAX_DOWNLOAD_ATTEMPTS})...")

        # 只在Windows平台下载
        if platform.system() != "Windows":
            self.logger.warning("⚠️ 自动下载仅支持Windows平台，请手动安装FFmpeg")
            return False

        # 尝试所有下载源
        for idx, url in enumerate(self.FFMPEG_DOWNLOAD_URLS, 1):
            self.logger.info(f"📥 尝试下载源 {idx}/{len(self.FFMPEG_DOWNLOAD_URLS)}: {url}")
            try:
                if self._download_and_extract(url):
                    self.logger.info("FFmpeg下载成功")
                    return True
            except Exception as e:
                self.logger.warning(f"⚠️ 下载源 {idx} 失败: {e}")
                continue

        self.logger.error("❌ 所有下载源均失败")
        return False

    def _download_and_extract(self, url: str) -> bool:
        """
        下载并解压FFmpeg

        Args:
            url: 下载链接

        Returns:
            bool: 是否成功
        """
        # 临时文件路径
        temp_zip = config.TEMP_DIR / "ffmpeg_download.zip"
        temp_extract = config.TEMP_DIR / "ffmpeg_extract"

        try:
            # 清理旧的临时文件
            if temp_zip.exists():
                temp_zip.unlink()
            if temp_extract.exists():
                shutil.rmtree(temp_extract)

            # 下载文件
            self.logger.info("📥 正在下载...")
            urllib.request.urlretrieve(url, temp_zip)

            if not temp_zip.exists() or temp_zip.stat().st_size < 1000:
                raise Exception("下载的文件无效或太小")

            self.logger.info(f"📦 下载完成，大小: {temp_zip.stat().st_size / 1024 / 1024:.1f} MB")

            # 解压文件
            self.logger.info("📦 正在解压...")
            temp_extract.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_extract)

            # 查找ffmpeg.exe
            ffmpeg_found = None
            for root, dirs, files in os.walk(temp_extract):
                if "ffmpeg.exe" in files:
                    ffmpeg_found = Path(root) / "ffmpeg.exe"
                    break

            if not ffmpeg_found:
                raise Exception("解压后未找到ffmpeg.exe")

            # 移动到目标位置
            self.logger.info(f"📁 正在移动文件到: {self.ffmpeg_dir}")
            self.ffmpeg_dir.mkdir(parents=True, exist_ok=True)

            # 复制ffmpeg.exe和相关文件
            bin_dir = ffmpeg_found.parent
            for file in bin_dir.glob("*.exe"):
                target = self.ffmpeg_dir / file.name
                shutil.copy2(file, target)
                self.logger.debug(f"复制: {file.name}")

            # 验证安装
            if not self._test_ffmpeg(str(self.ffmpeg_exe)):
                raise Exception("FFmpeg安装验证失败")

            self.logger.info("FFmpeg安装成功")

            # 清理临时文件
            self._cleanup_temp_files(temp_zip, temp_extract)

            return True

        except Exception as e:
            self.logger.error(f"❌ 下载或解压失败: {e}")
            # 清理临时文件
            self._cleanup_temp_files(temp_zip, temp_extract)
            return False

    def _cleanup_temp_files(self, temp_zip: Path, temp_extract: Path):
        """清理临时文件"""
        try:
            if temp_zip.exists():
                temp_zip.unlink()
            if temp_extract.exists():
                shutil.rmtree(temp_extract)
            self.logger.debug("临时文件已清理")
        except Exception as e:
            self.logger.warning(f"⚠️ 清理临时文件失败: {e}")

    def ensure_ffmpeg(self) -> str:
        """
        确保FFmpeg可用（检测 -> 下载 -> 回退）

        Returns:
            str: 可用的FFmpeg路径

        Raises:
            RuntimeError: 如果FFmpeg不可用且无法下载
        """
        # 检查FFmpeg
        available, path = self.check_ffmpeg()

        if available:
            return path

        # FFmpeg不可用，尝试下载
        self.logger.warning("⚠️ FFmpeg不可用，尝试自动下载...")
        print("\n" + "=" * 60)
        print("⚠️  警告：未检测到FFmpeg")
        print("=" * 60)
        print("FFmpeg是音视频处理的必需工具。")
        print("正在尝试自动下载并安装FFmpeg到项目目录...")
        print("=" * 60 + "\n")

        # 尝试下载
        download_success = self.download_ffmpeg()

        if download_success:
            # 重新检查
            available, path = self.check_ffmpeg()
            if available:
                print("\n" + "=" * 60)
                print("FFmpeg安装成功！")
                print(f"📁 安装位置: {path}")
                print("=" * 60 + "\n")
                return path

        # 下载失败，检查是否有系统FFmpeg
        if self._test_ffmpeg("ffmpeg"):
            self.logger.warning("⚠️ 下载失败，回退到系统FFmpeg")
            print("\n" + "=" * 60)
            print("⚠️  FFmpeg自动下载失败")
            print("=" * 60)
            print("已回退到使用系统环境变量中的FFmpeg。")
            print("建议手动下载FFmpeg并放置到项目目录以获得更好的兼容性。")
            print(f"目标位置: {self.ffmpeg_dir}")
            print("=" * 60 + "\n")
            return "ffmpeg"

        # 完全失败
        error_msg = (
            f"\n{'=' * 60}\n"
            f"❌ FFmpeg不可用且无法自动下载\n"
            f"{'=' * 60}\n"
            f"请手动安装FFmpeg：\n\n"
            f"方式1（推荐）：下载到项目目录\n"
            f"  1. 访问: https://github.com/BtbN/FFmpeg-Builds/releases\n"
            f"  2. 下载最新的 ffmpeg-master-latest-win64-gpl.zip\n"
            f"  3. 解压并将 bin 目录中的内容复制到: {self.ffmpeg_dir}\n\n"
            f"方式2：安装到系统\n"
            f"  1. 访问: https://ffmpeg.org/download.html\n"
            f"  2. 下载并安装FFmpeg\n"
            f"  3. 确保 ffmpeg 命令在系统 PATH 中\n"
            f"{'=' * 60}\n"
        )
        self.logger.error(error_msg)
        print(error_msg)
        raise RuntimeError("FFmpeg不可用，请手动安装")


# ========== 单例模式 ==========

_ffmpeg_manager_instance: Optional[FFmpegManager] = None


def get_ffmpeg_manager() -> FFmpegManager:
    """
    获取FFmpeg管理器实例（单例模式）

    Returns:
        FFmpegManager: FFmpeg管理器实例
    """
    global _ffmpeg_manager_instance
    if _ffmpeg_manager_instance is None:
        _ffmpeg_manager_instance = FFmpegManager()
    return _ffmpeg_manager_instance
