import os
import sys
import subprocess
import time
import shutil
import webbrowser
import hashlib
from pathlib import Path

# --- 配置区域 ---
PROJECT_ROOT = Path(__file__).resolve().parent
FFMPEG_DIR = PROJECT_ROOT / "tools"
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REQ_FILE = PROJECT_ROOT / "requirements.txt"
MARKER_FILE = PROJECT_ROOT / ".env_installed"  # 用于标记依赖是否已安装
REQ_HASH_FILE = PROJECT_ROOT / ".req_hash"  # 用于存储 requirements.txt 的哈希值

# ========================================
#  Python 路径配置 (双模式: 嵌入式优先, 回退venv)
# ========================================
EMBED_PYTHON_DIR = PROJECT_ROOT / "tools" / "python"
EMBED_PYTHON = EMBED_PYTHON_DIR / "python.exe"
EMBED_SITE_PACKAGES = EMBED_PYTHON_DIR / "Lib" / "site-packages"

VENV_DIR = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
VENV_SITE_PACKAGES = VENV_DIR / "Lib" / "site-packages"

# 检测Python模式: 优先嵌入式, 回退venv
if EMBED_PYTHON.exists():
    PYTHON_EXEC = EMBED_PYTHON
    SITE_PACKAGES = EMBED_SITE_PACKAGES
    PYTHON_MODE = "embedded"
elif VENV_PYTHON.exists():
    PYTHON_EXEC = VENV_PYTHON
    SITE_PACKAGES = VENV_SITE_PACKAGES
    PYTHON_MODE = "venv"
else:
    PYTHON_EXEC = None
    SITE_PACKAGES = None
    PYTHON_MODE = "not_found"

# 国内镜像源 (清华源)
PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def log(msg):
    print(f"[Bootloader] {msg}")


def get_file_hash(filepath: Path) -> str:
    """计算文件的 MD5 哈希值"""
    if not filepath.exists():
        return ""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_saved_hash() -> str:
    """获取保存的 requirements.txt 哈希值"""
    if REQ_HASH_FILE.exists():
        return REQ_HASH_FILE.read_text().strip()
    return ""


def save_hash(hash_value: str):
    """保存 requirements.txt 的哈希值"""
    REQ_HASH_FILE.write_text(hash_value)


def parse_requirements(filepath: Path) -> set:
    """解析 requirements.txt，返回包名集合（不含版本号）"""
    packages = set()
    if not filepath.exists():
        return packages

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行、注释和特殊指令
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # 提取包名（去除版本号和其他修饰符）
            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].split("<")[0].split(">")[0].strip()
            if pkg_name:
                packages.add(pkg_name.lower())
    return packages


def get_installed_packages() -> set:
    """获取当前已安装的包名集合"""
    if PYTHON_EXEC is None:
        return set()
    try:
        result = subprocess.run(
            [str(PYTHON_EXEC), "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, check=True
        )
        packages = set()
        for line in result.stdout.strip().split("\n"):
            if line and "==" in line:
                pkg_name = line.split("==")[0].strip().lower()
                packages.add(pkg_name)
        return packages
    except subprocess.CalledProcessError:
        return set()


def fix_pytorch_dll():
    """
    修复 PyTorch 在 Windows 上的 DLL 依赖问题。

    PyTorch 2.x+cu118 的 fbgemm.dll 依赖 libomp140.x86_64.dll (LLVM OpenMP)，
    但 Windows 系统默认不包含此 DLL。解决方案是将 PyTorch 自带的
    libiomp5md.dll (Intel OpenMP) 复制为 libomp140.x86_64.dll，
    两者 API 兼容。
    """
    if SITE_PACKAGES is None:
        return

    torch_lib = SITE_PACKAGES / "torch" / "lib"

    source_dll = torch_lib / "libiomp5md.dll"
    target_dll = torch_lib / "libomp140.x86_64.dll"

    if not torch_lib.exists():
        return  # PyTorch 未安装

    if target_dll.exists():
        return  # 已修复

    if source_dll.exists():
        log("Fixing PyTorch DLL (fbgemm.dll -> libomp140.x86_64.dll)...")
        shutil.copy(source_dll, target_dll)
        log("DLL fix completed!")


def sync_dependencies():
    """
    智能依赖同步：
    - 检测 requirements.txt 变化
    - 自动安装新增包
    """
    if PYTHON_EXEC is None:
        log("ERROR: Python not found!")
        log(f"  - Embedded Python path: {EMBED_PYTHON}")
        log(f"  - Venv Python path: {VENV_PYTHON}")
        log("Please install Python 3.10+ or place embedded Python in tools/python/")
        input("Press Enter to exit...")
        sys.exit(1)

    log(f"Python Mode: {PYTHON_MODE}")
    log(f"Python Path: {PYTHON_EXEC}")

    current_hash = get_file_hash(REQ_FILE)
    saved_hash = get_saved_hash()

    # 如果哈希值相同且标记文件存在，说明无变化，极速启动
    if current_hash == saved_hash and MARKER_FILE.exists():
        log("Dependencies unchanged, skipping (fast startup)...")
        return

    log("Detected requirements.txt change or first run, syncing dependencies...")

    # 解析当前 requirements.txt 中的包
    required_packages = parse_requirements(REQ_FILE)
    log(f"requirements.txt defines {len(required_packages)} packages")

    # 获取当前已安装的包
    installed_packages = get_installed_packages()

    # 计算需要安装的包
    to_install = required_packages - installed_packages

    if to_install:
        log(f"Need to install {len(to_install)} new packages: {', '.join(sorted(to_install))}")

    # 使用 pip install -r 来安装所有依赖
    log("Syncing dependencies...")
    cmd = [
        str(PYTHON_EXEC), "-m", "pip", "install",
        "-r", str(REQ_FILE),
        "-i", PYPI_MIRROR
    ]

    try:
        subprocess.run(cmd, check=True)

        # 安装成功后保存哈希值和标记文件
        save_hash(current_hash)
        MARKER_FILE.touch()
        log("Dependency sync completed!")

        # 修复 PyTorch DLL 依赖问题
        fix_pytorch_dll()

    except subprocess.CalledProcessError:
        log("ERROR: Dependency sync failed!")
        log("Hint: Check network or version constraints in requirements.txt")
        input("Press Enter to exit...")
        sys.exit(1)


def check_ffmpeg():
    """检查 FFmpeg 是否存在"""
    ffmpeg_exe = FFMPEG_DIR / "ffmpeg.exe"
    if not ffmpeg_exe.exists():
        log(f"ERROR: FFmpeg not found: {ffmpeg_exe}")
        log("Please download ffmpeg.exe and place it in the tools directory.")
        input("Press Enter to exit...")
        sys.exit(1)
    log("FFmpeg check passed.")


def setup_environment():
    """配置运行时的环境变量 (关键步骤)"""
    env = os.environ.copy()

    # 1. 添加 FFmpeg 到 PATH
    env["PATH"] = f"{FFMPEG_DIR};" + env["PATH"]

    # 2. 使用动态检测到的 site-packages 路径
    if SITE_PACKAGES is None:
        log("ERROR: Site packages path not determined!")
        sys.exit(1)

    # 3. 注入 CUDA 库路径 (解决 cu11 和 cu12 共存)
    # PyTorch (cu11) libs
    torch_lib = SITE_PACKAGES / "torch" / "lib"
    # Faster-Whisper (ctranslate2) 需要的 NVIDIA libs (cu12)
    nvidia_cudnn = SITE_PACKAGES / "nvidia" / "cudnn" / "bin"
    nvidia_cublas = SITE_PACKAGES / "nvidia" / "cublas" / "bin"

    # 将这些路径前置到 PATH
    extra_paths = [str(torch_lib), str(nvidia_cudnn), str(nvidia_cublas)]
    env["PATH"] = ";".join(extra_paths) + ";" + env["PATH"]

    # 4. 设置环境变量
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"

    return env


def start_services(env):
    """启动后端和前端"""
    processes = []

    try:
        # --- 1. 启动后端 (Uvicorn) ---
        log("Starting backend service...")
        backend_cmd = [
            str(PYTHON_EXEC), "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8000",
        ]
        backend_proc = subprocess.Popen(backend_cmd, cwd=str(BACKEND_DIR), env=env)
        processes.append(backend_proc)

        # --- 2. 等待后端启动后打开浏览器 ---
        time.sleep(3)
        webbrowser.open("http://127.0.0.1:8000/docs")

        log("All services started. Press Ctrl+C to stop.")

        # 守护进程：等待任意子进程结束
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                log("Backend service stopped.")
                break

    except KeyboardInterrupt:
        log("Stopping services...")
    finally:
        for p in processes:
            p.terminate()
        log("Exited.")


if __name__ == "__main__":
    print("="*40)
    print("   AnchorFlux - Smart Launcher")
    print("="*40)
    print(f"   Python Mode: {PYTHON_MODE}")
    print("="*40)

    # 1. 智能依赖同步
    sync_dependencies()

    # 2. FFmpeg 检查
    check_ffmpeg()

    # 3. 配置环境路径
    run_env = setup_environment()

    # 4. 启动服务
    start_services(run_env)
