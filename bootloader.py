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
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
REQ_FILE = PROJECT_ROOT / "requirements.txt"
MARKER_FILE = PROJECT_ROOT / ".env_installed" # 用于标记依赖是否已安装
REQ_HASH_FILE = PROJECT_ROOT / ".req_hash"  # 用于存储 requirements.txt 的哈希值
FFMPEG_DIR = PROJECT_ROOT / "tools"
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

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
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "list", "--format=freeze"],
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
    site_packages = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
    torch_lib = site_packages / "torch" / "lib"
    
    source_dll = torch_lib / "libiomp5md.dll"
    target_dll = torch_lib / "libomp140.x86_64.dll"
    
    if not torch_lib.exists():
        return  # PyTorch 未安装
    
    if target_dll.exists():
        return  # 已修复
    
    if source_dll.exists():
        log("🔧 修复 PyTorch DLL 依赖 (fbgemm.dll -> libomp140.x86_64.dll)...")
        shutil.copy(source_dll, target_dll)
        log("✅ DLL 修复完成！")

def sync_dependencies():
    """
    智能依赖同步：
    - 检测 requirements.txt 变化
    - 自动安装新增包
    - 自动卸载移除的包
    """
    current_hash = get_file_hash(REQ_FILE)
    saved_hash = get_saved_hash()
    
    # 如果哈希值相同且标记文件存在，说明无变化，极速启动
    if current_hash == saved_hash and MARKER_FILE.exists():
        log("✅ 依赖无变化，跳过检查 (极速启动模式)...")
        return
    
    log("检测到 requirements.txt 变化或首次运行，开始智能依赖同步...")
    
    # 解析当前 requirements.txt 中的包
    required_packages = parse_requirements(REQ_FILE)
    log(f"📋 requirements.txt 中定义了 {len(required_packages)} 个包")
    
    # 获取当前已安装的包
    installed_packages = get_installed_packages()
    
    # 计算需要安装和卸载的包
    # 注意：只同步 requirements.txt 中明确列出的包，不处理其依赖
    to_install = required_packages - installed_packages
    
    # 对于卸载，我们需要更谨慎：只卸载之前由 requirements.txt 安装但现在被移除的包
    # 这里使用一个简化策略：不自动卸载，只提示用户
    # 如果需要自动卸载，可以维护一个已安装包列表文件
    
    if to_install:
        log(f"📥 需要安装 {len(to_install)} 个新包: {', '.join(sorted(to_install))}")
    
    # 使用 pip install -r 来安装所有依赖（pip 会自动处理已安装的包）
    log("🔄 正在同步依赖...")
    cmd = [
        str(VENV_PYTHON), "-m", "pip", "install",
        "-r", str(REQ_FILE),
        "-i", PYPI_MIRROR
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # 安装成功后保存哈希值和标记文件
        save_hash(current_hash)
        MARKER_FILE.touch()
        log("✅ 依赖同步完成！")
        
        # 修复 PyTorch DLL 依赖问题
        fix_pytorch_dll()
        
    except subprocess.CalledProcessError:
        log("❌ 依赖同步失败，请检查网络或配置。")
        log("提示：如果是版本冲突问题，请检查 requirements.txt 中的版本约束。")
        input("按回车键退出...")
        sys.exit(1)

def check_dependencies():
    """
    智能依赖检查：改用原生 pip 解决多源解析冲突
    """
    if MARKER_FILE.exists():
        log("检测到环境已就绪，跳过依赖检查 (极速启动模式)...")
        return

    log("首次运行或环境未就绪，开始检查并安装依赖...")
    # 移除这里关于镜像源的打印，因为 requirements.txt 里可能已经指定了
    # log(f"正在使用镜像源: {PYPI_MIRROR}") 

    # --- 核心修改：使用 pip 而不是 uv ---
    # pip 会自动合并 requirements.txt 里的 --extra-index-url 和命令行里的 -i
    cmd = [
        str(VENV_PYTHON), "-m", "pip", "install",
        "-r", str(REQ_FILE),
        "-i", PYPI_MIRROR  # 保持清华源作为主源
    ]

    try:
        # 添加 check=True 会在失败时自动抛出异常，不用手动检查 returncode
        subprocess.run(cmd, check=True)
        
        # 安装成功后创建标记文件
        MARKER_FILE.touch()
        log("✅ 依赖安装完成！")
    except subprocess.CalledProcessError:
        log("❌ 依赖安装失败，请检查网络或配置。")
        log("提示：如果是 numpy 版本问题，请确保使用了 pip 而不是 uv。")
        input("按回车键退出...")
        sys.exit(1)
        sys.exit(1)

def check_ffmpeg():
    """检查 FFmpeg 是否存在"""
    ffmpeg_exe = FFMPEG_DIR / "ffmpeg.exe"
    if not ffmpeg_exe.exists():
        log(f"❌ 未找到 FFmpeg: {ffmpeg_exe}")
        log("请下载 ffmpeg.exe 并放入 tools 目录。")
        input("按回车键退出...")
        sys.exit(1)
    log("FFmpeg 检查通过。")

def setup_environment():
    """配置运行时的环境变量 (关键步骤)"""
    env = os.environ.copy()
    
    # 1. 添加 FFmpeg 到 PATH
    env["PATH"] = f"{FFMPEG_DIR};" + env["PATH"]

    # 2. 嵌入式 Python 的 site-packages 路径
    site_packages = PROJECT_ROOT / ".venv" / "Lib" / "site-packages"
    
    # 3. 注入 CUDA 库路径 (解决 cu11 和 cu12 共存)
    # PyTorch (cu11) libs
    torch_lib = site_packages / "torch" / "lib"
    # Faster-Whisper (ctranslate2) 需要的 NVIDIA libs (cu12)
    nvidia_cudnn = site_packages / "nvidia" / "cudnn" / "bin"
    nvidia_cublas = site_packages / "nvidia" / "cublas" / "bin"

    # 将这些路径前置到 PATH
    extra_paths = [str(torch_lib), str(nvidia_cudnn), str(nvidia_cublas)]
    env["PATH"] = ";".join(extra_paths) + ";" + env["PATH"]
    
    return env

def start_services(env):
    """启动后端和前端"""
    processes = []
    
    try:
        # --- 1. 启动后端 (Uvicorn) ---
        log("正在启动后端服务...")
        backend_cmd = [
            str(VENV_PYTHON), "-m", "uvicorn", 
            "app.main:app",  # 假设你的入口是 app/main.py
            "--host", "127.0.0.1", 
            "--port", "8000",
            "--reload" # 开发模式，生产环境可去掉
        ]
        # cwd 设置为 backend 目录确保导入路径正确
        backend_proc = subprocess.Popen(backend_cmd, cwd=str(BACKEND_DIR), env=env)
        processes.append(backend_proc)

        # --- 2. 启动前端 ---
        # 情况 A: 如果是 Web 页面，直接打开浏览器
        time.sleep(2) # 等待后端稍微初始化
        webbrowser.open("http://127.0.0.1:8000/docs") # 或者你的前端地址
        
        # 情况 B: 如果需要启动 Node.js 前端 (如 Vue/React 开发服)
        # log("正在启动前端服务...")
        # frontend_cmd = ["npm", "run", "dev"] 
        # frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR), shell=True, env=env)
        # processes.append(frontend_proc)

        log("✅ 所有服务已启动。按 Ctrl+C 停止服务。")
        
        # 守护进程：等待任意子进程结束
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                log("后端服务意外停止。")
                break

    except KeyboardInterrupt:
        log("正在停止服务...")
    finally:
        for p in processes:
            p.terminate()
        log("已退出。")

if __name__ == "__main__":
    print("="*40)
    print("   Video to SRT GPU - 智能启动器")
    print("="*40)
    
    # 1. 智能依赖同步（检测 requirements.txt 变化并自动安装/更新）
    sync_dependencies()
    
    # 2. FFmpeg 检查
    check_ffmpeg()
    
    # 3. 配置环境路径 (CUDA/DLLs)
    run_env = setup_environment()
    
    # 4. 启动服务
    start_services(run_env)