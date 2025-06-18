import os
# 设置环境变量 KMP_DUPLICATE_LIB_OK 为 TRUE, 允许加载重复的 OpenMP 运行时库.
# 这可以避免在某些环境中由于加载了多个 OpenMP 库 (例如, 一个由 NumPy/SciPy 提供, 另一个由 PyTorch 提供) 而导致的程序崩溃或错误.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import platform
import json
import threading
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc # Python 的垃圾回收模块.
from contextlib import redirect_stdout, redirect_stderr # 用于重定向标准输出和错误流.
import warnings
import sys
import io
import re # 正则表达式模块, 当前脚本中未直接大量使用, 可能为未来扩展或间接依赖项使用.
import logging # 日志模块, 用于记录程序运行信息.
import importlib.metadata # 用于获取已安装包的元数据, 如版本号.
import importlib.util   # 提供模块导入相关的实用功能, 如检查模块是否存在.

from rich.console import Console # Rich: 用于创建富文本终端输出.
from rich.panel import Panel # Rich: 用于在终端中创建面板.
from rich.text import Text # Rich: 用于创建和操作带样式的文本.
from rich.prompt import Prompt, Confirm, IntPrompt # Rich: 用于获取用户输入.
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn # Rich: 用于显示进度条.
from rich.table import Table # Rich: 用于在终端中创建表格.
from rich.live import Live # Rich: 用于实时更新终端显示内容.
from rich.box import ROUNDED # Rich: 定义面板和表格的边框样式.

# --- Rich 控制台实例 ---
# 全局 Console 对象, 用于整个应用程序的格式化输出.
console = Console()

# --- 全局日志记录器实例 ---
# 在 setup_logging() 函数中进行配置.
logger = logging.getLogger(__name__)
LOG_FILE_NAME = "app_runtime.log" # 日志文件名.
# log_file_path (日志文件的完整路径) 将在 APP_BASE_PATH 定义后设置.

# --- 忽略特定警告 ---
# 忽略 TensorFlow (如果通过某些库间接使用时) 可能发出的关于 TF32 禁用的警告.
warnings.filterwarnings("ignore", message="TensorFloat-32 \\(TF32\\) has been disabled")
# 忽略 PySoundFile 加载失败, 转而尝试 audioread 的警告 (soundfile 是首选).
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

class StderrFilter:
    """
    一个自定义的 stderr 包装器.
    主要用于过滤掉特定且通常无害的警告信息 (例如 'libpng warning').
    它还尝试模拟标准错误流的文件描述符 (fileno) 和 TTY (isatty) 属性,
    以提高与某些期望这些属性的库的兼容性.
    """
    def __init__(self):
        self.old_stderr = sys.stderr # 保存原始的 stderr 流.
        self._fileno = None # 初始化文件描述符编号.
        try:
            # 尝试获取原始 stderr 的文件描述符.
            self._fileno = self.old_stderr.fileno()
        except (AttributeError, io.UnsupportedOperation):
            # 如果获取失败 (例如, stderr 不是一个标准文件流), 则保持为 None.
            self._fileno = None

    def __enter__(self):
        # 进入上下文时, 将全局 sys.stderr 替换为当前 StderrFilter 实例.
        sys.stderr = self
        return self

    def __exit__(self, *args):
        # 退出上下文时, 恢复原始的 sys.stderr.
        sys.stderr = self.old_stderr

    def write(self, text):
        # 自定义写方法: 如果文本中不包含 'libpng warning', 则写入原始 stderr.
        if 'libpng warning' not in text:
            self.old_stderr.write(text)

    def flush(self):
        # 调用原始 stderr 的 flush 方法.
        self.old_stderr.flush()

    def fileno(self):
        # 提供 fileno 方法.
        if self._fileno is not None:
            return self._fileno
        # 如果初始时 _fileno 未设置成功, 再次尝试从 old_stderr 获取.
        if hasattr(self.old_stderr, 'fileno'):
             current_fileno = self.old_stderr.fileno()
             if isinstance(current_fileno, int): # 确保返回的是整数.
                 return current_fileno
        # 如果无法提供有效的文件描述符, 抛出异常.
        raise io.UnsupportedOperation("underlying stream does not support fileno or returned non-integer")

    def isatty(self):
        # 提供 isatty 方法, 判断流是否连接到 TTY (终端).
        if hasattr(self.old_stderr, 'isatty'):
            return self.old_stderr.isatty()
        return False # 默认返回 False.

# 应用 StderrFilter: 如果当前的 stderr 不是 StderrFilter 实例, 则替换它.
# 这可以防止重复包装.
if not isinstance(sys.stderr, StderrFilter):
    sys.stderr = StderrFilter()

# --- 环境变量设置 ---
# 设置 Python 的标准 I/O 编码为 UTF-8, 确保跨平台字符处理的一致性.
os.environ['PYTHONIOENCODING'] = 'utf-8'
# 尝试静默 Pillow (PIL Fork) 库可能产生的 libpng 警告 (StderrFilter 也处理这个, 这里是多一层保险).
os.environ['PILLOW_SILENCE_LIBPNG'] = '1'
# 设置 TensorFlow 的日志级别为 2 (错误), 以减少不必要的 INFO 和 WARNING 日志.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置 Hugging Face Hub 的 API 端点为国内镜像, 加快模型下载速度.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def get_app_base_path():
    """
    获取应用程序的基础路径.
    如果程序被 PyInstaller 等工具打包成单个可执行文件, sys.frozen 会被设置.
    此时, sys.executable 指向可执行文件本身. 其目录即为基础路径.
    否则 (在开发环境中直接运行 .py脚本时), 使用当前工作目录 (os.getcwd()).
    """
    if hasattr(sys, 'frozen'): # 判断是否为打包后的程序.
        return os.path.dirname(sys.executable)
    return os.getcwd() # 返回当前 Python 脚本的工作目录.

APP_BASE_PATH = get_app_base_path() # 应用程序的根目录.
log_file_path = os.path.join(APP_BASE_PATH, LOG_FILE_NAME) # 日志文件的完整路径.

def setup_logging():
    """配置全局日志记录器."""
    global logger, log_file_path # 使用全局 logger 和 log_file_path.
    logger.setLevel(logging.DEBUG) # 设置日志记录的最低级别为 DEBUG.

    # 文件处理器 (FileHandler): 将日志写入到文件.
    # 'a' 表示追加模式, encoding='utf-8' 确保中文日志正确记录.
    fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
    fh.setLevel(logging.DEBUG) # 文件处理器也记录 DEBUG 及以上级别.

    # 日志格式器 (Formatter): 定义日志信息的格式.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
    fh.setFormatter(formatter)

    # 将文件处理器添加到记录器, 但要避免重复添加.
    if not logger.handlers:
        logger.addHandler(fh)
        # 可以选择性地添加 StreamHandler 以同时在控制台输出日志 (主要用于调试脚本本身).
        # sh = logging.StreamHandler(sys.stdout)
        # sh.setLevel(logging.INFO) # 例如, 控制台只显示 INFO 及以上级别.
        # sh.setFormatter(formatter)
        # logger.addHandler(sh)

# --- 应用程序常量定义 ---
MODEL_CACHE_DIR = os.path.join(APP_BASE_PATH, "model_cache") # Whisper 模型下载和缓存的目录.
TEMP_DIR = os.path.join(APP_BASE_PATH, "temp_files")         # 存储临时文件 (如音频分段) 的目录.
STATUS_FILE = os.path.join(TEMP_DIR, "status.json")          # 存储处理状态的文件, 用于断点续传.

# 音频处理参数
SEGMENT_LENGTH_MS = 60 * 1000       # 音频分段的目标长度 (毫秒), 这里是 60 秒.
SILENCE_SEARCH_DURATION_MS = 2000 # 在分段末尾搜索静音的时长 (毫秒), 用于智能切分.
MIN_SILENCE_LEN = 300               # 检测到的静音的最小长度 (毫秒), 短于此的静音不用于切分.
SILENCE_THRESH = -40                # 静音检测的阈值 (dBFS), 低于此值被认为是静音.

# Whisper 模型和处理参数
DEVICE = "cpu"                                           # 指定运行设备为 CPU.
BATCH_SIZE = 16                                          # Whisper 模型转录时的批处理大小.
COMPUTE_TYPE = "int8"                                    # Whisper (Faster Whisper) 运算精度, "int8" 在 CPU 上有较好性能.
WHISPER_MODEL = "medium"                                 # 默认使用的 Whisper 模型大小.
USE_WORD_TIMESTAMPS_CONFIG = False                       # 是否在 SRT 字幕中使用词级别时间戳 (默认禁用, 使用句子/短语级别).
CPU_THREADS_PER_MODEL_INSTANCE = max(1, (os.cpu_count() or 4) // 2) # 模型内部使用的 CPU 线程数, 默认为 CPU核心数的一半 (至少为1, 最多不超过物理核心数).

# 全局 Whisper 模型实例和锁
whisper_model_global = None # 用于缓存加载的 Whisper 模型实例.
model_lock = threading.Lock() # 线程锁, 用于在多线程加载模型时保证线程安全.

# --- Python 依赖项及其期望版本 ---
# 格式: (pip 包名, "期望版本号", "导入时使用的模块名")
# 这个列表是程序正确运行的核心依赖, check_dependencies 函数会据此进行检查.
REQUIRED_PYTHON_PACKAGES = [
    ("torch", "2.7.0", "torch"),
    ("torchaudio", "2.7.0", "torchaudio"),
    ("whisperx", "3.3.4", "whisperx"),
    ("pydub", "0.25.1", "pydub"),
    ("rich", "14.0.0", "rich"),
    ("tqdm", "4.67.1", "tqdm"),
    ("transformers", "4.52.3", "transformers"),
    ("ffmpeg-python", "0.2.0", "ffmpeg"), # ffmpeg-python 包导入时模块名为 ffmpeg
    ("pytorch-lightning", "2.5.1.post0", "pytorch_lightning"), # pytorch-lightning 包导入时模块名为 pytorch_lightning
    ("faster-whisper", "1.1.1", "faster_whisper"),
    ("ctranslate2", "4.4.0", "ctranslate2"),
    ("soundfile", "0.13.1", "soundfile"),
    ("numpy", "2.2.6", "numpy"),
    ("onnxruntime", "1.22.0", "onnxruntime"),
]
# --- 依赖项定义结束 ---

def check_dependencies(verbose=True):
    """
    检查必要的 Python 依赖项和 ffmpeg 是否已正确安装并符合预期版本.
    参数:
        verbose (bool): 如果为 True, 则打印详细的检查过程和结果到控制台.
    返回:
        bool: 如果所有追踪的 Python 依赖项都满足要求则返回 True, 否则返回 False.
              ffmpeg 的问题会报告但不会直接导致此函数返回 False.
    """
    if verbose:
        console.print("[bold blue][INFO][/bold blue] 正在检查必要的依赖项 (基于最终列表)...")
    logger.info("开始检查依赖项 (基于最终列表)...")

    missing_deps_info = [] # 存储依赖问题的详细信息 (Rich Text 对象).
    installed_deps_messages = [] # 存储已成功安装的依赖信息.
    all_python_deps_ok = True # 标记所有 Python 依赖是否都 OK.

    for package_name, expected_version, module_name_to_check in REQUIRED_PYTHON_PACKAGES:
        try:
            # 1. 检查模块是否可以被 Python 找到.
            spec = importlib.util.find_spec(module_name_to_check)
            if spec is None:
                # 如果 find_spec 返回 None, 说明模块不存在, 抛出 ImportError.
                raise ImportError(f"模块 {module_name_to_check} (来自包 {package_name}) 未找到.")

            # 2. 获取已安装包的版本.
            installed_version = importlib.metadata.version(package_name)

            # 3. 比较版本.
            if installed_version == expected_version:
                msg = f"{package_name} (版本: {installed_version}) 已安装且版本正确."
                if verbose: # 只有在 verbose 模式下才加入成功安装列表以打印.
                    installed_deps_messages.append(Text.assemble((f"{package_name}", "green"), (f" (版本: {installed_version}) 已安装且版本正确.", "green")))
                logger.info(msg)
            else:
                all_python_deps_ok = False
                msg = f"{package_name} 版本不匹配. 期望: {expected_version}, 已安装: {installed_version}."
                rich_msg = Text.assemble(
                    (f"{package_name}", "yellow"),
                    ": 版本不匹配 (期望: ",
                    (expected_version, "bold white"),
                    ", 已安装: ",
                    (installed_version, "bold white"),
                    ")"
                )
                missing_deps_info.append(rich_msg)
                logger.warning(msg)
        except ImportError: # 捕获由 find_spec 引发的模块未找到错误.
            all_python_deps_ok = False
            msg = f"{package_name} (模块: {module_name_to_check}) 未安装."
            rich_msg = Text.assemble(
                (f"{package_name}", "red"),
                ": 未安装 (期望版本: ",
                (expected_version, "bold white"),
                ")"
            )
            missing_deps_info.append(rich_msg)
            logger.warning(msg)
        except importlib.metadata.PackageNotFoundError: # 捕获包元数据未找到错误.
            all_python_deps_ok = False
            msg = f"{package_name} 元数据未找到 (通常表示未正确安装)."
            rich_msg = Text.assemble(
                (f"{package_name}", "red"),
                ": 元数据未找到 (期望版本: ",
                (expected_version, "bold white"),
                ")"
            )
            missing_deps_info.append(rich_msg)
            logger.warning(msg)

    # 检查 ffmpeg (外部程序).
    ffmpeg_ok = False
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, encoding='utf-8', errors='replace')
        if result.returncode == 0:
            if verbose:
                installed_deps_messages.append(Text.assemble(("[bold green]ffmpeg (外部程序)[/bold green]", ": 已检测到.")))
            logger.info("ffmpeg (外部程序) 已检测到.")
            ffmpeg_ok = True
        else:
            msg = "ffmpeg (外部程序) 检测失败 (命令执行但返回非零代码). 可能需要手动安装或检查系统 PATH 配置."
            missing_deps_info.append(Text.assemble(("[yellow]ffmpeg (外部程序)[/yellow]", ": 检测失败 (返回非零代码).")))
            logger.warning(msg)
    except FileNotFoundError:
        msg = "未找到 ffmpeg (外部程序) 命令. 请安装 ffmpeg 并确保其已添加到系统 PATH 环境变量中."
        missing_deps_info.append(Text.assemble(("[red]ffmpeg (外部程序)[/red]", ": 未找到.")))
        logger.warning(msg)
    except Exception as e: # 其他执行 ffmpeg 命令时可能发生的错误.
        msg = f"检测 ffmpeg (外部程序) 时发生未知错误: {e}"
        missing_deps_info.append(Text.assemble(("[red]ffmpeg (外部程序)[/red]", f": 检测时出错 ({e}).")))
        logger.error(msg, exc_info=True)


    if verbose:
        for msg_console in installed_deps_messages:
            console.print(Text("[INFO] ").append(msg_console))

        if missing_deps_info:
            console.print("\n[bold yellow][WARNING][/bold yellow] 以下依赖项存在问题:")
            for dep_rich_text in missing_deps_info:
                console.print(Text("  - ").append(dep_rich_text))
            console.print(f"\n请根据安装教程指导, 安装或更新以上依赖项至指定版本.")
            console.print(f"教程中推荐的安装命令会确保这些依赖的版本正确.")
            console.print(f"如果脚本因依赖问题无法运行, 请优先参照安装教程解决.")


    if not all_python_deps_ok:
        logger.warning("依赖检查完成. 一个或多个 Python 依赖项存在问题.")
        return False # 只要 Python 依赖有问题, 就返回 False.
    
    # 如果 Python 依赖都 OK, 即使 ffmpeg 有问题, 也返回 True, 但在 verbose 模式下会提示 ffmpeg 问题.
    if verbose:
        console.print("[bold green][INFO][/bold green] 依赖检查完成. 所有追踪的 Python 依赖项均已安装且版本正确.")
        if not ffmpeg_ok and any("ffmpeg" in (item.plain if isinstance(item, Text) else str(item)).lower() for item in missing_deps_info):
             console.print("[bold yellow][INFO][/bold yellow] 注意: ffmpeg (外部程序) 检测存在问题, 请参照上方具体提示.")

    logger.info("依赖检查完成. 所有追踪的 Python 依赖项均已安装且版本正确.")
    return True

def ensure_app_dirs():
    """确保应用程序所需的目录 (临时目录, 模型缓存目录) 存在,如果不存在则创建."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        logger.info(f"临时目录已创建: {TEMP_DIR}")
        if console: console.print(f"[bold blue][INFO][/bold blue] 临时目录已创建: {TEMP_DIR}")
    if not os.path.exists(MODEL_CACHE_DIR):
        os.makedirs(MODEL_CACHE_DIR)
        logger.info(f"模型缓存目录已创建: {MODEL_CACHE_DIR}")
        if console: console.print(f"[bold blue][INFO][/bold blue] 模型缓存目录已创建: {MODEL_CACHE_DIR}")

def load_status():
    """从 status.json 文件加载先前的处理状态."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
                logger.info(f"状态已从 {STATUS_FILE} 加载.")
                return status_data
        except json.JSONDecodeError as e:
            logger.warning(f"状态文件 {STATUS_FILE} 已损坏: {e}. 将重新开始.")
            if console: console.print(f"[bold yellow][WARNING][/bold yellow] 状态文件 {STATUS_FILE} 已损坏. 将重新开始.")
            return {} # 返回空字典表示无效状态.
    logger.info(f"状态文件 {STATUS_FILE} 未找到. 返回空状态.")
    return {}

def save_status(status):
    """将当前处理状态保存到 status.json 文件."""
    ensure_app_dirs() # 确保临时目录存在.
    try:
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=4) # indent=4 使 JSON 文件更易读.
        logger.info(f"状态已保存到 {STATUS_FILE}.")
    except IOError as e:
        logger.error(f"保存状态到 {STATUS_FILE} 失败: {e}")
        if console: console.print(f"[bold red][ERROR][/bold red] 保存状态文件失败: {e}")


def cleanup_temp():
    """清理临时文件目录 (TEMP_DIR)."""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR) # 递归删除整个目录.
            logger.info(f"已清理临时目录: {TEMP_DIR}")
            if console: console.print(f"[bold blue][INFO][/bold blue] 已清理临时目录: {TEMP_DIR}")
        except OSError as e: # 处理删除时可能发生的错误.
            logger.error(f"清理临时文件目录 {TEMP_DIR} 失败: {e}")
            if console: console.print(f"[bold red][ERROR][/bold red] 清理临时文件目录 {TEMP_DIR} 失败: {e}")
    else:
        logger.info(f"临时目录 {TEMP_DIR} 不存在, 无需清理.")

def extract_audio(input_file_path, audio_output_path, force_extract=False):
    """
    使用 ffmpeg 从输入的媒体文件中提取音频, 并转换为单声道、16kHz采样率的 PCM s16le 格式 WAV 文件.
    参数:
        input_file_path (str): 输入媒体文件的路径.
        audio_output_path (str): 输出 WAV 音频文件的路径.
        force_extract (bool): 如果为 True, 即使输出文件已存在也强制重新提取.
    返回:
        bool: 提取成功返回 True, 否则返回 False.
    """
    logger.info(f"请求提取音频: input='{input_file_path}', output='{audio_output_path}', force={force_extract}")
    if not force_extract and os.path.exists(audio_output_path):
        logger.info(f"标准化的音频文件已存在, 跳过提取: {audio_output_path}")
        if console: console.print(f"[bold blue][INFO][/bold blue] 标准化的音频文件已存在. 跳过提取: {audio_output_path}")
        return True

    # ffmpeg 命令参数:
    # -y: 覆盖输出文件而不询问.
    # -i: 指定输入文件.
    # -vn: 禁用视频录制 (只提取音频).
    # -ac 1: 设置音频通道为 1 (单声道).
    # -ar 16000: 设置音频采样率为 16000 Hz.
    # -acodec pcm_s16le: 设置音频编解码器为 PCM signed 16-bit little-endian.
    command = [
        "ffmpeg", "-y", "-i", input_file_path,
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        audio_output_path
    ]
    logger.info(f"执行 ffmpeg 命令: {' '.join(command)}")
    if console: console.print(f"[bold blue][INFO][/bold blue] 正在处理输入文件 '{os.path.basename(input_file_path)}' 以生成标准化的音频...")
    
    try:
        # 使用 Rich Progress 显示 ffmpeg 处理状态.
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
            progress.add_task("FFmpeg 处理中...", total=None) # total=None 表示不确定进度的任务.
            # 执行 ffmpeg 命令. check=True 表示如果命令返回非零退出码则抛出 CalledProcessError.
            # capture_output=True 捕获标准输出和标准错误.
            process_result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        logger.info(f"标准化的音频已成功生成: {audio_output_path}. FFmpeg stdout (前200字符): {process_result.stdout[:200]}...")
        if console: console.print(f"[bold green][INFO][/bold green] 标准化的音频已成功生成: {audio_output_path}")
        return True
    except FileNotFoundError: # 如果 ffmpeg 命令未找到.
        logger.error("未找到 ffmpeg. 请确保已安装 ffmpeg 并将其添加到系统 PATH 环境变量中.")
        if console: console.print("[bold red][ERROR][/bold red] 未找到 ffmpeg. 请确保已安装 ffmpeg 并将其添加到系统 PATH.")
        return False
    except subprocess.CalledProcessError as e: # 如果 ffmpeg 命令执行出错.
        logger.error(f"使用 ffmpeg 处理文件 '{os.path.basename(input_file_path)}' 时出错. "
                     f"命令: {' '.join(e.cmd)}, 返回码: {e.returncode}\n"
                     f"Stderr: {e.stderr}\nStdout: {e.stdout}", exc_info=False) # exc_info=False 因为我们手动记录了 stderr 和 stdout.
        if console:
            console.print(f"[bold red][ERROR][/bold red] 使用 ffmpeg 处理文件 '{os.path.basename(input_file_path)}' 时出错.")
            console.print(f"命令: {' '.join(e.cmd)}\n返回码: {e.returncode}")
            if e.stderr:
                console.print(f"FFmpeg 错误输出:\n[dim]{e.stderr.strip()}[/dim]")
            else:
                console.print("FFmpeg 未产生 stderr 输出 (或输出为空).")
            if e.stdout and e.stdout.strip(): # 只在 stdout 非空时打印.
                 console.print(f"FFmpeg 标准输出:\n[dim]{e.stdout.strip()}[/dim]")
        return False
    except Exception as e: # 捕获其他潜在错误.
        logger.error(f"提取音频时发生未知错误: {e}", exc_info=True)
        if console: console.print(f"[bold red][ERROR][/bold red] 提取音频时发生未知错误: {e}")
        return False


def get_existing_segments(temp_dir_path):
    """获取临时目录中已存在的所有音频分段文件 (.wav)."""
    segments = sorted(glob.glob(os.path.join(temp_dir_path, "segment_*.wav"))) # 查找匹配 "segment_*.wav" 的文件.
    logger.debug(f"在 {temp_dir_path} 中找到 {len(segments)} 个现有分段.")
    return segments

def split_audio(audio_path, force_split=False):
    """
    将标准化后的音频文件分割成多个小段.
    分割逻辑: 优先按 SEGMENT_LENGTH_MS 分割, 如果在分段末尾的 SILENCE_SEARCH_DURATION_MS 区域内
    检测到长度大于 MIN_SILENCE_LEN 的静音, 则在静音开始处切分.
    参数:
        audio_path (str): 标准化后的 WAV 音频文件路径.
        force_split (bool): 如果为 True, 即使存在有效的分段信息或文件, 也强制重新分割.
    返回:
        list: 包含分段信息 (文件路径, 开始时间) 的字典列表. 如果失败则返回空列表.
    """
    logger.info(f"请求分割音频: path='{audio_path}', force_split={force_split}")
    try:
        from pydub import AudioSegment, silence # pydub 用于音频操作和静音检测.
        logger.debug("pydub 导入成功.")
    except ImportError:
        logger.critical("pydub 未安装, 无法进行音频分割.", exc_info=True)
        if console: console.print("[bold red][CRITICAL][/bold red] pydub 库未安装, 无法分割音频. 请安装 pydub.")
        return []


    existing_segment_files = get_existing_segments(TEMP_DIR) # 获取已存在的分段文件.
    status = load_status() # 加载状态.

    # 检查是否可以重用之前的分段结果.
    if not force_split and status.get("segments_info") and existing_segment_files:
        # 如果状态文件中的分段数量与实际文件数量匹配.
        if len(existing_segment_files) == len(status["segments_info"]):
            logger.info("使用先前保存的分段信息和文件.")
            if console: console.print("[bold blue][INFO][/bold blue] 使用先前保存的分段信息和文件.")
            return status["segments_info"]
        else:
            logger.info("现有的分段文件数量与状态记录不匹配. 将重新分段.")
            if console: console.print("[bold yellow][WARNING][/bold yellow] 现有的分段文件数量与状态记录不匹配. 将重新分段.")
    elif not force_split and existing_segment_files: # 如果有文件但无状态信息.
        logger.info("检测到现有的分段文件但无状态信息. 将重新分段以确保时间戳和信息正确.")
        if console: console.print("[bold yellow][WARNING][/bold yellow] 检测到现有的分段文件但无状态信息. 将重新分段.")

    if console: console.print("[bold blue][INFO][/bold blue] 开始音频分段...")
    logger.info("开始音频分段...")
    try:
        # 确保使用 soundfile (如果可用) 来加载 WAV, pydub 会自动尝试.
        if importlib.util.find_spec("soundfile"):
            logger.debug("Pydub 将尝试使用 soundfile 加载 WAV 文件 (如果 soundfile 已正确安装并被 pydub 识别).")
        audio = AudioSegment.from_wav(audio_path)
        logger.info(f"音频文件 {audio_path} 加载成功, 长度: {len(audio)} ms.")
    except FileNotFoundError:
        logger.error(f"音频文件未找到: {audio_path}")
        if console: console.print(f"[bold red][ERROR][/bold red] 音频文件未找到: {audio_path}")
        return []
    except Exception as e: # 捕获 pydub 加载时可能发生的其他错误 (如格式不支持, 文件损坏).
        logger.error(f"使用 pydub 加载音频文件失败 ({audio_path}): {e}", exc_info=True)
        if console: console.print(f"[bold red][ERROR][/bold red] 使用 pydub 加载音频文件失败 ({audio_path}): {e}")
        return []

    audio_length_ms = len(audio)
    segments_info = [] # 存储新生成的分段信息.
    current_pos_ms = 0
    segment_idx = 0

    # 如果存在旧的分段文件, 先清理它们.
    if existing_segment_files:
        logger.info("正在清理旧的分段文件...")
        if console: console.print("[bold blue][INFO][/bold blue] 正在清理旧的分段文件...")
        for f_path in existing_segment_files:
            try:
                os.remove(f_path)
                logger.debug(f"已删除旧分段文件: {f_path}")
            except OSError as e:
                logger.warning(f"清理旧分段文件 {f_path} 失败: {e}")
                if console: console.print(f"[bold yellow][WARNING][/bold yellow] 清理旧分段文件 {f_path} 失败: {e}")

    # 使用 Rich Progress 显示分段进度.
    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("({task.completed}/{task.total} ms)"),
                  TimeRemainingColumn(), TimeElapsedColumn(),
                  console=console, transient=False) as progress_bar: # transient=False 使进度条在完成后保留.
        
        segment_task = progress_bar.add_task("音频分段进度", total=audio_length_ms)
        logger.info(f"开始循环分段, 总长度: {audio_length_ms} ms.")
        while current_pos_ms < audio_length_ms:
            end_pos_ms = current_pos_ms + SEGMENT_LENGTH_MS # 目标结束位置.
            actual_end_pos_ms = min(end_pos_ms, audio_length_ms) # 实际结束位置 (不能超过音频总长).

            if actual_end_pos_ms <= current_pos_ms: # 如果没有更多内容可分割.
                logger.debug(f"分段结束: actual_end_pos_ms ({actual_end_pos_ms}) <= current_pos_ms ({current_pos_ms}).")
                break

            # 智能切分逻辑: 在目标分段的末尾区域搜索静音.
            # 条件: 1. 不是最后一个分段. 2. 当前分段长度大于静音搜索窗口.
            if actual_end_pos_ms < audio_length_ms and \
               (actual_end_pos_ms - current_pos_ms) > SILENCE_SEARCH_DURATION_MS:
                
                # 定义搜索静音的区域 (在当前目标分段的尾部).
                search_start_ms = max(current_pos_ms, actual_end_pos_ms - SILENCE_SEARCH_DURATION_MS)
                search_chunk = audio[search_start_ms:actual_end_pos_ms] # 获取该区域的音频数据.
                logger.debug(f"分段 {segment_idx}: 在 {search_start_ms}-{actual_end_pos_ms} ms 范围内搜索静音.")
                try:
                    # 使用 pydub 检测静音.
                    silence_ranges = silence.detect_silence(
                        search_chunk, 
                        min_silence_len=MIN_SILENCE_LEN, 
                        silence_thresh=SILENCE_THRESH
                    )
                    if silence_ranges: # 如果找到静音.
                        # 获取搜索区域内第一个静音段的开始时间 (相对于 search_chunk).
                        first_silence_start_in_search_chunk = silence_ranges[0][0]
                        # 计算该静音点在整个音频中的绝对时间.
                        potential_new_end_ms = search_start_ms + first_silence_start_in_search_chunk
                        # 确保新的结束点不会太靠前 (至少比当前开始点多 MIN_SILENCE_LEN).
                        if potential_new_end_ms > (current_pos_ms + MIN_SILENCE_LEN): 
                            logger.debug(f"分段 {segment_idx}: 在 {potential_new_end_ms} ms 处找到有意义的静音点, 更新结束时间.")
                            actual_end_pos_ms = potential_new_end_ms # 更新实际结束位置.
                        else:
                            logger.debug(f"分段 {segment_idx}: 静音点 ({potential_new_end_ms}ms) 太靠前, 忽略.")
                    else:
                        logger.debug(f"分段 {segment_idx}: 在尾部搜索区域未找到静音点.")
                except Exception as e: # 静音检测时发生错误.
                    logger.warning(f"在分段 {segment_idx} 静音检测时出错: {e}", exc_info=True)
                    if console: progress_bar.console.print(f"\n[bold yellow][WARNING][/bold yellow] 在分段 {segment_idx} 静音检测时出错: {e}")
            
            segment_audio_chunk = audio[current_pos_ms:actual_end_pos_ms] # 获取当前分段的音频数据.
            segment_filename = os.path.join(TEMP_DIR, f"segment_{segment_idx}.wav") # 构建分段文件名.
            
            try:
                segment_audio_chunk.export(segment_filename, format="wav") # 导出分段为 WAV 文件.
                segments_info.append({"file": segment_filename, "start_ms": current_pos_ms})
                logger.info(f"已导出分段 {segment_idx}: {segment_filename}, start_ms: {current_pos_ms}, end_ms: {actual_end_pos_ms}")
            except Exception as e: # 导出文件时发生错误.
                logger.error(f"导出分段 {segment_filename} 失败: {e}", exc_info=True)
                if console: progress_bar.console.print(f"\n[bold red][ERROR][/bold red] 导出分段 {segment_filename} 失败: {e}")
            
            progress_bar.update(segment_task, advance=(actual_end_pos_ms - current_pos_ms)) # 更新进度条.
            current_pos_ms = actual_end_pos_ms # 更新当前处理位置.
            segment_idx += 1
    
    logger.info(f"音频分段完成. 共创建 {len(segments_info)} 个分段.")
    if console: console.print(f"[bold green][INFO][/bold green] 音频分段完成. 共创建 {len(segments_info)} 个分段.")
    
    # 保存分段信息到状态文件.
    status = load_status() # 重新加载状态以防其他地方修改.
    status["segments_info"] = segments_info
    save_status(status)
    
    return segments_info

def load_whisper_model_rich():
    """
    加载 WhisperX 模型. 如果模型已加载, 则返回缓存的实例.
    使用 Rich Progress 显示加载状态.
    返回:
        whisperx.WhisperPipeline: 加载的 WhisperX 模型实例, 或在失败时返回 None.
    """
    logger.info("请求加载 WhisperX 模型.")
    try:
        import whisperx # 确保 whisperx 可用.
        logger.debug("whisperx 导入成功.")
    except ImportError:
        logger.critical("whisperx 未安装, 无法加载模型.", exc_info=True)
        if console: console.print("[bold red][CRITICAL][/bold red] whisperx 库未安装. 请参照教程安装.")
        return None

    global whisper_model_global, WHISPER_MODEL, COMPUTE_TYPE, DEVICE, CPU_THREADS_PER_MODEL_INSTANCE, MODEL_CACHE_DIR
    
    if whisper_model_global is None: # 如果模型尚未加载.
        logger.info(f"全局 Whisper 模型为 None, 开始加载. 模型: {WHISPER_MODEL}, 计算类型: {COMPUTE_TYPE}, 设备: {DEVICE}")
        effective_compute_type = COMPUTE_TYPE
        # 检查 CPU 计算类型是否为推荐类型.
        if DEVICE == "cpu" and COMPUTE_TYPE not in ["int8", "float32", "int16"]:
            msg = f"当前计算类型 '{COMPUTE_TYPE}' 可能不是 CPU 上的最优选择. CPU 通常推荐 'int8', 'float32', 或 'int16'. WhisperX 可能会自动调整或报错."
            logger.warning(msg)
            if console: console.print(f"[bold yellow][WARNING][/bold yellow] {msg}")

        if console:
            console.print(f"[bold blue][INFO][/bold blue] 正在加载 WhisperX 模型 ([cyan]{WHISPER_MODEL}[/cyan], compute: [cyan]{effective_compute_type}[/cyan], device: [cyan]{DEVICE}[/cyan])...")
            console.print(f"[INFO] 首次加载新模型需要从 HuggingFace (镜像源: '{os.environ.get('HF_ENDPOINT', '默认源')}') 下载到 '{MODEL_CACHE_DIR}', 请耐心等待...")
        logger.info(f"模型下载源 (HF_ENDPOINT): {os.environ.get('HF_ENDPOINT', '默认源')}, 模型缓存目录: {MODEL_CACHE_DIR}")

        asr_options = {} # asr_options 用于传递给 transcribe 方法的选项
        model_load_kwargs = {
            "device": DEVICE,
            "compute_type": effective_compute_type,
            "download_root": MODEL_CACHE_DIR, # 指定模型下载和缓存的根目录.
            # "asr_options": asr_options, # asr_options 通常在 transcribe 时指定更合适
        }
        
        if DEVICE == "cpu":
            # threads 参数用于 faster-whisper 模型加载时的 CPU 线程数
            model_load_kwargs["threads"] = CPU_THREADS_PER_MODEL_INSTANCE
            logger.info(f"WhisperX 模型 (faster-whisper) 将使用 {CPU_THREADS_PER_MODEL_INSTANCE} 个 CPU 线程 (通过 'threads' 参数传递给 load_model).")
            if console: console.print(f"[INFO] WhisperX 模型 (faster-whisper) 将使用 [cyan]{CPU_THREADS_PER_MODEL_INSTANCE}[/cyan] 个 CPU 线程 (通过 'threads' 参数).")
        
        # 如果有其他需要传递给 faster-whisper transcribe 时的选项, 可以在这里设置 asr_options
        # 例如: asr_options["beam_size"] = 5
        # 然后在调用 model_instance.transcribe 时传入 asr_options
        # 目前脚本中 transcribe 调用时未使用额外的 asr_options, 所以这里保持为空或不设置
        # 如果确实有需要传递给 load_model 的 asr_options (非 cpu_threads), 则保留下面这行
        if asr_options: 
             model_load_kwargs["asr_options"] = asr_options
            
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
                progress.add_task("加载 Whisper 模型中...", total=None)
                with model_lock: # 使用锁确保线程安全地加载模型.
                    if whisper_model_global is None: # 双重检查锁定模式.
                         logger.info(f"实际执行 whisperx.load_model with kwargs: {model_load_kwargs}")
                         whisper_model_global = whisperx.load_model(WHISPER_MODEL, **model_load_kwargs)
            logger.info("WhisperX 模型加载成功.")
            if console: console.print("[bold green][INFO][/bold green] WhisperX 模型加载成功.")
        except Exception as e:
            logger.error(f"加载 WhisperX 模型失败: {e}", exc_info=True)
            if console:
                console.print(f"[bold red][ERROR][/bold red] 加载 WhisperX 模型失败: {e}")
                console.print(f"请检查: 模型名称 ('{WHISPER_MODEL}'), 计算类型 ('{effective_compute_type}'), 设备 ('{DEVICE}'), CPU 线程数 ('{CPU_THREADS_PER_MODEL_INSTANCE}'), 模型缓存路径 ('{MODEL_CACHE_DIR}').")
                console.print("确保 WhisperX 及相关依赖 (如 PyTorch, faster-whisper, ctranslate2, onnxruntime) 已按教程正确安装且与 CPU 兼容.")
            return None
    else: # 模型已加载, 使用缓存实例.
        logger.info("WhisperX 模型已加载, 使用缓存的实例.")
    return whisper_model_global

def transcribe_and_align_segment(segment_info, model_instance, align_model_cache):
    """
    对单个音频分段进行转录和对齐.
    参数:
        segment_info (dict): 包含分段文件路径和开始时间信息的字典.
        model_instance (whisperx.WhisperPipeline): 已加载的 WhisperX 主模型实例.
        align_model_cache (dict): 用于缓存已加载的对齐模型的字典 (按语言代码).
    返回:
        dict: 包含转录和对齐结果的字典, 或在出错时包含错误信息的字典.
              成功时格式: {"segments": [...], "word_segments": [...] (可选)}
              失败时格式: {"error": "...", "segment_basename": "..."}
    """
    segment_file = segment_info["file"]
    segment_start_ms = segment_info["start_ms"]
    segment_basename = os.path.basename(segment_file) # 用于日志和错误报告.
    logger.info(f"开始处理分段: {segment_basename}, start_ms: {segment_start_ms}")
    
    try:
        import whisperx # 再次确认 whisperx 可用.
    except ImportError: # 这通常不应发生, 因为 load_whisper_model_rich 已检查.
        logger.critical("whisperx 在 transcribe_and_align_segment 中导入失败.", exc_info=True)
        return {"error": "whisperx not installed during segment processing", "segment_basename": segment_basename}

    # 从分段信息中获取预先检测到的语言 (如果有).
    detected_language_for_segment = segment_info.get("detected_language")
    logger.debug(f"分段 {segment_basename}: 使用预检测语言 (如有): {detected_language_for_segment}")

    audio_data = None # 初始化音频数据变量.
    try:
        # 使用 whisperx.load_audio 加载音频, 它内部会尝试 soundfile.
        audio_data = whisperx.load_audio(segment_file)
        logger.debug(f"分段 {segment_basename}: 音频加载成功.")
        
        # --- 转录 ---
        # language 参数: 如果为 None, whisperx 会自动检测语言. 如果提供了, 则使用指定语言.
        transcription_result = model_instance.transcribe(audio_data, batch_size=BATCH_SIZE, language=detected_language_for_segment)
        logger.debug(f"分段 {segment_basename}: 转录完成. 结果片段数: {len(transcription_result.get('segments', [])) if transcription_result else 'N/A'}, 检测/使用语言: {transcription_result.get('language', 'N/A')}")

        if not transcription_result or not transcription_result.get("segments"):
            logger.warning(f"分段 {segment_basename}: 转录结果为空或不包含 'segments'.")
            return None # 返回 None 表示此分段没有有效的转录结果.

        # --- 对齐 ---
        # 确定用于加载对齐模型的语言代码.
        # 如果预检测语言有效, 使用它; 否则使用转录时 whisperx 检测到的语言.
        lang_code_for_align = detected_language_for_segment if detected_language_for_segment else transcription_result["language"]
        logger.debug(f"分段 {segment_basename}: 用于加载对齐模型的语言代码: {lang_code_for_align}")
        
        # 从缓存加载或新建对齐模型.
        align_model, align_metadata = align_model_cache.get(lang_code_for_align, (None, None))
        
        if align_model is None: # 如果缓存中没有此语言的对齐模型.
            logger.info(f"分段 {segment_basename}: 语言 '{lang_code_for_align}' 的对齐模型不在缓存中, 开始加载 (缓存目录: {MODEL_CACHE_DIR}).")
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=lang_code_for_align,
                    device=DEVICE,
                    model_dir=MODEL_CACHE_DIR # 指定对齐模型的下载和缓存目录.
                )
                align_model_cache[lang_code_for_align] = (align_model, align_metadata) # 存入缓存.
                logger.info(f"分段 {segment_basename}: 语言 '{lang_code_for_align}' 的对齐模型加载并缓存成功.")
            except Exception as e:
                logger.error(f"加载语言 '{lang_code_for_align}' 的对齐模型失败: {e}", exc_info=True)
                # 如果对齐模型加载失败, 可以选择是否继续 (只返回转录结果) 或标记为错误.
                # 当前实现: 标记为错误, 因为对齐是流程的一部分.
                return {"error": f"加载对齐模型 ({lang_code_for_align}) 失败: {e}", "segment_basename": segment_basename}
        else:
            logger.debug(f"分段 {segment_basename}: 使用缓存的语言 '{lang_code_for_align}' 对齐模型.")
                
        # 执行对齐.
        aligned_result = whisperx.align(
            transcription_result["segments"], # 注意: whisperx.align 期望的是一个 segments 列表.
            align_model, 
            align_metadata, 
            audio_data, 
            device=DEVICE
        )
        logger.debug(f"分段 {segment_basename}: 对齐完成.")

        # --- 调整时间戳 ---
        # 将对齐结果中的时间戳从相对于分段开始调整为相对于整个音频文件的开始.
        segment_start_sec = segment_start_ms / 1000.0
        final_adjusted_alignment = {"segments": []} # 初始化最终的、调整时间戳后的结果.

        # 处理词级别时间戳 (如果对齐结果中有).
        if "word_segments" in aligned_result and aligned_result["word_segments"] is not None:
            final_adjusted_alignment["word_segments"] = []
            for word_info in aligned_result["word_segments"]:
                if "start" in word_info and isinstance(word_info["start"], (float, int)):
                    word_info["start"] += segment_start_sec
                if "end" in word_info and isinstance(word_info["end"], (float, int)):
                    word_info["end"] += segment_start_sec
                final_adjusted_alignment["word_segments"].append(word_info)
        
        # 处理片段级别时间戳 (这是 whisperx.align 的主要输出).
        # aligned_result 本身就是包含对齐后 segments 的字典.
        for seg in aligned_result["segments"]:
            if "start" in seg and isinstance(seg["start"], (float, int)):
                seg["start"] += segment_start_sec
            if "end" in seg and isinstance(seg["end"], (float, int)):
                seg["end"] += segment_start_sec
            # 如果 aligned_result 的 segments 内部还嵌套了 "words", 也需要调整.
            if "words" in seg and isinstance(seg["words"], list):
                for word_info_in_seg in seg["words"]:
                    if "start" in word_info_in_seg and isinstance(word_info_in_seg["start"], (float, int)):
                        word_info_in_seg["start"] += segment_start_sec
                    if "end" in word_info_in_seg and isinstance(word_info_in_seg["end"], (float, int)):
                        word_info_in_seg["end"] += segment_start_sec
            final_adjusted_alignment["segments"].append(seg)
        
        logger.info(f"分段 {segment_basename}: 成功处理并调整时间戳.")
        return final_adjusted_alignment # 返回调整后的结果.
    except Exception as e: # 捕获转录或对齐过程中发生的其他所有错误.
        logger.error(f"处理分段 {segment_basename} 时发生未知错误: {e}", exc_info=True)
        return {"error": f"未知错误: {str(e)}", "segment_basename": segment_basename}
    finally:
        # 清理音频数据以释放内存.
        if audio_data is not None:
            del audio_data
            gc.collect()
            logger.debug(f"分段 {segment_basename}: 音频数据已清理.")

def process_all_segments(segments_info_list, current_status):
    """
    处理所有音频分段: 进行转录和对齐.
    参数:
        segments_info_list (list): 包含所有分段信息 (文件路径, 开始时间) 的列表.
        current_status (dict): 当前的处理状态, 用于续传和保存中间结果.
    返回:
        list: 包含每个分段处理结果 (或错误信息) 的列表. 如果发生严重错误 (如模型加载失败), 返回 None.
    """
    logger.info(f"开始处理所有 {len(segments_info_list)} 个分段.")
    try:
        import whisperx # 再次确认 whisperx 可用.
    except ImportError:
        logger.critical("whisperx 在 process_all_segments 中导入失败.", exc_info=True)
        if console: console.print("[bold red][CRITICAL][/bold red] whisperx 库未安装. 请参照教程安装.")
        return None # 严重错误, 无法继续.

    model_instance = load_whisper_model_rich() # 加载/获取 Whisper 主模型.
    if model_instance is None:
        logger.error("Whisper 模型未能加载. 中止处理所有分段.")
        # console.print 已在 load_whisper_model_rich 中处理.
        return None # 严重错误.

    processed_results_map = current_status.get("processed_results", {}) # 从状态加载已处理的分段结果.
    all_segment_results = [None] * len(segments_info_list) # 初始化用于存储所有分段结果的列表.
    tasks_to_submit_for_processing = [] # 需要新处理或重试的任务列表.
    already_processed_count = 0 # 已成功处理并可跳过的分段计数.

    # --- 步骤 1: 整体语言检测 (如果尚未进行) ---
    overall_detected_language = current_status.get("detected_language")
    # 条件: 未指定语言, 且有分段, 且第一个分段文件存在.
    if overall_detected_language is None and segments_info_list and os.path.exists(segments_info_list[0]["file"]):
        logger.info("正在使用第一个分段检测音频的整体语言...")
        if console: console.print(f"[bold blue][INFO][/bold blue] 正在使用第一个分段检测音频的整体语言...")
        first_segment_audio = None
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress_lang:
                progress_lang.add_task("语言检测中...", total=None)
                first_segment_audio = whisperx.load_audio(segments_info_list[0]["file"])
                # 使用 model_instance.transcribe 来获取语言信息, 因为它更可靠.
                # 注意: transcribe 可能会比较耗时, 但为了准确的语言代码, 这是值得的.
                # 如果 whisperx 有单独的 detect_language API 且可靠, 可以考虑使用.
                # 当前 whisperx.WhisperPipeline 没有独立的 detect_language.
                initial_transcription_result = model_instance.transcribe(first_segment_audio, batch_size=BATCH_SIZE) 
            
            if initial_transcription_result and "language" in initial_transcription_result:
                overall_detected_language = initial_transcription_result["language"]
                logger.info(f"音频整体语言已确定为: {overall_detected_language}")
                if console: console.print(f"[bold green][INFO][/bold green] 音频整体语言已确定为: [cyan]{overall_detected_language}[/cyan]")
                current_status["detected_language"] = overall_detected_language
                save_status(current_status) # 保存检测到的语言到状态文件.
            else:
                logger.warning("无法从第一个分段确定音频语言. 后续转录时将由模型自动判断各分段语言.")
                if console: console.print(f"[bold yellow][WARNING][/bold yellow] 无法确定音频语言. 转录时将由模型自动判断各分段语言.")
        except Exception as e:
            logger.warning(f"初始语言检测失败: {e}. 后续转录时将由模型自动判断各分段语言.", exc_info=True)
            if console: console.print(f"[bold yellow][WARNING][/bold yellow] 初始语言检测失败: {e}. 转录时将由模型自动判断各分段语言.")
        finally:
            if first_segment_audio is not None: 
                del first_segment_audio; gc.collect()
                logger.debug("用于语言检测的第一个分段音频数据已清理.")

    # 将整体检测到的语言代码应用于所有分段信息 (如果分段尚未指定语言).
    if overall_detected_language:
        logger.info(f"将整体检测到的语言 '{overall_detected_language}' 应用于所有未指定特定语言的分段.")
        for seg_info in segments_info_list:
            if "detected_language" not in seg_info or seg_info["detected_language"] is None:
                seg_info["detected_language"] = overall_detected_language
    
    # --- 步骤 2: 预加载对齐模型 (如果整体语言已知) ---
    alignment_model_cache = {} # 用于缓存对齐模型, key 为语言代码.
    if overall_detected_language:
        logger.info(f"正在为语言 '{overall_detected_language}' 预加载对齐模型 (设备: {DEVICE}, 缓存目录: {MODEL_CACHE_DIR})...")
        if console:
            console.print(f"[bold blue][INFO][/bold blue] 正在为语言 [cyan]{overall_detected_language}[/cyan] 预加载对齐模型 (设备: {DEVICE})...")
            console.print(f"[INFO] 对齐模型将通过 HuggingFace (镜像源: '{os.environ.get('HF_ENDPOINT', '默认源')}') 下载, 并缓存到 '{MODEL_CACHE_DIR}'.")
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress_align:
                progress_align.add_task(f"加载 {overall_detected_language} 对齐模型中...", total=None)
                with model_lock: # 确保线程安全.
                    # 检查缓存中是否已存在 (理论上首次预加载不会).
                    if overall_detected_language not in alignment_model_cache:
                        align_model, align_metadata = whisperx.load_align_model(
                            language_code=overall_detected_language, 
                            device=DEVICE,
                            model_dir=MODEL_CACHE_DIR # 指定缓存目录.
                        )
                        alignment_model_cache[overall_detected_language] = (align_model, align_metadata)
            logger.info(f"语言 '{overall_detected_language}' 的对齐模型已预加载并缓存.")
            if console: console.print(f"[bold green][INFO][/bold green] 语言 [cyan]{overall_detected_language}[/cyan] 的对齐模型已预加载.")
        except Exception as e:
            logger.warning(f"为语言 '{overall_detected_language}' 预加载对齐模型失败: {e}. 后续处理中将按需加载.", exc_info=True)
            if console: console.print(f"[bold yellow][WARNING][/bold yellow] 为语言 '{overall_detected_language}' 预加载对齐模型失败: {e}. 后续处理中将按需加载.")

    # --- 步骤 3: 筛选需要处理的分段 ---
    for idx, seg_info in enumerate(segments_info_list):
        idx_str = str(idx) # JSON key 必须是字符串.
        # 检查此分段是否已在状态中成功处理过.
        if idx_str in processed_results_map and \
           processed_results_map[idx_str] is not None and \
           not ("error" in processed_results_map[idx_str] and processed_results_map[idx_str]["error"]): # 确保 error 字段不存在或其值为假.
            all_segment_results[idx] = processed_results_map[idx_str] # 使用缓存的结果.
            already_processed_count += 1
            logger.debug(f"分段 {idx}: 使用已缓存的成功结果.")
        else: # 如果未处理或先前处理失败.
            if os.path.exists(seg_info["file"]): # 再次确认分段文件存在.
                tasks_to_submit_for_processing.append((idx, seg_info)) # 加入待处理列表.
                logger.debug(f"分段 {idx} (文件: {seg_info['file']}): 添加到待处理列表.")
            else: # 如果分段文件丢失.
                logger.warning(f"分段文件 {seg_info['file']} (索引 {idx}) 未找到. 跳过此分段.")
                if console: console.print(f"[bold yellow][WARNING][/bold yellow] 分段文件 {seg_info['file']} (索引 {idx}) 未找到. 跳过.")
                error_info = {"error": f"分段文件 {os.path.basename(seg_info.get('file', '未知文件'))} 未找到.", "segment_basename": os.path.basename(seg_info.get("file", f"segment_{idx}_NA"))}
                all_segment_results[idx] = error_info # 记录错误.
                processed_results_map[idx_str] = error_info # 更新状态.

    # 如果所有分段都已成功处理 (无需新任务).
    if not tasks_to_submit_for_processing and already_processed_count == len(segments_info_list):
        logger.info("所有分段先前均已成功处理, 无需新的转录任务.")
        if console: console.print("[bold blue][INFO][/bold blue] 所有分段先前均已成功处理.")
        save_status(current_status) # 确保状态是最新的.
        return all_segment_results

    # --- 步骤 4: 使用线程池并行处理分段 ---
    cpu_core_count = os.cpu_count() or 1 # 获取 CPU 核心数 (至少为1).
    # 计算线程池的工作线程数 (num_workers).
    # 每个 Whisper 模型实例 (faster-whisper) 内部会使用 CPU_THREADS_PER_MODEL_INSTANCE 个线程.
    # 因此, 线程池的并发任务数应受此限制, 以避免 CPU 过度竞争.
    threads_per_task = CPU_THREADS_PER_MODEL_INSTANCE if CPU_THREADS_PER_MODEL_INSTANCE > 0 else 1
    num_workers = max(1, cpu_core_count // threads_per_task) 
    num_workers = min(num_workers, 4) # 额外限制: 最大并发任务数不超过 4, 以免内存和 I/O 瓶颈.
                                      # 这个限制可以根据实际情况调整.
    
    logger.info(f"准备转录 {len(tasks_to_submit_for_processing)} 个新的/失败的分段 (共 {len(segments_info_list)} 个分段). "
                f"使用 {num_workers} 个 Python 工作线程 (并发任务). 每个任务内部 (faster-whisper) 将使用 {CPU_THREADS_PER_MODEL_INSTANCE} 个 CPU 线程.")
    if console:
        console.print(f"[bold blue][INFO][/bold blue] 准备转录 [cyan]{len(tasks_to_submit_for_processing)}[/cyan] 个新的/失败的分段 (总计 [cyan]{len(segments_info_list)}[/cyan] 个).")
        console.print(f"[INFO] 使用 [cyan]{num_workers}[/cyan] 个 Python 工作线程进行并发处理.")
        console.print(f"[INFO] 每个转录任务内部 (faster-whisper) 将尝试使用 [cyan]{CPU_THREADS_PER_MODEL_INSTANCE}[/cyan] 个 CPU 线程.")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("[cyan]{task.completed}/{task.total}[/cyan] 段"), TimeElapsedColumn(), TimeRemainingColumn(),
                  console=console, transient=False, refresh_per_second=1) as progress_bar:
        
        transcribe_task = progress_bar.add_task("转录进度", total=len(segments_info_list), completed=already_processed_count)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 构建 future 到原始索引的映射, 以便在任务完成时能对应到正确的位置.
            future_to_idx_map = {
                executor.submit(transcribe_and_align_segment, seg_info_item, model_instance, alignment_model_cache): original_idx 
                for original_idx, seg_info_item in tasks_to_submit_for_processing
            }
            
            for future in as_completed(future_to_idx_map): # 按任务完成顺序处理结果.
                original_idx = future_to_idx_map[future] # 获取此任务对应的原始分段索引.
                idx_str = str(original_idx)
                # 获取分段基本名用于错误报告 (即使 seg_info 有问题也能提供一些信息).
                segment_basename_for_error = os.path.basename(segments_info_list[original_idx].get("file", f"segment_{original_idx}_NA"))
                try:
                    segment_result = future.result() # 获取任务的执行结果.
                    if segment_result is not None:
                        # 检查结果中是否有 error 标记且其值为真.
                        if "error" in segment_result and segment_result["error"]:
                            logger.warning(f"分段 {original_idx} ({segment_result.get('segment_basename', segment_basename_for_error)}) 处理时返回错误: {segment_result['error']}")
                            if console: progress_bar.console.print(f"\n[bold yellow][WARNING][/bold yellow] 分段 {original_idx} ({segment_result.get('segment_basename', segment_basename_for_error)}) 处理错误: {segment_result['error']}")
                            all_segment_results[original_idx] = segment_result # 存储错误结果.
                            processed_results_map[idx_str] = segment_result
                        else: # 处理成功.
                            logger.info(f"分段 {original_idx} ({segment_basename_for_error}) 处理成功.")
                            all_segment_results[original_idx] = segment_result # 存储成功结果.
                            processed_results_map[idx_str] = segment_result
                    else: # 如果任务返回 None (表示该分段无有效转录结果).
                        logger.warning(f"分段 {original_idx} ({segment_basename_for_error}) 未能成功产生结果 (返回 None).")
                        if console: progress_bar.console.print(f"\n[bold yellow][WARNING][/bold yellow] 分段 {original_idx} ({segment_basename_for_error}) 未能成功产生结果 (返回 None).")
                        error_info = {"error": "No result returned from processing", "segment_basename": segment_basename_for_error}
                        all_segment_results[original_idx] = error_info
                        processed_results_map[idx_str] = error_info
                    
                    # 每次有任务完成 (成功或失败), 都更新并保存状态.
                    current_status["processed_results"] = processed_results_map
                    save_status(current_status)
                except Exception as exc: # 捕获线程中未被 transcribe_and_align_segment 内部捕获的意外异常.
                    logger.error(f'分段 {original_idx} ({segment_basename_for_error}) 执行时产生意外异常: {exc}', exc_info=True)
                    if console: progress_bar.console.print(f'\n[bold red][ERROR][/bold red] 分段 {original_idx} ({segment_basename_for_error}) 执行时产生意外异常: {exc}')
                    error_info = {"error": f"意外异常: {str(exc)}", "segment_basename": segment_basename_for_error, "unexpected_exception": True}
                    all_segment_results[original_idx] = error_info
                    processed_results_map[idx_str] = error_info
                    current_status["processed_results"] = processed_results_map
                    save_status(current_status)
                finally:
                    progress_bar.update(transcribe_task, advance=1) # 更新总进度条.
    
    # --- 步骤 5: 完成后总结 ---
    successful_count = sum(1 for r in all_segment_results if r is not None and not ("error" in r and r["error"]))
    failed_count = len(segments_info_list) - successful_count
    if failed_count > 0:
        logger.warning(f"所有分段处理完成. 成功: {successful_count}, 失败: {failed_count}.")
        if console: console.print(f"[bold yellow][WARNING][/bold yellow] 所有分段处理完成. 成功: [cyan]{successful_count}[/cyan], 失败: [cyan]{failed_count}[/cyan].")
    else:
        logger.info("所有分段均已成功转录和对齐.")
        if console: console.print("[bold green][INFO][/bold green] 所有分段均已成功转录和对齐.")
    
    del alignment_model_cache; gc.collect() # 清理对齐模型缓存以释放内存.
    logger.debug("对齐模型缓存已清理.")
    return all_segment_results

def format_timestamp(seconds_float):
    """
    将浮点数表示的秒数格式化为 SRT 时间戳字符串 (HH:MM:SS,mmm).
    参数:
        seconds_float (float | int): 秒数.
    返回:
        str: SRT 格式的时间戳.
    """
    # 对输入进行更严格的检查, 确保是数字且非负.
    if seconds_float is None or not isinstance(seconds_float, (int, float)) or seconds_float < 0:
        logger.debug(f"格式化时间戳时遇到无效输入: {seconds_float}, 将其视为 0.0.")
        seconds_float = 0.0 # 默认为 0.
    
    total_seconds_int = int(seconds_float) # 取整数秒.
    milliseconds = int(round((seconds_float - total_seconds_int) * 1000)) # 计算毫秒部分并四舍五入.
    
    # 计算时、分、秒.
    hours, remainder = divmod(total_seconds_int, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 处理毫秒和秒的进位.
    if milliseconds >= 1000: 
        seconds += milliseconds // 1000
        milliseconds %= 1000
    if seconds >= 60:
        minutes += seconds // 60
        seconds %= 60
    if minutes >= 60: # 虽然罕见 (对于单个字幕条目), 但完整处理.
        hours += minutes // 60
        minutes %= 60
        
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt(all_transcription_results, srt_output_path, use_word_timestamps=False):
    """
    根据转录和对齐结果生成 SRT 字幕文件.
    参数:
        all_transcription_results (list): 包含所有分段处理结果的列表.
        srt_output_path (str): 输出 SRT 文件的路径.
        use_word_timestamps (bool): 是否使用词级别时间戳.
    返回:
        bool: SRT 文件成功生成 (即使内容为空) 返回 True, 写入失败返回 False.
    """
    logger.info(f"开始生成 SRT 文件: {srt_output_path}, 使用词级别时间戳: {use_word_timestamps}")
    if console: console.print("[bold blue][INFO][/bold blue] 正在生成 SRT 字幕文件...")
    srt_content_lines = []
    subtitle_entry_counter = 1

    if all_transcription_results is None: # 检查整体结果列表是否为 None.
        logger.error("转录结果列表为 None, 无法生成 SRT.")
        if console: console.print("[bold red][ERROR][/bold red] 转录结果列表为 None, 无法生成 SRT.")
        return False

    for i, single_segment_result in enumerate(all_transcription_results):
        # 跳过处理失败或为空 (None) 的单个分段结果.
        if single_segment_result is None or \
           ("error" in single_segment_result and single_segment_result["error"]): # 确保 error 字段存在且为真.
            logger.warning(f"跳过为 SRT 生成内容的分段 {i} (原因: 处理失败或结果为空). "
                           f"错误信息 (如有): {single_segment_result.get('error', 'N/A') if isinstance(single_segment_result, dict) else '结果为 None'}")
            continue

        segments_to_process_for_srt = [] # 存储当前分段中要写入 SRT 的条目.
        # 根据配置选择使用词级别还是片段级别时间戳.
        if use_word_timestamps and "word_segments" in single_segment_result and single_segment_result["word_segments"]:
            logger.debug(f"SRT 生成: 分段 {i} 使用词级别时间戳.")
            for word_info in single_segment_result["word_segments"]:
                start_time = word_info.get("start")
                end_time = word_info.get("end")
                text = word_info.get("word", "").strip() # 使用 .get 获取, 避免 KeyError.
                # 确保时间戳有效 (start < end) 且文本非空.
                if start_time is not None and end_time is not None and text and end_time > start_time:
                    segments_to_process_for_srt.append({"start": start_time, "end": end_time, "text": text})
        # 默认或当词级别时间戳不可用/未启用时, 使用片段 (句子/短语) 级别时间戳.
        elif "segments" in single_segment_result and single_segment_result["segments"]:
            logger.debug(f"SRT 生成: 分段 {i} 使用句子/短语级别时间戳.")
            for seg_data in single_segment_result["segments"]:
                start_time = seg_data.get("start")
                end_time = seg_data.get("end")
                text = seg_data.get("text", "").strip()
                if start_time is not None and end_time is not None and text and end_time > start_time:
                    segments_to_process_for_srt.append({"start": start_time, "end": end_time, "text": text})
        
        # 将筛选出的条目格式化并添加到 srt_content_lines.
        for srt_entry_data in segments_to_process_for_srt:
            start_time_sec, end_time_sec, text_content = srt_entry_data["start"], srt_entry_data["end"], srt_entry_data["text"]
            
            # 再次确认时间戳有效性和文本内容 (理论上上面已过滤, 此处为保险).
            if end_time_sec <= start_time_sec or not text_content:
                logger.debug(f"SRT 生成: 跳过无效的字幕条目: start={start_time_sec}, end={end_time_sec}, text='{text_content}'")
                continue
            
            srt_content_lines.append(str(subtitle_entry_counter))
            srt_content_lines.append(f"{format_timestamp(start_time_sec)} --> {format_timestamp(end_time_sec)}")
            srt_content_lines.append(text_content)
            srt_content_lines.append("") # SRT 条目之间需要一个空行.
            subtitle_entry_counter += 1

    if not srt_content_lines: # 如果没有生成任何有效的字幕行.
        logger.warning("SRT 生成: 未生成任何有效的字幕行. SRT 文件将为空或只包含 BOM (如果写入).")
        if console: console.print("[bold yellow][WARNING][/bold yellow] 未生成任何有效的字幕行. SRT 文件内容将为空.")
        # 即使内容为空, 也尝试创建文件, 这表示一个“技术上成功”的生成过程.
        try:
            with open(srt_output_path, "w", encoding="utf-8") as f:
                # f.write('\ufeff') # 可选: 写入 UTF-8 BOM (字节顺序标记), 某些播放器可能需要.
                pass # 创建一个空文件.
            logger.info(f"SRT 文件已生成但内容为空: {srt_output_path}")
            return True # 表示文件创建成功.
        except IOError as e: # 处理写入空文件时的错误.
            logger.error(f"写入空的 SRT 文件 '{srt_output_path}' 失败: {e}", exc_info=True)
            if console: console.print(f"[bold red][ERROR][/bold red] 写入空的 SRT 文件 '{srt_output_path}' 失败: {e}")
            return False


    try: # 写入包含内容的 SRT 文件.
        with open(srt_output_path, "w", encoding="utf-8") as f:
            # f.write('\ufeff') # 可选: 写入 UTF-8 BOM.
            f.write("\n".join(srt_content_lines))
        
        abs_srt_path = os.path.abspath(srt_output_path) # 获取绝对路径用于日志和链接.
        logger.info(f"SRT 字幕文件已成功生成: {abs_srt_path}")
        if console: console.print(f"[bold green][SUCCESS][/bold green] SRT 字幕文件已生成: [link=file://{abs_srt_path}]{srt_output_path}[/link]")
        return True
    except IOError as e: # 处理文件写入错误.
        logger.error(f"写入 SRT 文件 '{srt_output_path}' 失败: {e}", exc_info=True)
        if console: console.print(f"[bold red][ERROR][/bold red] 写入 SRT 文件 '{srt_output_path}' 失败: {e}")
        return False

def get_file_path_with_dialog():
    """
    使用 Tkinter 文件对话框让用户选择媒体文件.
    在调用 Tkinter 前后会尝试重定向和恢复 stderr, 以避免 Tkinter 的一些内部消息干扰控制台.
    返回:
        str: 用户选择的文件路径, 或在取消/失败时返回空字符串.
    """
    logger.info("尝试打开文件选择对话框 (Tkinter).")
    selected_file_path = ""
    # 用于 stderr 重定向的变量.
    original_stderr_fileno = -1 
    saved_stderr_fd_copy = -1
    dev_null_fd = -1

    try:
        # 步骤 1: 安全地获取当前 stderr 的文件描述符.
        try:
            current_stderr_stream = sys.stderr
            if hasattr(current_stderr_stream, 'fileno') and callable(current_stderr_stream.fileno):
                original_stderr_fileno = current_stderr_stream.fileno()
                if not isinstance(original_stderr_fileno, int): # 如果 fileno() 返回非整数.
                    original_stderr_fileno = sys.__stderr__.fileno() # 退回到原始 stderr.
            else: # 如果 stderr 没有 fileno 方法 (例如某些自定义流).
                original_stderr_fileno = sys.__stderr__.fileno()
        except (AttributeError, io.UnsupportedOperation, ValueError) as e_fileno:
             # 如果所有尝试都失败, 默认使用标准错误的文件描述符 2.
            original_stderr_fileno = 2 
            logger.warning(f"获取 stderr.fileno() 失败 (错误: {e_fileno}), 默认使用 fd 2.")
        logger.debug(f"用于 Tkinter 重定向的原始 stderr 文件描述符: {original_stderr_fileno}")

        # 步骤 2: 复制原始 stderr 文件描述符, 并将 stderr 重定向到 /dev/null (或 NUL).
        saved_stderr_fd_copy = os.dup(original_stderr_fileno) # 复制 fd.
        dev_null_path = os.devnull # 获取平台特定的 null 设备路径.
        dev_null_fd = os.open(dev_null_path, os.O_RDWR) # 打开 null 设备.
        os.dup2(dev_null_fd, original_stderr_fileno) # 将 stderr 指向 null 设备.
        logger.debug(f"stderr 已重定向到 {dev_null_path} 以进行 Tkinter 文件对话框操作.")

        # 步骤 3: 调用 Tkinter 文件对话框.
        import tkinter as tk # 动态导入 tkinter, 仅在此函数中使用.
        from tkinter import filedialog
        os.environ['TK_SILENCE_DEPRECATION'] = '1' # 尝试静默 macOS 上的 Tk 版本警告.
        root = tk.Tk() # 创建 Tkinter 根窗口.
        root.withdraw() # 隐藏主窗口, 只显示对话框.
        if console: console.print("[bold blue][INFO][/bold blue] 请在弹出的对话框中选择要处理的视频或音频文件...")
        
        dialog_selected_path = filedialog.askopenfilename(
            title="选择视频或音频文件",
            filetypes=[ # 定义文件类型过滤器.
                ("媒体文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.mp3 *.wav *.flac *.m4a *.aac *.ogg"),
                ("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("音频文件", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg"),
                ("所有文件", "*.*")
            ]
        )
        root.destroy() # 完成后销毁 Tkinter 根窗口.
        selected_file_path = dialog_selected_path if dialog_selected_path else "" # 如果用户取消则为空字符串.
        if selected_file_path:
            logger.info(f"用户通过文件对话框选择了文件: {selected_file_path}")
        else:
            logger.info("用户取消了文件选择对话框, 或未选择任何文件.")

    except Exception as e: # 捕获所有可能的异常, 包括 Tkinter 初始化失败.
        logger.error(f"打开文件选择对话框时发生错误: {e}", exc_info=True)
        # 即使在异常情况下, 也尝试恢复 stderr.
        if saved_stderr_fd_copy != -1 and original_stderr_fileno != -1:
            try: 
                os.dup2(saved_stderr_fd_copy, original_stderr_fileno)
                logger.debug("stderr 在 Tkinter 异常处理中尝试恢复.")
            except OSError as oe_restore:
                # 如果恢复失败, 记录严重错误到原始 stderr (如果可能) 和日志.
                sys.__stderr__.write(f"CRITICAL: 在 Tkinter 异常处理中恢复 stderr 失败: {oe_restore}\n")
                logger.critical(f"在 Tkinter 异常处理中恢复 stderr 失败: {oe_restore}")
        
        if console:
            console.print(f"[bold red][ERROR][/bold red] 打开文件选择对话框时出错: {e}")
            console.print("[INFO] 如果无法使用图形化文件选择器, 请考虑修改脚本以接受命令行参数或手动输入文件路径.")
        selected_file_path = "" # 确保出错时返回空字符串.
    finally:
        # 步骤 4: 在 finally 块中确保关闭所有打开的文件描述符并恢复 stderr.
        if dev_null_fd != -1: # 关闭指向 null 设备的 fd.
            try: os.close(dev_null_fd)
            except OSError: pass # 忽略关闭错误.
        
        if saved_stderr_fd_copy != -1 and original_stderr_fileno != -1: # 恢复 stderr.
            try:
                os.dup2(saved_stderr_fd_copy, original_stderr_fileno)
                logger.debug("stderr 已在 finally 块中成功恢复.")
            except OSError as oe_final_restore:
                 # 如果最终恢复失败, 记录严重错误.
                 sys.__stderr__.write(f"CRITICAL: 在 finally 块中恢复 stderr 失败: {oe_final_restore}\n")
                 logger.critical(f"在 finally 块中恢复 stderr 失败: {oe_final_restore}")
        
        if saved_stderr_fd_copy != -1: # 关闭复制的原始 stderr fd.
            try: os.close(saved_stderr_fd_copy)
            except OSError: pass
            
    return selected_file_path

def handle_import_and_process():
    """处理导入媒体文件并生成字幕的整个流程."""
    logger.info("开始“导入媒体文件并生成字幕”流程.")
    global WHISPER_MODEL, COMPUTE_TYPE # 确保引用全局配置.
    
    selected_input_file = ""
    # --- 文件选择循环 ---
    while True:
        selected_input_file = get_file_path_with_dialog() # 获取用户选择的文件.
        if not selected_input_file: # 如果用户取消或对话框出错.
            logger.info("未选择文件或文件选择被取消/失败.")
            if not Confirm.ask("未选择文件. 是否重试选择文件?", default=True, console=console):
                logger.info("用户选择不重试文件选择, 返回主菜单.")
                if console: console.print("[bold blue][INFO][/bold blue] 返回主菜单.")
                return # 退出当前处理流程.
            else:
                logger.info("用户选择重试文件选择.")
                continue # 继续下一次文件选择循环.
        
        # 验证选择的是否是文件 (而不是目录等).
        if not os.path.isfile(selected_input_file):
            logger.warning(f"用户选择的路径不是一个有效的文件: '{selected_input_file}'.")
            if console:
                rich_error_msg = Text.assemble(("选择的路径 '", "yellow"), (selected_input_file, "cyan"), ("' 不是一个有效的文件.\n是否重试选择文件?", "yellow"))
                if not Confirm.ask(rich_error_msg, default=True, console=console):
                    logger.info("用户选择不重试 (因路径无效), 返回主菜单.")
                    console.print("[bold blue][INFO][/bold blue] 返回主菜单.")
                    return # 退出当前处理流程.
                else:
                    logger.info("用户选择重试文件选择 (因路径无效).")
                    continue # 继续下一次文件选择循环.
        else: # 文件有效.
            logger.info(f"用户已选择有效文件: {selected_input_file}")
            if console: console.print(f"[bold blue][INFO][/bold blue] 已选择文件: [cyan]{selected_input_file}[/cyan]")
            break # 跳出文件选择循环.

    # --- 准备处理 ---
    input_filename_base = os.path.basename(selected_input_file) # 获取文件名 (不含路径).
    srt_output_filepath = os.path.splitext(selected_input_file)[0] + ".srt" # 构建 SRT 输出路径.
    temp_audio_filepath = os.path.join(TEMP_DIR, "audio.wav") # 标准化后的音频文件的临时路径.
    logger.info(f"输入文件名: {input_filename_base}, SRT 输出路径: {srt_output_filepath}, 临时音频路径: {temp_audio_filepath}")

    status_data = load_status() # 加载先前的处理状态.
    force_audio_extraction, force_audio_split = False, False # 默认不强制重新处理.

    # --- 检查是否可以从先前状态继续 (断点续传逻辑) ---
    # 条件: 输入文件名, 使用的模型, 计算类型都与上次相同.
    if status_data.get("input_file") == input_filename_base and \
       status_data.get("model_used") == WHISPER_MODEL and \
       status_data.get("compute_type_used") == COMPUTE_TYPE:
        logger.info(f"找到与当前文件 '{input_filename_base}' (模型: {WHISPER_MODEL}, 计算类型: {COMPUTE_TYPE}) 相关的先前处理状态.")
        if console: console.print(f"[bold blue][INFO][/bold blue] 找到与当前文件 '{input_filename_base}' (模型: {WHISPER_MODEL}, 计算类型: {COMPUTE_TYPE}) 相关的先前处理状态.")
        if not Confirm.ask("是否尝试从上次中断处继续处理?", default=True, console=console):
            logger.info("用户选择不继续处理, 将重新开始所有步骤.")
            if console: console.print("[bold blue][INFO][/bold blue] 将重新开始处理 (清理先前状态和临时文件).")
            status_data = {} # 清空状态.
            force_audio_extraction = True # 强制重新提取音频.
            force_audio_split = True    # 强制重新分割音频.
            cleanup_temp()              # 清理临时文件目录.
            ensure_app_dirs()           # 确保临时目录等已创建.
        else:
            logger.info("用户选择从上次中断处继续处理.")
    else: # 如果输入文件、模型或计算类型与上次不同, 或无先前状态.
        if status_data: # 如果存在旧状态但不匹配.
            logger.info("输入文件、模型或计算类型已更改, 或无先前有效状态. 开始新的处理流程.")
            if console: console.print("[bold blue][INFO][/bold blue] 输入文件、模型或计算类型已更改. 开始新的处理 (清理先前状态和临时文件).")
        else: # 完全没有先前状态.
            logger.info("无先前处理状态. 开始新的处理流程.")
        status_data = {} # 清空/初始化状态.
        force_audio_extraction = True
        force_audio_split = True
        cleanup_temp()
        ensure_app_dirs()

    # 更新当前处理的状态信息并保存.
    status_data["input_file"] = input_filename_base
    status_data["model_used"] = WHISPER_MODEL 
    status_data["compute_type_used"] = COMPUTE_TYPE
    # 可以考虑加入更多状态信息, 如脚本版本, 时间戳等.
    save_status(status_data)
    logger.info(f"当前处理状态已更新并保存: input_file='{input_filename_base}', model_used='{WHISPER_MODEL}', compute_type_used='{COMPUTE_TYPE}'")

    # --- 步骤 1: 提取音频 ---
    logger.info("流程步骤 1: 提取音频")
    if not extract_audio(selected_input_file, temp_audio_filepath, force_audio_extraction):
        logger.error("音频提取失败. 中止当前文件的处理流程.")
        # console.print 已在 extract_audio 中处理.
        return # 中止流程.

    # --- 步骤 2: 音频分段 ---
    logger.info("流程步骤 2: 音频分段")
    audio_segments = split_audio(temp_audio_filepath, force_audio_split)
    if not audio_segments: # 如果分段失败或列表为空 (没有有效分段).
        logger.error("音频分段失败或未产生任何有效分段. 中止当前文件的处理流程.")
        # console.print 已在 split_audio 中处理 (如果 pydub 未安装).
        if console and not audio_segments: # 如果返回空列表但无特定错误打印.
             console.print("[bold red][ERROR][/bold red] 音频分段失败或未产生任何有效分段.")
        return # 中止流程.

    # --- 步骤 3: 转录和对齐所有分段 ---
    logger.info("流程步骤 3: 转录和对齐所有分段")
    transcription_results = process_all_segments(audio_segments, status_data) # 传递当前状态用于续传.
    if transcription_results is None: # 如果模型加载等严重错误导致返回 None.
        logger.error("转录和对齐过程中发生严重错误 (例如 Whisper 模型未能加载). 中止当前文件的处理流程.")
        # console.print 已在 process_all_segments 或其调用的函数中处理.
        return # 中止流程.
    
    # 检查是否有任何分段成功处理.
    successful_segment_count = sum(1 for r in transcription_results if r is not None and not ("error" in r and r["error"]))
    if successful_segment_count == 0:
        logger.error("所有音频分段均未能成功转录. 无法生成 SRT 文件.")
        if console: console.print("[bold red][ERROR][/bold red] 所有音频分段均未能成功转录. 无法生成 SRT 文件.")
        return # 中止流程.
    elif successful_segment_count < len(audio_segments): # 如果部分分段处理失败.
        logger.warning(f"处理完成, 但只有 {successful_segment_count}/{len(audio_segments)} 个分段成功转录.")
        if console: console.print(f"[bold yellow][WARNING][/bold yellow] 处理完成, 但只有 {successful_segment_count}/{len(audio_segments)} 个分段成功转录.")
        # 询问用户是否仍要根据已成功的部分生成 SRT. 默认改为 True, 因为部分结果也好过没有.
        if not Confirm.ask("是否仍要根据已成功处理的部分生成 SRT 文件?", default=True, console=console):
            logger.info("用户选择不生成部分 SRT 文件.")
            if console: console.print("[bold blue][INFO][/bold blue] 已取消生成 SRT 文件.")
            return # 中止流程.
        logger.info("用户同意根据已成功处理的部分生成 SRT 文件.")
            
    # --- 步骤 4: 生成 SRT 文件 ---
    logger.info("流程步骤 4: 生成 SRT 文件")
    if not generate_srt(transcription_results, srt_output_filepath, USE_WORD_TIMESTAMPS_CONFIG):
        logger.warning("SRT 文件生成过程完成, 但可能未成功写入内容或内容为空.")
        # console.print 已在 generate_srt 中处理.
    
    # --- 步骤 5: 清理临时文件 (可选) ---
    logger.info("流程步骤 5: 清理临时文件 (可选)")
    if Confirm.ask("是否清理本次处理产生的临时文件 (包括分段和状态文件)?", default=True, console=console):
        logger.info("用户选择清理临时文件.")
        cleanup_temp() # 清理 TEMP_DIR.
        # 注意: cleanup_temp 会删除 status.json. 如果希望保留 input_file, model_used 等顶级状态,
        # 则不应在每次成功处理后都删除 status.json, 或 status.json 应存放在 TEMP_DIR 之外.
        # 当前设计: status.json 在 TEMP_DIR 中, 所以会被一起清理.
    else:
        logger.info("用户选择保留临时文件.")
        if console: console.print(f"[bold blue][INFO][/bold blue] 保留临时文件. 您可以稍后在 '{TEMP_DIR}' 手动删除, 或在下次处理不同文件时它们可能会被自动清理.")
    
    logger.info("“导入媒体文件并生成字幕”流程结束.")
    if console: console.print("\n[bold blue][INFO][/bold blue] 当前文件处理流程结束. 返回主菜单.")

def handle_dependencies_check_ui():
    """处理“检查与管理依赖项”菜单选项的用户界面和逻辑."""
    logger.info("进入“检查与管理依赖项”界面.")
    if console: console.print(Panel(Text("依赖检查与管理", justify="center", style="bold cyan"), box=ROUNDED, expand=False))
    
    logger.info("自动执行依赖项检查 (详细模式).")
    dependencies_ok = check_dependencies(verbose=True) # 执行检查并获取结果.

    if not dependencies_ok:
        if console:
            console.print("\n[bold red]重要提示:[/bold red] 检测到依赖项问题 (详情见上).")
            console.print("强烈建议您在继续操作前, 先根据安装教程解决这些依赖问题, 以确保程序正常运行.")
        # 即使检查失败, 也允许用户返回主菜单, 但给出强烈建议.
        # if not Confirm.ask("是否仍要返回主菜单 (不推荐, 除非您知道如何手动解决依赖问题)?", default=False, console=console):
        #     logger.info("用户选择停留在依赖检查提示页面 (或选择退出).")
        #     if console: console.print("[INFO] 请关闭程序并参照教程解决依赖问题, 或选择返回主菜单以尝试其他操作.")
            # sys.exit(1) # 或者不直接退出, 让用户决定.
        # else:
        #     logger.info("用户选择忽略依赖问题警告并返回主菜单.")
    
    logger.info("“检查与管理依赖项”界面结束.")
    if console: console.print("\n[bold blue][INFO][/bold blue] 返回主菜单.")

def handle_model_selection_ui():
    """处理“配置 Whisper 模型与参数”菜单选项的用户界面和逻辑."""
    logger.info("进入“配置 Whisper 模型与参数”界面.")
    global WHISPER_MODEL, COMPUTE_TYPE, BATCH_SIZE, whisper_model_global, CPU_THREADS_PER_MODEL_INSTANCE
    
    if console: console.print(Panel(Text("Whisper 模型与参数配置 (CPU)", justify="center", style="bold cyan"), box=ROUNDED, expand=False))
    
    # 显示当前配置.
    settings_table = Table(title="当前模型设置", box=ROUNDED, show_lines=True)
    settings_table.add_column("参数", style="magenta", no_wrap=True)
    settings_table.add_column("当前值", style="green")
    settings_table.add_row("Whisper 模型 (WHISPER_MODEL)", WHISPER_MODEL)
    settings_table.add_row("计算类型 (COMPUTE_TYPE)", COMPUTE_TYPE)
    settings_table.add_row("运行设备 (DEVICE)", DEVICE) # DEVICE 当前固定为 "cpu".
    settings_table.add_row("批处理大小 (BATCH_SIZE)", str(BATCH_SIZE))
    settings_table.add_row("模型内部 CPU 线程数 (CPU_THREADS_PER_MODEL_INSTANCE)", str(CPU_THREADS_PER_MODEL_INSTANCE))
    if console: console.print(settings_table)
    logger.info(f"显示当前模型设置: Model={WHISPER_MODEL}, Compute={COMPUTE_TYPE}, Batch={BATCH_SIZE}, CPUThreads={CPU_THREADS_PER_MODEL_INSTANCE}")

    if console:
        console.print("\n[bold]可配置选项:[/bold]")
        console.print("  [1] 选择 Whisper 模型大小 (例如: tiny, base, small, medium, large-v1/v2/v3)")
        console.print("  [2] 设置计算类型 (CPU 推荐: int8, float32, int16)")
        console.print("  [3] 设置批处理大小 (整数, 例如: 8, 16, 32)")
        console.print(f"  [4] 设置模型内部 CPU 线程数 (整数, 1 到 {os.cpu_count() or 'N/A'})")
        console.print("  [5] 返回主菜单")
    
    choice = Prompt.ask("请选择要修改的配置项", choices=["1", "2", "3", "4", "5"], default="5", console=console)
    logger.info(f"用户在模型配置中选择: {choice}")
    
    # 临时变量存储待修改的配置, 避免直接修改全局变量直到用户确认.
    temp_model = WHISPER_MODEL
    temp_compute = COMPUTE_TYPE
    temp_batch = BATCH_SIZE
    temp_cpu_threads = CPU_THREADS_PER_MODEL_INSTANCE

    if choice == "1":
        new_model = Prompt.ask(f"输入新的 Whisper 模型名称 (当前: {temp_model})", default=temp_model, console=console).strip()
        if new_model: temp_model = new_model # 只有非空输入才更新.
        logger.info(f"临时模型名称更改为: {temp_model}")
    elif choice == "2":
        allowed_cpu_compute_types = ["int8", "float32", "int16"] # faster-whisper 支持的 CPU 计算类型.
        new_compute = Prompt.ask(f"输入新的计算类型 (当前: {temp_compute}, CPU 推荐: {', '.join(allowed_cpu_compute_types)})", default=temp_compute, console=console).strip()
        if new_compute in allowed_cpu_compute_types: # 检查是否为允许的类型.
            temp_compute = new_compute
            logger.info(f"临时计算类型更改为: {temp_compute}")
        else:
            logger.warning(f"用户输入了无效的计算类型: '{new_compute}'. 保持当前设置 '{temp_compute}'.")
            if console: console.print(f"[yellow]无效的计算类型: '{new_compute}'. 推荐使用 {', '.join(allowed_cpu_compute_types)}. 保持当前设置.[/yellow]")
    elif choice == "3":
        new_batch_size = IntPrompt.ask(f"输入新的批处理大小 (当前: {temp_batch}, 必须为正整数)", default=temp_batch, console=console)
        if new_batch_size > 0: # 批处理大小必须为正.
            temp_batch = new_batch_size
            logger.info(f"临时批处理大小更改为: {temp_batch}")
        else:
            logger.warning(f"用户输入了无效的批处理大小: {new_batch_size}. 保持当前设置 {temp_batch}.")
            if console: console.print(f"[yellow]无效的批处理大小: {new_batch_size}. 必须大于 0. 保持当前设置.[/yellow]")
    elif choice == "4":
        max_suggested_threads = os.cpu_count() or 8 # 如果无法获取 CPU 核心数, 默认为 8 作为建议上限.
        new_cpu_threads = IntPrompt.ask(f"输入新的模型内部 CPU 线程数 (当前: {temp_cpu_threads}, 建议 1-{max_suggested_threads})", default=temp_cpu_threads, console=console)
        # 允许设置的线程数可以略高于物理核心数, 但不宜过高. 这里限制在 max_suggested_threads * 2.
        if 0 < new_cpu_threads <= (max_suggested_threads * 2) : 
            if new_cpu_threads > max_suggested_threads: # 如果超过建议的物理核心数.
                 if console: console.print(f"[yellow]注意: 设置的线程数 ({new_cpu_threads}) 超过 CPU 物理核心数 ({max_suggested_threads}). 这可能不会带来额外性能提升, 甚至可能因过度竞争而降低性能.[/yellow]")
            temp_cpu_threads = new_cpu_threads
            logger.info(f"临时内部 CPU 线程数更改为: {temp_cpu_threads}")
        else: # 如果输入无效 (例如 <=0 或远超核心数).
            logger.warning(f"用户输入了无效的内部 CPU 线程数: {new_cpu_threads}. 保持当前设置 {temp_cpu_threads}.")
            if console: console.print(f"[yellow]无效的内部 CPU 线程数: {new_cpu_threads}. 必须大于 0 且不宜过大. 保持当前设置.[/yellow]")
    elif choice == "5": # 用户选择返回主菜单.
        logger.info("用户选择返回主菜单, 未作任何模型配置更改.")
        if console: console.print("[bold blue][INFO][/bold blue] 未作更改. 返回主菜单.")
        return

    # 如果用户做了更改选择 (即 choice 不是 "5").
    if choice != "5":
        if console:
            console.print(f"\n[bold]建议的更改预览:[/bold]\n"
                          f"  模型: {temp_model}\n"
                          f"  计算类型: {temp_compute}\n"
                          f"  批处理大小: {temp_batch}\n"
                          f"  内部 CPU 线程数: {temp_cpu_threads}")
        if Confirm.ask("\n是否应用以上更改?", default=True, console=console):
            logger.info("用户确认应用模型配置更改.")
            # 检查是否有影响模型重新加载的关键配置发生变化.
            model_config_changed = (WHISPER_MODEL != temp_model or 
                                    COMPUTE_TYPE != temp_compute or 
                                    CPU_THREADS_PER_MODEL_INSTANCE != temp_cpu_threads) # DEVICE 不变.
            
            # 应用更改到全局变量.
            WHISPER_MODEL = temp_model
            COMPUTE_TYPE = temp_compute
            BATCH_SIZE = temp_batch
            CPU_THREADS_PER_MODEL_INSTANCE = temp_cpu_threads
            logger.info(f"模型配置已更新: Model={WHISPER_MODEL}, Compute={COMPUTE_TYPE}, Batch={BATCH_SIZE}, CPUThreads={CPU_THREADS_PER_MODEL_INSTANCE}")
            
            # 如果关键配置已更改且全局模型实例已存在, 则清除它, 以便下次重新加载.
            if model_config_changed and whisper_model_global is not None:
                logger.info("模型关键配置已更改. 清除已缓存的全局 Whisper 模型实例.")
                if console: console.print("[INFO] 模型关键配置已更改. 已缓存的 Whisper 模型实例已清除. 下次使用时将根据新配置重新加载.")
                del whisper_model_global # 删除旧模型实例.
                whisper_model_global = None # 重置为 None.
                gc.collect() # 尝试垃圾回收.
            if console: console.print("[bold green][SUCCESS][/bold green] 模型配置已更新!")
        else: # 用户取消应用更改.
            logger.info("用户取消应用模型配置更改.")
            if console: console.print("[bold blue][INFO][/bold blue] 配置更改已取消.")
    
    logger.info("“配置 Whisper 模型与参数”界面结束.")
    if console: console.print("\n[bold blue][INFO][/bold blue] 返回主菜单.")

def handle_advanced_settings_ui():
    """处理“高级设置”菜单选项的用户界面和逻辑, 例如 SRT 时间戳级别."""
    logger.info("进入“高级设置”界面.")
    global USE_WORD_TIMESTAMPS_CONFIG # 引用全局配置变量.
    
    if console: console.print(Panel(Text("高级设置", justify="center", style="bold cyan"), box=ROUNDED, expand=False))
    
    # 显示当前 SRT 时间戳模式.
    current_status_text = "[bold green]启用 (词级别时间戳)[/bold green]" if USE_WORD_TIMESTAMPS_CONFIG else "[bold red]禁用 (使用句子/短语级别时间戳)[/bold red]"
    if console: console.print(f"当前 SRT 时间戳模式: {current_status_text}")
    logger.info(f"当前 SRT 时间戳模式: {'词级别' if USE_WORD_TIMESTAMPS_CONFIG else '句子/短语级别'}")
    
    # 构建切换提示文本.
    prompt_text = f"是否 {'禁用词级别时间戳 (切换到句子/短语级别)' if USE_WORD_TIMESTAMPS_CONFIG else '启用词级别时间戳'}?"
    
    # 询问用户是否切换, 默认选项是切换到当前状态的相反状态.
    if Confirm.ask(prompt_text, default=not USE_WORD_TIMESTAMPS_CONFIG, console=console):
        USE_WORD_TIMESTAMPS_CONFIG = not USE_WORD_TIMESTAMPS_CONFIG # 切换状态.
        new_status_text = "[bold green]启用 (词级别时间戳)[/bold green]" if USE_WORD_TIMESTAMPS_CONFIG else "[bold red]禁用 (使用句子/短语级别时间戳)[/bold red]"
        logger.info(f"SRT 时间戳模式已更改为: {'词级别' if USE_WORD_TIMESTAMPS_CONFIG else '句子/短语级别'}")
        if console: console.print(f"[bold green][SUCCESS][/bold green] SRT 时间戳模式已设置为: {new_status_text}")
    else: # 用户选择不切换.
        logger.info("SRT 时间戳模式设置未更改.")
        if console: console.print("[bold blue][INFO][/bold blue] SRT 时间戳模式设置未更改.")
        
    logger.info("“高级设置”界面结束.")
    if console: console.print("\n[bold blue][INFO][/bold blue] 返回主菜单.")

def display_main_menu_ui():
    """显示主菜单并获取用户选择."""
    if console:
        console.rule("[bold cyan]Video-to-SRT 字幕生成工具 (CPU 版)[/bold cyan]", style="cyan")
        
        menu_text_obj = Text("\n请选择操作:\n\n", justify="left")
        menu_text_obj.append("  [1] 导入媒体文件并生成字幕 (默认)\n", style="yellow")
        menu_text_obj.append("  [2] 检查与管理依赖项\n", style="yellow")
        menu_text_obj.append("  [3] 配置 Whisper 模型与参数\n", style="yellow")
        menu_text_obj.append("  [4] 高级设置 (SRT 时间戳级别等)\n", style="yellow") # 更具体的描述.
        menu_text_obj.append("  [5] 退出程序\n\n", style="yellow")
        
        console.print(Panel(menu_text_obj, title="主菜单", border_style="magenta", padding=(1, 2), expand=False, box=ROUNDED))
    
    choice = Prompt.ask("输入选项 [1-5]", choices=["1", "2", "3", "4", "5"], default="1", console=console)
    logger.debug(f"主菜单用户选择: {choice}")
    return choice

def main_cli_loop():
    """主命令行界面循环."""
    ensure_app_dirs() # 确保应用所需的基础目录存在.
    logger.info("进入主命令行界面 (CLI) 循环.")
    
    # --- 启动时自动进行一次依赖检查 (非详细模式) ---
    if console: console.print("[INFO] 正在进行启动时依赖项检查...")
    # verbose=False 表示如果一切正常, 不会打印过多信息; 如果有问题, check_dependencies 内部会打印警告.
    if not check_dependencies(verbose=False): 
        if console:
            console.print("\n[bold red]警告: 启动时依赖检查发现问题![/bold red]")
            console.print("检测到部分依赖项未正确安装或版本不匹配.")
            console.print("请稍后进入菜单 [2] “检查与管理依赖项” 以查看详细信息,")
            console.print("并参照安装教程解决这些问题, 否则程序可能无法正常运行或出现错误.")
            console.line(1) # 添加空行以分隔.
        logger.warning("启动时依赖检查发现问题. 已提示用户.")
    else:
        if console: console.print("[INFO] 启动时依赖项检查通过.")
        logger.info("启动时依赖项检查通过.")


    # --- 主循环 ---
    while True:
        user_action = display_main_menu_ui() # 显示菜单并获取用户选择.
        logger.info(f"用户在主菜单选择了操作: {user_action}")
        
        if user_action == "1":
            handle_import_and_process()
        elif user_action == "2":
            handle_dependencies_check_ui()
        elif user_action == "3":
            handle_model_selection_ui()
        elif user_action == "4":
            handle_advanced_settings_ui()
        elif user_action == "5": # 退出程序.
            if Confirm.ask("您确定要退出程序吗?", default=True, console=console): 
                logger.info("用户确认退出程序.")
                if console: console.print("[bold blue]感谢使用本工具, 程序已退出.[/bold blue]")
                break # 跳出主循环, 结束程序.
            else:
                logger.info("用户取消退出程序.")
        
        if console: console.line(2) # 在每次操作后添加一些空行, 使界面更清晰.

if __name__ == "__main__":
    setup_logging() # 首先配置日志记录器.
    # --- 记录程序启动信息 ---
    logger.info("======================================================================")
    logger.info(f"应用程序 '{os.path.basename(sys.argv[0] if hasattr(sys, 'argv') and sys.argv else __file__)}' 启动")
    logger.info(f"  程序基础路径 (APP_BASE_PATH): {APP_BASE_PATH}")
    logger.info(f"  日志文件路径 (log_file_path): {log_file_path}")
    logger.info(f"  模型缓存目录 (MODEL_CACHE_DIR): {MODEL_CACHE_DIR}")
    logger.info(f"  临时文件目录 (TEMP_DIR): {TEMP_DIR}")
    logger.info(f"  状态文件路径 (STATUS_FILE): {STATUS_FILE}")
    logger.info(f"  Python 版本: {sys.version.splitlines()[0] if sys.version else '未知'}") # 只取版本号第一行.
    logger.info(f"  操作系统: {platform.system()} {platform.release()} ({platform.version()})")
    logger.info(f"  CPU 核心数: {os.cpu_count() or '未知'}") # 如果无法获取则显示未知.
    logger.info(f"  HuggingFace Endpoint (HF_ENDPOINT): {os.environ.get('HF_ENDPOINT', '未设置')}")
    logger.info("======================================================================")

    try:
        main_cli_loop() # 进入主程序循环.
    except KeyboardInterrupt: # 处理用户按 Ctrl+C 中断程序.
        logger.warning("用户通过 KeyboardInterrupt (Ctrl+C) 中断了程序.", exc_info=False) # exc_info=False 因为这是预期的用户行为.
        if console: console.print("\n[bold yellow]用户中断了程序运行 (Ctrl+C). 正在退出...[/bold yellow]")
    except Exception as e: # 捕获主循环中未被处理的任何其他严重错误.
        logger.critical("在主程序循环 (main_cli_loop) 中发生未处理的严重错误:", exc_info=True) # exc_info=True 会自动记录完整的堆栈跟踪.
        if console:
            console.print("\n[bold red][CRITICAL ERROR][/bold red] 程序遇到未处理的严重错误, 即将终止.")
            # 使用 Rich 的 print_exception 打印更详细、格式化的异常信息到控制台.
            console.print_exception(show_locals=False, width=None) # show_locals=False 避免暴露过多局部变量信息.
            console.print(f"[bold red]错误详情 (也已记录到日志文件 '{LOG_FILE_NAME}'): {str(e)}[/bold red]")
            console.print(f"建议将上述错误信息截图或复制, 或查看位于以下路径的日志文件 '{LOG_FILE_NAME}':\n  '{os.path.join(APP_BASE_PATH, LOG_FILE_NAME)}'\n以便排查问题.")
    finally: # 程序结束前的最终清理和日志记录.
        logger.info("应用程序终止.")
        logger.info("======================================================================\n")
        if console: console.print("程序已终止.", style="dim")
