@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

:: 设置窗口标题
title Video to SRT GPU Dashboard

echo [INFO] 正在启动 Video to SRT GPU...

:: 1. 获取项目根目录
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

:: 2. 设置 Python 路径
set "VENV_ROOT=%PROJECT_ROOT%\.venv"
set "PYTHON_EXEC=%VENV_ROOT%\Scripts\python.exe"

:: 检查 Python 解释器是否存在
if not exist "%PYTHON_EXEC%" (
    echo [ERROR] 未找到 Python 解释器: %PYTHON_EXEC%
    echo [INFO] 请先执行: uv venv .venv
    pause
    exit /b 1
)

:: 3. 设置 CUDA DLL 路径（混合环境关键！）
:: PyTorch 的 CUDA 11 库
set "TORCH_LIB=%VENV_ROOT%\Lib\site-packages\torch\lib"
:: Faster-Whisper 需要的 CUDA 12 库
set "NVIDIA_CUDNN=%VENV_ROOT%\Lib\site-packages\nvidia\cudnn\bin"
set "NVIDIA_CUBLAS=%VENV_ROOT%\Lib\site-packages\nvidia\cublas\bin"

:: 4. 设置工具路径（FFmpeg/FFprobe）
set "TOOLS_DIR=%PROJECT_ROOT%\tools"

:: 检查 FFmpeg 是否存在
if not exist "%TOOLS_DIR%\ffmpeg.exe" (
    echo [ERROR] 未找到 FFmpeg: %TOOLS_DIR%\ffmpeg.exe
    echo [INFO] 请下载 ffmpeg.exe 并放入 tools 目录
    pause
    exit /b 1
)

:: 检查 FFprobe 是否存在
if not exist "%TOOLS_DIR%\ffprobe.exe" (
    echo [ERROR] 未找到 FFprobe: %TOOLS_DIR%\ffprobe.exe
    echo [INFO] 请下载 ffprobe.exe 并放入 tools 目录
    pause
    exit /b 1
)

echo [INFO] FFmpeg 检查通过: %TOOLS_DIR%\ffmpeg.exe
echo [INFO] FFprobe 检查通过: %TOOLS_DIR%\ffprobe.exe

:: 5. 更新 PATH（CUDA库 + 工具目录 优先）
set "PATH=%TORCH_LIB%;%NVIDIA_CUDNN%;%NVIDIA_CUBLAS%;%TOOLS_DIR%;%PATH%"

echo [INFO] 环境变量已配置
echo [INFO] - CUDA 11 (PyTorch): %TORCH_LIB%
echo [INFO] - CUDA 12 (cuDNN): %NVIDIA_CUDNN%
echo [INFO] - CUDA 12 (cuBLAS): %NVIDIA_CUBLAS%
echo [INFO] - Tools: %TOOLS_DIR%

:: 6. 启动 Python 启动器（bootloader.py 会处理依赖检查和服务启动）
echo [INFO] 正在启动服务...
"%PYTHON_EXEC%" "%PROJECT_ROOT%\bootloader.py"

:: 检查 Python 脚本异常退出则暂停显示错误
if %errorlevel% neq 0 (
    echo.
    echo [程序异常退出]
    pause
)
