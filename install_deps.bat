@echo off
chcp 65001 >nul 2>&1

title AnchorFlux - Install Dependencies

echo.
echo ========================================
echo   AnchorFlux - Dependency Installer
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "TOOLS_DIR=%PROJECT_ROOT%\tools"
set "REQ_FILE=%PROJECT_ROOT%\requirements.txt"
set "MARKER_FILE=%PROJECT_ROOT%\.env_installed"
set "PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"

REM ========================================
REM  Python 路径配置 (双模式: 嵌入式优先, 回退venv)
REM ========================================
set "EMBED_PYTHON=%TOOLS_DIR%\python\python.exe"
set "EMBED_SITE_PACKAGES=%TOOLS_DIR%\python\Lib\site-packages"
set "VENV_ROOT=%PROJECT_ROOT%\.venv"
set "VENV_PYTHON=%VENV_ROOT%\Scripts\python.exe"
set "VENV_SITE_PACKAGES=%VENV_ROOT%\Lib\site-packages"

REM 检测Python模式: 优先嵌入式, 回退venv
if exist "%EMBED_PYTHON%" (
    set "PYTHON_EXEC=%EMBED_PYTHON%"
    set "SITE_PACKAGES=%EMBED_SITE_PACKAGES%"
    set "PYTHON_MODE=embedded"
    goto python_found
)

if exist "%VENV_PYTHON%" (
    set "PYTHON_EXEC=%VENV_PYTHON%"
    set "SITE_PACKAGES=%VENV_SITE_PACKAGES%"
    set "PYTHON_MODE=venv"
    goto python_found
)

REM 两者都不存在, 尝试用系统Python创建venv
echo [INFO] No Python environment found, trying to create venv...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo [INFO] Please install Python 3.10+ or place embedded Python in tools\python\
    pause
    exit /b 1
)

echo [INFO] Creating virtual environment...
python -m venv "%VENV_ROOT%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
set "PYTHON_EXEC=%VENV_PYTHON%"
set "SITE_PACKAGES=%VENV_SITE_PACKAGES%"
set "PYTHON_MODE=venv"
echo [OK] Virtual environment created

:python_found
echo [Config] Project: %PROJECT_ROOT%
echo [Config] Python Mode: %PYTHON_MODE%
echo [Config] Python: %PYTHON_EXEC%
echo.

REM 删除标记文件以强制重新安装
if exist "%MARKER_FILE%" (
    echo [INFO] Removing old installation marker...
    del "%MARKER_FILE%"
)

echo [Step 1/3] Upgrading pip...
"%PYTHON_EXEC%" -m pip install --upgrade pip -i %PYPI_MIRROR% -q
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)
echo [OK] pip upgraded
echo.

echo [Step 2/3] Installing dependencies (this may take 10-30 minutes)...
echo [INFO] Using mirror: %PYPI_MIRROR%
echo.
"%PYTHON_EXEC%" -m pip install -r "%REQ_FILE%" -i %PYPI_MIRROR%
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies
    echo [INFO] Common issues:
    echo        1. Network connection problem
    echo        2. Insufficient disk space
    echo        3. Version conflict
    echo.
    pause
    exit /b 1
)
echo.
echo [OK] Dependencies installed
echo.

echo [Step 3/4] Fixing onnxruntime version conflict...
REM funasr-onnx 会自动安装 onnxruntime (CPU版), 但我们只需要 onnxruntime-gpu
REM onnxruntime-gpu 完全兼容 CPU 推理, 可以满足所有依赖
REM 两个包共用 onnxruntime 目录, 卸载时可能留下残留, 需要清理后重装
"%PYTHON_EXEC%" -m pip uninstall onnxruntime -y >nul 2>&1
if exist "%SITE_PACKAGES%\onnxruntime" (
    echo [INFO] Cleaning residual onnxruntime directory...
    rmdir /s /q "%SITE_PACKAGES%\onnxruntime" >nul 2>&1
)
REM 重新安装 GPU 版本以确保完整性
echo [INFO] Reinstalling onnxruntime-gpu to ensure integrity...
"%PYTHON_EXEC%" -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.18.0 -i %PYPI_MIRROR% -q
echo [OK] onnxruntime-gpu installed (CPU version removed)
echo.

echo [Step 4/4] Fixing PyTorch DLL dependencies...
if exist "%SITE_PACKAGES%\torch\lib\libiomp5md.dll" (
    if not exist "%SITE_PACKAGES%\torch\lib\libomp140.x86_64.dll" (
        echo [INFO] Copying libiomp5md.dll to libomp140.x86_64.dll...
        copy "%SITE_PACKAGES%\torch\lib\libiomp5md.dll" "%SITE_PACKAGES%\torch\lib\libomp140.x86_64.dll" >nul 2>&1
        echo [OK] DLL fix applied
    ) else (
        echo [OK] DLL already fixed
    )
) else (
    echo [INFO] PyTorch not installed or DLL not found, skipping fix
)
echo.

REM 创建安装标记文件
echo installed > "%MARKER_FILE%"

echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo   Python Mode: %PYTHON_MODE%
echo.
echo   You can now run the application with:
echo   - run.bat (recommended)
echo.
echo ========================================

pause
