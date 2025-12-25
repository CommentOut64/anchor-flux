@echo off
chcp 65001 >nul 2>&1

title AnchorFlux - Build Release Package

echo.
echo ========================================
echo   AnchorFlux - Release Builder
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

REM ========================================
REM  版本配置
REM ========================================
set "VERSION=3.1.0"
set "RELEASE_NAME=AnchorFlux_v%VERSION%_Portable"
set "BUILD_DIR=%PROJECT_ROOT%\build"
set "RELEASE_DIR=%BUILD_DIR%\%RELEASE_NAME%"

echo [Config] Project: %PROJECT_ROOT%
echo [Config] Version: %VERSION%
echo [Config] Output: %RELEASE_DIR%
echo.

REM ========================================
REM  Step 1: 清理旧的构建目录
REM ========================================
echo [Step 1/7] Cleaning old build directory...
if exist "%BUILD_DIR%" (
    rmdir /s /q "%BUILD_DIR%"
)
mkdir "%RELEASE_DIR%"
echo [OK] Build directory created
echo.

REM ========================================
REM  Step 2: 复制核心文件
REM ========================================
echo [Step 2/7] Copying core files...

REM 启动脚本
copy "%PROJECT_ROOT%\run.bat" "%RELEASE_DIR%\" >nul
copy "%PROJECT_ROOT%\install_deps.bat" "%RELEASE_DIR%\" >nul
copy "%PROJECT_ROOT%\bootloader.py" "%RELEASE_DIR%\" >nul
copy "%PROJECT_ROOT%\requirements.txt" "%RELEASE_DIR%\" >nul

REM 配置文件
if exist "%PROJECT_ROOT%\user_config.json" (
    copy "%PROJECT_ROOT%\user_config.json" "%RELEASE_DIR%\" >nul
)
if exist "%PROJECT_ROOT%\pyproject.toml" (
    copy "%PROJECT_ROOT%\pyproject.toml" "%RELEASE_DIR%\" >nul
)
if exist "%PROJECT_ROOT%\.python-version" (
    copy "%PROJECT_ROOT%\.python-version" "%RELEASE_DIR%\" >nul
)

echo [OK] Core files copied
echo.

REM ========================================
REM  Step 3: 复制后端代码
REM ========================================
echo [Step 3/7] Copying backend code...
xcopy "%PROJECT_ROOT%\backend" "%RELEASE_DIR%\backend\" /E /I /Q /Y >nul

REM 排除不需要的文件
if exist "%RELEASE_DIR%\backend\__pycache__" rmdir /s /q "%RELEASE_DIR%\backend\__pycache__"
if exist "%RELEASE_DIR%\backend\tests" rmdir /s /q "%RELEASE_DIR%\backend\tests"
if exist "%RELEASE_DIR%\backend\scripts" rmdir /s /q "%RELEASE_DIR%\backend\scripts"
if exist "%RELEASE_DIR%\backend\jobs" rmdir /s /q "%RELEASE_DIR%\backend\jobs"

REM 递归删除所有 __pycache__ 目录
for /d /r "%RELEASE_DIR%\backend" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d"
)

REM 删除 .pyc 文件
del /s /q "%RELEASE_DIR%\backend\*.pyc" >nul 2>&1

echo [OK] Backend code copied
echo.

REM ========================================
REM  Step 4: 构建并复制前端代码
REM ========================================
echo [Step 4/7] Building and copying frontend code...
mkdir "%RELEASE_DIR%\frontend"

REM 先构建前端 (需要 Node.js)
where node >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js not found! Cannot build frontend.
    echo [INFO] Please install Node.js 18+ and try again.
    pause
    exit /b 1
)

echo [INFO] Installing frontend dependencies...
cd /d "%PROJECT_ROOT%\frontend"
call npm install >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies
    pause
    exit /b 1
)

echo [INFO] Building frontend for production...
call npm run build
if errorlevel 1 (
    echo [ERROR] Failed to build frontend
    pause
    exit /b 1
)
cd /d "%PROJECT_ROOT%"
echo [OK] Frontend built successfully

REM 复制已构建的dist目录
if exist "%PROJECT_ROOT%\frontend\dist" (
    echo [INFO] Copying frontend dist...
    xcopy "%PROJECT_ROOT%\frontend\dist" "%RELEASE_DIR%\frontend\dist\" /E /I /Q /Y >nul
) else (
    echo [ERROR] Frontend dist not found after build!
    pause
    exit /b 1
)

REM 复制前端配置文件 (用于开发模式)
copy "%PROJECT_ROOT%\frontend\package.json" "%RELEASE_DIR%\frontend\" >nul
copy "%PROJECT_ROOT%\frontend\package-lock.json" "%RELEASE_DIR%\frontend\" >nul 2>&1
copy "%PROJECT_ROOT%\frontend\vite.config.js" "%RELEASE_DIR%\frontend\" >nul
copy "%PROJECT_ROOT%\frontend\index.html" "%RELEASE_DIR%\frontend\" >nul
if exist "%PROJECT_ROOT%\frontend\jsconfig.json" copy "%PROJECT_ROOT%\frontend\jsconfig.json" "%RELEASE_DIR%\frontend\" >nul

REM 复制前端源码 (用于开发模式)
xcopy "%PROJECT_ROOT%\frontend\src" "%RELEASE_DIR%\frontend\src\" /E /I /Q /Y >nul
xcopy "%PROJECT_ROOT%\frontend\public" "%RELEASE_DIR%\frontend\public\" /E /I /Q /Y >nul 2>&1

echo [OK] Frontend code copied
echo.

REM ========================================
REM  Step 5: 复制工具目录 (含嵌入式Python)
REM ========================================
echo [Step 5/7] Copying tools directory...
xcopy "%PROJECT_ROOT%\tools" "%RELEASE_DIR%\tools\" /E /I /Q /Y >nul

REM 清理嵌入式Python中不需要的文件
if exist "%RELEASE_DIR%\tools\python\Lib\site-packages" (
    REM 只保留pip/setuptools/wheel，其他依赖由用户在线安装
    echo [INFO] Cleaning embedded Python site-packages...
    for /d %%d in ("%RELEASE_DIR%\tools\python\Lib\site-packages\*") do (
        set "dirname=%%~nxd"
        setlocal enabledelayedexpansion
        if /i not "!dirname!"=="pip" (
        if /i not "!dirname!"=="pip-25.3.dist-info" (
        if /i not "!dirname!"=="setuptools" (
        if /i not "!dirname!"=="setuptools-80.9.0.dist-info" (
        if /i not "!dirname!"=="wheel" (
        if /i not "!dirname!"=="wheel-0.45.1.dist-info" (
        if /i not "!dirname!"=="pkg_resources" (
        if /i not "!dirname!"=="_distutils_hack" (
            rmdir /s /q "%%d" 2>nul
        ))))))))
        endlocal
    )
    del /q "%RELEASE_DIR%\tools\python\Lib\site-packages\distutils-precedence.pth" 2>nul
)

echo [OK] Tools directory copied
echo.

REM ========================================
REM  Step 6: 创建空目录结构
REM ========================================
echo [Step 6/7] Creating directory structure...
mkdir "%RELEASE_DIR%\input" 2>nul
mkdir "%RELEASE_DIR%\output" 2>nul
mkdir "%RELEASE_DIR%\jobs" 2>nul
mkdir "%RELEASE_DIR%\temp" 2>nul
mkdir "%RELEASE_DIR%\logs" 2>nul
mkdir "%RELEASE_DIR%\models" 2>nul
mkdir "%RELEASE_DIR%\models\huggingface" 2>nul
mkdir "%RELEASE_DIR%\models\torch" 2>nul

REM 创建占位文件
echo. > "%RELEASE_DIR%\input\.gitkeep"
echo. > "%RELEASE_DIR%\output\.gitkeep"
echo. > "%RELEASE_DIR%\jobs\.gitkeep"
echo. > "%RELEASE_DIR%\temp\.gitkeep"
echo. > "%RELEASE_DIR%\logs\.gitkeep"
echo. > "%RELEASE_DIR%\models\.gitkeep"

echo [OK] Directory structure created
echo.

REM ========================================
REM  Step 7: 创建README
REM ========================================
echo [Step 7/7] Creating README...
(
echo ========================================
echo   AnchorFlux v%VERSION% - Portable Edition
echo ========================================
echo.
echo [Quick Start]
echo.
echo 1. Double-click "run.bat" to start
echo 2. First run will install dependencies automatically ^(10-30 min^)
echo 3. Browser will open automatically when ready
echo.
echo [System Requirements]
echo.
echo - Windows 10/11 64-bit
echo - NVIDIA GPU with CUDA 11.8 support
echo - At least 8GB VRAM ^(16GB recommended^)
echo - At least 16GB RAM
echo - Node.js 18+ ^(for frontend dev mode^)
echo.
echo [File Structure]
echo.
echo run.bat           - Main startup script
echo install_deps.bat  - Manual dependency installer
echo tools/python/     - Embedded Python 3.10
echo tools/ffmpeg.exe  - FFmpeg binary
echo backend/          - Backend source code
echo frontend/         - Frontend source code
echo models/           - AI models ^(auto-download^)
echo input/            - Input video files
echo output/           - Output subtitle files
echo.
echo [Notes]
echo.
echo - First run requires internet connection
echo - Models will be downloaded from hf-mirror.com
echo - Dependencies will be installed from tsinghua mirror
echo.
echo ========================================
) > "%RELEASE_DIR%\README.txt"

echo [OK] README created
echo.

REM ========================================
REM  完成统计
REM ========================================
echo ========================================
echo   Build Complete!
echo ========================================
echo.
echo   Output: %RELEASE_DIR%
echo.

REM 计算目录大小
for /f "tokens=3" %%a in ('dir "%RELEASE_DIR%" /s /-c 2^>nul ^| findstr "File(s)"') do set SIZE=%%a
echo   Size: %SIZE% bytes
echo.

echo   Next steps:
echo   1. Compress %RELEASE_NAME% folder to .7z or .zip
echo   2. Recommended: 7z a -t7z -mx=9 %RELEASE_NAME%.7z %RELEASE_NAME%
echo.
echo ========================================

pause
