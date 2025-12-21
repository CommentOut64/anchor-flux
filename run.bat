@echo off
chcp 65001 >nul 2>&1

title Video to SRT GPU

echo.
echo ========================================
echo   AnchorFlux - Starting...
echo ========================================
echo.

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "VENV_ROOT=%PROJECT_ROOT%\.venv"
set "PYTHON_EXEC=%VENV_ROOT%\Scripts\python.exe"
set "TOOLS_DIR=%PROJECT_ROOT%\tools"
set "BACKEND_DIR=%PROJECT_ROOT%\backend"
set "FRONTEND_DIR=%PROJECT_ROOT%\frontend"
set "REQ_FILE=%PROJECT_ROOT%\requirements.txt"
set "MARKER_FILE=%PROJECT_ROOT%\.env_installed"
set "PYPI_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "HF_ENDPOINT=https://hf-mirror.com"

echo [Config] Project: %PROJECT_ROOT%
echo.

echo [Step 1/6] Checking virtual environment...
if not exist "%VENV_ROOT%" (
    echo [INFO] Creating virtual environment...
    python -m venv "%VENV_ROOT%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

if not exist "%PYTHON_EXEC%" (
    echo [ERROR] Python not found: %PYTHON_EXEC%
    pause
    exit /b 1
)
echo.

echo [Step 2/6] Checking Python dependencies...
if exist "%MARKER_FILE%" (
    echo [INFO] Dependencies already installed, skipping...
) else (
    echo [INFO] Installing dependencies...
    "%PYTHON_EXEC%" -m pip install --upgrade pip -i %PYPI_MIRROR% -q
    "%PYTHON_EXEC%" -m pip install -r "%REQ_FILE%" -i %PYPI_MIRROR%
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo installed > "%MARKER_FILE%"
    echo [OK] Dependencies installed
)
echo.

echo [Step 3/6] Checking frontend dependencies...
set "SKIP_FRONTEND=0"
where node >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Node.js not found, frontend will not start
    set "SKIP_FRONTEND=1"
    goto skip_frontend
)
if not exist "%FRONTEND_DIR%\node_modules" (
    echo [INFO] Installing frontend dependencies...
    cd /d "%FRONTEND_DIR%"
    call npm install
    cd /d "%PROJECT_ROOT%"
    echo [OK] Frontend dependencies installed
) else (
    echo [OK] Frontend dependencies exist
)
:skip_frontend
echo.

echo [Step 4/6] Checking FFmpeg...
if exist "%TOOLS_DIR%\ffmpeg.exe" (
    echo [OK] FFmpeg found
) else (
    echo [WARNING] FFmpeg not found in tools folder
)
echo.

echo [Step 5/6] Configuring environment...
set "TORCH_LIB=%VENV_ROOT%\Lib\site-packages\torch\lib"
set "NVIDIA_CUDNN=%VENV_ROOT%\Lib\site-packages\nvidia\cudnn\bin"
set "NVIDIA_CUBLAS=%VENV_ROOT%\Lib\site-packages\nvidia\cublas\bin"
set "PATH=%TORCH_LIB%;%NVIDIA_CUDNN%;%NVIDIA_CUBLAS%;%TOOLS_DIR%;%PATH%"
echo [OK] Environment configured
echo.

echo [Step 6/6] Starting services...
echo.

REM 先清理可能残留的旧进程
echo [Cleanup] Checking for old processes...

REM 清理 FFmpeg 进程
taskkill /F /IM ffmpeg.exe >nul 2>&1
if %ERRORLEVEL%==0 echo [Cleanup] FFmpeg processes terminated

REM 清理 ffprobe 进程
taskkill /F /IM ffprobe.exe >nul 2>&1

REM 检查并释放端口 8000（旧后端进程）
echo [Cleanup] Checking port 8000...
set "CLEANED_8000=0"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    if not "%%a"=="" (
        echo [Cleanup] Found process on port 8000 - PID: %%a
        taskkill /F /PID %%a >nul 2>&1
        if not ERRORLEVEL 1 (
            echo [Cleanup] Killed process PID %%a
            set "CLEANED_8000=1"
        )
    )
)

REM 检查并释放端口 5173（旧前端进程）
echo [Cleanup] Checking port 5173...
set "CLEANED_5173=0"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5173" ^| findstr "LISTENING"') do (
    if not "%%a"=="" (
        echo [Cleanup] Found process on port 5173 - PID: %%a
        taskkill /F /PID %%a >nul 2>&1
        if not ERRORLEVEL 1 (
            echo [Cleanup] Killed process PID %%a
            set "CLEANED_5173=1"
        )
    )
)

REM 使用 wmic 查找并终止残留的 Python 进程（运行 uvicorn）
echo [Cleanup] Checking for old uvicorn processes...
for /f "skip=1 tokens=1" %%p in ('wmic process where "commandline like '%%uvicorn%%app.main%%'" get processid 2^>nul') do (
    if not "%%p"=="" (
        echo [Cleanup] Found old uvicorn process - PID: %%p
        taskkill /F /PID %%p >nul 2>&1
    )
)

REM 使用 wmic 查找并终止残留的 Node 进程（运行 vite）
echo [Cleanup] Checking for old vite processes...
for /f "skip=1 tokens=1" %%p in ('wmic process where "commandline like '%%vite%%'" get processid 2^>nul') do (
    if not "%%p"=="" (
        echo [Cleanup] Found old vite process - PID: %%p
        taskkill /F /PID %%p >nul 2>&1
    )
)

REM 等待进程完全退出
timeout /t 2 /nobreak >nul
echo [OK] Old processes cleanup completed
echo.

echo [Starting] Backend service on port 8000...
start "Video2SRT Backend" cmd /c "title Video2SRT Backend && cd /d %BACKEND_DIR% && set KMP_DUPLICATE_LIB_OK=TRUE && set PATH=%TORCH_LIB%;%NVIDIA_CUDNN%;%NVIDIA_CUBLAS%;%TOOLS_DIR%;%PATH% && %PYTHON_EXEC% -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo [Waiting] Backend initializing...
timeout /t 5 /nobreak >nul

if "%SKIP_FRONTEND%"=="0" (
    echo [Starting] Frontend service on port 5173...
    start "Video2SRT Frontend" cmd /c "title Video2SRT Frontend && cd /d %FRONTEND_DIR% && npm run dev"
)

cd /d "%PROJECT_ROOT%"

timeout /t 3 /nobreak >nul

REM 不在这里打开浏览器，由后端统一控制
REM start "" "http://localhost:5173"

echo.
echo ========================================
echo   Services Started!
echo ========================================
echo.
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
echo   [!] This window will close automatically
echo       when you click "Exit System" button.
echo.
echo   [!] Do NOT close this window manually!
echo.
echo ========================================

REM 循环检测后端进程是否还在运行
REM 如果后端进程退出（用户点击"退出系统"），则自动关闭主窗口
:wait_loop
timeout /t 3 /nobreak >nul

REM 检查端口 8000 是否还有进程监听
netstat -ano 2>nul | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] Backend service has stopped. Closing...
    REM 终止前端进程
    taskkill /F /IM node.exe /FI "WINDOWTITLE eq Video2SRT*" >nul 2>&1
    timeout /t 1 /nobreak >nul
    exit /b 0
)

goto wait_loop
