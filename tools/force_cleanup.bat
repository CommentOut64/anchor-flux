@echo off
chcp 65001 >nul
title 强制清理进程

echo ==========================================
echo     强制清理 Video2SRT 相关进程
echo ==========================================
echo.
echo 警告：此操作将终止所有 Video2SRT 相关进程！
echo.
pause

echo.
echo [Step 1] 终止 FFmpeg 进程...
taskkill /F /IM ffmpeg.exe 2>nul
taskkill /F /IM ffprobe.exe 2>nul
echo.

echo [Step 2] 清理端口 8000 (后端)...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    if not "%%a"=="" (
        echo   终止进程 PID: %%a
        taskkill /F /PID %%a >nul 2>&1
    )
)
echo.

echo [Step 3] 清理端口 5173 (前端)...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5173" ^| findstr "LISTENING"') do (
    if not "%%a"=="" (
        echo   终止进程 PID: %%a
        taskkill /F /PID %%a >nul 2>&1
    )
)
echo.

echo [Step 4] 清理 uvicorn 进程...
for /f "skip=1 tokens=1" %%p in ('wmic process where "commandline like '%%uvicorn%%app.main%%'" get processid 2^>nul') do (
    if not "%%p"=="" (
        echo   终止 uvicorn 进程 PID: %%p
        taskkill /F /PID %%p >nul 2>&1
    )
)
echo.

echo [Step 5] 清理 vite 进程...
for /f "skip=1 tokens=1" %%p in ('wmic process where "commandline like '%%vite%%'" get processid 2^>nul') do (
    if not "%%p"=="" (
        echo   终止 vite 进程 PID: %%p
        taskkill /F /PID %%p >nul 2>&1
    )
)
echo.

echo ==========================================
echo 清理完成！等待 2 秒...
timeout /t 2 /nobreak >nul
echo.

echo [验证] 检查端口状态...
netstat -ano | findstr ":8000\|:5173" | findstr "LISTENING"
if %ERRORLEVEL% NEQ 0 (
    echo   所有端口已释放
)
echo.

echo ==========================================
echo 清理操作完成
echo ==========================================
pause
