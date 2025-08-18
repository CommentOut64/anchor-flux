@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置标题
title Video-to-SRT 一键启动器

:: 设置代理绕过本地服务 (Clash 等)
set NO_PROXY=127.0.0.1,localhost
set no_proxy=127.0.0.1,localhost

:: 确保在正确的目录
cd /d "%~dp0"

echo ==========================================================
echo 🚀 Video-to-SRT 一键启动器
echo ==========================================================
echo.

:: 检查并安装必要依赖
echo [信息] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Python 未安装或未加入PATH
    echo 请确保已安装Python并添加到系统PATH中
    goto :error_exit
)

python -c "import psutil, requests" >nul 2>&1
if errorlevel 1 (
    echo [信息] 安装必要的依赖包...
    pip install psutil requests
    if errorlevel 1 (
        echo [错误] 依赖包安装失败
        goto :error_exit
    )
)

:: 清理之前的进程
echo [信息] 清理之前的进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5174') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: 启动后端服务
echo [信息] 启动后端服务...
start "Video-to-SRT 后端" /MIN cmd /c "cd /d %~dp0backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

:: 等待后端启动
echo [信息] 等待后端服务启动...
set /a retries=0
:wait_backend
powershell -Command "try { (Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/ping' -TimeoutSec 2).StatusCode -eq 200 } catch { $false }" | findstr True >nul
if not errorlevel 1 goto backend_ready

set /a retries+=1
if !retries! geq 30 (
    echo [错误] 后端服务启动超时
    goto :error_exit
)
timeout /t 1 >nul
goto wait_backend

:backend_ready
echo [成功] 后端服务已启动

:: 检查前端依赖
if not exist "%~dp0frontend\node_modules" (
    echo [信息] 安装前端依赖...
    cd /d "%~dp0frontend"
    npm install
    if errorlevel 1 (
        echo [错误] 前端依赖安装失败
        goto :error_exit
    )
    cd /d "%~dp0"
)

:: 启动前端服务
echo [信息] 启动前端服务...
start "Video-to-SRT 前端" /MIN cmd /c "cd /d %~dp0frontend && npm run dev"

:: 等待前端启动
echo [信息] 等待前端服务启动...
set /a retries=0
:wait_frontend
powershell -Command "try { (Invoke-WebRequest -UseBasicParsing 'http://localhost:5174' -TimeoutSec 2).StatusCode -eq 200 } catch { $false }" | findstr True >nul
if not errorlevel 1 goto frontend_ready

set /a retries+=1
if !retries! geq 30 (
    echo [错误] 前端服务启动超时
    goto :error_exit
)
timeout /t 1 >nul
goto wait_frontend

:frontend_ready
echo [成功] 前端服务已启动

:: 打开浏览器
echo [信息] 打开浏览器...
start http://localhost:5174

echo.
echo ==========================================================
echo ✅ 服务启动完成！
echo 前端地址: http://localhost:5174
echo 后端地址: http://127.0.0.1:8000  
echo ==========================================================
echo.
echo 📌 注意事项：
echo    • 保持此窗口打开以监控服务状态
echo    • 关闭此窗口会自动停止所有服务
echo    • 如遇问题，请检查防火墙和杀毒软件设置
echo.
echo 按任意键退出并停止所有服务...
pause >nul

:: 清理进程
echo.
echo [信息] 正在停止服务...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5174') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo [成功] 所有服务已停止
exit /b 0

:error_exit
echo.
echo [错误] 启动失败，请检查上述错误信息
pause
exit /b 1
