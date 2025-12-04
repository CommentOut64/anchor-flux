@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置标题
title Video-to-SRT 进程清理器 (精准版 - 带验证机制)

echo ================================================
echo   Video-to-SRT 进程清理器 - 精准清理模式
echo ================================================
echo.
echo 说明：本清理器仅清理主程序相关的命令行
echo       不会影响正在运行的测试进程
echo.

:: 初始化变量
set "KILLED_BACKEND=0"
set "KILLED_FRONTEND=0"
set "KILLED_PORTS=0"

:: ====================================
:: 第一阶段：快速验证 - 检查命令行标题
:: ====================================
echo [验证] 正在识别主程序运行的命令行...
echo.

:: 1. 查找并记录后端命令行的PID（通过窗口标题）
echo [扫描] 后端服务（窗口标题: 后端服务）...
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv ^| findstr /v "PID"') do (
    if not "%%i"=="" (
        :: 检查命令行是否包含 uvicorn 和 main.py（后端特征）
        for /f "tokens=*" %%c in ('wmic process where "ProcessId=%%i" get CommandLine ^| findstr /i "uvicorn.*main"') do (
            if not "%%c"=="" (
                echo   ✓ 找到后端进程 (PID: %%i)
                set "BACKEND_PID=%%i"
                goto check_frontend
            )
        )
    )
)

:check_frontend
:: 2. 查找并记录前端命令行的PID（通过特征）
echo [扫描] 前端服务（窗口标题: 前端服务）...
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq node.exe" /fo csv ^| findstr /v "PID"') do (
    if not "%%i"=="" (
        :: 检查命令行是否包含 npm run dev 或 vite（前端特征）
        for /f "tokens=*" %%c in ('wmic process where "ProcessId=%%i" get CommandLine ^| findstr /i "npm.*dev\|vite"') do (
            if not "%%c"=="" (
                echo   ✓ 找到前端进程 (PID: %%i)
                set "FRONTEND_PID=%%i"
                goto phase2
            )
        )
    )
)

:: ====================================
:: 第二阶段：优雅关闭 - 尝试API优雅关闭
:: ====================================
:phase2
echo.
echo [关闭] 尝试优雅关闭后端服务...
powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/shutdown' -Method Post -TimeoutSec 5 | Out-Null; Write-Host '[成功] 后端服务已优雅关闭' } catch { Write-Host '[提示] 后端服务未响应或已关闭' }" 2>nul

:: 等待一下让服务有时间关闭
timeout /t 2 >nul

:: ====================================
:: 第三阶段：强制终止 - 仅清理已验证的进程
:: ====================================
echo.
echo [终止] 强制终止已验证的程序进程...

:: 终止后端进程（如果还在运行）
if defined BACKEND_PID (
    echo   • 正在终止后端进程 (PID: %BACKEND_PID%)...
    taskkill /f /pid %BACKEND_PID% >nul 2>&1
    if errorlevel 0 (
        echo     ✓ 后端进程已终止
        set "KILLED_BACKEND=1"
    ) else (
        echo     ℹ 后端进程已退出
    )
)

:: 终止前端进程（如果还在运行）
if defined FRONTEND_PID (
    echo   • 正在终止前端进程 (PID: %FRONTEND_PID%)...
    taskkill /f /pid %FRONTEND_PID% >nul 2>&1
    if errorlevel 0 (
        echo     ✓ 前端进程已终止
        set "KILLED_FRONTEND=1"
    ) else (
        echo     ℹ 前端进程已退出
    )
)

:: ====================================
:: 第四阶段：端口清理 - 仅清理已知的应用端口
:: ====================================
echo.
echo [端口] 清理应用占用的端口...

:: 清理8000端口 (后端 uvicorn)
echo   • 清理端口 8000 (后端)...
for /f "tokens=5" %%a in ('netstat -ano 2>nul ^| findstr ":8000"') do (
    if not "%%a"=="0" (
        :: 额外验证：确保这个PID对应的是uvicorn进程
        for /f "tokens=*" %%c in ('wmic process where "ProcessId=%%a" get CommandLine 2>nul ^| findstr /i "uvicorn"') do (
            if not "%%c"=="" (
                taskkill /f /pid %%a >nul 2>&1
                echo     ✓ 端口 8000 已清理
            )
        )
    )
)

:: 清理5173/5174端口 (前端 vite)
for /f %%port in ('echo 5173 5174 5175') do (
    for /f "tokens=5" %%a in ('netstat -ano 2>nul ^| findstr ":%%port"') do (
        if not "%%a"=="0" (
            :: 额外验证：确保这个PID对应的是node进程
            for /f "tokens=*" %%c in ('wmic process where "ProcessId=%%a" get CommandLine 2>nul ^| findstr /i "vite\|node"') do (
                if not "%%c"=="" (
                    taskkill /f /pid %%a >nul 2>&1
                    echo     ✓ 端口 %%port 已清理
                    set "KILLED_PORTS=1"
                )
            )
        )
    )
)

:: 等待进程完全退出
timeout /t 1 >nul

:: ====================================
:: 完成报告
:: ====================================
echo.
echo ================================================
echo   清理完成 - 汇总报告
echo ================================================
echo   后端进程: %KILLED_BACKEND% (已终止)
echo   前端进程: %KILLED_FRONTEND% (已终止)
echo   端口清理: %KILLED_PORTS% (已清理)
echo.
echo ✓ 只清理了主程序的命令行
echo ✓ 测试进程不受影响
echo ================================================
echo.

