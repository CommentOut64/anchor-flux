@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置标题
title Video-to-SRT 进程诊断工具

echo ================================================
echo   Video-to-SRT 进程诊断工具
echo ================================================
echo.
echo 本工具将显示当前运行的主程序相关进程
echo.

:: ====================================
:: 检查后端进程
:: ====================================
echo [诊断] 后端进程...
echo.

set "BACKEND_COUNT=0"
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv ^| findstr /v "PID"') do (
    if not "%%i"=="" (
        for /f "tokens=*" %%c in ('wmic process where "ProcessId=%%i" get CommandLine 2>nul') do (
            if "%%c"=="" goto next_python
            
            :: 显示完整命令行
            if not "%%c"=="CommandLine" (
                echo   PID: %%i
                echo   命令: %%c
                
                :: 检查是否为后端
                echo %%c | findstr /i "uvicorn.*main" >nul
                if not errorlevel 1 (
                    echo   ✓ 确认：这是后端服务进程
                    set /a BACKEND_COUNT+=1
                ) else (
                    echo   ✗ 检查结果：不是后端服务进程
                )
                echo.
            )
        )
    )
    :next_python
)

if %BACKEND_COUNT% equ 0 (
    echo   ℹ️  未找到后端服务进程
    echo.
)

:: ====================================
:: 检查前端进程
:: ====================================
echo [诊断] 前端进程...
echo.

set "FRONTEND_COUNT=0"
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq node.exe" /fo csv ^| findstr /v "PID"') do (
    if not "%%i"=="" (
        for /f "tokens=*" %%c in ('wmic process where "ProcessId=%%i" get CommandLine 2>nul') do (
            if "%%c"=="" goto next_node
            
            :: 显示完整命令行
            if not "%%c"=="CommandLine" (
                echo   PID: %%i
                echo   命令: %%c
                
                :: 检查是否为前端
                echo %%c | findstr /i "npm.*dev\|vite" >nul
                if not errorlevel 1 (
                    echo   ✓ 确认：这是前端服务进程
                    set /a FRONTEND_COUNT+=1
                ) else (
                    echo   ✗ 检查结果：不是前端服务进程
                )
                echo.
            )
        )
    )
    :next_node
)

if %FRONTEND_COUNT% equ 0 (
    echo   ℹ️  未找到前端服务进程
    echo.
)

:: ====================================
:: 检查端口占用
:: ====================================
echo [诊断] 端口占用...
echo.

:: 检查 8000 端口
echo   端口 8000 (后端):
netstat -ano 2>nul | findstr ":8000" >nul
if not errorlevel 1 (
    for /f "tokens=5" %%a in ('netstat -ano 2>nul ^| findstr ":8000"') do (
        echo     • PID %%a 正在占用此端口
        for /f "tokens=*" %%c in ('wmic process where "ProcessId=%%a" get CommandLine 2>nul ^| findstr /v "CommandLine"') do (
            if not "%%c"=="" (
                echo       命令: %%c
            )
        )
    )
) else (
    echo     ✓ 端口空闲
)
echo.

:: 检查 5173-5175 端口
echo   端口 5173-5175 (前端):
for /f %%port in ('echo 5173 5174 5175') do (
    netstat -ano 2>nul | findstr ":%%port" >nul
    if not errorlevel 1 (
        for /f "tokens=5" %%a in ('netstat -ano 2>nul ^| findstr ":%%port"') do (
            echo     • PID %%a 正在占用端口 %%port
            for /f "tokens=*" %%c in ('wmic process where "ProcessId=%%a" get CommandLine 2>nul ^| findstr /v "CommandLine"') do (
                if not "%%c"=="" (
                    echo       命令: %%c
                )
            )
        )
    ) else (
        echo     • 端口 %%port - 空闲
    )
)
echo.

:: ====================================
:: 诊断报告
:: ====================================
echo ================================================
echo   诊断报告摘要
echo ================================================
echo   后端进程: %BACKEND_COUNT%
echo   前端进程: %FRONTEND_COUNT%
echo.

if %BACKEND_COUNT% gtr 0 (
    echo ✓ 后端服务已运行
) else (
    echo ℹ️  后端服务未运行
)

if %FRONTEND_COUNT% gtr 0 (
    echo ✓ 前端服务已运行
) else (
    echo ℹ️  前端服务未运行
)

echo.
echo ================================================

pause
