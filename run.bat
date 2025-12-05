@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

:: 设置窗口标题
title Video to SRT GPU Dashboard

:: 获取当前目录
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

:: 定义 Python 解释器路径
set "PYTHON_EXEC=%PROJECT_ROOT%\.venv\Scripts\python.exe"

:: 检查 Python 解释器是否存在
if not exist "%PYTHON_EXEC%" (
    echo [ERROR] 未找到 Python 解释器
    echo [INFO] 正在尝试初始化 UV 环境...
    
    :: 如果你希望脚本自动创建 venv，可在下面添加命令
    :: 否则报错提示用户
    echo 请执行: uv venv .venv
    pause
    exit /b 1
)

:: 启动 Python 启动器
"%PYTHON_EXEC%" "%PROJECT_ROOT%\bootloader.py"

:: 检查 Python 脚本异常退出则暂停显示错误
if %errorlevel% neq 0 (
    echo.
    echo [程序异常退出]
    pause
)