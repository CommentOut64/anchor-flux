@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: 设置标题
title Video-to-SRT 进程清理器 (全清理版 - 谨慎使用！)

echo ================================================
echo   Video-to-SRT 进程清理器 - 全清理模式
echo ================================================
echo.
echo ⚠️  警告：此脚本将清理所有 Python 和 Node.js 进程
echo           包括其他测试和开发进程！
echo.
echo 请确认是否继续？(Y/N)
set /p confirm=">> "

if /i not "%confirm%"=="Y" (
    echo 已取消操作
    pause
    exit /b
)

echo.
echo [信息] 正在全量清理所有相关进程...
echo.

:: 尝试优雅关闭后端服务
echo [信息] 尝试优雅关闭后端服务...
powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/shutdown' -Method Post -TimeoutSec 5 | Out-Null; Write-Host '[成功] 后端服务已优雅关闭' } catch { Write-Host '[提示] 后端服务未响应或已关闭' }" 2>nul

:: 等待一下让服务有时间关闭
timeout /t 2 >nul

echo [信息] 强制终止所有 Python 和 Node.js 进程...

:: 终止所有 Python 进程
echo   • 清理 python.exe...
taskkill /f /im python.exe >nul 2>&1

echo   • 清理 python3.exe...
taskkill /f /im python3.exe >nul 2>&1

echo   • 清理 pythonw.exe...
taskkill /f /im pythonw.exe >nul 2>&1

:: 终止所有 Node.js 进程
echo   • 清理 node.exe...
taskkill /f /im node.exe >nul 2>&1

echo   • 清理 npm.exe...
taskkill /f /im npm.exe >nul 2>&1

echo   • 清理 npx.exe...
taskkill /f /im npx.exe >nul 2>&1

:: 清理占用端口的进程（无需验证）
echo.
echo [信息] 清理占用的应用端口...

:: 清理 8000 端口（后端）
echo   • 清理端口 8000...
for /f "tokens=5" %%a in ('netstat -ano 2>nul ^| findstr ":8000"') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: 清理 5173/5174/5175 端口（前端）
echo   • 清理端口 5173-5175...
for /f "tokens=5" %%a in ('netstat -ano 2>nul ^| findstr ":517[3-5]"') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: 清理 3000 端口（备用前端）
echo   • 清理端口 3000...
for /f "tokens=5" %%a in ('netstat -ano 2>nul ^| findstr ":3000"') do (
    taskkill /f /pid %%a >nul 2>&1
)

:: 等待进程完全退出
timeout /t 1 >nul

echo.
echo ================================================
echo   清理完成
echo ================================================
echo.
echo ✓ 已清理所有 Python 进程
echo ✓ 已清理所有 Node.js 进程
echo ✓ 已清理所有应用占用的端口
echo.
echo ⚠️  其他 Python/Node 应用可能也被关闭了
echo.

pause
