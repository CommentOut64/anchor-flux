@echo off
chcp 65001 >nul
echo ========================================
echo   视频转录工具 - 一键启动
echo ========================================
echo.

REM 默认启用镜像源（提升国内下载速度）
set USE_HF_MIRROR=true
echo [配置] 🌐 已启用 HuggingFace 镜像源（国内访问优化）
echo.

REM 启动后端服务
echo [启动] 🚀 正在启动后端服务...
cd /d "%~dp0backend"
start "后端服务" cmd /k "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

REM 等待后端启动
echo [等待] ⏳ 等待后端服务启动...
timeout /t 5 /nobreak >nul

REM 启动前端服务
echo [启动] 🎨 正在启动前端服务...
cd /d "%~dp0frontend"
start "前端服务" cmd /k "npm run dev"

echo.
echo ========================================
echo   ✅ 服务启动完成！
echo ========================================
echo.
echo 📌 前端地址: http://localhost:5173
echo 📌 后端地址: http://localhost:8000
echo 📌 镜像源: https://mirrors.tuna.tsinghua.edu.cn（已启用）
echo.
echo 💡 如需禁用镜像源使用官方源，请：
echo    1. 关闭所有服务窗口
echo    2. 运行：一键启动_官方源.bat
echo.
echo ⚠️ 请不要关闭此窗口！
echo ⚠️ 如需停止服务，请关闭后端和前端窗口
echo.
pause
