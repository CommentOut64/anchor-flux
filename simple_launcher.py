#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-to-SRT 简化启动器
避免exe打包问题，直接使用Python运行
"""

import os
import sys
import time
import signal
import psutil
import subprocess
import threading
import webbrowser
from pathlib import Path
import requests
from datetime import datetime

class SimpleVideoToSRTLauncher:
    def __init__(self):
        self.script_dir = Path(__file__).parent.absolute()
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
        # 配置
        self.backend_port = 8000
        self.frontend_port = 5174
        self.backend_host = "127.0.0.1"
    
    def log(self, message, level="INFO"):
        """统一日志输出"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icon = {"INFO": "ℹ️", "ERROR": "❌", "SUCCESS": "✅", "WARNING": "⚠️"}.get(level, "ℹ️")
        print(f"[{timestamp}] {icon} {message}")
        sys.stdout.flush()
    
    def kill_existing_processes(self):
        """清理之前的进程"""
        self.log("清理之前的进程...")
        
        processes_killed = 0
        for port in [self.backend_port, self.frontend_port]:
            try:
                for conn in psutil.net_connections():
                    if hasattr(conn, 'laddr') and conn.laddr.port == port:
                        try:
                            process = psutil.Process(conn.pid)
                            process.terminate()
                            processes_killed += 1
                        except:
                            pass
            except:
                pass
        
        if processes_killed > 0:
            self.log(f"已清理 {processes_killed} 个进程", "SUCCESS")
            time.sleep(2)
        else:
            self.log("没有发现需要清理的进程")
    
    def start_backend(self):
        """启动后端服务"""
        self.log("启动后端服务...")
        
        backend_dir = self.script_dir / "backend"
        if not backend_dir.exists():
            self.log("backend 目录不存在", "ERROR")
            return False
        
        try:
            # 直接使用python main.py而不是uvicorn
            cmd = [sys.executable, "app/main.py"]
            
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=backend_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
                env=os.environ.copy()
            )
            
            # 等待后端启动
            for i in range(30):
                try:
                    response = requests.get(f"http://{self.backend_host}:{self.backend_port}/api/ping", timeout=2)
                    if response.status_code == 200:
                        self.log("后端服务已启动", "SUCCESS")
                        return True
                except:
                    pass
                time.sleep(1)
            
            self.log("后端服务启动超时", "ERROR")
            return False
            
        except Exception as e:
            self.log(f"启动后端服务失败: {e}", "ERROR")
            return False
    
    def find_npm_path(self):
        """查找npm的完整路径"""
        import shutil
        
        # 首先尝试shutil.which
        npm_path = shutil.which('npm')
        if npm_path and os.path.exists(npm_path):
            return npm_path
        
        # 如果找不到，尝试常见路径
        possible_paths = [
            r"C:\Program Files\nodejs\npm.cmd",
            r"C:\Program Files (x86)\nodejs\npm.cmd",
        ]
        
        username = os.environ.get('USERNAME', '')
        if username:
            possible_paths.append(rf"C:\Users\{username}\AppData\Roaming\npm\npm.cmd")
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def start_frontend(self):
        """启动前端服务"""
        self.log("启动前端服务...")
        
        frontend_dir = self.script_dir / "frontend"
        if not frontend_dir.exists():
            self.log("frontend 目录不存在", "ERROR")
            return False
        
        # 查找npm路径
        npm_path = self.find_npm_path()
        if not npm_path:
            self.log("未找到npm命令，请确保Node.js已正确安装", "ERROR")
            return False
        
        # 检查 node_modules
        if not (frontend_dir / "node_modules").exists():
            self.log("安装前端依赖...")
            try:
                subprocess.run([npm_path, "install"], cwd=frontend_dir, check=True)
                self.log("前端依赖安装完成", "SUCCESS")
            except Exception as e:
                self.log(f"前端依赖安装失败: {e}", "ERROR")
                return False
        
        try:
            cmd = [npm_path, "run", "dev"]
            
            self.frontend_process = subprocess.Popen(
                cmd,
                cwd=frontend_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
                env=os.environ.copy()
            )
            
            # 等待前端启动
            for i in range(30):
                try:
                    response = requests.get(f"http://localhost:{self.frontend_port}", timeout=2)
                    if response.status_code == 200:
                        self.log("前端服务已启动", "SUCCESS")
                        return True
                except:
                    pass
                time.sleep(1)
            
            self.log("前端服务启动超时", "ERROR")
            return False
            
        except Exception as e:
            self.log(f"启动前端服务失败: {e}", "ERROR")
            return False
    
    def cleanup(self):
        """清理资源"""
        self.log("正在关闭服务...")
        self.running = False
        
        for process, name in [(self.backend_process, "后端"), (self.frontend_process, "前端")]:
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    self.log(f"{name}服务已关闭", "SUCCESS")
                except:
                    try:
                        process.kill()
                    except:
                        pass
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        self.log("接收到关闭信号，正在清理...")
        self.cleanup()
        sys.exit(0)
    
    def run(self):
        """主运行函数"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            print("=" * 60)
            self.log("🚀 Video-to-SRT 应用启动器")
            print("=" * 60)
            
            # 1. 清理之前的进程
            self.kill_existing_processes()
            
            # 2. 启动后端
            if not self.start_backend():
                self.log("后端启动失败，退出", "ERROR")
                input("按回车键退出...")
                return False
            
            # 3. 启动前端
            if not self.start_frontend():
                self.log("前端启动失败，退出", "ERROR")
                self.cleanup()
                input("按回车键退出...")
                return False
            
            # 4. 打开浏览器
            try:
                url = f"http://localhost:{self.frontend_port}"
                self.log(f"打开浏览器: {url}")
                webbrowser.open(url)
            except Exception as e:
                self.log(f"打开浏览器失败: {e}", "WARNING")
            
            # 5. 显示成功信息
            print("=" * 60)
            self.log("✅ 服务启动完成！", "SUCCESS")
            self.log(f"前端地址: http://localhost:{self.frontend_port}")
            self.log(f"后端地址: http://{self.backend_host}:{self.backend_port}")
            print("=" * 60)
            print()
            print("📌 注意事项：")
            print("   • 前后端服务在独立的命令行窗口中运行")
            print("   • 请保持这些窗口打开")
            print("   • 按 Ctrl+C 退出并停止所有服务")
            print()
            
            # 主线程等待
            while self.running:
                time.sleep(1)
            
            return True
            
        except KeyboardInterrupt:
            self.log("用户中断，正在退出...")
        except Exception as e:
            self.log(f"运行时出错: {e}", "ERROR")
            input("按回车键退出...")
        finally:
            self.cleanup()
        
        return False

def main():
    launcher = SimpleVideoToSRTLauncher()
    success = launcher.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
