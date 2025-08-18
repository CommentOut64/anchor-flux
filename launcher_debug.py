#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-to-SRT 应用启动器 - 调试版本
增加详细的错误输出和调试信息
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
import traceback

class VideoToSRTLauncher:
    def __init__(self):
        # 正确获取脚本/exe所在目录
        if getattr(sys, 'frozen', False):
            # 如果是打包的exe文件
            self.script_dir = Path(sys.executable).parent.absolute()
        else:
            # 如果是Python脚本
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
        print(f"[{timestamp}] [{level}] {message}")
        sys.stdout.flush()  # 确保立即输出
    
    def check_environment(self):
        """检查运行环境"""
        self.log("=== 环境检查 ===")
        
        # 检查Python
        self.log(f"Python版本: {sys.version}")
        self.log(f"Python路径: {sys.executable}")
        
        # 检查工作目录
        self.log(f"工作目录: {self.script_dir}")
        self.log(f"后端目录存在: {(self.script_dir / 'backend').exists()}")
        self.log(f"前端目录存在: {(self.script_dir / 'frontend').exists()}")
        
        # 检查必要的Python包
        packages_to_check = ['uvicorn', 'fastapi', 'psutil', 'requests']
        for package in packages_to_check:
            try:
                __import__(package)
                self.log(f"✅ {package} 已安装")
            except ImportError:
                self.log(f"❌ {package} 未安装", "ERROR")
                return False
        
        # 检查Node.js
        npm_path = self.find_npm_path()
        if npm_path:
            self.log(f"✅ npm路径: {npm_path}")
            try:
                result = subprocess.run([npm_path, "--version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.log(f"✅ npm版本: {result.stdout.strip()}")
                else:
                    self.log(f"❌ npm版本检查失败: {result.stderr}", "ERROR")
            except Exception as e:
                self.log(f"❌ npm版本检查异常: {e}", "ERROR")
        else:
            self.log("❌ 未找到npm", "ERROR")
            return False
        
        return True
    
    def kill_existing_processes(self):
        """清理之前的进程"""
        self.log("检测并清理之前的进程...")
        
        processes_killed = 0
        
        # 清理占用端口的进程
        for port in [self.backend_port, self.frontend_port]:
            try:
                for conn in psutil.net_connections():
                    if conn.laddr.port == port and conn.status == 'LISTEN':
                        try:
                            process = psutil.Process(conn.pid)
                            process_name = process.name()
                            self.log(f"终止占用端口 {port} 的进程: {process_name} (PID: {conn.pid})")
                            process.terminate()
                            process.wait(timeout=3)
                            processes_killed += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                            try:
                                process.kill()
                            except:
                                pass
            except Exception as e:
                self.log(f"清理端口 {port} 时出错: {e}", "WARNING")
        
        if processes_killed > 0:
            self.log(f"已清理 {processes_killed} 个进程")
            time.sleep(2)
        else:
            self.log("没有发现需要清理的进程")
    
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
    
    def wait_for_service(self, url, service_name, timeout=60):
        """等待服务启动"""
        self.log(f"等待 {service_name} 启动...")
        
        for i in range(timeout):
            if not self.running:
                return False
                
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    self.log(f"{service_name} 已启动并正常响应")
                    return True
            except requests.exceptions.RequestException as e:
                if i == 0:  # 只在第一次记录详细错误
                    self.log(f"等待 {service_name} 时的连接错误: {e}", "DEBUG")
            
            if i % 10 == 0 and i > 0:
                self.log(f"等待 {service_name} 启动中... ({i}s)")
            
            time.sleep(1)
        
        self.log(f"{service_name} 启动超时", "ERROR")
        return False
    
    def start_backend(self):
        """启动后端服务"""
        self.log("启动后端服务...")
        
        backend_dir = self.script_dir / "backend"
        if not backend_dir.exists():
            self.log("backend 目录不存在", "ERROR")
            return False
        
        # 检查main.py文件
        main_py = backend_dir / "app" / "main.py"
        if not main_py.exists():
            self.log(f"未找到 {main_py}", "ERROR")
            return False
        
        try:
            # 使用 uvicorn 启动
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "app.main:app", 
                "--host", self.backend_host,
                "--port", str(self.backend_port),
                "--reload"
            ]
            
            self.log(f"后端启动命令: {' '.join(cmd)}")
            self.log(f"工作目录: {backend_dir}")
            
            # 确保环境变量完整传递
            env = os.environ.copy()
            
            # Windows下隐藏窗口（仅在exe模式下）
            startupinfo = None
            creationflags = 0
            
            if os.name == 'nt' and getattr(sys, 'frozen', False):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                startupinfo=startupinfo,
                creationflags=creationflags,
                text=True
            )
            
            # 检查进程是否立即退出
            time.sleep(2)
            if self.backend_process.poll() is not None:
                stdout, stderr = self.backend_process.communicate()
                self.log(f"后端进程立即退出，返回码: {self.backend_process.returncode}", "ERROR")
                self.log(f"标准输出: {stdout}", "ERROR")
                self.log(f"错误输出: {stderr}", "ERROR")
                return False
            
            # 等待后端启动
            if self.wait_for_service(f"http://{self.backend_host}:{self.backend_port}/api/ping", "后端服务"):
                return True
            else:
                # 如果启动失败，获取错误信息
                if self.backend_process.poll() is not None:
                    stdout, stderr = self.backend_process.communicate()
                    self.log(f"后端启动失败，输出: {stdout}", "ERROR")
                    self.log(f"后端启动失败，错误: {stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"启动后端服务时出错: {e}", "ERROR")
            self.log(f"详细错误: {traceback.format_exc()}", "ERROR")
            return False
    
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
        
        self.log(f"使用npm路径: {npm_path}")
        
        # 检查 package.json
        package_json = frontend_dir / "package.json"
        if not package_json.exists():
            self.log("未找到 package.json", "ERROR")
            return False
        
        # 检查 node_modules
        if not (frontend_dir / "node_modules").exists():
            self.log("安装前端依赖...")
            try:
                npm_install = subprocess.run(
                    [npm_path, "install"], 
                    cwd=frontend_dir, 
                    capture_output=True, 
                    text=True,
                    env=os.environ.copy(),
                    timeout=300
                )
                if npm_install.returncode != 0:
                    self.log(f"npm install 失败: {npm_install.stderr}", "ERROR")
                    self.log(f"npm install 输出: {npm_install.stdout}", "ERROR")
                    return False
                self.log("前端依赖安装完成")
            except Exception as e:
                self.log(f"安装前端依赖时出错: {e}", "ERROR")
                return False
        
        try:
            cmd = [npm_path, "run", "dev"]
            self.log(f"前端启动命令: {' '.join(cmd)}")
            
            # 确保环境变量完整传递
            env = os.environ.copy()
            
            # Windows下隐藏窗口（仅在exe模式下）
            startupinfo = None
            creationflags = 0
            
            if os.name == 'nt' and getattr(sys, 'frozen', False):
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            
            self.frontend_process = subprocess.Popen(
                cmd,
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                startupinfo=startupinfo,
                creationflags=creationflags,
                text=True
            )
            
            # 等待前端启动
            if self.wait_for_service(f"http://localhost:{self.frontend_port}", "前端服务"):
                return True
            else:
                # 如果启动失败，获取错误信息
                if self.frontend_process.poll() is not None:
                    stdout, stderr = self.frontend_process.communicate()
                    self.log(f"前端启动失败，输出: {stdout}", "ERROR")
                    self.log(f"前端启动失败，错误: {stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"启动前端服务时出错: {e}", "ERROR")
            self.log(f"详细错误: {traceback.format_exc()}", "ERROR")
            return False
    
    def open_browser(self):
        """打开浏览器"""
        try:
            url = f"http://localhost:{self.frontend_port}"
            self.log(f"打开浏览器: {url}")
            webbrowser.open(url)
        except Exception as e:
            self.log(f"打开浏览器失败: {e}", "WARNING")
    
    def monitor_processes(self):
        """监控进程状态"""
        while self.running:
            try:
                # 检查后端进程
                if self.backend_process and self.backend_process.poll() is not None:
                    self.log("后端进程意外退出", "ERROR")
                    stdout, stderr = self.backend_process.communicate()
                    self.log(f"后端退出输出: {stdout}", "ERROR")
                    self.log(f"后端退出错误: {stderr}", "ERROR")
                    self.running = False
                    break
                
                # 检查前端进程
                if self.frontend_process and self.frontend_process.poll() is not None:
                    self.log("前端进程意外退出", "ERROR")
                    stdout, stderr = self.frontend_process.communicate()
                    self.log(f"前端退出输出: {stdout}", "ERROR")
                    self.log(f"前端退出错误: {stderr}", "ERROR")
                    self.running = False
                    break
                
                time.sleep(5)
            except Exception as e:
                self.log(f"监控进程时出错: {e}", "WARNING")
                time.sleep(5)
    
    def cleanup(self):
        """清理资源"""
        self.log("正在关闭服务...")
        self.running = False
        
        # 终止进程
        for process, name in [(self.backend_process, "后端"), (self.frontend_process, "前端")]:
            if process:
                try:
                    if os.name == 'nt':
                        # Windows使用taskkill
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], 
                                     capture_output=True)
                    else:
                        process.terminate()
                        process.wait(timeout=5)
                    self.log(f"{name}服务已关闭")
                except Exception as e:
                    self.log(f"关闭{name}服务时出错: {e}", "WARNING")
                    try:
                        if process.poll() is None:
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
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            self.log("=== Video-to-SRT 应用启动器 (调试版本) ===")
            
            # 0. 环境检查
            if not self.check_environment():
                self.log("环境检查失败，请检查上述错误", "ERROR")
                input("按回车键退出...")
                return False
            
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
            self.open_browser()
            
            # 5. 开始监控
            self.log("=== 服务启动完成 ===")
            self.log(f"前端地址: http://localhost:{self.frontend_port}")
            self.log(f"后端地址: http://{self.backend_host}:{self.backend_port}")
            self.log("按 Ctrl+C 退出")
            
            monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
            monitor_thread.start()
            
            # 主线程等待
            while self.running:
                time.sleep(1)
            
            return True
            
        except KeyboardInterrupt:
            self.log("用户中断，正在退出...")
        except Exception as e:
            self.log(f"运行时出错: {e}", "ERROR")
            self.log(f"详细错误: {traceback.format_exc()}", "ERROR")
            input("按回车键退出...")
        finally:
            self.cleanup()
        
        return False

def main():
    launcher = VideoToSRTLauncher()
    success = launcher.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
