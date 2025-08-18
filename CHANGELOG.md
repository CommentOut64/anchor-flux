# Changelog

All notable changes to the Video-to-SRT project will be documented in this file.

## [2.0.0] - 2025-08-18 - 🚀 全面架构升级

### 🎯 Major Changes - 重大变更

- **完全重构**：从单一Python脚本重构为现代化前后端分离架构
- **Web界面**：新增Vue.js前端，提供直观的用户界面
- **API后端**：采用FastAPI构建RESTful API服务
- **并发处理**：支持多任务并发处理，提升处理效率

### ✨ New Features - 新功能

#### 🖥️ Frontend (前端)

- **现代化UI**：基于Vue 3 + Vite的响应式Web界面
- **文件上传**：拖拽上传支持，支持多种视频格式
- **实时进度**：实时显示转录进度和状态
- **结果预览**：在线预览和下载SRT字幕文件
- **任务管理**：查看历史任务和状态

#### 🔧 Backend (后端)
- **FastAPI框架**：高性能异步API服务
- **任务队列**：基于线程的任务处理系统
- **状态管理**：完整的任务生命周期管理
- **错误处理**：详细的错误信息和恢复机制
- **文件管理**：安全的文件上传和存储机制

#### 🤖 AI Processing (AI处理)
- **WhisperX集成**：使用最新的WhisperX模型
- **GPU加速**：支持CUDA GPU加速处理
- **多语言支持**：自动语言检测和多语言转录
- **音频分段**：智能音频分段，提升长视频处理效果
- **批量处理**：支持多文件批量转录

#### 🚀 Launcher System (启动器系统)
- **一键启动**：智能启动器，自动处理依赖和环境
- **进程管理**：自动清理残留进程，确保干净启动
- **错误诊断**：详细的错误诊断和解决建议
- **多种启动方式**：
  - `一键启动.bat` - 生产环境推荐
  - `start.bat` - 开发调试使用
  - `simple_launcher.py` - Python启动器
  - `launcher_debug.py` - 调试版本

### 🛠️ Technical Improvements - 技术改进

#### 架构升级
- **前后端分离**：解耦前端UI和后端处理逻辑
- **RESTful API**：标准化的API接口设计
- **异步处理**：提升并发处理能力
- **模块化设计**：更好的代码组织和可维护性

#### 开发体验
- **热重载**：前后端开发时支持热重载
- **环境隔离**：清晰的开发和生产环境配置
- **依赖管理**：规范化的依赖管理
- **文档完善**：详细的使用说明和API文档

#### 性能优化
- **内存优化**：改进的内存使用和垃圾回收
- **处理速度**：优化的音频处理管道
- **资源利用**：更好的CPU和GPU资源利用

### 🔧 Configuration - 配置

#### 新增配置文件

```
frontend/
├── package.json          # 前端依赖配置
├── vite.config.js        # Vite构建配置
└── src/                  # 前端源码

backend/
├── app/
│   ├── main.py          # FastAPI应用入口
│   └── processor.py     # 转录处理核心
└── requirements.txt     # 后端依赖

launchers/
├── 一键启动.bat         # 推荐启动方式
├── simple_launcher.py  # Python启动器
└── launcher_debug.py   # 调试启动器
```

### 📁 File Structure - 文件结构变更

#### 新增目录
- `frontend/` - Vue.js前端应用
- `backend/` - FastAPI后端服务
- `jobs/` - 任务文件存储目录

#### 已废弃文件
- `video_to_srt.py` - 原始单文件脚本（保留作参考）
- `run.bat` - 旧版启动脚本

### 🚀 Migration Guide - 迁移指南

#### 从1.x版本升级

1. **备份数据**：备份现有的视频文件和字幕文件
2. **安装依赖**：
   ```bash
   # Python依赖
   pip install -r requirements.txt
   
   # 前端依赖
   cd frontend
   npm install
   ```
3. **启动应用**：
   ```bash
   # 推荐方式
   双击 一键启动.bat
   ```

#### 主要变化
- **UI界面**：从命令行界面迁移到Web界面
- **文件处理**：通过Web界面上传，而非命令行参数
- **结果获取**：通过Web界面下载，而非文件系统查找

### Documentation - 文档

- 新增 `LAUNCHER_README.md` - 启动器使用说明
- 更新 `README.md` - 项目概述和快速开始
- 新增 API 文档（通过 FastAPI 自动生成）

### What's Next - 下一步计划

- [ ] 字幕编辑和后处理功能
- [ ] Docker容器化部署

---

## [1.1.0] - 2024-06-18 - 🎬 初始版本

### Features - 功能

- **基础转录**：使用Whisper模型进行视频到字幕转录
- **命令行界面**：Rich库提供的美观命令行界面
- **多格式支持**：支持常见的视频和音频格式
- **GPU加速**：支持CUDA GPU加速处理

### Technical Stack - 技术栈

- Python 3.8+
- Whisper/WhisperX
- Rich (CLI UI)
- FFmpeg (音频处理)

