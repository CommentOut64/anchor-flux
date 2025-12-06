# 废弃文件归档

这些文件已被废弃，仅保留作为历史参考。

## 目录结构

```
archive/
├── README.md              # 本文件
├── requirements.txt       # 旧的 conda 环境导出（含 whisperx）
├── backend_legacy/        # 废弃的后端代码
│   ├── debug_main.py
│   ├── main_refactored.py
│   ├── main_simple.py
│   ├── processor.py
│   └── video_to_srt_old.py
└── frontend_legacy/       # 废弃的前端代码
    ├── index.html
    ├── package.json
    ├── public/
    └── src/
```

## 后端废弃文件 (backend_legacy/)

### 早期主入口文件

- `main_refactored.py` - 早期重构尝试（68行）
  - 曾尝试使用 transcription_service 的简化版本
  - 已被完整版 main.py 替代
  - 归档时间：2025-11-18

- `main_simple.py` - 简化版本（94行）
  - 早期的简化实现
  - 已被完整版 main.py 替代
  - 归档时间：2025-11-18

- `debug_main.py` - 调试用主入口（39行）
  - 含硬编码绝对路径 (F:\video_to_srt_gpu)
  - 简单的调试API接口
  - 已被完整版 main.py 替代
  - 归档时间：2025-11-18

- `video_to_srt_old.py` - 原始 CLI 版本
  - 使用 Rich 库的命令行界面
  - 已被 Web 前端替代
  - 归档时间：2025-11-18

### 核心处理模块

- `processor.py` - 原处理器模块（82行简化版）
  - **原始功能**：转录处理 + 模型管理 + CPU亲和性（533行）
  - **迁移记录**：
    - `TranscriptionProcessor` → `services/transcription_service.py`
    - `CPUAffinityManager` → `services/cpu_affinity_service.py`
    - `initialize_model_manager`等模型管理函数 → `services/model_preload_manager.py`
  - **简化版**：仅保留模型管理门面函数（最终也被替代）
  - 归档时间：2025-11-18

## 前端废弃文件 (frontend_legacy/)

- 完整的旧版 Vue 3 前端代码
- 包含组件、服务、stores 等
- 已被当前 frontend/ 目录下的新版本替代
- 归档时间：2025-11-28

## 其他文件

- `requirements.txt` - 旧的 conda 环境导出
  - 包含 whisperx 等已废弃的依赖
  - 当前项目使用 uv 管理依赖

## 当前唯一入口

**backend/app/main.py** - 项目的唯一主入口文件

## 说明

- 这些文件不应再被使用或修改
- 如需参考历史代码，请查看 git 历史记录
- 所有新功能都应在 backend/app/main.py 和对应的服务模块中开发
- 模型管理功能现在完全由 `services/model_preload_manager.py` 提供

## 最后更新

- 更新日期：2025-12-06
- 更新内容：合并 _deprecated 和 archive 目录，统一归档结构
