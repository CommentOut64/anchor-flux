# Silero VAD ONNX 集成方案总结

## 已完成的工作

### 1. 下载并内置 Silero VAD ONNX 模型

- **源地址**: `https://huggingface.co/onnx-community/silero-vad`
- **模型文件**: `model.onnx` (2.14MB)
- **目标路径**: `backend/app/assets/silero/silero_vad.onnx`
- **下载方式**: 优先从 HuggingFace 镜像站下载（hf-mirror.com），失败后尝试官方源

### 2. 修改代码使用内置模型

#### 2.1 添加依赖（`requirements.txt`）
```
silero-vad
```

#### 2.2 修改 `transcription_service.py`
- 移除 `torch.hub` 依赖（避免 GitHub 访问问题）
- 使用 `silero-vad` 库的 `OnnxWrapper` 类
- 直接加载本地 ONNX 模型文件
- 使用 onnxruntime 推理（跨平台兼容）

**关键代码 (lines 1289-1388)**:
```python
from silero_vad import get_speech_timestamps
from silero_vad.utils_vad import OnnxWrapper

# 使用项目内置的 ONNX 模型
builtin_model_path = PathlibPath(__file__).parent.parent / "assets" / "silero" / "silero_vad.onnx"

# 加载ONNX模型（直接从本地路径）
model = OnnxWrapper(str(builtin_model_path), force_onnx_cpu=False)
```

#### 2.3 集成到模型管理器 (`model_manager_service.py`)
- 添加 `silero_vad_status` 状态字段
- 添加 `_check_silero_vad()` 检测方法
- 启动时自动检测内置模型是否存在

**关键代码 (lines 220-248)**:
```python
def _check_silero_vad(self):
    """检查内置 Silero VAD 模型是否存在"""
    builtin_model_path = PathlibPath(__file__).parent.parent / "assets" / "silero" / "silero_vad.onnx"

    if builtin_model_path.exists():
        self.silero_vad_status = {
            "status": "ready",
            "path": str(builtin_model_path),
            "size_mb": round(file_size_mb, 2),
            "type": "built-in"
        }
```

#### 2.4 优先预加载 (`model_preload_manager.py`)
- 在预加载 Whisper 模型之前先加载 Silero VAD
- 由于模型只有 2.14MB，加载速度极快

**关键代码 (lines 206-230)**:
```python
# ===== 优先加载 Silero VAD 模型（内置，快速） =====
builtin_model_path = PathlibPath(__file__).parent.parent / "assets" / "silero" / "silero_vad.onnx"

if builtin_model_path.exists():
    # 预加载 Silero VAD（验证可用性）
    _ = OnnxWrapper(str(builtin_model_path), force_onnx_cpu=False)
    self.logger.info("✅ Silero VAD 模型预加载成功（内置模型）")
```

### 3. 测试验证

创建了全面的测试脚本 `test_silero_vad.py`，包含4个测试用例：

1. ✅ 内置模型文件存在
2. ✅ silero-vad 库能正确加载内置模型
3. ✅ 模型管理器检测到 Silero VAD
4. ✅ 转录服务可以使用 Silero VAD

**测试结果**: 所有测试通过

## 技术架构

### 模型加载流程

```
启动应用
  ↓
模型管理器初始化
  ↓
检测内置 Silero VAD
  ├─ 存在 → status="ready"
  └─ 缺失 → status="missing"
  ↓
预加载管理器启动
  ↓
优先加载 Silero VAD (2MB, 快速)
  ↓
加载 Whisper 模型
```

### VAD 分段流程

```
音频数组 (memory mode)
  ↓
调用 _vad_silero()
  ↓
加载内置 ONNX 模型
  ↓
执行 VAD 推理 (onnxruntime)
  ↓
返回语音段元数据
  ├─ 检测到语音 → 返回 VAD 分段
  └─ 未检测到语音 → 降级到能量检测分段
```

## 优势

### 1. 无需网络下载
- **用户体验**: 首次使用即可工作，无需等待下载
- **网络问题**: 不受 GitHub 访问限制
- **离线使用**: 完全离线环境也能使用

### 2. 快速加载
- **模型大小**: 仅 2.14MB（比 Whisper tiny 模型小 30+ 倍）
- **加载速度**: < 1秒
- **内存占用**: 极低

### 3. 跨平台兼容
- **推理引擎**: 使用 onnxruntime（跨平台）
- **无特殊依赖**: 不依赖 torch.hub、GitHub 等外部服务

### 4. 高性能
- **ONNX 加速**: 根据官方文档，ONNX 推理速度比 JIT 快 4-5 倍
- **CPU 友好**: 支持 CPU 推理，GPU 可用时自动使用
- **低延迟**: 单个音频块（30ms+）处理时间 < 1ms

## 兼容性

### 新用户
- 项目默认包含 Silero VAD 模型
- 无需任何额外配置
- 开箱即用

### 旧用户
- 更新代码后，`silero-vad` 库会自动安装（requirements.txt）
- 内置模型随源码发布（不在 .gitignore 中）
- 自动降级机制：如果模型缺失，会提示用户重新获取项目文件

### 环境要求
- Python >= 3.8
- onnxruntime >= 1.16.1（已在 requirements.txt 中）
- torch >= 1.12.0（项目已有）
- torchaudio >= 0.12.0（项目已有）

## 文件变更

### 新增文件
1. `backend/app/assets/silero/silero_vad.onnx` (2.14MB)
2. `backend/test_silero_vad.py` (测试脚本)

### 修改文件
1. `backend/app/services/transcription_service.py` (lines 1289-1388)
2. `backend/app/services/model_manager_service.py` (lines 136, 149, 220-248)
3. `backend/app/services/model_preload_manager.py` (lines 206-230)
4. `requirements.txt` (+1 行: silero-vad)

## 后续建议

1. **前端展示**: 在模型管理页面显示 Silero VAD 状态（已内置）
2. **文档更新**: 在用户手册中说明 VAD 模型为内置，无需下载
3. **版本管理**: 未来如需更新 VAD 模型，可提供自动更新机制

## 技术参考

- Silero VAD GitHub: https://github.com/snakers4/silero-vad
- ONNX Community: https://huggingface.co/onnx-community/silero-vad
- silero-vad PyPI: https://pypi.org/project/silero-vad/
- 官方文档: https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies
