# Demucs 模型下载说明

## 重要说明

⚠️ **Demucs 模型不支持国内镜像加速**

Demucs 官方模型托管在 Facebook CDN (`dl.fbaipublicfiles.com`)，**不是** HuggingFace。
目前没有可用的国内镜像源。

官方下载地址示例：
```
https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th
```

## 模型下载加速方案

### 方案1：使用代理（推荐）

配置系统代理或使用 VPN 加速下载。

**环境变量方式：**
```bash
# Windows (PowerShell)
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"

# Linux/Mac
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
```

### 方案2：手动下载模型

1. **找到模型缓存目录：**
   - Windows: `%USERPROFILE%\.cache\torch\hub\checkpoints\`
   - Linux/Mac: `~/.cache/torch/hub/checkpoints/`

2. **手动下载模型文件：**

   | 模型 | 下载链接 |
   |------|---------|
   | `htdemucs_ft` | https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th |
   | `htdemucs` | https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th |
   | `mdx_extra` | https://dl.fbaipublicfiles.com/demucs/mdx_final/0d19c1c6-0f06f20e.th |
   | `mdx_extra_q` | https://dl.fbaipublicfiles.com/demucs/mdx_final/83fc094f-4a16d450.th |

3. **放到缓存目录即可**

### 方案3：云服务器中转

在海外云服务器上下载模型，然后传到本地。

## 模型大小参考

| 模型 | 大小 | 用途 |
|------|------|------|
| `htdemucs_ft` | ~70MB | 默认模型（推荐） |
| `htdemucs` | ~80MB | 快速模式 |
| `mdx_extra_q` | ~25MB | 强BGM场景 |
| `mdx_extra` | ~100MB | 升级/兜底模型 |

## 下载日志示例

```
[INFO] 加载模型: htdemucs_ft (官方源)
Downloading: "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th"
[INFO] ✓ 模型加载成功: htdemucs_ft
```

## 缓存说明

模型首次下载后会被缓存，后续使用无需重新下载：

```
~/.cache/torch/hub/checkpoints/
├── f7e0c4bc-ba3fe64a.th   # htdemucs_ft
├── 955717e8-8726e21a.th   # htdemucs
├── 0d19c1c6-0f06f20e.th   # mdx_extra
└── 83fc094f-4a16d450.th   # mdx_extra_q
```

## 故障排除

### 下载超时

如果下载超时，可以：
1. 使用代理
2. 手动下载模型文件到缓存目录
3. 多次重试（已下载部分会被缓存）

### 模型加载失败

检查缓存目录中的模型文件是否完整：
```bash
# 检查文件大小
dir %USERPROFILE%\.cache\torch\hub\checkpoints\
```

如果文件不完整，删除后重新下载。

## 自定义镜像源

### 方式1：环境变量（推荐）

在启动应用前设置环境变量：

**Windows (PowerShell)：**
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python main.py
```

**Windows (CMD)：**
```cmd
set HF_ENDPOINT=https://hf-mirror.com
python main.py
```

**Linux/Mac：**
```bash
export HF_ENDPOINT="https://hf-mirror.com"
python main.py
```

### 方式2：配置文件

在 `user_config.json` 中添加：
```json
{
  "hf_endpoint": "https://hf-mirror.com"
}
```

（需要代码支持此配置源）

## 可用镜像源

| 镜像源 | URL | 地区 | 说明 |
|--------|-----|------|------|
| HF Mirror | `https://hf-mirror.com` | 国内 | ✅ 推荐（最快） |
| 官方源 | `https://huggingface.co` | 国外 | 可能较慢 |

## 模型大小参考

| 模型 | 大小 | 场景 |
|------|------|------|
| `htdemucs_ft` | ~70MB | 默认模型（推荐） |
| `mdx_extra_q` | ~25MB | 强BGM场景 |
| `mdx_extra` | ~100MB | 兜底升级模型 |

## 模型缓存位置

首次下载后，模型会被缓存到本地，后续使用无需重新下载：

- **Windows**：`%USERPROFILE%\.cache\demucs\`
- **Linux/Mac**：`~/.cache/demucs/`

缓存的模型文件：
```
~/.cache/demucs/
├── htdemucs_ft.pt     # Fine-tuned 模型
├── mdx_extra_q.pt     # 量化模型
└── mdx_extra.pt       # 完整模型
```

## 故障排除

### 确认镜像已生效

检查日志中的下载 URL：

❌ **未生效（官方源）：**
```
Downloading: "https://dl.fbaipublicfiles.com/demucs/..."
```

✅ **已生效（国内镜像）：**
```
[INFO] 从国内镜像加载: htdemucs_ft
[INFO] 下载模型: https://hf-mirror.com/facebook/demucs/resolve/main/htdemucs_ft.th
```

### 镜像无法访问

如果国内镜像不可用或连接超时，系统会自动回退到官方源。检查日志确认：

```
[WARNING] 镜像加载失败: ..., 切换到官方源
[INFO] 从官方源加载: htdemucs_ft
[INFO] ✓ 官方源加载成功
```

### 模型下载失败

检查网络连接并清除缓存：

```bash
# Windows
rmdir /s %USERPROFILE%\.cache\demucs\

# Linux/Mac
rm -rf ~/.cache/demucs/
```

然后重新启动应用，会重新下载模型。

### 下载速度缓慢

1. 确认国内镜像已启用（日志中应有相关信息）
2. 检查网络连接质量
3. 如果还是慢，可尝试提前下载模型到本地

## 预下载模型（可选）

如果需要提前下载模型到本地缓存，可使用：

```bash
# 使用 Demucs 官方工具下载
python -m demucs.separate --download htdemucs_ft --dry-run

# 或编写脚本预下载
python -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')"
```

## 日志查看

应用启动时会输出镜像配置信息：

```
[INFO] 已启用国内加速镜像: https://hf-mirror.com
[INFO] 加载Demucs模型: htdemucs_ft
[INFO] Demucs模型 htdemucs_ft 已加载到GPU
```

## 性能提示

- ✅ 首次运行时会自动下载模型，需要耐心等待（取决于网速）
- ✅ 之后的运行会直接使用缓存模型，速度很快
- ✅ 不同任务可能加载不同模型，会触发模型切换（需要几秒钟）
- ✅ 国内镜像通常比官方源快 5-10 倍

