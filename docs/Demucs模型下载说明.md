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

   | 模型 | 文件名 | 下载链接 |
   |------|--------|---------|
   | `htdemucs_ft` | f7e0c4bc-ba3fe64a.th | [下载](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th) |
   | `htdemucs` | 955717e8-8726e21a.th | [下载](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th) |
   | `mdx_extra` | 0d19c1c6-0f06f20e.th | [下载](https://dl.fbaipublicfiles.com/demucs/mdx_final/0d19c1c6-0f06f20e.th) |
   | `mdx_extra_q` | 83fc094f-4a16d450.th | [下载](https://dl.fbaipublicfiles.com/demucs/mdx_final/83fc094f-4a16d450.th) |

3. **将下载的文件放到缓存目录即可**

### 方案3：云服务器中转

在海外云服务器上下载模型，然后传到本地。

## 模型大小参考

| 模型 | 大小 | 用途 |
|------|------|------|
| `htdemucs_ft` | ~70MB | 默认模型（Fine-tuned，推荐） |
| `htdemucs` | ~80MB | 快速模式 |
| `mdx_extra_q` | ~25MB | 强BGM场景（量化版） |
| `mdx_extra` | ~100MB | 升级/兜底模型（最高质量） |

## 下载日志示例

```
[INFO] 加载模型: htdemucs_ft (官方源)
Downloading: "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th"
  to %USERPROFILE%\.cache\torch\hub\checkpoints\f7e0c4bc-ba3fe64a.th
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

## 预下载模型（可选）

如果需要提前下载模型到本地缓存：

```bash
# 使用 Python 预下载
python -c "from demucs.pretrained import get_model; get_model('htdemucs_ft')"
```

## 故障排除

### 下载超时

如果下载超时，可以：
1. 使用代理
2. 手动下载模型文件到缓存目录
3. 多次重试（PyTorch 会断点续传）

### 模型加载失败

检查缓存目录中的模型文件是否完整：
```powershell
# Windows - 检查文件大小
Get-ChildItem "$env:USERPROFILE\.cache\torch\hub\checkpoints\*.th" | Format-Table Name, Length
```

如果文件不完整（大小明显小于预期），删除后重新下载：
```powershell
# 删除损坏的文件
Remove-Item "$env:USERPROFILE\.cache\torch\hub\checkpoints\f7e0c4bc-ba3fe64a.th"
```

## 性能提示

- ✅ 首次运行时会自动下载模型，需要耐心等待（取决于网速）
- ✅ 之后的运行会直接使用缓存模型，速度很快
- ✅ 不同任务可能加载不同模型，会触发模型切换（需要几秒钟）
- ✅ 建议使用代理加速首次下载
