# Demucs 预置模型目录

此目录存放随项目发布的预置 Demucs 模型。

## 默认模型

| 文件名 | 模型名 | 大小 | 说明 |
|--------|--------|------|------|
| `htdemucs.th` | htdemucs | ~80MB | Hybrid Transformer（默认推荐） |

## 工作原理

程序启动时会检查此目录：
1. 如果 `htdemucs.th` 存在，且 torch 缓存中没有对应模型
2. 自动复制到 `models/torch/hub/checkpoints/955717e8-8726e21a.th`
3. Demucs 直接从缓存加载，无需网络下载

## 添加其他模型

如需添加其他预置模型，需要：
1. 下载模型文件
2. 重命名并放到此目录
3. 在 `demucs_service.py` 的 `_install_pretrained_model` 方法中添加映射

### 模型下载地址

| 模型名 | 原始文件名 | 下载链接 |
|--------|-----------|----------|
| htdemucs | 955717e8-8726e21a.th | [下载](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th) |
| htdemucs_ft | f7e0c4bc-ba3fe64a.th | [下载](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th) |
| mdx_extra | 0d19c1c6-0f06f20e.th | [下载](https://dl.fbaipublicfiles.com/demucs/mdx_final/0d19c1c6-0f06f20e.th) |
| mdx_extra_q | 83fc094f-4a16d450.th | [下载](https://dl.fbaipublicfiles.com/demucs/mdx_final/83fc094f-4a16d450.th) |

> **注意**：`htdemucs_ft` 是 Bag of Models，实际需要 4 个文件，不推荐使用。
