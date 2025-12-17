# 预置模型目录

此目录存放随项目发布的预置模型，无需额外下载。

## 模型列表

### Demucs (人声分离)

| 文件名 | 模型名 | 大小 | 说明 |
|--------|--------|------|------|
| `htdemucs.th` | htdemucs | ~80MB | Hybrid Transformer（默认推荐） |

**工作原理**：
1. 程序启动时检查 `backend/models/pretrained/htdemucs.th`
2. 如果 torch 缓存中没有对应模型，自动复制到 `models/torch/hub/checkpoints/955717e8-8726e21a.th`
3. Demucs 直接从缓存加载，无需网络下载

### SenseVoice (语音识别)

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `sensevoice/model_quant.onnx` | ~230MB | 量化 ONNX 模型（推荐） |
| `sensevoice/tokens.json` | ~344KB | 词汇表 |
| `sensevoice/configuration.json` | ~396B | 配置文件 |
| `sensevoice/config.yaml` | ~1.8KB | 配置文件 |
| `sensevoice/chn_jpn_yue_eng_ko_spectok.bpe.model` | ~368KB | 分词模型 |

**工作原理**：
1. 程序自动检测 `backend/models/pretrained/sensevoice/`
2. 优先使用预置的量化 ONNX 模型
3. 支持中、粤、英、日、韩多语言识别

## 添加其他模型

### Demucs

如需添加其他 Demucs 预置模型：
1. 下载模型文件
2. 重命名并放到此目录
3. 在 `demucs_service.py` 的 `_install_pretrained_model` 方法中添加映射

#### Demucs 模型下载地址

| 模型名 | 原始文件名 | 下载链接 |
|--------|-----------|----------|
| htdemucs | 955717e8-8726e21a.th | [下载](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th) |
| htdemucs_ft | f7e0c4bc-ba3fe64a.th | [下载](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th) |
| mdx_extra | 0d19c1c6-0f06f20e.th | [下载](https://dl.fbaipublicfiles.com/demucs/mdx_final/0d19c1c6-0f06f20e.th) |
| mdx_extra_q | 83fc094f-4a16d450.th | [下载](https://dl.fbaipublicfiles.com/demucs/mdx_final/83fc094f-4a16d450.th) |

> **注意**：`htdemucs_ft` 是 Bag of Models，实际需要 4 个文件，不推荐使用。

### SenseVoice

SenseVoice 模型已预置，如需更新：
1. 运行导出脚本：`python scripts/export_sensevoice_onnx.py`
2. 复制生成的文件到 `backend/models/pretrained/sensevoice/`

**必需文件**：
- `model_quant.onnx` (量化模型) 或 `model.onnx` (原始模型)
- `tokens.json` (词汇表)
- `configuration.json`, `config.yaml` (配置文件)
- `chn_jpn_yue_eng_ko_spectok.bpe.model` (分词模型)

**模型来源**：
- ModelScope: https://www.modelscope.cn/models/iic/SenseVoiceSmall
- HuggingFace: https://huggingface.co/FunAudioLLM/SenseVoiceSmall
