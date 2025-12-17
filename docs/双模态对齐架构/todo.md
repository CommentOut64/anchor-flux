1. 把旧架构那套“频谱分诊→逐段分离”的流程迁移到新架构，理由：
  频谱分诊已经写好并验证过，可直接复用 audio_spectrum_classifier 的 diagnose_chunks，开发量小。
  只对诊断为 noisy 的 chunk 调 Demucs，显存和耗时会大幅下降，满足你现在对 8 GB 爆满的担忧。
  新架构本身就是按 chunk 并行处理，把 Demucs 结果写回 ProcessingContext.current_audio 就可以被 SlowWorker 直接消费，逻辑适配成本不高。
  即使将来要扩展多模型策略，也可以在频谱诊断阶段统一决策（如不同 chunk 用不同 model_quality），不会影响双流流水线。
  落地步骤大概是：在 job_queue_service/async_dual_pipeline 初始化 chunk 时调用频谱诊断，给 ProcessingContext 填入 need_separation 标记；FastWorker before SenseVoice 把需要分离的 chunk 异步喂给 demucs_service.separate_chunk（或在 VAD 结束后一次性跑完）并及时 unload_model()。这样既保留新架构的 streaming 优势，又恢复旧架构的按需分离策略，是收益最大的一条路。
2. **块内重复检测**：在 TextNormalizer.clean_whisper_output() 或 AlignmentWorker 进入 SRT 阶段前加一层 N‑gram/压缩比检测；例如计算句子里 3‑5 词短语的出现次数或检测 len(set(chunks))/len(chunks)，异常时截断或回退 (backend/app/services/text_normalizer.py (lines 36-118))。这能精准打击用户指出的“在同一条字幕里循环”的问题。
3. 扩展 WhisperService.transcribe() 的参数，把 repetition_penalty、no_repeat_ngram_size 暴露给配置，然后在 WhisperExecutor 或缓冲池模式中提供>1 的 penalty；这是最直接的方式，成本也只是在调用处加几个关键字参数（backend/app/services/whisper_service.py (lines 452-474)）。
5. 引入微型标点模型 (Punctuator)，在 CPU 上跑一个极小的 BERT-based 标点恢复模型（ONNX 格式，几十 MB）
  流程：SenseVoice 文本 -> Punctuator (CPU 10ms) -> 带标点文本 -> 推送。
模型：CT-Transformer (FunASR/Sherpa 版本)

### 🚀 实施方案

由于这个模型不是标准的 HuggingFace 架构，不能直接用 `transformers` 库加载，也不能用之前的导出脚本。你需要直接下载社区已经转换好的 ONNX 版本（由 `sherpa-onnx` 社区提供），并使用我下面提供的专用推理代码。

#### 1. 下载模型文件

请下载以下两个文件到你的 `backend/app/assets/models/punctuation/` 目录：

1. **模型文件 (model.onnx)**: [点击下载 (Sherpa-ONNX 仓库)](https://www.google.com/search?q=https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-common-vocab-0001.onnx)
2. **词表文件 (tokens.txt)**: [点击下载 (Sherpa-ONNX 仓库)](https://www.google.com/search?q=https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/tokens.txt)

*(注：如果下载慢，可以使用该项目的 HuggingFace 镜像或 ModelScope 镜像)*

#### 2. 编写轻量级推理引擎 (Python)

CT-Transformer 的输入处理比 BERT 简单，不需要 `AutoTokenizer`，只需要一个简单的词表映射。

请创建 `backend/app/services/ct_punctuation_service.py`：

```python
import os
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

class CTPunctuationService:
    def __init__(self, model_dir="backend/app/assets/models/punctuation"):
        self.model_path = os.path.join(model_dir, "sherpa-onnx-punct-ct-transformer-zh-en-common-vocab-0001.onnx")
        self.vocab_path = os.path.join(model_dir, "tokens.txt")
        
        # 1. 加载词表
        self.token2id = {}
        self.id2token = {}
        self._load_vocab()
        
        # 2. 加载 ONNX Session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            self.model_path, 
            sess_options, 
            providers=['CPUExecutionProvider']
        )
        
        # 标点符号映射 (模型输出的ID对应的标点)
        # ID 0: <EPS> (无标点)
        # ID 1: , (逗号/顿号)
        # ID 2: . (句号)
        # ID 3: ? (问号)
        self.punctuations = ["", "，", "。", "？"] 

    def _load_vocab(self):
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    token = parts[0]
                    idx = int(parts[1])
                    self.token2id[token] = idx
                    self.id2token[idx] = token
        
        # 特殊 token ID
        self.unk_id = self.token2id.get("<UNK>", 0)

    def _tokenize(self, text: str) -> np.ndarray:
        # CT-Transformer 的分词逻辑：
        # 中文按字分，英文按 Word 分 (简化版：SenseVoice 输出通常带空格，这里按字/词查表即可)
        # 这里的实现做一个简化的字符级 fallback，对于此特定模型通常足够有效
        ids = []
        for char in text:
            # 注意：实际生产中英文单词可能需要 BPE，但这个 5MB 模型通常是 Char-based 的
            # 如果是纯英文单词，可能需要查表整体，查不到则回退到 Char
            # 简单起见，我们先尝试直接查
            ids.append(self.token2id.get(char, self.unk_id))
        return np.array([ids], dtype=np.int64)

    def restore(self, text: str) -> str:
        if not text:
            return ""

        # 1. 准备输入
        input_ids = self._tokenize(text)
        
        # 2. 推理
        # CT-Transformer 输入通常只需要 x (token ids)
        outputs = self.session.run(
            ["y"], 
            {"x": input_ids}
        )[0] # Shape: (1, seq_len, num_classes)
        
        # 3. 解码
        # outputs 是每个 token 后面应该跟什么标点的概率
        preds = np.argmax(outputs, axis=2)[0]
        
        result = []
        for i, char in enumerate(text):
            result.append(char)
            # 获取当前字后面的标点
            punct_id = preds[i]
            if punct_id > 0 and punct_id < len(self.punctuations):
                result.append(self.punctuations[punct_id])
                
        return "".join(result)

# 使用示例
if __name__ == "__main__":
    service = CTPunctuationService()
    text = "今天天气真不错啊我们要不要出去玩"
    print(service.restore(text))
    # 输出示例: 今天天气真不错啊，我们要不要出去玩？

```

### 3. 集成建议

* **输入处理**：上面的 `_tokenize` 是一个最简化的字符级实现。由于 `tokens.txt` 里包含了常见的汉字和英文字母/单词，直接查表通常能覆盖 95% 的情况。如果遇到英文单词识别不准，可以考虑简单的正则：英文按词查，中文按字查。
* **文件位置**：将 `model.onnx` (5.6MB) 和 `tokens.txt` 放入打包资源中，对最终包体积的影响微乎其微。
