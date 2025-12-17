"""
Whisper 幻觉词 Token ID 提取脚本

用于提取常见幻觉词的 Token ID，供 suppress_tokens 参数使用。
支持 faster-whisper (CTranslate2 格式) 模型。

使用方法:
    python scripts/extract_hallucination_tokens.py --model medium
    python scripts/extract_hallucination_tokens.py --model large-v3
"""
import argparse
import os
import sys
from pathlib import Path

# 解决 OMP 冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# 配置 HuggingFace 镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
print(f"HuggingFace 镜像源: {os.environ['HF_ENDPOINT']}")
print(f"提示：如需使用官方源，请设置环境变量 USE_HF_MIRROR=false")


# 常见幻觉词汇表 (根据社区反馈和实际测试)
HALLUCINATION_TEXTS = {
    # ========== 连续符号类 ==========
    "_": "单个下划线 (CTC 解码残留)",
    "__": "双下划线",
    "___": "三下划线",
    "____": "四下划线",
    "______": "六下划线 (常见幻觉)",
    "...": "省略号",
    "..": "双点",
    ".": "单点",

    # ========== YouTube 风格幻觉 ==========
    "Questions": "Questions 幻觉开头",
    " Questions": "带空格的 Questions",
    "Subtitles": "Subtitles 幻觉",
    " Subtitles": "带空格的 Subtitles",
    "Subtitles by": "Subtitles by 幻觉",
    "Copyright": "Copyright 幻觉",
    " Copyright": "带空格的 Copyright",
    "Thanks for watching": "结束语幻觉",
    "Please subscribe": "订阅幻觉",
    "Amara.org": "字幕网站幻觉",
    " Amara": "带空格的 Amara",

    # ========== 音乐/音效类 ==========
    "♪": "音符符号",
    "[Music]": "音乐标记",
    "[Applause]": "掌声标记",
    "[Laughter]": "笑声标记",

    # ========== 中文幻觉 ==========
    "字幕": "中文字幕幻觉",
    "谢谢观看": "中文结束语幻觉",
    "请订阅": "中文订阅幻觉",
}


def extract_token_ids(model_name: str = "medium") -> dict:
    """
    提取幻觉词的 Token ID

    Args:
        model_name: 模型名称 (medium, large-v3 等)

    Returns:
        dict: {text: {"ids": [...], "first_id": int, "description": str}, ...}
    """
    print(f"\n{'='*60}")
    print(f"正在加载模型 {model_name} 的分词器...")
    print(f"{'='*60}")

    try:
        # 方法 1: 尝试使用 faster-whisper 内置的 tokenizer
        from faster_whisper import WhisperModel
        from app.core import config

        print(f"使用 faster-whisper 加载模型...")
        print(f"缓存目录: {config.HF_CACHE_DIR}")

        # 加载模型 (CPU 模式，仅用于获取 tokenizer)
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            download_root=str(config.HF_CACHE_DIR)
        )

        tokenizer = model.hf_tokenizer
        print(f"分词器加载成功!")

    except Exception as e:
        print(f"faster-whisper 加载失败: {e}")
        print(f"\n尝试使用 transformers 加载...")

        try:
            from transformers import WhisperTokenizer

            # 构造 HuggingFace 仓库 ID
            repo_id = f"openai/whisper-{model_name}"
            print(f"仓库 ID: {repo_id}")

            tokenizer = WhisperTokenizer.from_pretrained(repo_id)
            print(f"分词器加载成功!")

        except Exception as e2:
            print(f"transformers 加载失败: {e2}")
            print(f"\n请确保已安装 faster-whisper 或 transformers:")
            print(f"  pip install faster-whisper")
            print(f"  或")
            print(f"  pip install transformers")
            return {}

    # 提取 Token ID
    print(f"\n{'='*60}")
    print(f"开始提取幻觉词 Token ID")
    print(f"{'='*60}")

    results = {}
    suppress_candidates = set()  # 收集建议封杀的 ID

    print(f"\n{'文本':<25} | {'Token IDs':<30} | {'首个ID':<8} | 说明")
    print("-" * 90)

    for text, description in HALLUCINATION_TEXTS.items():
        try:
            # 编码文本 - faster-whisper 使用 tokenizers 库，返回 Encoding 对象
            encoding = tokenizer.encode(text, add_special_tokens=False)

            # 获取 token IDs - Encoding 对象使用 .ids 属性
            if hasattr(encoding, 'ids'):
                ids = list(encoding.ids)
            elif isinstance(encoding, list):
                ids = encoding
            else:
                # 尝试转换为列表
                ids = list(encoding)

            if ids:
                first_id = ids[0]
                suppress_candidates.add(first_id)
            else:
                first_id = None

            results[text] = {
                "ids": ids,
                "first_id": first_id,
                "description": description
            }

            # 显示格式化的 ID 列表
            ids_str = str(ids) if len(str(ids)) <= 25 else str(ids)[:22] + "..."
            first_str = str(first_id) if first_id else "N/A"

            print(f"{repr(text):<25} | {ids_str:<30} | {first_str:<8} | {description}")

        except Exception as e:
            print(f"{repr(text):<25} | ERROR: {e}")
            results[text] = {"ids": [], "first_id": None, "description": description, "error": str(e)}

    print("-" * 90)

    # 生成配置代码
    print(f"\n{'='*60}")
    print(f"建议的 suppress_tokens 配置 (模型: {model_name})")
    print(f"{'='*60}")

    # 按 ID 排序
    sorted_ids = sorted(suppress_candidates)

    print(f"\n# 将以下代码添加到 backend/app/config/model_config.py")
    print(f"# 模型: {model_name}")
    print(f"# 总计: {len(sorted_ids)} 个 Token ID")
    print(f"\nSUPPRESS_TOKENS_{model_name.upper().replace('-', '_')} = [")

    # 分行显示，每行 10 个 ID
    for i in range(0, len(sorted_ids), 10):
        chunk = sorted_ids[i:i+10]
        print(f"    {', '.join(map(str, chunk))},")

    print(f"]")

    # 生成完整配置
    print(f"\n\n# 完整配置示例:")
    print(f"""
WHISPER_SUPPRESS_TOKENS = {{
    "{model_name}": {sorted_ids},
}}

def get_suppress_tokens(model_name: str) -> list:
    '''获取指定模型的幻觉抑制 Token ID 列表'''
    for key, tokens in WHISPER_SUPPRESS_TOKENS.items():
        if key in model_name:
            return tokens
    return []
""")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="提取 Whisper 幻觉词的 Token ID"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="medium",
        help="模型名称 (默认: medium)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径 (可选)"
    )

    args = parser.parse_args()

    results = extract_token_ids(args.model)

    if args.output and results:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {output_path}")

    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
