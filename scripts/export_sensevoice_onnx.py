"""
SenseVoice ONNX 模型导出脚本

使用方法:
    python scripts/export_sensevoice_onnx.py

依赖安装:
    pip install -U funasr funasr-onnx

导出后模型位置:
    ~/.cache/modelscope/hub/iic/SenseVoiceSmall/model.onnx (或 model_quant.onnx)
"""
import os
import sys
from pathlib import Path

def check_dependencies():
    """检查依赖"""
    try:
        import funasr
        print(f"[OK] funasr 版本: {funasr.__version__}")
    except ImportError:
        print("[ERROR] funasr 未安装，请运行: pip install -U funasr")
        return False

    try:
        import funasr_onnx
        print(f"[OK] funasr-onnx 已安装")
    except ImportError:
        print("[ERROR] funasr-onnx 未安装，请运行: pip install -U funasr-onnx")
        return False

    return True

def export_onnx_model(quantize: bool = True):
    """
    导出 SenseVoice ONNX 模型

    Args:
        quantize: 是否使用量化模型 (INT8)，推荐 True
    """
    from funasr_onnx import SenseVoiceSmall
    from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

    model_dir = "iic/SenseVoiceSmall"

    print(f"\n{'='*60}")
    print(f"开始导出 SenseVoice ONNX 模型")
    print(f"模型: {model_dir}")
    print(f"量化: {quantize}")
    print(f"{'='*60}\n")

    # 首次加载会自动下载模型并导出 ONNX
    # ONNX 模型会导出到 ~/.cache/modelscope/hub/iic/SenseVoiceSmall/
    print("[1/3] 加载模型并导出 ONNX（首次会下载模型，约 500MB）...")
    model = SenseVoiceSmall(model_dir, batch_size=1, quantize=quantize)

    # 获取模型路径
    home = Path.home()
    model_path = home / ".cache" / "modelscope" / "hub" / "iic" / "SenseVoiceSmall"

    print(f"\n[2/3] 检查导出结果...")
    onnx_file = model_path / ("model_quant.onnx" if quantize else "model.onnx")
    tokens_file = model_path / "tokens.json"

    if onnx_file.exists():
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        print(f"  [OK] ONNX 模型: {onnx_file} ({size_mb:.1f} MB)")
    else:
        # 检查其他可能的文件名
        for name in ["model.onnx", "model_quant.onnx"]:
            alt_file = model_path / name
            if alt_file.exists():
                size_mb = alt_file.stat().st_size / (1024 * 1024)
                print(f"  [OK] ONNX 模型: {alt_file} ({size_mb:.1f} MB)")
                break
        else:
            print(f"  [WARNING] 未找到 ONNX 文件，请检查 {model_path}")

    if tokens_file.exists():
        print(f"  [OK] 词汇表: {tokens_file}")
    else:
        print(f"  [WARNING] 未找到词汇表文件")

    # 测试推理
    print(f"\n[3/3] 测试 ONNX 推理...")
    try:
        # 使用模型目录中的示例音频测试
        example_audio = model_path / "example" / "zh.mp3"
        if example_audio.exists():
            res = model([str(example_audio)], language="auto", use_itn=True)
            text = rich_transcription_postprocess(res[0])
            print(f"  测试音频: {example_audio.name}")
            print(f"  识别结果: {text[:100]}..." if len(text) > 100 else f"  识别结果: {text}")
            print(f"\n  [OK] ONNX 推理测试通过!")
        else:
            print(f"  [SKIP] 未找到示例音频，跳过测试")
    except Exception as e:
        print(f"  [ERROR] 推理测试失败: {e}")

    print(f"\n{'='*60}")
    print(f"导出完成!")
    print(f"模型位置: {model_path}")
    print(f"{'='*60}\n")

    return str(model_path)

def copy_to_project(source_path: str, target_dir: str = None):
    """
    可选：复制模型到项目目录

    Args:
        source_path: 源模型路径
        target_dir: 目标目录，默认为 models/sensevoice
    """
    import shutil

    if target_dir is None:
        # 获取项目根目录
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        target_dir = project_root / "models" / "sensevoice"
    else:
        target_dir = Path(target_dir)

    source_path = Path(source_path)

    print(f"\n复制模型到项目目录: {target_dir}")

    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)

    # 复制必要文件
    files_to_copy = [
        "model.onnx",
        "model_quant.onnx",
        "tokens.json",
        "vocab.json",
        "config.yaml",
        "configuration.json",
        "chn_jpn_yue_eng_ko_spectok.bpe.model"
    ]

    copied = 0
    for filename in files_to_copy:
        src_file = source_path / filename
        if src_file.exists():
            dst_file = target_dir / filename
            shutil.copy2(src_file, dst_file)
            print(f"  [COPIED] {filename}")
            copied += 1

    print(f"\n复制完成，共 {copied} 个文件")
    return str(target_dir)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SenseVoice ONNX 模型导出工具")
    print("="*60 + "\n")

    # 检查依赖
    if not check_dependencies():
        print("\n请先安装依赖:")
        print("  pip install -U funasr funasr-onnx")
        sys.exit(1)

    # 导出模型
    model_path = export_onnx_model(quantize=True)

    # 询问是否复制到项目目录
    print("\n是否复制模型到项目目录 (models/sensevoice)?")
    print("  输入 'y' 复制，其他跳过: ", end="")

    try:
        choice = input().strip().lower()
        if choice == 'y':
            copy_to_project(model_path)
    except EOFError:
        print("跳过复制")

    print("\n导出完成! 现在可以使用 SenseVoice 工作流了。")
