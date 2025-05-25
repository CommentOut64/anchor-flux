# build.py
import sys, subprocess
from pkg_resources import working_set

# 1. 明确你程序真正需要 metadata 支持的包列表
needed_pkgs = {
    "whisperx",
    "faster-whisper",    # 注意 pip 名称
    "ctranslate2",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "onnxruntime",
    "pydub",
    "soundfile",
    "audioread",
    "rich",
    "importlib-metadata",
    # …如果还有其它运行时会读取 metadata 的包，就加在这里
}

python_exe = sys.executable

# 2. Nuitka 基础参数
cmd = [
    python_exe, "-m", "nuitka",
    "--standalone",
    "--jobs=8",
    "--remove-output",
    "--output-dir=VideoToSRT_CPU_Nuitka_Dist",
    "--output-filename=VideoToSRT_CPU",
    "--show-progress",
    "--lto=no",
    "--windows-console-mode=force",
    "--disable-plugin=transformers",
    "--disable-plugin=torch",
    "--enable-plugin=tk-inter",
    "--nofollow-import-to=transformers.models.nllb",
    "--nofollow-import-to=transformers.models.dinov2*",
    # 包含你程序直接 import 的包
    "--include-package=whisperx",
    "--include-package=faster_whisper",
    "--include-package=ctranslate2",
    "--include-package=transformers",
    "--include-package=tokenizers",
    "--include-package=huggingface_hub",
    "--include-package=safetensors",
    "--include-package=onnxruntime",
    "--include-module=onnxruntime.capi._pybind_state",
    "--include-package=pydub",
    "--include-package=soundfile",
    "--include-package=audioread",
    "--include-package=rich",
    "--include-package=pkg_resources",
    "--include-package=importlib_metadata",
]

# 3. 只为 needed_pkgs 加 metadata
for dist in working_set:
    name = dist.project_name  # 例如 "PyYAML"、"transformers"、"rich"
    if name.lower() in {p.lower() for p in needed_pkgs}:
        cmd.append(f"--include-distribution-metadata={dist.project_name}")

# 4. 程序脚本（唯一的位置参数）
cmd.append("modified_script.py")

# 5. 执行打包
print("Running Nuitka with", len(cmd), "arguments…")
subprocess.check_call(cmd)
