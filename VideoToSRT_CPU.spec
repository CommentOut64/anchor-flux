# VideoToSRT_CPU.spec (V8)
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None
conda_env_site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')

binaries_to_collect = []
# PyTorch 核心 DLLs
torch_lib_path = os.path.join(conda_env_site_packages, 'torch', 'lib')
if os.path.isdir(torch_lib_path):
    critical_torch_dlls = [
        'c10.dll', 'torch.dll', 'torch_cpu.dll',
        'torch_global_deps.dll', 'fbgemm.dll', 'asmjit.dll',
        'libiomp5md.dll',
    ]
    for dll_name in critical_torch_dlls:
        dll_src_path = os.path.join(torch_lib_path, dll_name)
        if os.path.exists(dll_src_path):
            binaries_to_collect.append((dll_src_path, '.'))
# torchvision._C.pyd
torchvision_c_pyd_path = os.path.join(conda_env_site_packages, 'torchvision', '_C.pyd')
if os.path.exists(torchvision_c_pyd_path):
    binaries_to_collect.append((torchvision_c_pyd_path, 'torchvision'))
else:
    print(f"INFO: torchvision._C.pyd not found: {torchvision_c_pyd_path}", file=sys.stderr)

datas_to_collect = []
def add_data_files_to_subdir(package_name, dest_subdir_name, current_datas_list, include_py_files=False):
    try:
        collected_files = collect_data_files(package_name, include_py_files=include_py_files)
        for src_path, relative_dest_path_in_pkg in collected_files:
            current_datas_list.append((src_path, os.path.join(dest_subdir_name, relative_dest_path_in_pkg)))
    except Exception as e:
        print(f"Error collecting data for {package_name}: {e}", file=sys.stderr)

add_data_files_to_subdir('transformers', 'transformers', datas_to_collect, include_py_files=False)
try:
    import transformers
    transformers_pkg_path = os.path.dirname(transformers.__file__)
    key_subdirs_for_init_py = ["models", "pipelines", "tokenization", "utils"]
    for subdir_name in key_subdirs_for_init_py:
        src_init_path = os.path.join(transformers_pkg_path, subdir_name, '__init__.py')
        if not os.path.exists(src_init_path):
            print(f"WARNING: Expected __init__.py not found at {src_init_path} for transformers subdir {subdir_name}", file=sys.stderr)
except ImportError:
    print("WARNING: transformers package not found during .spec file processing for __init__.py files.", file=sys.stderr)
except Exception as e:
    print(f"Error trying to assess specific transformers __init__.py files: {e}", file=sys.stderr)

add_data_files_to_subdir('huggingface_hub', 'huggingface_hub', datas_to_collect, include_py_files=False)
add_data_files_to_subdir('whisperx', 'whisperx', datas_to_collect, include_py_files=False)
add_data_files_to_subdir('faster_whisper', 'faster_whisper', datas_to_collect, include_py_files=False)
add_data_files_to_subdir('soundfile', 'soundfile', datas_to_collect, include_py_files=False)

a = Analysis(
    ['modified_script.py'],
    pathex=[conda_env_site_packages, '.'],
    binaries=binaries_to_collect,
    datas=datas_to_collect,
    hiddenimports=[
        'tkinter', 'tkinter.filedialog',
        'pydub', 'pydub.utils', 'pydub.scipy_effects',
        'soundfile', 'audioread', '_soundfile_data',
        'torch', 'torchvision', 'torchvision.ops', 'torchaudio',
        'transformers',
        'transformers.models.auto.modeling_auto',
        'transformers.models.auto.configuration_auto',
        'transformers.models.auto.tokenization_auto',
        'transformers.pipelines',
        'transformers.tokenization_utils_base',
        'transformers.models.wav2vec2.modeling_wav2vec2',
        'transformers.models.wav2vec2.configuration_wav2vec2',
        'transformers.models.whisper.modeling_whisper',
        'transformers.models.whisper.processing_whisper',
        'transformers.models.whisper.feature_extraction_whisper',
        'ctranslate2',
        'faster_whisper',
        'whisperx', 'whisperx.asr', 'whisperx.alignment', 'whisperx.utils',
        'onnxruntime', 'onnxruntime.capi._pybind_state',
        'safetensors',
        'pytorch_lightning',
        'huggingface_hub', 'huggingface_hub.inference_api', 'huggingface_hub.utils', 'huggingface_hub.constants',
        'filelock',
        'requests',
        'tqdm',
        'regex',
        'packaging', 'packaging.version', 'packaging.specifiers',
        'tiktoken',
        'pkg_resources', 'pkg_resources.py2_warn',
        'rich', 'rich.themes',
        'shutil', 'json', 'threading', 'glob', 'gc', 'warnings', 'io', 'subprocess',
        'concurrent.futures',
        'platform',
        'importlib_metadata',
        'charset_normalizer',
        'idna',
        'pdb',
        'unittest', 'unittest.mock',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'doctest', 'test', 'tests',
        'tkinter.test', 'tkinter.tix', 'FixTk',
        'PIL.ImageTk', 'PIL._tkinter_finder',
        'cv2',
        'matplotlib', 'pandas',
        'IPython', 'jupyter_client', 'jupyter_core',
        'PyQt5', 'PySide2', 'wx',
        'torch.utils.tensorboard',
        'onnxruntime.training',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=True
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],  # 已移动到关键字参数前
    binaries=a.binaries,
    zipfiles=a.zipfiles,
    datas=a.datas,
    name='VideoToSRT_CPU',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='VideoToSRT_CPU'
)
