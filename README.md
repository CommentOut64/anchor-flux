# Video-to-SRT 字幕生成工具

> 仅供学习与研究，禁止任何形式的商业使用。作者不对使用后果承担责任。

这是一个视频字幕自动生成工具，将音频分割成合适长度的小段，使用 WhisperX 进行语音识别和时间戳对齐，快速将视频转换成 SRT 字幕文件。

该项目最初的设想是本地生成neuro直播回放的双语字幕，用于~~英语听力练习~~（助眠）

该分支为cpu-only版本，如果需要GPU加速版本，可以切换到gpu-acceleration分支

## 📋 功能特点

* 🎬 支持多种视频格式
* 📊 高精度音频分段与处理，提高长视频的转录质量
* ⏱️ 词级的时间戳对齐
* 🔄 支持断点续传，中断后可从上次进度继续

## 系统要求

* Windows 10 或更高版本
* Python 3.10
* conda 环境
* 至少 8GB RAM
* 足够的磁盘空间用于临时文件
* 稳定的网络连接 (用于下载依赖和模型)

## 安装步骤

### 第一步：安装 Miniconda

Miniconda 是一个轻量级的 Conda 发行版，用于管理 Python 环境和包。

1.  **下载 Miniconda**:
    访问 Miniconda 官方下载页面：[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
    根据您的 Windows 系统选择最新的 Python 3.x 版本的 Miniconda 安装程序 (例如 `Miniconda3-latest-Windows-x86_64.exe`)。

2.  **安装 Miniconda**:

    *   运行下载的安装程序。
    *   遵循安装向导的指示。建议选择 “Just Me” 安装。
    *   在 "Advanced Installation Options" 步骤中，**不建议**勾选 "Add Anaconda to my PATH environment variable" (这是 Miniconda 安装程序的默认建议)。我们将通过 Anaconda Prompt 来管理环境，并通过 `conda init` 使 `conda` 命令在标准命令提示符中可用。
    *   保持其他选项为默认设置，完成安装。

3.  **初始化 Conda**:

    *   安装完成后，从 Windows 开始菜单找到并打开 "Anaconda Prompt (Miniconda3)"。

    *   在打开的 Anaconda Prompt 窗口中，执行以下命令来初始化 Conda 以便在标准的命令提示符 (`cmd.exe`) 中使用 (这将使 `run.bat` 文件能够正确激活 conda 环境)：

        ```bash
        conda init cmd.exe
        ```

    *   执行完毕后，关闭当前的 Anaconda Prompt 窗口，然后**重新打开一个新的 Anaconda Prompt 窗口**或**重启您的计算机**，以使更改生效。

4.  **验证安装**:
    重新打开一个 Anaconda Prompt (Miniconda3) 窗口，输入以下命令并按回车：

    ```bash
    conda --version
    ```

    如果安装成功，您将看到 Conda 的版本号。

### 第二步：下载并配置 FFmpeg

FFmpeg 是一个处理多媒体数据的开源工具集，本程序用其进行音频提取。

1.  **下载 FFmpeg**:

    *   推荐从 [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) 下载 FFmpeg 的 Windows 构建版本。
    *   在页面中找到 "release builds" 部分，下载 `ffmpeg-release-essentials.zip` (这是一个较小的版本，包含了运行程序所需的基本功能)。

2.  **解压 FFmpeg**:

    *   将下载的 `ffmpeg-release-essentials.zip` 文件解压到您选择的一个稳定位置。例如，您可以解压到 `C:\ffmpeg`。
    *   解压后，您会得到一个类似 `ffmpeg-7.0-essentials_build` (版本号可能不同) 的文件夹，其中包含一个 `bin` 目录，`ffmpeg.exe` 就在这个 `bin` 目录里。

3.  **添加到系统环境变量 PATH**:

    *   在 Windows 搜索栏中搜索“环境变量”，然后选择“编辑系统环境变量”。
    *   在“系统属性”窗口中，点击“高级”选项卡下的“环境变量(N)...”按钮。
    *   在“环境变量”窗口的“系统变量(S)”区域中，找到名为 `Path` 的变量，选中它，然后点击“编辑(E)...”。
    *   在“编辑环境变量”窗口中，点击“新建(N)”，然后添加您 FFmpeg `bin` 目录的完整路径。例如，如果您将 FFmpeg 解压到 `C:\ffmpeg` 并且解压后的文件夹名为 `ffmpeg-7.0-essentials_build`，那么您需要添加的路径是 `C:\ffmpeg\ffmpeg-7.0-essentials_build\bin`。
    *   点击“确定”保存所有打开的窗口。

4.  **验证安装**:

    *   打开一个新的命令提示符窗口 (CMD) 或 Anaconda Prompt (Miniconda3) 窗口（**必须是新打开的窗口**，以加载更新后的环境变量）。

    *   输入以下命令并按回车：

        ```bash
        ffmpeg -version
        ```

    *   如果配置正确，您将看到 FFmpeg 的版本信息。

### 第三步：创建并激活 Conda 环境

我们将创建一个独立的 Conda 环境来安装程序的 Python 依赖，以避免与系统中的其他 Python 包冲突。

1.  **打开 Anaconda Prompt**:
    从 Windows 开始菜单打开 "Anaconda Prompt (Miniconda3)"。

2.  **创建 Conda 环境**:
    在 Anaconda Prompt 中，执行以下命令创建一个名为 `srt_packer` 并使用 Python 3.10 的环境 (与您提供的依赖列表中的 Python 版本一致)：

    ```bash
    conda create -n srt_packer python=3.10 -y
    ```

3.  **激活 Conda 环境**:
    创建成功后，执行以下命令激活新环境：

    ```bash
    conda activate srt_packer
    ```

    激活环境后，您会看到命令提示符前缀变为 `(srt_packer)`。**后续所有 Conda 和 pip 命令都必须在此激活的环境中执行。**

### 第四步：配置 Pip 清华镜像源 (可选但推荐)

为了加快 Pip 包的下载速度，建议配置国内的镜像源。

1.  **确保 Conda 环境已激活**:
    如果您的 Anaconda Prompt 不是 `(srt_packer)` 开头，请先执行 `conda activate srt_packer`。

2.  **配置清华大学 Pip 镜像源**:
    在激活的 `srt_packer` 环境的 Anaconda Prompt 中，执行以下命令：

    ```bash
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    ```

    此命令会将 Pip 的全局默认下载源设置为清华大学的 PyPI 镜像。

    **注意**:

    *   后续步骤中安装 PyTorch 时，由于命令中明确指定了 `--index-url https://download.pytorch.org/whl/cpu`，因此 PyTorch 的下载将不受此全局配置影响，会从其官方指定地址下载。
    *   其他不指定 `--index-url` 的 `pip install` 命令（如安装 WhisperX 和其余依赖）将会使用此处配置的清华镜像源。

### 第五步：安装 Python 依赖库

在此步骤中，我们将安装程序运行所需的所有 Python 库。请确保 `srt_packer` 环境已激活。

1.  **安装 WhisperX**:

    ```bash
    pip install git+https://github.com/m-bain/whisperx.git
    ```

2.  **安装 PyTorch (CPU 版本)**:

    ```bash
    pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
    ```

3.  **安装其他依赖**:

    ```bash
    pip install pydub==0.25.1 tqdm transformers ffmpeg-python pytorch-lightning==2.5.1.post0 rich soundfile
    ```

    *   如果遇到 `ffmpeg-python` 安装问题，请确保您已正确完成第二步 (配置 FFmpeg)。

### 第六步：下载并解压程序文件

1.  **下载程序**:
    *   访问项目的 GitHub Releases 页面 (https://github.com/CommentOut64/video_to_srt/releases)。
    *   下载最新的包含 `modified_script.py` 和 `run.bat` 的程序压缩包 (例如 `video_to_srt_cpu_vx.x.x.zip`)。

2.  **解压文件**:
    *   将下载的压缩包解压到您希望存放程序的目录，例如 `D:\tools\video_to_srt`。
    *   解压后，确保 `modified_script.py` 和 `run.bat` 文件位于同一目录下。

### 第七步：运行程序

1.  **进入程序目录**:
    打开文件资源管理器，导航到您在第六步中解压程序文件的目录。

2.  **运行 `run.bat`**:

    *   以管理员身份运行 `run.bat` 文件。

        此脚本会自动激活 `srt_packer` Conda 环境，然后运行 `modified_script.py` Python 脚本。

    *   首次运行时，程序可能需要下载 Whisper 模型文件，请耐心等待模型下载完成。

    程序启动后，您应该会在命令行窗口中看到主菜单界面。按照菜单提示操作即可。

## 注意事项

*   **关于警告：**脚本运行时可能显示关于依赖版本不匹配或模型版本的警告，**请忽略此类警告。**
*   **下载模型：**程序首次使用特定模型时，会从网络下载模型文件并缓存到程序目录下的 `model_cache` 文件夹。这可能需要一些时间，具体取决于您的网络速度和模型大小。
*   **模型的选择：**默认使用的模型是**medium**，它在测试中表现更好，您也可以在“配置 Whisper 模型与参数”选项中切换为small等来提高推理速度，但这可能会降低生成字幕的准确率
*   **临时文件**: 程序在处理过程中会在 `temp_files` 目录下生成临时文件。您可以根据程序提示选择在处理完成后是否清理这些文件。
*   **日志文件**: 程序运行日志会保存在程序根目录下的 `app_runtime.log` 文件中。如果遇到问题，此日志文件可能包含有用的诊断信息。

## 故障排除

*   **`conda` 或 `ffmpeg` 命令找不到**: 确保您已正确配置系统环境变量 `PATH` (针对 FFmpeg)，并已正确初始化 Conda (`conda init cmd.exe`) 且重启了命令提示符或计算机。始终建议在 "Anaconda Prompt (Miniconda3)" 中执行 `conda` 和 `pip` 命令。
*   **依赖安装失败**: 检查网络连接。仔细核对命令是否输入正确。查看 pip 或 conda 的错误输出获取详细信息。如果下载速度慢，请确认是否已按第四步配置 Pip 镜像源。
*   **程序报错**: 查看命令行窗口的错误信息以及 `app_runtime.log` 日志文件。

## 🔍 TODO

* 部分警告无法完全去除
* 未完成依赖检测部分
* 未能成功打包
* 首次转录时有概率卡在进度条100%时而无法继续删除缓存文件（或许是GPU锁或线程阻塞问题）

## 📄 许可证

MIT License

## 🙏 致谢

* [WhisperX](vscode-file://vscode-app/d:/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) - 提供了核心的转录和对齐功能
* [OpenAI Whisper](vscode-file://vscode-app/d:/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) - 基础的语音识别模型
* [FFmpeg](vscode-file://vscode-app/d:/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) - 视频和音频处理
* 所有开源库的贡献者们
