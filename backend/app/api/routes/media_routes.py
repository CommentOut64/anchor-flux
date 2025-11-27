"""
媒体资源路由 - 为前端编辑器提供视频、音频、波形数据等资源
支持:
- 视频流（Range请求支持，大视频拖拽跳转）
- 音频文件
- 波形峰值数据（流式计算，避免OOM；支持FFmpeg加速）
- Proxy视频自动生成（解决浏览器格式兼容性，支持SSE进度推送）
- 视频缩略图（支持Sprite图优化）
"""
import os
import io
import json
import wave
import struct
import asyncio
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from core.config import config

router = APIRouter(prefix="/api/media", tags=["media"])

# 浏览器兼容的视频格式（H.264编码的MP4）
BROWSER_COMPATIBLE_FORMATS = {'.mp4', '.webm'}

# 需要转码的格式
NEED_TRANSCODE_FORMATS = {'.mkv', '.avi', '.mov', '.wmv', '.flv', '.m4v'}

# Proxy生成状态追踪（job_id -> {progress, status, start_time}）
_proxy_generation_status = {}


def _find_video_file(job_dir: Path) -> Optional[Path]:
    """在任务目录中查找视频文件"""
    video_exts = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv', '.m4v']
    for file in job_dir.iterdir():
        if file.is_file() and file.suffix.lower() in video_exts:
            return file
    return None


def _serve_file_with_range(file_path: Path, request: Request, media_type: str):
    """
    支持HTTP Range请求的文件流式传输（允许拖拽进度条）
    """
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        # 无Range请求，返回完整文件
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name
        )

    # 解析Range头：bytes=start-end
    try:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1

        # 确保范围有效
        if start >= file_size:
            raise HTTPException(status_code=416, detail="请求范围无效")
        end = min(end, file_size - 1)

    except ValueError:
        raise HTTPException(status_code=400, detail="无效的Range头格式")

    # 返回部分内容（状态码206）
    def file_iterator():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            chunk_size = 8192  # 8KB
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                yield data
                remaining -= len(data)

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    }

    return StreamingResponse(
        file_iterator(),
        status_code=206,
        media_type=media_type,
        headers=headers
    )


async def _get_video_resolution(video_path: Path) -> tuple:
    """使用FFprobe获取视频分辨率"""
    ffmpeg_cmd = config.get_ffmpeg_command()
    ffprobe_cmd = ffmpeg_cmd.replace('ffmpeg', 'ffprobe')

    cmd = [
        ffprobe_cmd, '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        str(video_path)
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        data = json.loads(stdout.decode())
        streams = data.get('streams', [])
        if streams:
            width = streams[0].get('width', 0)
            height = streams[0].get('height', 0)
            return (width, height)
    except:
        pass
    return (0, 0)


async def _get_video_duration(video_path: Path) -> float:
    """使用FFprobe获取视频时长"""
    ffmpeg_cmd = config.get_ffmpeg_command()
    ffprobe_cmd = ffmpeg_cmd.replace('ffmpeg', 'ffprobe')

    # 检查视频文件是否存在
    if not video_path.exists():
        print(f"[media] 视频文件不存在: {video_path}")
        return 0.0

    cmd = [
        ffprobe_cmd, '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    
    print(f"[media] FFprobe 命令: {' '.join(cmd)}")

    try:
        # 使用同步 subprocess 来避免异步事件循环问题
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        if result.returncode != 0:
            print(f"[media] FFprobe 失败 (返回码 {result.returncode}): {result.stderr}")
            return 0.0
            
        duration_str = result.stdout.strip()
        if not duration_str:
            print(f"[media] FFprobe 返回空时长，视频路径: {video_path}")
            return 0.0
        
        print(f"[media] FFprobe 成功获取时长: {duration_str}秒")
        return float(duration_str)
    except subprocess.TimeoutExpired:
        print(f"[media] FFprobe 超时，视频路径: {video_path}")
        return 0.0
    except Exception as e:
        print(f"[media] 获取视频时长异常: {e}, 路径: {video_path}")
        return 0.0


async def _generate_proxy_video_with_progress(source: Path, output: Path, job_id: str):
    """
    异步生成Proxy视频（带进度追踪和SSE推送）
    720p, H.264, 适合浏览器预览
    """
    global _proxy_generation_status

    if output.exists():
        return True

    # 初始化状态
    _proxy_generation_status[job_id] = {
        "status": "starting",
        "progress": 0,
        "start_time": time.time(),
        "duration": 0
    }

    try:
        # 获取视频时长（用于计算进度）
        duration = await _get_video_duration(source)
        _proxy_generation_status[job_id]["duration"] = duration

        ffmpeg_cmd = config.get_ffmpeg_command()
        cmd = [
            ffmpeg_cmd,
            '-i', str(source),
            '-vf', 'scale=-2:720',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '28',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-progress', 'pipe:1',  # 输出进度到stdout
            '-y',
            str(output)
        ]

        print(f"[media] 开始生成Proxy视频: {source.name} -> proxy.mp4")
        _proxy_generation_status[job_id]["status"] = "processing"

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # 解析FFmpeg进度输出
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            line_str = line.decode().strip()
            if line_str.startswith('out_time_ms='):
                try:
                    out_time_ms = int(line_str.split('=')[1])
                    if duration > 0:
                        progress = min(100, (out_time_ms / 1000000) / duration * 100)
                        _proxy_generation_status[job_id]["progress"] = round(progress, 1)

                        # 推送SSE进度（如果有连接的话）
                        _push_proxy_progress(job_id, progress)
                except:
                    pass

        await process.wait()

        if process.returncode == 0 and output.exists():
            _proxy_generation_status[job_id]["status"] = "completed"
            _proxy_generation_status[job_id]["progress"] = 100
            print(f"[media] Proxy视频生成完成: {output}")
            _push_proxy_progress(job_id, 100, completed=True)
            return True
        else:
            _proxy_generation_status[job_id]["status"] = "failed"
            print(f"[media] Proxy视频生成失败")
            return False

    except Exception as e:
        _proxy_generation_status[job_id]["status"] = "failed"
        print(f"[media] 生成Proxy视频异常: {e}")
        return False


def _push_proxy_progress(job_id: str, progress: float, completed: bool = False):
    """推送Proxy生成进度到SSE"""
    try:
        from services.sse_service import get_sse_manager
        sse_manager = get_sse_manager()

        channel_id = f"job:{job_id}"
        event_type = "proxy_complete" if completed else "proxy_progress"

        sse_manager.broadcast_sync(
            channel_id,
            event_type,
            {
                "job_id": job_id,
                "progress": progress,
                "completed": completed
            }
        )
    except Exception as e:
        # SSE推送失败不影响Proxy生成
        pass


def _generate_peaks_with_ffmpeg(audio_path: Path, samples: int = 2000) -> Tuple[list, float]:
    """
    使用FFmpeg生成波形峰值数据（比Python wave更高效）
    适用于大文件（>100MB）
    """
    ffmpeg_cmd = config.get_ffmpeg_command()
    ffprobe_cmd = ffmpeg_cmd.replace('ffmpeg', 'ffprobe')

    # 获取音频时长
    probe_cmd = [
        ffprobe_cmd, '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]

    try:
        result = subprocess.run(
            probe_cmd, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        duration = float(result.stdout.strip())
    except:
        duration = 0

    if duration <= 0:
        return [], 0

    # 使用FFmpeg提取原始PCM数据（降采样到8kHz单声道以减少数据量）
    cmd = [
        ffmpeg_cmd,
        '-i', str(audio_path),
        '-ac', '1',           # 单声道
        '-ar', '8000',        # 8kHz采样率
        '-f', 's16le',        # 16-bit PCM
        '-acodec', 'pcm_s16le',
        '-'
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        if result.returncode != 0:
            return [], duration

        # 解析PCM数据
        pcm_data = result.stdout
        sample_count = len(pcm_data) // 2  # 16-bit = 2 bytes

        if sample_count < samples:
            samples = sample_count

        chunk_size = sample_count // samples
        peaks = []

        for i in range(samples):
            start_idx = i * chunk_size * 2
            end_idx = min(start_idx + chunk_size * 2, len(pcm_data))
            chunk_bytes = pcm_data[start_idx:end_idx]

            if len(chunk_bytes) < 2:
                peaks.extend([0.0, 0.0])
                continue

            # 解析为16-bit整数
            fmt = f'<{len(chunk_bytes) // 2}h'
            try:
                samples_chunk = struct.unpack(fmt, chunk_bytes)
                normalized = [s / 32768.0 for s in samples_chunk]
                peaks.append(float(min(normalized)))
                peaks.append(float(max(normalized)))
            except:
                peaks.extend([0.0, 0.0])

        return peaks, duration

    except Exception as e:
        print(f"[media] FFmpeg波形生成失败: {e}")
        return [], duration


def _generate_peaks_with_wave(audio_path: Path, samples: int = 2000) -> Tuple[list, float]:
    """
    使用Python wave模块生成波形峰值数据（流式读取，低内存占用）
    """
    try:
        with wave.open(str(audio_path), 'rb') as wav:
            frame_rate = wav.getframerate()
            n_frames = wav.getnframes()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            duration = n_frames / frame_rate

            if n_frames < samples:
                samples = n_frames
            chunk_frames = n_frames // samples
            peaks = []

            for i in range(samples):
                frames_data = wav.readframes(chunk_frames)

                if not frames_data:
                    peaks.extend([0.0, 0.0])
                    continue

                try:
                    if sample_width == 2:  # 16-bit
                        fmt = f'<{len(frames_data) // 2}h'
                        samples_chunk = struct.unpack(fmt, frames_data)
                        max_val = 32768.0
                    elif sample_width == 4:  # 32-bit
                        fmt = f'<{len(frames_data) // 4}i'
                        samples_chunk = struct.unpack(fmt, frames_data)
                        max_val = 2147483648.0
                    elif sample_width == 1:  # 8-bit
                        samples_chunk = list(frames_data)
                        max_val = 128.0
                    else:
                        samples_chunk = [0]
                        max_val = 1.0

                    if n_channels > 1:
                        samples_chunk = samples_chunk[::n_channels]

                    if samples_chunk:
                        normalized = [s / max_val for s in samples_chunk]
                        peaks.append(float(min(normalized)))
                        peaks.append(float(max(normalized)))
                    else:
                        peaks.extend([0.0, 0.0])
                except:
                    peaks.extend([0.0, 0.0])

            return peaks, duration

    except Exception as e:
        print(f"[media] Wave波形生成失败: {e}")
        return [], 0


async def _generate_sprite_thumbnails(video_path: Path, output_path: Path, count: int = 20, cols: int = 5) -> dict:
    """
    生成Sprite雪碧图（将多个缩略图合并为单张图片，减少HTTP请求）

    Args:
        video_path: 视频文件路径
        output_path: 输出目录
        count: 缩略图数量
        cols: 每行列数

    Returns:
        dict: { sprite_url, thumb_width, thumb_height, cols, rows, timestamps }
    """
    import base64

    ffmpeg_cmd = config.get_ffmpeg_command()
    print(f"[media] 生成Sprite图，视频: {video_path}, FFmpeg: {ffmpeg_cmd}")

    # 获取视频时长
    duration = await _get_video_duration(video_path)
    print(f"[media] Sprite - 视频时长: {duration}秒")
    if duration <= 0:
        print(f"[media] Sprite - 无法获取时长，跳过Sprite生成")
        return None

    # 计算时间点
    interval = duration / count
    timestamps = [i * interval for i in range(count)]

    # 计算行数
    rows = (count + cols - 1) // cols

    # 使用FFmpeg的tile滤镜生成Sprite图
    # 先提取帧，再合并
    sprite_file = output_path / "sprite.jpg"

    # 生成用于提取帧的时间点参数
    select_expr = '+'.join([f'eq(n,{int(t * 25)})' for t in timestamps])  # 假设25fps

    # 使用更可靠的方法：逐帧提取再合并
    temp_frames = []
    thumb_width = 160
    thumb_height = None

    for i, ts in enumerate(timestamps):
        cmd = [
            ffmpeg_cmd,
            '-ss', str(ts),
            '-i', str(video_path),
            '-vframes', '1',
            '-vf', f'scale={thumb_width}:-1',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-q:v', '5',
            '-'
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                timeout=10
            )
            if result.returncode == 0 and result.stdout:
                temp_frames.append(result.stdout)
            else:
                temp_frames.append(None)
        except:
            temp_frames.append(None)

    # 使用PIL合并图片（如果可用）
    try:
        from PIL import Image

        # 获取第一张有效图片的尺寸
        first_valid = None
        for frame_data in temp_frames:
            if frame_data:
                first_valid = Image.open(io.BytesIO(frame_data))
                thumb_width, thumb_height = first_valid.size
                break

        if not first_valid:
            return None

        # 创建Sprite画布
        sprite_width = thumb_width * cols
        sprite_height = thumb_height * rows
        sprite_img = Image.new('RGB', (sprite_width, sprite_height), (0, 0, 0))

        # 粘贴各帧
        for i, frame_data in enumerate(temp_frames):
            if frame_data:
                try:
                    img = Image.open(io.BytesIO(frame_data))
                    col = i % cols
                    row = i // cols
                    x = col * thumb_width
                    y = row * thumb_height
                    sprite_img.paste(img, (x, y))
                except:
                    pass

        # 保存Sprite图
        sprite_img.save(str(sprite_file), 'JPEG', quality=80)

        # 转为base64
        with open(sprite_file, 'rb') as f:
            sprite_base64 = base64.b64encode(f.read()).decode('utf-8')

        return {
            "sprite": f"data:image/jpeg;base64,{sprite_base64}",
            "sprite_url": f"/api/media/{output_path.name}/sprite.jpg",
            "thumb_width": thumb_width,
            "thumb_height": thumb_height,
            "cols": cols,
            "rows": rows,
            "count": count,
            "timestamps": timestamps,
            "duration": duration
        }

    except ImportError:
        # PIL不可用，回退到单独的缩略图
        print("[media] PIL不可用，使用单独缩略图模式")
        return None


@router.get("/{job_id}/video")
async def get_video(job_id: str, request: Request):
    """
    获取视频文件（支持Range请求，自动Proxy转码）

    优先返回Proxy视频（如果存在），否则返回源视频
    对于不兼容的格式，会触发异步生成Proxy
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 1. 查找Proxy视频（优先）
    proxy_video = job_dir / "proxy.mp4"
    if proxy_video.exists():
        return _serve_file_with_range(proxy_video, request, 'video/mp4')

    # 2. 查找源视频
    video_file = _find_video_file(job_dir)
    if not video_file:
        raise HTTPException(status_code=404, detail="视频文件不存在")

    # 3. 检查是否需要生成Proxy
    if video_file.suffix.lower() in NEED_TRANSCODE_FORMATS:
        # 检查是否已在生成中
        if job_id in _proxy_generation_status and _proxy_generation_status[job_id]["status"] == "processing":
            raise HTTPException(
                status_code=202,
                detail={
                    "message": "Proxy视频生成中...",
                    "progress": _proxy_generation_status[job_id].get("progress", 0),
                    "proxy_generating": True
                }
            )

        # 异步生成Proxy视频（带进度追踪）
        asyncio.create_task(_generate_proxy_video_with_progress(video_file, proxy_video, job_id))

        raise HTTPException(
            status_code=202,
            detail={
                "message": "视频格式不兼容，正在生成预览版本...",
                "format": video_file.suffix,
                "proxy_generating": True
            }
        )

    # 4. 返回兼容格式的源视频
    return _serve_file_with_range(video_file, request, 'video/mp4')


@router.get("/{job_id}/audio")
async def get_audio(job_id: str, request: Request):
    """获取音频文件（支持Range请求）"""
    job_dir = config.JOBS_DIR / job_id
    audio_file = job_dir / "audio.wav"

    if not audio_file.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")

    return _serve_file_with_range(audio_file, request, 'audio/wav')


@router.get("/{job_id}/peaks")
async def get_audio_peaks(job_id: str, samples: int = 2000, method: str = "auto"):
    """
    获取音频波形峰值数据

    Args:
        job_id: 任务ID
        samples: 采样点数（默认2000）
        method: 生成方法 - "auto"(自动选择), "ffmpeg"(FFmpeg加速), "wave"(Python原生)

    Returns:
        JSON: { peaks: [min, max, min, max, ...], duration: 180.5, method: "ffmpeg" }
    """
    job_dir = config.JOBS_DIR / job_id
    audio_file = job_dir / "audio.wav"
    peaks_cache_file = job_dir / f"peaks_{samples}.json"

    # 检查缓存
    if peaks_cache_file.exists():
        try:
            with open(peaks_cache_file, 'r') as f:
                return JSONResponse(json.load(f))
        except:
            pass

    if not audio_file.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")

    # 根据文件大小选择方法
    file_size_mb = audio_file.stat().st_size / (1024 * 1024)

    if method == "auto":
        # 大于50MB使用FFmpeg加速
        use_ffmpeg = file_size_mb > 50
    elif method == "ffmpeg":
        use_ffmpeg = True
    else:
        use_ffmpeg = False

    try:
        if use_ffmpeg:
            peaks, duration = _generate_peaks_with_ffmpeg(audio_file, samples)
            used_method = "ffmpeg"
        else:
            peaks, duration = _generate_peaks_with_wave(audio_file, samples)
            used_method = "wave"

        # 如果FFmpeg失败，回退到wave
        if not peaks and use_ffmpeg:
            peaks, duration = _generate_peaks_with_wave(audio_file, samples)
            used_method = "wave_fallback"

        result = {
            "peaks": peaks,
            "duration": duration,
            "method": used_method,
            "samples": len(peaks) // 2
        }

        # 缓存结果
        try:
            with open(peaks_cache_file, 'w') as f:
                json.dump(result, f)
        except:
            pass

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"波形数据生成失败: {str(e)}")


@router.get("/{job_id}/proxy-status")
async def check_proxy_status(job_id: str):
    """
    检查Proxy视频生成状态（支持进度查询）

    Returns:
        JSON: { exists, size, generating, progress, status }
    """
    job_dir = config.JOBS_DIR / job_id
    proxy_video = job_dir / "proxy.mp4"

    if proxy_video.exists():
        return JSONResponse({
            "exists": True,
            "size": proxy_video.stat().st_size,
            "generating": False,
            "progress": 100,
            "status": "completed"
        })

    # 检查生成状态
    if job_id in _proxy_generation_status:
        status_info = _proxy_generation_status[job_id]
        return JSONResponse({
            "exists": False,
            "size": 0,
            "generating": status_info["status"] == "processing",
            "progress": status_info.get("progress", 0),
            "status": status_info["status"],
            "elapsed": time.time() - status_info.get("start_time", time.time())
        })

    # 检查源视频是否需要转码
    video_file = _find_video_file(job_dir)
    needs_proxy = video_file and video_file.suffix.lower() in NEED_TRANSCODE_FORMATS

    return JSONResponse({
        "exists": False,
        "size": 0,
        "generating": False,
        "progress": 0,
        "status": "not_started",
        "needs_proxy": needs_proxy
    })


@router.get("/{job_id}/thumbnail")
async def get_thumbnail(job_id: str):
    """
    获取任务的单个缩略图（第一帧）用于任务卡片展示（第二阶段修复：实时更新）

    Args:
        job_id: 任务ID

    Returns:
        Base64编码的缩略图 or JSON占位符
    """
    job_dir = config.JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    # 检查缓存的缩略图文件
    cached_thumbnail = job_dir / "thumbnail.jpg"
    if cached_thumbnail.exists():
        try:
            import base64
            with open(cached_thumbnail, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            return JSONResponse({
                "thumbnail": f"data:image/jpeg;base64,{img_base64}",
                "cached": True
            })
        except:
            pass  # 缓存读取失败，重新生成

    try:
        import base64

        # 查找视频文件
        video_file = _find_video_file(job_dir)
        if not video_file:
            proxy = job_dir / "proxy.mp4"
            if proxy.exists():
                video_file = proxy
            else:
                return JSONResponse({
                    "thumbnail": None,
                    "message": "视频文件不存在"
                })

        # 获取视频分辨率（决定缩略图大小）
        width, height = await _get_video_resolution(video_file)

        # 计算缩略图尺寸：只有4K及以上视频才压缩，其他保持原分辨率（但限制最大1920px）
        if width > 3840:  # 4K视频
            thumb_width = 1920
        elif width > 1920:  # 大于1080p但不到4K
            thumb_width = width  # 保持原分辨率
        elif width > 0:
            thumb_width = width  # 保持原分辨率
        else:
            thumb_width = 1280  # 默认值（无法检测分辨率时）

        # 获取第一帧作为缩略图
        ffmpeg_cmd = config.get_ffmpeg_command()

        cmd = [
            ffmpeg_cmd,
            '-ss', '1',  # 从1秒开始（避免纯黑帧）
            '-i', str(video_file),
            '-vframes', '1',  # 只提取1帧
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-vf', f'scale={thumb_width}:-1',  # 根据原视频分辨率设置宽度
            '-q:v', '2',  # 高质量（2-5之间，数字越小质量越高）
            '-'
        ]

        result = subprocess.run(
            cmd, capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            timeout=10
        )

        if result.returncode == 0 and result.stdout:
            # 缓存到文件系统
            try:
                with open(cached_thumbnail, 'wb') as f:
                    f.write(result.stdout)
                print(f"[media] 缩略图已缓存: {job_id} (宽度: {thumb_width}px)")
            except Exception as e:
                print(f"[media] 缓存缩略图失败: {e}")

            img_base64 = base64.b64encode(result.stdout).decode('utf-8')
            return JSONResponse({
                "thumbnail": f"data:image/jpeg;base64,{img_base64}",
                "width": thumb_width,
                "cached": False
            })
        else:
            return JSONResponse({
                "thumbnail": None,
                "message": "无法生成缩略图"
            })

    except Exception as e:
        return JSONResponse({
            "thumbnail": None,
            "message": str(e)
        })


@router.get("/{job_id}/thumbnails")
async def get_thumbnails(job_id: str, count: int = 10, sprite: bool = True):
    """
    获取视频缩略图

    Args:
        job_id: 任务ID
        count: 缩略图数量（默认10张）
        sprite: 是否使用Sprite雪碧图（默认True，减少请求数）

    Returns:
        - sprite=True: { sprite, thumb_width, thumb_height, cols, rows, timestamps }
        - sprite=False: { thumbnails: [base64_img1, ...], timestamps: [...] }
    """
    job_dir = config.JOBS_DIR / job_id

    # 检查Sprite缓存
    sprite_cache = job_dir / f"sprite_{count}.json"
    thumbnails_cache = job_dir / f"thumbnails_{count}.json"

    if sprite and sprite_cache.exists():
        try:
            with open(sprite_cache, 'r') as f:
                return JSONResponse(json.load(f))
        except:
            pass

    if not sprite and thumbnails_cache.exists():
        try:
            with open(thumbnails_cache, 'r') as f:
                return JSONResponse(json.load(f))
        except:
            pass

    # 查找视频文件
    video_file = _find_video_file(job_dir)
    if not video_file:
        proxy = job_dir / "proxy.mp4"
        if proxy.exists():
            video_file = proxy
        else:
            raise HTTPException(status_code=404, detail="视频文件不存在")

    try:
        import base64

        # 尝试生成Sprite图（如果请求且PIL可用）
        if sprite:
            sprite_result = await _generate_sprite_thumbnails(video_file, job_dir, count)
            if sprite_result:
                # 缓存结果
                try:
                    with open(sprite_cache, 'w') as f:
                        json.dump(sprite_result, f)
                except:
                    pass
                return JSONResponse(sprite_result)

        # 回退到单独缩略图模式
        print(f"[media] 回退到单独缩略图模式，视频文件: {video_file}")
        ffmpeg_cmd = config.get_ffmpeg_command()
        duration = await _get_video_duration(video_file)
        print(f"[media] 获取到视频时长: {duration}秒")

        if duration <= 0:
            raise HTTPException(status_code=500, detail=f"无法获取视频时长，视频路径: {video_file}")

        interval = duration / count
        thumbnails = []
        timestamps = []

        for i in range(count):
            timestamp = i * interval
            timestamps.append(timestamp)

            cmd = [
                ffmpeg_cmd,
                '-ss', str(timestamp),
                '-i', str(video_file),
                '-vframes', '1',
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',
                '-vf', 'scale=160:-1',
                '-q:v', '5',
                '-'
            ]

            result = subprocess.run(
                cmd, capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                timeout=10
            )

            if result.returncode == 0 and result.stdout:
                img_base64 = base64.b64encode(result.stdout).decode('utf-8')
                thumbnails.append(f"data:image/jpeg;base64,{img_base64}")
            else:
                thumbnails.append(None)

        result = {
            "thumbnails": thumbnails,
            "timestamps": timestamps,
            "duration": duration,
            "sprite": False
        }

        # 缓存结果
        try:
            with open(thumbnails_cache, 'w') as f:
                json.dump(result, f)
        except:
            pass

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缩略图生成失败: {str(e)}")


@router.get("/{job_id}/sprite.jpg")
async def get_sprite_image(job_id: str):
    """直接获取Sprite图片文件"""
    job_dir = config.JOBS_DIR / job_id
    sprite_file = job_dir / "sprite.jpg"

    if not sprite_file.exists():
        raise HTTPException(status_code=404, detail="Sprite图不存在")

    return FileResponse(str(sprite_file), media_type="image/jpeg")


@router.post("/{job_id}/post-process")
async def post_process_transcription(job_id: str):
    """
    转录后处理：预生成编辑器所需的所有数据
    在转录完成后调用，异步生成波形、缩略图、Proxy视频

    Returns:
        JSON: { peaks: bool, thumbnails: bool, proxy: bool, sprite: bool }
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    results = {
        "peaks": False,
        "thumbnails": False,
        "sprite": False,
        "proxy": False,
        "proxy_needed": False
    }

    # 1. 生成波形峰值
    try:
        audio_file = job_dir / "audio.wav"
        if audio_file.exists():
            peaks_cache = job_dir / "peaks_2000.json"
            if not peaks_cache.exists():
                await get_audio_peaks(job_id, 2000)
            results["peaks"] = True
    except Exception as e:
        print(f"[media] 波形生成失败: {e}")

    # 2. 生成缩略图（优先Sprite图）
    try:
        sprite_cache = job_dir / "sprite_10.json"
        thumbnails_cache = job_dir / "thumbnails_10.json"
        if not sprite_cache.exists() and not thumbnails_cache.exists():
            await get_thumbnails(job_id, 10, sprite=True)
        results["thumbnails"] = True
        results["sprite"] = sprite_cache.exists()
    except Exception as e:
        print(f"[media] 缩略图生成失败: {e}")

    # 3. 检查是否需要生成Proxy
    video_file = _find_video_file(job_dir)
    if video_file and video_file.suffix.lower() in NEED_TRANSCODE_FORMATS:
        results["proxy_needed"] = True
        proxy_video = job_dir / "proxy.mp4"
        if not proxy_video.exists():
            asyncio.create_task(_generate_proxy_video_with_progress(video_file, proxy_video, job_id))
            results["proxy"] = True

    return JSONResponse(results)


@router.get("/{job_id}/srt")
async def get_srt_content(job_id: str):
    """
    获取SRT字幕文件内容

    Returns:
        JSON: { job_id, filename, content, encoding }
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    srt_file = None
    for file in job_dir.iterdir():
        if file.suffix.lower() == '.srt':
            srt_file = file
            break

    if not srt_file:
        raise HTTPException(status_code=404, detail="SRT文件不存在")

    try:
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        return JSONResponse({
            "job_id": job_id,
            "filename": srt_file.name,
            "content": content,
            "encoding": "utf-8"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取SRT文件失败: {str(e)}")


@router.post("/{job_id}/srt")
async def save_srt_content(job_id: str, request: Request):
    """
    保存编辑后的SRT字幕文件

    Body:
        { content: "1\n00:00:01,000 --> 00:00:03,000\n..." }
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    try:
        body = await request.json()
        content = body.get("content")

        if not content:
            raise HTTPException(status_code=400, detail="缺少content参数")

        srt_file = None
        for file in job_dir.iterdir():
            if file.suffix.lower() == '.srt':
                srt_file = file
                break

        if not srt_file:
            video_file = _find_video_file(job_dir)
            if video_file:
                srt_file = job_dir / f"{video_file.stem}.srt"
            else:
                srt_file = job_dir / "output.srt"

        # 备份原文件
        if srt_file.exists():
            backup_file = job_dir / f"{srt_file.stem}.srt.bak"
            import shutil
            shutil.copy2(srt_file, backup_file)

        with open(srt_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return JSONResponse({
            "success": True,
            "message": "SRT文件保存成功",
            "filename": srt_file.name
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存SRT文件失败: {str(e)}")


@router.get("/{job_id}/info")
async def get_media_info(job_id: str):
    """
    获取任务的媒体信息摘要

    Returns:
        JSON: 包含视频、音频、SRT等文件的可用状态
    """
    job_dir = config.JOBS_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在")

    video_file = _find_video_file(job_dir)
    audio_file = job_dir / "audio.wav"
    proxy_video = job_dir / "proxy.mp4"
    peaks_cache = job_dir / "peaks_2000.json"
    sprite_cache = job_dir / "sprite_10.json"
    thumbnails_cache = job_dir / "thumbnails_10.json"

    srt_file = None
    for file in job_dir.iterdir():
        if file.suffix.lower() == '.srt':
            srt_file = file
            break

    needs_proxy = video_file and video_file.suffix.lower() in NEED_TRANSCODE_FORMATS

    # 获取Proxy生成状态
    proxy_status = None
    if job_id in _proxy_generation_status:
        proxy_status = _proxy_generation_status[job_id]

    return JSONResponse({
        "job_id": job_id,
        "video": {
            "exists": video_file is not None,
            "filename": video_file.name if video_file else None,
            "format": video_file.suffix if video_file else None,
            "needs_proxy": needs_proxy,
            "proxy_exists": proxy_video.exists(),
            "proxy_generating": proxy_status and proxy_status.get("status") == "processing",
            "proxy_progress": proxy_status.get("progress", 0) if proxy_status else 0,
            "url": f"/api/media/{job_id}/video" if video_file or proxy_video.exists() else None
        },
        "audio": {
            "exists": audio_file.exists(),
            "url": f"/api/media/{job_id}/audio" if audio_file.exists() else None
        },
        "peaks": {
            "exists": peaks_cache.exists(),
            "url": f"/api/media/{job_id}/peaks" if audio_file.exists() else None
        },
        "thumbnails": {
            "exists": thumbnails_cache.exists() or sprite_cache.exists(),
            "sprite_exists": sprite_cache.exists(),
            "url": f"/api/media/{job_id}/thumbnails" if video_file else None,
            "sprite_url": f"/api/media/{job_id}/sprite.jpg" if sprite_cache.exists() else None
        },
        "srt": {
            "exists": srt_file is not None,
            "filename": srt_file.name if srt_file else None,
            "url": f"/api/media/{job_id}/srt" if srt_file else None
        }
    })
