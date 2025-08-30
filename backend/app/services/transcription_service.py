"""
转录处理服务
"""
import os, subprocess, uuid, threading, json, math, gc, logging
from typing import List, Dict, Optional, Tuple
from pydub import AudioSegment, silence
import whisperx
import torch
import shutil

from models.job_models import JobSettings, JobState
from models.hardware_models import HardwareInfo, OptimizationConfig
from services.hardware_service import get_hardware_detector, get_hardware_optimizer

# 全局模型缓存 (按 (model, compute_type, device) 键)
_model_cache: Dict[Tuple[str, str, str], object] = {}
_align_model_cache: Dict[str, Tuple[object, object]] = {}
_model_lock = threading.Lock()
_align_lock = threading.Lock()

# 音频处理配置
SEGMENT_LEN_MS = 60_000
SILENCE_SEARCH_MS = 2_000
MIN_SILENCE_LEN_MS = 300
SILENCE_THRESH_DBFS = -40

# 进度权重配置
PHASE_WEIGHTS = {
    "extract": 5,
    "split": 10,
    "transcribe": 80,
    "srt": 5
}
TOTAL_WEIGHT = sum(PHASE_WEIGHTS.values())


class TranscriptionService:
    """转录处理服务"""
    
    def __init__(self, jobs_root: str):
        self.jobs_root = jobs_root
        os.makedirs(self.jobs_root, exist_ok=True)
        self.jobs: Dict[str, JobState] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # 初始化硬件检测和优化
        self.hardware_detector = get_hardware_detector()
        self.hardware_optimizer = get_hardware_optimizer()
        self._hardware_info: Optional[HardwareInfo] = None
        self._optimization_config: Optional[OptimizationConfig] = None
        
        # 执行硬件检测
        self._detect_hardware()

    def _detect_hardware(self):
        """执行硬件检测并生成优化配置"""
        try:
            self.logger.info("开始硬件检测...")
            self._hardware_info = self.hardware_detector.detect()
            self._optimization_config = self.hardware_optimizer.get_optimization_config(self._hardware_info)
            
            # 记录检测结果
            hw = self._hardware_info
            opt = self._optimization_config
            self.logger.info(f"硬件检测完成 - GPU: {'✓' if hw.cuda_available else '✗'}, "
                           f"CPU: {hw.cpu_cores}核/{hw.cpu_threads}线程, "
                           f"内存: {hw.memory_total_mb}MB, "
                           f"优化配置: batch={opt.batch_size}, device={opt.recommended_device}")
        except Exception as e:
            self.logger.error(f"硬件检测失败: {e}")
    
    def get_hardware_info(self) -> Optional[HardwareInfo]:
        """获取硬件信息"""
        return self._hardware_info
    
    def get_optimization_config(self) -> Optional[OptimizationConfig]:
        """获取优化配置"""  
        return self._optimization_config
    
    def get_optimized_job_settings(self, base_settings: Optional[JobSettings] = None) -> JobSettings:
        """获取基于硬件优化的任务设置"""
        # 使用硬件优化配置作为默认值
        if self._optimization_config:
            optimized = JobSettings(
                model=base_settings.model if base_settings else "medium",
                compute_type=base_settings.compute_type if base_settings else "float16",
                device=self._optimization_config.recommended_device,
                batch_size=self._optimization_config.batch_size,
                word_timestamps=base_settings.word_timestamps if base_settings else False
            )
            return optimized
        
        # 如果没有硬件信息，使用传入的设置或默认设置
        return base_settings or JobSettings()

    def create_job(self, filename: str, src_path: str, settings: JobSettings, job_id: Optional[str] = None) -> JobState:
        """创建转录任务"""
        job_id = job_id or uuid.uuid4().hex
        job_dir = os.path.join(self.jobs_root, job_id)
        os.makedirs(job_dir, exist_ok=True)
        dest_path = os.path.join(job_dir, filename)
        
        # 复制文件到任务目录
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            try:
                shutil.copyfile(src_path, dest_path)
            except Exception:
                pass
        
        job = JobState(
            job_id=job_id, 
            filename=filename, 
            dir=job_dir, 
            input_path=src_path,  # 保存原始输入路径
            settings=settings, 
            status="uploaded", 
            phase="pending", 
            message="文件已上传"
        )
        
        with self.lock:
            self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[JobState]:
        """获取任务状态"""
        with self.lock:
            return self.jobs.get(job_id)

    def start_job(self, job_id: str):
        """启动转录任务"""
        job = self.get_job(job_id)
        if not job or job.status not in ("uploaded", "failed"):
            return
        
        job.canceled = False
        job.error = None
        job.status = "processing"
        job.message = "开始处理"
        threading.Thread(target=self._run_pipeline, args=(job,), daemon=True).start()

    def cancel_job(self, job_id: str) -> bool:
        """取消转录任务"""
        job = self.get_job(job_id)
        if not job:
            return False
        job.canceled = True
        job.message = "取消中..."
        return True

    def _update_progress(self, job: JobState, phase: str, phase_ratio: float, message: str = ""):
        """更新任务进度"""
        job.phase = phase
        # 计算累计进度
        done_weight = 0
        for p, w in PHASE_WEIGHTS.items():
            if p == phase:
                break
            done_weight += w
        current_weight = PHASE_WEIGHTS.get(phase, 0) * max(0.0, min(1.0, phase_ratio))
        job.progress = round((done_weight + current_weight) / TOTAL_WEIGHT * 100, 2)
        if message:
            job.message = message

    def _run_pipeline(self, job: JobState):
        """执行转录处理管道"""
        try:
            if job.canceled:
                job.status = 'canceled'
                job.message = '已取消'
                return

            input_path = os.path.join(job.dir, job.filename)
            audio_path = os.path.join(job.dir, 'audio.wav')
            
            # 1. 提取音频
            self._update_progress(job, 'extract', 0, '提取音频中')
            if job.canceled: 
                raise RuntimeError('任务已取消')
            if not self._extract_audio(input_path, audio_path):
                raise RuntimeError('FFmpeg 提取音频失败')
            self._update_progress(job, 'extract', 1, '音频提取完成')
            
            if job.canceled: 
                raise RuntimeError('任务已取消')
            
            # 2. 分段
            self._update_progress(job, 'split', 0, '音频分段中')
            segments = self._split_audio(audio_path)
            if job.canceled: 
                raise RuntimeError('任务已取消')
            job.segments = segments
            job.total = len(segments)
            self._update_progress(job, 'split', 1, f'分段完成 共{job.total}段')
            
            # 3. 转录
            self._update_progress(job, 'transcribe', 0, '加载模型中')
            if job.canceled: 
                raise RuntimeError('任务已取消')
            model = self._get_model(job.settings)
            align_cache = {}
            processed_results = []
            
            for idx, seg in enumerate(segments):
                if job.canceled:
                    raise RuntimeError('任务已取消')
                ratio = idx / max(1, len(segments))
                self._update_progress(job, 'transcribe', ratio, f'转录 {idx+1}/{len(segments)}')
                seg_result = self._transcribe_segment(seg, model, job, align_cache)
                if seg_result:
                    processed_results.append(seg_result)
                job.processed = idx + 1
            
            self._update_progress(job, 'transcribe', 1, '转录完成 生成字幕中')
            if job.canceled: 
                raise RuntimeError('任务已取消')
            
            # 4. 生成SRT
            srt_path = os.path.join(job.dir, os.path.splitext(job.filename)[0] + '.srt')
            self._update_progress(job, 'srt', 0, '写入 SRT...')
            self._generate_srt(processed_results, srt_path, job.settings.word_timestamps)
            self._update_progress(job, 'srt', 1, '处理完成')
            job.srt_path = srt_path
            
            if job.canceled:
                job.status = 'canceled'
                job.message = '已取消'
            else:
                job.status = 'finished'
                job.message = '完成'
                
        except Exception as e:
            if job.canceled and '取消' in str(e):
                job.status = 'canceled'
                job.message = '已取消'
            else:
                job.status = 'failed'
                job.message = f'失败: {e}'
                job.error = str(e)
        finally:
            gc.collect()

    def _extract_audio(self, input_file: str, audio_out: str) -> bool:
        """使用FFmpeg提取音频"""
        if os.path.exists(audio_out):
            return True
        cmd = ['ffmpeg', '-y', '-i', input_file, '-vn', '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', audio_out]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc.returncode == 0 and os.path.exists(audio_out)

    def _split_audio(self, audio_path: str) -> List[Dict]:
        """将音频分段处理"""
        audio = AudioSegment.from_wav(audio_path)
        length = len(audio)
        segments = []
        pos = 0
        idx = 0
        
        while pos < length:
            end = min(pos + SEGMENT_LEN_MS, length)
            # 静音搜索优化分段点
            if end < length and (end - pos) > SILENCE_SEARCH_MS:
                search_start = max(pos, end - SILENCE_SEARCH_MS)
                search_chunk = audio[search_start:end]
                try:
                    silences = silence.detect_silence(
                        search_chunk, 
                        min_silence_len=MIN_SILENCE_LEN_MS, 
                        silence_thresh=SILENCE_THRESH_DBFS
                    )
                    if silences:
                        # 使用第一个静音开始
                        silence_start = silences[0][0]
                        new_end = search_start + silence_start
                        if new_end - pos > MIN_SILENCE_LEN_MS:
                            end = new_end
                except Exception:
                    pass
            
            chunk = audio[pos:end]
            seg_file = os.path.join(os.path.dirname(audio_path), f'segment_{idx}.wav')
            chunk.export(seg_file, format='wav')
            segments.append({'file': seg_file, 'start_ms': pos})
            pos = end
            idx += 1
        return segments

    def _get_model(self, settings: JobSettings):
        """获取或缓存WhisperX模型"""
        key = (settings.model, settings.compute_type, settings.device)
        with _model_lock:
            if key in _model_cache:
                return _model_cache[key]
            m = whisperx.load_model(settings.model, settings.device, compute_type=settings.compute_type)
            _model_cache[key] = m
            return m

    def _get_align_model(self, lang: str, device: str):
        """获取或缓存对齐模型"""
        with _align_lock:
            if lang in _align_model_cache:
                return _align_model_cache[lang]
            am, meta = whisperx.load_align_model(language_code=lang, device=device)
            _align_model_cache[lang] = (am, meta)
            return am, meta

    def _transcribe_segment(self, seg: Dict, model, job: JobState, align_cache: Dict):
        """转录单个音频段"""
        audio = whisperx.load_audio(seg['file'])
        rs = model.transcribe(audio, batch_size=job.settings.batch_size, verbose=False, language=job.language)
        if not rs or 'segments' not in rs:
            return None
        
        # 检测语言
        if not job.language and 'language' in rs:
            job.language = rs['language']
        lang = job.language or rs.get('language')
        
        # 对齐模型
        if lang not in align_cache:
            am, meta = self._get_align_model(lang, job.settings.device)
            align_cache[lang] = (am, meta)
        am, meta = align_cache[lang]
        aligned = whisperx.align(rs['segments'], am, meta, audio, job.settings.device)
        
        # 调整时间偏移
        start_offset = seg['start_ms'] / 1000.0
        final = {'segments': []}
        if 'segments' in aligned:
            for s in aligned['segments']:
                if 'start' in s: 
                    s['start'] += start_offset
                if 'end' in s: 
                    s['end'] += start_offset
                final['segments'].append(s)
        if 'word_segments' in aligned:
            final['word_segments'] = []
            for w in aligned['word_segments']:
                if 'start' in w: 
                    w['start'] += start_offset
                if 'end' in w: 
                    w['end'] += start_offset
                final['word_segments'].append(w)
        del audio
        return final

    def _format_ts(self, sec: float) -> str:
        """格式化时间戳为SRT格式"""
        if sec < 0: 
            sec = 0
        ms = int(round(sec * 1000))
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def _generate_srt(self, results: List[Dict], path: str, word_level: bool):
        """生成SRT字幕文件"""
        lines = []
        n = 1
        for r in results:
            if not r: 
                continue
            entries = []
            
            if word_level and r.get('word_segments'):
                for w in r['word_segments']:
                    if w.get('start') is not None and w.get('end') is not None:
                        txt = (w.get('word') or '').strip()
                        if txt:
                            entries.append({'start': w['start'], 'end': w['end'], 'text': txt})
            elif r.get('segments'):
                for s in r['segments']:
                    if s.get('start') is not None and s.get('end') is not None:
                        txt = (s.get('text') or '').strip()
                        if txt:
                            entries.append({'start': s['start'], 'end': s['end'], 'text': txt})
            
            for e in entries:
                if e['end'] <= e['start']: 
                    continue
                lines.append(str(n))
                lines.append(f"{self._format_ts(e['start'])} --> {self._format_ts(e['end'])}")
                lines.append(e['text'])
                lines.append("")
                n += 1
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


# 单例处理器
_service_instance: Optional[TranscriptionService] = None


def get_transcription_service(root: str) -> TranscriptionService:
    """获取转录服务实例（单例模式）"""
    global _service_instance
    if _service_instance is None:
        _service_instance = TranscriptionService(root)
    return _service_instance