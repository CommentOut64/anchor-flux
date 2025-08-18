import os, subprocess, uuid, threading, json, math, gc
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pydub import AudioSegment, silence
import whisperx
import torch
import shutil

# 全局模型缓存 (按 (model, compute_type, device) 键)
_model_cache: Dict[Tuple[str, str, str], object] = {}
_align_model_cache: Dict[str, Tuple[object, object]] = {}
_model_lock = threading.Lock()
_align_lock = threading.Lock()

SEGMENT_LEN_MS = 60_000
SILENCE_SEARCH_MS = 2_000
MIN_SILENCE_LEN_MS = 300
SILENCE_THRESH_DBFS = -40

PHASE_WEIGHTS = {
    "extract": 5,
    "split": 10,
    "transcribe": 80,
    "srt": 5
}
TOTAL_WEIGHT = sum(PHASE_WEIGHTS.values())

@dataclass
class JobSettings:
    model: str = "medium"
    compute_type: str = "float16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    word_timestamps: bool = False

@dataclass
class JobState:
    job_id: str
    filename: str
    dir: str
    settings: JobSettings
    status: str = "queued"  # queued, processing, finished, failed, canceled
    phase: str = "pending"  # extract, split, transcribe, srt
    progress: float = 0.0
    message: str = "等待开始"
    error: Optional[str] = None
    segments: List[Dict] = field(default_factory=list)
    processed: int = 0
    total: int = 0
    language: Optional[str] = None
    srt_path: Optional[str] = None
    canceled: bool = False  # 新增取消标记

    def to_dict(self):
        d = asdict(self)
        d.pop('segments', None)  # 不透出内部详情
        return d

class TranscriptionProcessor:
    def __init__(self, jobs_root: str):
        self.jobs_root = jobs_root
        os.makedirs(self.jobs_root, exist_ok=True)
        self.jobs: Dict[str, JobState] = {}
        self.lock = threading.Lock()

    def create_job(self, filename: str, src_path: str, settings: JobSettings, job_id: Optional[str] = None) -> JobState:
        job_id = job_id or uuid.uuid4().hex
        job_dir = os.path.join(self.jobs_root, job_id)
        os.makedirs(job_dir, exist_ok=True)
        dest_path = os.path.join(job_dir, filename)
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            try:
                shutil.copyfile(src_path, dest_path)
            except Exception:
                pass
        job = JobState(job_id=job_id, filename=filename, dir=job_dir, settings=settings, status="uploaded", phase="pending", message="文件已上传")
        with self.lock:
            self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[JobState]:
        with self.lock:
            return self.jobs.get(job_id)

    def start_job(self, job_id: str):
        job = self.get_job(job_id)
        if not job or job.status not in ("uploaded", "failed"):
            return
        job.canceled = False
        job.error = None
        job.status = "processing"
        job.message = "开始处理"
        threading.Thread(target=self._run_pipeline, args=(job,), daemon=True).start()

    def cancel_job(self, job_id: str):
        job = self.get_job(job_id)
        if not job:
            return False
        job.canceled = True
        job.message = "取消中..."
        return True

    def _update_progress(self, job: JobState, phase: str, phase_ratio: float, message: str = ""):
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
        try:
            if job.canceled:
                job.status = 'canceled'
                job.message = '已取消'
                return
            input_path = os.path.join(job.dir, job.filename)
            audio_path = os.path.join(job.dir, 'audio.wav')
            # 提取音频
            self._update_progress(job, 'extract', 0, '提取音频中')
            if job.canceled: raise RuntimeError('任务已取消')
            if not self._extract_audio(input_path, audio_path):
                raise RuntimeError('FFmpeg 提取音频失败')
            self._update_progress(job, 'extract', 1, '音频提取完成')
            if job.canceled: raise RuntimeError('任务已取消')
            # 分段
            self._update_progress(job, 'split', 0, '音频分段中')
            segments = self._split_audio(audio_path)
            if job.canceled: raise RuntimeError('任务已取消')
            job.segments = segments
            job.total = len(segments)
            self._update_progress(job, 'split', 1, f'分段完成 共{job.total}段')
            # 转录
            self._update_progress(job, 'transcribe', 0, '加载模型中')
            if job.canceled: raise RuntimeError('任务已取消')
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
            if job.canceled: raise RuntimeError('任务已取消')
            # 生成SRT
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

    # ---------- 核心步骤实现 ----------
    def _extract_audio(self, input_file: str, audio_out: str) -> bool:
        if os.path.exists(audio_out):
            return True
        cmd = ['ffmpeg', '-y', '-i', input_file, '-vn', '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', audio_out]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return proc.returncode == 0 and os.path.exists(audio_out)

    def _split_audio(self, audio_path: str) -> List[Dict]:
        audio = AudioSegment.from_wav(audio_path)
        length = len(audio)
        segments = []
        pos = 0
        idx = 0
        while pos < length:
            end = min(pos + SEGMENT_LEN_MS, length)
            # 静音搜索
            if end < length and (end - pos) > SILENCE_SEARCH_MS:
                search_start = max(pos, end - SILENCE_SEARCH_MS)
                search_chunk = audio[search_start:end]
                try:
                    silences = silence.detect_silence(search_chunk, min_silence_len=MIN_SILENCE_LEN_MS, silence_thresh=SILENCE_THRESH_DBFS)
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
        key = (settings.model, settings.compute_type, settings.device)
        with _model_lock:
            if key in _model_cache:
                return _model_cache[key]
            m = whisperx.load_model(settings.model, settings.device, compute_type=settings.compute_type)
            _model_cache[key] = m
            return m

    def _get_align_model(self, lang: str, device: str):
        with _align_lock:
            if lang in _align_model_cache:
                return _align_model_cache[lang]
            am, meta = whisperx.load_align_model(language_code=lang, device=device)
            _align_model_cache[lang] = (am, meta)
            return am, meta

    def _transcribe_segment(self, seg: Dict, model, job: JobState, align_cache: Dict):
        audio = whisperx.load_audio(seg['file'])
        rs = model.transcribe(audio, batch_size=job.settings.batch_size, verbose=False, language=job.language)
        if not rs or 'segments' not in rs:
            return None
        # 语言
        if not job.language and 'language' in rs:
            job.language = rs['language']
        lang = job.language or rs.get('language')
        # 对齐模型
        if lang not in align_cache:
            am, meta = self._get_align_model(lang, job.settings.device)
            align_cache[lang] = (am, meta)
        am, meta = align_cache[lang]
        aligned = whisperx.align(rs['segments'], am, meta, audio, job.settings.device)
        # 调整时间
        start_offset = seg['start_ms'] / 1000.0
        final = {'segments': []}
        if 'segments' in aligned:
            for s in aligned['segments']:
                if 'start' in s: s['start'] += start_offset
                if 'end' in s: s['end'] += start_offset
                final['segments'].append(s)
        if 'word_segments' in aligned:
            final['word_segments'] = []
            for w in aligned['word_segments']:
                if 'start' in w: w['start'] += start_offset
                if 'end' in w: w['end'] += start_offset
                final['word_segments'].append(w)
        del audio
        return final

    def _format_ts(self, sec: float) -> str:
        if sec < 0: sec = 0
        ms = int(round(sec * 1000))
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def _generate_srt(self, results: List[Dict], path: str, word_level: bool):
        lines = []
        n = 1
        for r in results:
            if not r: continue
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
                if e['end'] <= e['start']: continue
                lines.append(str(n))
                lines.append(f"{self._format_ts(e['start'])} --> {self._format_ts(e['end'])}")
                lines.append(e['text'])
                lines.append("")
                n += 1
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

# 单例处理器
processor: Optional[TranscriptionProcessor] = None

def get_processor(root: str) -> TranscriptionProcessor:
    global processor
    if processor is None:
        processor = TranscriptionProcessor(root)
    return processor
