"""
频谱指纹分诊台

对每个VAD Chunk进行频谱分析，决定是否需要人声分离。
在VAD切分之后、转录之前执行。

V2 更新 (2025-12-21):
- 新增 YAMNet 探针模式语义级分类器，替代基于规则的频谱分析
- 优先使用 YAMNet，回退到规则方法
- 解决人声被误判为音乐的问题 (谐波比阈值过低)
"""
import numpy as np
import logging
from typing import List, Tuple, Optional

from app.models.circuit_breaker_models import (
    SpectrumFeatures, SpectrumDiagnosis, DiagnosisResult
)
from app.core.spectrum_thresholds import SpectrumThresholds, DEFAULT_SPECTRUM_THRESHOLDS

logger = logging.getLogger(__name__)


# ========== YAMNet 集成 ==========

def _get_yamnet_classifier():
    """懒加载 YAMNet 分类器（避免循环导入）"""
    try:
        from app.services.yamnet_classifier import get_yamnet_classifier
        return get_yamnet_classifier()
    except Exception as e:
        logger.warning(f"YAMNet 分类器加载失败: {e}")
        return None


class AudioSpectrumClassifier:
    """
    音频频谱分诊器

    支持两种模式：
    1. YAMNet 探针模式（默认）：使用预训练模型进行语义级分类，准确区分人声和音乐
    2. 规则模式（回退）：基于频谱特征的规则判断
    """

    def __init__(
        self,
        thresholds: SpectrumThresholds = None,
        use_yamnet: bool = True
    ):
        """
        初始化分诊器

        Args:
            thresholds: 频谱阈值配置
            use_yamnet: 是否使用 YAMNet 语义分类器（默认 True）
        """
        self.thresholds = thresholds or DEFAULT_SPECTRUM_THRESHOLDS
        self._librosa = None  # 懒加载
        self._use_yamnet = use_yamnet
        self._yamnet = None  # 懒加载

    def _get_yamnet(self):
        """获取 YAMNet 分类器实例"""
        if self._yamnet is None and self._use_yamnet:
            self._yamnet = _get_yamnet_classifier()
        return self._yamnet

    def _ensure_librosa(self):
        """确保 librosa 已加载"""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa

    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> SpectrumFeatures:
        """
        提取频谱特征

        Args:
            audio: 音频数组 (单声道)
            sr: 采样率

        Returns:
            SpectrumFeatures: 提取的特征
        """
        librosa = self._ensure_librosa()

        features = SpectrumFeatures()

        # 确保音频有效
        if len(audio) < sr * 0.1:  # 至少0.1秒
            return features

        try:
            # 1. 过零率 (ZCR)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.zcr = float(np.mean(zcr))
            features.zcr_variance = float(np.var(zcr))

            # 2. 频谱特征
            # 短时傅里叶变换
            stft = np.abs(librosa.stft(audio))

            # 谱质心
            cent = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
            features.spectral_centroid = float(np.mean(cent))

            # 谱带宽
            bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
            features.spectral_bandwidth = float(np.mean(bandwidth))

            # 频谱平坦度
            flatness = librosa.feature.spectral_flatness(S=stft)[0]
            features.spectral_flatness = float(np.mean(flatness))

            # 频谱滚降点 (85%能量点)
            rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
            features.spectral_rolloff = float(np.mean(rolloff))

            # 3. 谐波比 (简化计算)
            harmonic, percussive = librosa.effects.hpss(audio)
            h_energy = np.sum(harmonic ** 2)
            total_energy = np.sum(audio ** 2)
            features.harmonic_ratio = float(h_energy / (total_energy + 1e-10))

            # 4. 能量特征
            rms = librosa.feature.rms(y=audio)[0]
            features.rms_energy = float(np.mean(rms))
            features.energy_variance = float(np.var(rms))

            # 高频能量占比 (4kHz以上)
            freq_bins = librosa.fft_frequencies(sr=sr)
            high_freq_idx = freq_bins >= 4000
            if np.any(high_freq_idx):
                high_freq_energy = np.sum(stft[high_freq_idx, :] ** 2)
                total_spectral_energy = np.sum(stft ** 2)
                features.high_freq_ratio = float(high_freq_energy / (total_spectral_energy + 1e-10))

            # 5. 节奏特征
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            features.onset_strength = float(np.mean(onset_env))

            # 估计BPM
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            features.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])

        except Exception as e:
            logger.warning(f"特征提取失败: {e}")

        return features

    def diagnose_chunk(
        self,
        audio: np.ndarray,
        chunk_index: int,
        sr: int = 16000
    ) -> SpectrumDiagnosis:
        """
        对单个Chunk进行频谱分诊

        优先使用 YAMNet 语义分类器，回退到规则方法。

        Args:
            audio: 音频数组
            chunk_index: Chunk索引
            sr: 采样率

        Returns:
            SpectrumDiagnosis: 分诊结果
        """
        duration_sec = len(audio) / sr

        # 极短片段（< 0.5秒）直接返回 CLEAN，样本量不足无法可靠分析
        if duration_sec < 0.5:
            logger.debug(f"Chunk {chunk_index}: 极短片段({duration_sec:.2f}s)，跳过分诊")
            return SpectrumDiagnosis(
                chunk_index=chunk_index,
                diagnosis=DiagnosisResult.CLEAN,
                need_separation=False,
                music_score=0.0,
                noise_score=0.0,
                clean_score=1.0,
                recommended_model=None,
                features=SpectrumFeatures(),
                reason=f"极短片段({duration_sec:.2f}s)，跳过分诊"
            )

        # 尝试使用 YAMNet 语义分类器
        yamnet = self._get_yamnet()
        if yamnet is not None and yamnet.is_available():
            return self._diagnose_with_yamnet(audio, chunk_index, sr, yamnet)

        # 回退到规则方法
        return self._diagnose_with_rules(audio, chunk_index, sr)

    def _diagnose_with_yamnet(
        self,
        audio: np.ndarray,
        chunk_index: int,
        sr: int,
        yamnet
    ) -> SpectrumDiagnosis:
        """
        使用 YAMNet 进行语义级分诊

        Args:
            audio: 音频数组
            chunk_index: Chunk索引
            sr: 采样率
            yamnet: YAMNet 分类器实例

        Returns:
            SpectrumDiagnosis: 分诊结果
        """
        th = self.thresholds

        # YAMNet 分类
        result = yamnet.classify_chunk(audio, chunk_id=chunk_index)

        # 转换为 SpectrumDiagnosis 格式
        if result.is_music:
            diagnosis = DiagnosisResult.MUSIC
            need_separation = True
            # 统一使用 htdemucs（shift=1 模式）
            recommended_model = "htdemucs"
            reason = f"[YAMNet] 检测到音乐 (score={result.music_score:.2f})"
        else:
            diagnosis = DiagnosisResult.CLEAN
            need_separation = False
            recommended_model = None
            reason = f"[YAMNet] {', '.join(result.tags)} (speech={result.speech_score:.2f})"

        # 日志记录
        logger.debug(
            f"Chunk {chunk_index} [YAMNet]: is_music={result.is_music}, "
            f"music={result.music_score:.3f}, speech={result.speech_score:.3f}, "
            f"tags={result.tags}"
        )

        return SpectrumDiagnosis(
            chunk_index=chunk_index,
            diagnosis=diagnosis,
            need_separation=need_separation,
            music_score=result.music_score,
            noise_score=0.0,  # YAMNet 不单独计算噪音分数
            clean_score=result.speech_score,  # 用 speech_score 作为 clean_score
            recommended_model=recommended_model,
            features=SpectrumFeatures(),  # YAMNet 模式不提取传统特征
            reason=reason
        )

    def _diagnose_with_rules(
        self,
        audio: np.ndarray,
        chunk_index: int,
        sr: int = 16000
    ) -> SpectrumDiagnosis:
        """
        使用规则方法进行频谱分诊（回退方法）

        Args:
            audio: 音频数组
            chunk_index: Chunk索引
            sr: 采样率

        Returns:
            SpectrumDiagnosis: 分诊结果
        """
        th = self.thresholds
        duration_sec = len(audio) / sr

        # 短 Chunk 保守策略：降低敏感度
        short_chunk_threshold = 2.0
        is_short_chunk = duration_sec < short_chunk_threshold

        features = self.extract_features(audio, sr)

        # 计算各项得分
        music_score = self._calculate_music_score(features)
        noise_score = self._calculate_noise_score(features)
        clean_score = 1.0 - max(music_score, noise_score)

        # 短 Chunk 敏感度调整：提高阈值 30%
        if is_short_chunk:
            effective_music_threshold = th.music_score_threshold * 1.3
            effective_noise_threshold = th.noise_score_threshold * 1.3
            effective_mixed_threshold = 0.2 * 1.3
            logger.debug(
                f"Chunk {chunk_index}: 短片段({duration_sec:.2f}s)，"
                f"提高阈值 music>{effective_music_threshold:.2f}, noise>{effective_noise_threshold:.2f}"
            )
        else:
            effective_music_threshold = th.music_score_threshold
            effective_noise_threshold = th.noise_score_threshold
            effective_mixed_threshold = 0.2

        # 综合判定
        diagnosis = DiagnosisResult.CLEAN
        need_separation = False
        recommended_model = None
        reason = "纯净人声"

        if is_short_chunk:
            reason = f"纯净人声 (短片段{duration_sec:.1f}s)"

        if music_score >= effective_music_threshold:
            diagnosis = DiagnosisResult.MUSIC
            need_separation = True
            # 统一使用 htdemucs（shift=1 模式）
            recommended_model = "htdemucs"
            reason = f"检测到音乐 (score={music_score:.2f})"

        elif noise_score >= effective_noise_threshold:
            diagnosis = DiagnosisResult.NOISE
            need_separation = True
            recommended_model = "htdemucs"
            reason = f"检测到噪音 (score={noise_score:.2f})"

        elif music_score > effective_mixed_threshold and noise_score > effective_mixed_threshold:
            diagnosis = DiagnosisResult.MIXED
            need_separation = True
            recommended_model = "htdemucs"
            reason = f"混合噪音 (music={music_score:.2f}, noise={noise_score:.2f})"

        return SpectrumDiagnosis(
            chunk_index=chunk_index,
            diagnosis=diagnosis,
            need_separation=need_separation,
            music_score=music_score,
            noise_score=noise_score,
            clean_score=clean_score,
            recommended_model=recommended_model,
            features=features,
            reason=reason
        )

    def _calculate_music_score(self, f: SpectrumFeatures) -> float:
        """计算音乐得分"""
        th = self.thresholds
        score = 0.0

        # 谐波比高 → 音乐
        if f.harmonic_ratio >= th.harmonic_ratio_music:
            score += 0.35
        elif f.harmonic_ratio >= th.harmonic_ratio_music * 0.7:
            score += 0.2

        # 谱质心在音乐范围内
        if th.spectral_centroid_music_low <= f.spectral_centroid <= th.spectral_centroid_music_high:
            score += 0.25

        # 能量有节奏性波动
        if f.energy_variance >= th.energy_variance_music:
            score += 0.2

        # 有明显节拍
        if f.onset_strength >= th.onset_strength_music:
            score += 0.2

        return min(score, 1.0)

    def _calculate_noise_score(self, f: SpectrumFeatures) -> float:
        """计算噪音得分"""
        th = self.thresholds
        score = 0.0

        # 过零率高 → 噪音
        if f.zcr >= th.zcr_noise_high:
            score += 0.3
            # ZCR方差小说明是稳态噪音（如白噪声）
            if f.zcr_variance <= th.zcr_variance_noise:
                score += 0.15

        # 高频能量占比高 → 噪音
        if f.high_freq_ratio >= th.high_freq_ratio_noise:
            score += 0.25

        # 频谱平坦 → 噪音
        if f.spectral_flatness >= th.spectral_flatness_noise:
            score += 0.2

        # 谐波比低 → 噪音
        if f.harmonic_ratio < 0.3:
            score += 0.1

        return min(score, 1.0)

    def diagnose_chunks(
        self,
        chunks: List[Tuple[np.ndarray, float, float]],
        sr: int = 16000
    ) -> List[SpectrumDiagnosis]:
        """
        批量分诊多个Chunk

        Args:
            chunks: [(audio_array, start_time, end_time), ...]
            sr: 采样率

        Returns:
            List[SpectrumDiagnosis]: 分诊结果列表
        """
        results = []
        for i, (audio, start, end) in enumerate(chunks):
            diag = self.diagnose_chunk(audio, i, sr)
            logger.debug(
                f"Chunk {i} [{start:.1f}s-{end:.1f}s]: "
                f"{diag.diagnosis.value}, need_sep={diag.need_separation}, "
                f"model={diag.recommended_model}"
            )
            results.append(diag)

        # 统计日志
        need_sep_count = sum(1 for d in results if d.need_separation)
        logger.info(
            f"频谱分诊完成: {len(results)} chunks, "
            f"{need_sep_count} 需要分离 ({need_sep_count/len(results)*100:.1f}%)"
        )

        return results

    def quick_global_diagnosis(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        sample_duration: float = 10.0
    ) -> Tuple[str, float]:
        """
        快速全局预判（替代旧的 detect_background_music_level）

        采用分位数采样策略，快速判断整体音频的 BGM 情况。
        这是一个轻量级预判，不需要运行 Demucs。

        采样策略：取音频时长的 15%、50%、85% 处各截取 sample_duration 秒

        Args:
            audio: 完整音频数组 (samples,) 单声道
            sr: 采样率（默认 16000）
            sample_duration: 每个采样点的时长（秒，默认 10）

        Returns:
            Tuple[str, float]: (建议级别 "none"/"light"/"heavy", 平均音乐得分)
        """
        duration_sec = len(audio) / sr

        # 音频太短，无法可靠预判
        if duration_sec < sample_duration * 2:
            logger.warning(f"音频太短({duration_sec:.1f}s)，跳过全局预判")
            return "unknown", 0.0

        # 分位数采样位置
        sample_positions = [0.15, 0.50, 0.85]
        music_scores = []
        noise_scores = []

        for pos in sample_positions:
            start_time = duration_sec * pos
            # 确保不超出边界
            if start_time + sample_duration > duration_sec:
                start_time = duration_sec - sample_duration
            if start_time < 0:
                start_time = 0

            # 截取采样片段
            start_sample = int(start_time * sr)
            end_sample = int((start_time + sample_duration) * sr)
            chunk = audio[start_sample:end_sample]

            # 提取特征并计算得分
            features = self.extract_features(chunk, sr)
            music_score = self._calculate_music_score(features)
            noise_score = self._calculate_noise_score(features)

            music_scores.append(music_score)
            noise_scores.append(noise_score)

            logger.debug(
                f"全局采样 {pos*100:.0f}% ({start_time:.1f}s): "
                f"music={music_score:.2f}, noise={noise_score:.2f}"
            )

        # 使用最大值判断（保守策略：只要有一处 BGM 很重，就视为 heavy）
        avg_music = sum(music_scores) / len(music_scores)
        max_music = max(music_scores)
        max_noise = max(noise_scores)

        logger.info(
            f"全局预判完成: music_scores={[f'{s:.2f}' for s in music_scores]}, "
            f"avg={avg_music:.2f}, max={max_music:.2f}"
        )

        # 决策逻辑
        th = self.thresholds
        if max_music >= th.heavy_bgm_threshold:  # 默认 0.6
            return "heavy", avg_music
        elif max_music >= th.light_bgm_threshold:  # 默认 0.35
            return "light", avg_music
        elif max_noise >= th.noise_score_threshold:  # 默认 0.45
            return "light", avg_music  # 有噪音也建议轻度处理
        else:
            return "none", avg_music


# ========== 单例访问 ==========

_classifier_instance = None


def get_spectrum_classifier() -> AudioSpectrumClassifier:
    """获取频谱分诊器单例"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = AudioSpectrumClassifier()
    return _classifier_instance
