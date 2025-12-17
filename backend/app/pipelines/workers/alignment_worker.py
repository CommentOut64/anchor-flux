"""
AlignmentWorker - 对齐层 Worker（CPU）

职责：
1. 执行双流对齐（三级降级策略）
2. 分句 + 语义分组（使用慢流策略）
3. 推送定稿到 SSE（Chunk 级别批量替换）
4. 填充 ProcessingContext.final_sentences

特点：
- 质量优先（~0.5秒/Chunk）
- 三级降级：双模态对齐 → Whisper 伪对齐 → SenseVoice 草稿
- 分句策略：依赖 Whisper 的精准标点
- 语义分组：依赖语义完整性（续接词、从句判断）
"""
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.services.alignment.alignment_service import AlignmentService, AlignmentConfig
from app.services.sentence_splitter import SentenceSplitter, SplitConfig
from app.services.semantic_grouper import SemanticGrouper, GroupConfig
from app.services.streaming_subtitle import StreamingSubtitleManager, get_streaming_subtitle_manager
from app.services.pseudo_alignment import PseudoAlignment
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp


class AlignmentLevel(Enum):
    """对齐级别（三级降级策略）"""
    DUAL_MODAL = "dual_modal"           # Level 1: 双模态对齐（黄金标准）
    WHISPER_PSEUDO = "whisper_pseudo"   # Level 2: Whisper 伪对齐（银标准）
    SENSEVOICE_ONLY = "sensevoice_only" # Level 3: SenseVoice 草稿（铜标准）


class AlignmentWorker:
    """
    AlignmentWorker - 对齐层 Worker（CPU）

    在三级流水线中负责：
    1. 双流对齐（三级降级）
    2. 慢流分句（依赖标点）
    3. 推送定稿（Chunk 级别）
    """

    def __init__(
        self,
        job_id: str,
        final_split_config: Optional[SplitConfig] = None,
        final_group_config: Optional[GroupConfig] = None,
        alignment_config: Optional[AlignmentConfig] = None,
        enable_semantic_grouping: bool = True,
        alignment_score_threshold: float = 0.3,
        enable_fallback: bool = True,
        alignment_service: Optional[AlignmentService] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 AlignmentWorker

        Args:
            job_id: 任务 ID
            final_split_config: 慢流分句配置
            final_group_config: 慢流语义分组配置
            alignment_config: 对齐服务配置
            enable_semantic_grouping: 是否启用语义分组
            alignment_score_threshold: 对齐质量阈值
            enable_fallback: 是否启用降级策略
            alignment_service: 对齐服务
            logger: 日志记录器
        """
        self.job_id = job_id
        self.enable_semantic_grouping = enable_semantic_grouping
        self.alignment_score_threshold = alignment_score_threshold
        self.enable_fallback = enable_fallback
        self.logger = logger or logging.getLogger(__name__)

        # 初始化对齐服务
        if alignment_config is None:
            alignment_config = AlignmentConfig()
        self.alignment_service = alignment_service or AlignmentService(
            config=alignment_config,
            logger=self.logger
        )

        # 初始化慢流分句器（依赖 Whisper 的精准标点）
        if final_split_config is None:
            final_split_config = SplitConfig(
                prefer_punctuation_break=True,   # 依赖标点
                use_dynamic_pause=True,
                pause_threshold=0.5,
                max_duration=5.0,                # 软上限（监控点）
                enable_hard_limit=True,          # 启用硬上限（异常保护，设置很宽松）
                hard_limit_duration=20.0,        # 20秒硬上限（远大于正常句子）
                delay_split_to_punctuation=True, # 延迟切分到标点
                delay_split_max_wait=15.0,       # 允许延迟15秒（总计20秒）
                merge_short_sentences=True
            )
        self.final_splitter = SentenceSplitter(final_split_config)

        # 初始化慢流语义分组器（依赖语义完整性）
        if final_group_config is None:
            final_group_config = GroupConfig(
                max_group_gap=2.0,
                max_group_duration=10.0,
                max_group_sentences=5,
                enable_overlap_detection=True
            )
        self.final_grouper = SemanticGrouper(final_group_config)

        # 获取流式字幕管理器
        self.subtitle_manager = get_streaming_subtitle_manager(job_id)

    async def process(self, ctx: ProcessingContext):
        """
        处理单个 Chunk（对齐层）

        流程：
        1. 双流对齐（三级降级）
        2. 分句 + 语义分组
        3. 推送定稿
        4. 填充 ctx.final_sentences

        Args:
            ctx: 处理上下文
        """
        chunk = ctx.audio_chunk

        # 阶段 1: 双流对齐（三级降级策略）
        self.logger.debug(f"Chunk {ctx.chunk_index}: 双流对齐")
        final_sentences, alignment_level = await self._align_and_fallback(
            ctx.whisper_result,
            ctx.sv_result,
            chunk
        )

        ctx.final_sentences = final_sentences

        # 阶段 2: 推送定稿（使用 Chunk 级别的批量替换）
        self.subtitle_manager.replace_chunk(ctx.chunk_index, final_sentences)

        self.logger.info(
            f"Chunk {ctx.chunk_index}: 定稿已推送 "
            f"({len(final_sentences)} 个句子, 对齐级别={alignment_level.value})"
        )

    async def _align_and_fallback(
        self,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
        chunk: AudioChunk
    ) -> tuple[List[SentenceSegment], AlignmentLevel]:
        """
        双流对齐 + 三级降级策略 + 分句

        Level 1: 双模态对齐（黄金标准）
        Level 2: Whisper 伪对齐（银标准）
        Level 3: SenseVoice 草稿（铜标准）

        Args:
            whisper_result: Whisper 推理结果
            sv_result: SenseVoice 推理结果
            chunk: AudioChunk

        Returns:
            tuple[List[SentenceSegment], AlignmentLevel]: 最终句子列表和对齐级别
        """
        # 获取 Whisper 检测到的语言，用于语义完整性检查
        detected_language = whisper_result.get('language', 'auto')
        self.final_splitter.config.language = detected_language
        self.logger.debug(f"使用 Whisper 检测到的语言: {detected_language}")

        # ========== 早期拦截：长度暴涨检测 ==========
        whisper_text = whisper_result.get('text', '').strip()
        sv_text_clean = sv_result.get('text_clean', '').strip()

        # 检测 Whisper 输出是否异常膨胀
        if sv_text_clean:  # 仅当 SenseVoice 有输出时检查
            len_w = len(whisper_text)
            len_sv = len(sv_text_clean)

            # 长度暴涨公式: len(whisper) > 3 * len(sensevoice) + 10
            if len_w > 3 * len_sv + 10:
                # 检查 Whisper 置信度
                whisper_confidence = whisper_result.get('confidence', 0.5)

                # 置信度阈值：0.5 对应 avg_logprob ≈ -0.5
                if whisper_confidence < 0.5:
                    self.logger.warning(
                        f"Whisper 长度暴涨且置信度低，直接降级到 Level 3: "
                        f"len(whisper)={len_w} > 3*{len_sv}+10, "
                        f"confidence={whisper_confidence:.2f} < 0.5, "
                        f"whisper='{whisper_text[:50]}...', sv='{sv_text_clean[:50]}...'"
                    )
                    # 直接降级到 Level 3（SenseVoice 草稿）
                    sentences = self._split_sentences_from_sv(sv_result, chunk)
                    for sentence in sentences:
                        sentence.is_finalized = True
                        sentence.is_draft = False
                    return sentences, AlignmentLevel.SENSEVOICE_ONLY
                else:
                    # 特权放行：置信度高，可能是 SenseVoice 漏识别
                    self.logger.info(
                        f"Whisper 长度暴涨但置信度高，特权放行: "
                        f"len(whisper)={len_w} > 3*{len_sv}+10, "
                        f"confidence={whisper_confidence:.2f} >= 0.5"
                    )

        try:
            # Level 1: 尝试双模态对齐
            sv_words_data = sv_result.get('words', [])

            if not whisper_text or not sv_words_data:
                raise ValueError("Whisper 或 SenseVoice 结果为空")

            # 转换 SenseVoice words 为 WordTimestamp 列表
            sv_tokens = [
                WordTimestamp(
                    word=w.get('word', ''),
                    start=w.get('start', 0.0),
                    end=w.get('end', 0.0),
                    confidence=w.get('confidence', 1.0)
                )
                for w in sv_words_data
            ]

            # 执行对齐
            aligned_subtitle = await self.alignment_service.align(
                whisper_text=whisper_text,
                sv_tokens=sv_tokens,
                vad_range=(0.0, chunk.duration),  # 使用相对时间
                chunk_offset=chunk.start,
                audio_array=chunk.audio,
                sample_rate=chunk.sample_rate
            )

            # 检查对齐质量
            if aligned_subtitle.alignment_score < self.alignment_score_threshold:
                raise ValueError(f"对齐质量过低: {aligned_subtitle.alignment_score:.2f}")

            # 转换对齐结果为 WordTimestamp 列表
            aligned_words = [
                WordTimestamp(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    confidence=w.final_confidence,
                    is_pseudo=w.is_pseudo
                )
                for w in aligned_subtitle.words
            ]

            # 重新分句（基于 Whisper 的标点，使用慢流分句器）
            sentences = self.final_splitter.split(aligned_words, whisper_text)

            # 语义分组（使用慢流语义分组器）
            if self.enable_semantic_grouping:
                sentences = self.final_grouper.group(sentences)

            # 设置状态
            for sentence in sentences:
                sentence.source = TextSource.WHISPER_PATCH
                sentence.is_finalized = True
                sentence.is_draft = False
                sentence.alignment_score = aligned_subtitle.alignment_score
                sentence.matched_ratio = aligned_subtitle.matched_ratio
                sentence.whisper_text = whisper_text

            self.logger.info(
                f"双模态对齐成功: alignment_score={aligned_subtitle.alignment_score:.2f}, "
                f"matched_ratio={aligned_subtitle.matched_ratio:.2f}, "
                f"分句数={len(sentences)}"
            )

            return sentences, AlignmentLevel.DUAL_MODAL

        except Exception as e:
            self.logger.warning(f"双模态对齐失败: {e}, 降级到 Whisper 伪对齐")

            if not self.enable_fallback:
                raise

            try:
                # Level 2: Whisper 伪对齐
                whisper_text = whisper_result.get('text', '').strip()

                if not whisper_text:
                    raise ValueError("Whisper 结果为空")

                # 使用伪对齐生成字级时间戳
                words = PseudoAlignment.apply(
                    original_start=0.0,
                    original_end=chunk.duration,
                    new_text=whisper_text
                )

                # 调整时间偏移
                for word in words:
                    word.start += chunk.start
                    word.end += chunk.start

                # 分句（使用慢流分句器）
                sentences = self.final_splitter.split(words, whisper_text)

                # 语义分组（使用慢流语义分组器）
                if self.enable_semantic_grouping:
                    sentences = self.final_grouper.group(sentences)

                # 设置状态
                for sentence in sentences:
                    sentence.source = TextSource.WHISPER_PATCH
                    sentence.is_finalized = True
                    sentence.is_draft = False
                    sentence.alignment_score = 0.5  # 伪对齐的默认分数
                    sentence.whisper_text = whisper_text

                self.logger.info(f"Whisper 伪对齐成功, 分句数={len(sentences)}")

                return sentences, AlignmentLevel.WHISPER_PSEUDO

            except Exception as e2:
                self.logger.error(f"Whisper 伪对齐失败: {e2}, 降级到 SenseVoice 草稿")

                # Level 3: SenseVoice 草稿
                sentences = self._split_sentences_from_sv(sv_result, chunk)

                # 设置为定稿状态（虽然是草稿，但作为最终结果）
                for sentence in sentences:
                    sentence.is_finalized = True
                    sentence.is_draft = False

                self.logger.info(f"使用 SenseVoice 草稿作为最终结果, 分句数={len(sentences)}")

                return sentences, AlignmentLevel.SENSEVOICE_ONLY

    def _split_sentences_from_sv(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk
    ) -> List[SentenceSegment]:
        """
        从 SenseVoice 结果分句（降级方案）

        Args:
            sv_result: SenseVoice 推理结果
            chunk: AudioChunk

        Returns:
            List[SentenceSegment]: 分句后的句子列表
        """
        text_clean = sv_result.get('text_clean', '')
        words_data = sv_result.get('words', [])

        # 转换 words 为 WordTimestamp 列表
        words = [
            WordTimestamp(
                word=w.get('word', ''),
                start=w.get('start', 0.0),
                end=w.get('end', 0.0),
                confidence=w.get('confidence', 1.0)
            )
            for w in words_data
        ]

        if not words:
            self.logger.warning("SenseVoice 结果没有字级时间戳，无法分句")
            return []

        # 使用慢流分句器（但是基于 SenseVoice 的数据）
        sentences = self.final_splitter.split(words, text_clean)

        # 语义分组
        if self.enable_semantic_grouping:
            sentences = self.final_grouper.group(sentences)

        # 调整时间偏移和设置状态
        for sentence in sentences:
            sentence.start += chunk.start
            sentence.end += chunk.start
            sentence.source = TextSource.SENSEVOICE

            # 调整 words 的时间偏移
            for word in sentence.words:
                word.start += chunk.start
                word.end += chunk.start

        return sentences
