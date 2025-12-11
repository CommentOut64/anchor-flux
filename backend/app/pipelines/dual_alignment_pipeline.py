"""
DualAlignmentPipeline - 双流对齐流水线

Phase 4 实现 - 2025-12-10（修订版：集成分句功能）

核心流程：
1. 快流（SenseVoice）：秒级推理 → 分句（Layer 1 + Layer 2）→ 推送草稿
2. 慢流（Whisper）：深度推理 → 对齐 → 分句（Layer 1 + Layer 2）→ 推送定稿
3. 双流对齐：Needleman-Wunsch 序列对齐
4. 三级降级：双模态对齐 → Whisper伪对齐 → SenseVoice草稿

设计决策（基于用户裁决）：
- 数据模型：统一使用 SentenceSegment（废弃 AlignedSubtitle）
- SSE 推送：复用 StreamingSubtitleManager
- Prompt 构建：上一个 Chunk 的完整定稿文本（最后200字符）
- 并发策略：串行处理 Chunk，单消费者队列模型
- 错误处理：三级降级策略
- 分句策略：推送前必须完成 Layer 1 + Layer 2 分句
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from app.services.audio.chunk_engine import AudioChunk
from app.services.inference.sensevoice_executor import SenseVoiceExecutor
from app.services.inference.whisper_executor import WhisperExecutor
from app.services.alignment.alignment_service import AlignmentService, AlignmentConfig
from app.services.alignment.keyword_extractor import KeywordExtractor, KeywordExtractionConfig
from app.services.streaming_subtitle import StreamingSubtitleManager, get_streaming_subtitle_manager
from app.services.pseudo_alignment import PseudoAlignment
from app.services.sentence_splitter import SentenceSplitter, SplitConfig
from app.services.semantic_grouper import SemanticGrouper, GroupConfig
from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.models.confidence_models import AlignedSubtitle, AlignedWord, AlignmentStatus
from app.utils.prompt_builder import get_prompt_builder


class AlignmentLevel(Enum):
    """对齐级别（三级降级策略）"""
    DUAL_MODAL = "dual_modal"           # Level 1: 双模态对齐（黄金标准）
    WHISPER_PSEUDO = "whisper_pseudo"   # Level 2: Whisper 伪对齐（银标准）
    SENSEVOICE_ONLY = "sensevoice_only" # Level 3: SenseVoice 草稿（铜标准）


@dataclass
class DualAlignmentConfig:
    """双流对齐流水线配置"""
    # 模型配置
    sensevoice_language: str = "auto"
    whisper_language: str = "auto"

    # Prompt 配置
    prompt_max_length: int = 200        # Prompt 最大长度（字符）
    use_keyword_extraction: bool = True  # 是否使用关键词提取
    user_glossary: Optional[List[str]] = None  # 用户词表

    # 对齐配置
    alignment_config: Optional[AlignmentConfig] = None
    keyword_config: Optional[KeywordExtractionConfig] = None

    # 快流分句配置（SenseVoice Draft）
    draft_split_config: Optional[SplitConfig] = None
    draft_group_config: Optional[GroupConfig] = None

    # 慢流分句配置（Whisper Final）
    final_split_config: Optional[SplitConfig] = None
    final_group_config: Optional[GroupConfig] = None

    enable_semantic_grouping: bool = True  # 是否启用语义分组

    # 降级策略配置
    alignment_score_threshold: float = 0.3  # 对齐质量阈值（低于此值降级）
    enable_fallback: bool = True            # 是否启用降级策略

    def __post_init__(self):
        """初始化后处理"""
        if self.alignment_config is None:
            self.alignment_config = AlignmentConfig()
        if self.keyword_config is None:
            self.keyword_config = KeywordExtractionConfig()

        # 快流配置：主要依赖 VAD 停顿，不依赖标点
        if self.draft_split_config is None:
            self.draft_split_config = SplitConfig(
                prefer_punctuation_break=False,  # 不依赖标点
                use_dynamic_pause=True,          # 使用动态停顿
                pause_threshold=0.5,             # 停顿阈值
                max_duration=5.0,                # 最大时长
                merge_short_sentences=True       # 合并短句
                # max_chars 默认为 0，不启用字符数限制
            )

        # 快流语义分组：主要依赖物理约束（时间间隔、句子长度）
        if self.draft_group_config is None:
            self.draft_group_config = GroupConfig(
                max_group_gap=2.0,               # 组内最大间隔
                max_group_duration=10.0,         # 单组最大时长
                max_group_sentences=5,           # 单组最大句子数
                enable_overlap_detection=True    # 启用重叠检测
            )

        # 慢流配置：依赖 Whisper 的精准标点
        if self.final_split_config is None:
            self.final_split_config = SplitConfig(
                prefer_punctuation_break=True,   # 依赖标点
                use_dynamic_pause=True,          # 使用动态停顿
                pause_threshold=0.5,             # 停顿阈值
                max_duration=5.0,                # 最大时长
                merge_short_sentences=True       # 合并短句
                # max_chars 默认为 0，不启用字符数限制
            )

        # 慢流语义分组：依赖语义完整性（续接词、从句判断）
        if self.final_group_config is None:
            self.final_group_config = GroupConfig(
                max_group_gap=2.0,               # 组内最大间隔
                max_group_duration=10.0,         # 单组最大时长
                max_group_sentences=5,           # 单组最大句子数
                enable_overlap_detection=True    # 启用重叠检测
            )


@dataclass
class ChunkProcessingResult:
    """Chunk 处理结果"""
    chunk_index: int
    sentences: List[SentenceSegment]
    alignment_level: AlignmentLevel
    processing_time: float
    error: Optional[str] = None


class DualAlignmentPipeline:
    """
    双流对齐流水线

    核心职责：
    1. 串行处理 AudioChunk 队列
    2. 快流（SenseVoice）推理 + 分句 + 立即推送草稿
    3. 慢流（Whisper）推理 + 智能 Prompt 构建
    4. 双流对齐 + 分句 + 三级降级策略
    5. SSE 事件推送（草稿和定稿）
    """

    def __init__(
        self,
        job_id: str,
        config: Optional[DualAlignmentConfig] = None,
        sensevoice_executor: Optional[SenseVoiceExecutor] = None,
        whisper_executor: Optional[WhisperExecutor] = None,
        alignment_service: Optional[AlignmentService] = None,
        keyword_extractor: Optional[KeywordExtractor] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化双流对齐流水线

        Args:
            job_id: 任务 ID
            config: 流水线配置
            sensevoice_executor: SenseVoice 执行器
            whisper_executor: Whisper 执行器
            alignment_service: 对齐服务
            keyword_extractor: 关键词提取器
            logger: 日志记录器
        """
        self.job_id = job_id
        self.config = config or DualAlignmentConfig()
        self.logger = logger or logging.getLogger(__name__)

        # 初始化执行器和服务
        self.sensevoice_executor = sensevoice_executor or SenseVoiceExecutor()
        self.whisper_executor = whisper_executor or WhisperExecutor()
        self.alignment_service = alignment_service or AlignmentService(
            config=self.config.alignment_config,
            logger=self.logger
        )
        self.keyword_extractor = keyword_extractor or KeywordExtractor(
            config=self.config.keyword_config,
            logger=self.logger
        )

        # 初始化快流分句器和语义分组器（SenseVoice Draft）
        self.draft_splitter = SentenceSplitter(self.config.draft_split_config)
        self.draft_grouper = SemanticGrouper(self.config.draft_group_config)

        # 初始化慢流分句器和语义分组器（Whisper Final）
        self.final_splitter = SentenceSplitter(self.config.final_split_config)
        self.final_grouper = SemanticGrouper(self.config.final_group_config)

        # 获取流式字幕管理器
        self.subtitle_manager = get_streaming_subtitle_manager(job_id)

        # Prompt 缓存（上一个 Chunk 的定稿文本）
        self.previous_whisper_text: Optional[str] = None

        # 统计信息
        self.total_chunks = 0
        self.processed_chunks = 0
        self.alignment_levels: Dict[AlignmentLevel, int] = {
            AlignmentLevel.DUAL_MODAL: 0,
            AlignmentLevel.WHISPER_PSEUDO: 0,
            AlignmentLevel.SENSEVOICE_ONLY: 0
        }

        # 错误统计
        self.error_count = 0
        self.error_details: List[Dict[str, Any]] = []

        # 显存使用统计
        self.vram_usage: Dict[str, Any] = {
            'initial_vram': 0,
            'peak_vram': 0,
            'final_vram': 0,
            'sensevoice_vram': 0,
            'whisper_vram': 0
        }

    async def run(
        self,
        chunks: List[AudioChunk],
        progress_callback: Optional[callable] = None
    ) -> List[ChunkProcessingResult]:
        """
        运行双流对齐流水线

        串行处理所有 Chunk，每个 Chunk 执行：
        1. SenseVoice 快流推理 → 分句 → 推送草稿
        2. Whisper 慢流推理 → 对齐 → 分句 → 推送定稿

        Args:
            chunks: AudioChunk 列表
            progress_callback: 进度回调 callback(progress: float, message: str)

        Returns:
            List[ChunkProcessingResult]: 处理结果列表
        """
        self.total_chunks = len(chunks)
        self.logger.info(f"开始双流对齐流水线: {self.total_chunks} 个 Chunk")

        # 记录初始显存使用
        self._record_vram_usage('initial')

        # 加载模型（如果尚未加载）
        try:
            if not self.sensevoice_executor.is_loaded():
                self.logger.info("加载 SenseVoice 模型...")
                self.sensevoice_executor.service.load_model()
                self.logger.info("SenseVoice 模型加载完成")
        except Exception as e:
            self.logger.error(f"SenseVoice 模型加载失败: {e}", exc_info=True)
            raise RuntimeError(f"SenseVoice 模型加载失败，无法继续处理: {e}")

        try:
            if not self.whisper_executor.is_loaded():
                self.logger.info("加载 Whisper 模型...")
                self.whisper_executor.service.load_model()
                self.logger.info("Whisper 模型加载完成")
        except Exception as e:
            self.logger.error(f"Whisper 模型加载失败: {e}", exc_info=True)
            if not self.config.enable_fallback:
                raise RuntimeError(f"Whisper 模型加载失败，无法继续处理: {e}")
            else:
                self.logger.warning("Whisper 模型加载失败，将使用 SenseVoice 草稿作为最终结果")

        results = []

        for i, chunk in enumerate(chunks):
            self.logger.info(f"处理 Chunk {i+1}/{self.total_chunks}")

            # 处理单个 Chunk
            result = await self._process_chunk(chunk, i)
            results.append(result)

            self.processed_chunks += 1

            # 进度回调
            if progress_callback:
                progress = (i + 1) / self.total_chunks
                progress_callback(progress, f"处理 Chunk {i+1}/{self.total_chunks}")

        self.logger.info(f"双流对齐流水线完成: {self.processed_chunks}/{self.total_chunks} 个 Chunk")

        # 记录最终显存使用
        self._record_vram_usage('final')

        self._log_statistics()

        return results

    async def _process_chunk(
        self,
        chunk: AudioChunk,
        chunk_index: int
    ) -> ChunkProcessingResult:
        """
        处理单个 Chunk

        流程：
        1. SenseVoice 快流推理 → 分句 → 推送草稿
        2. Whisper 慢流推理（使用智能 Prompt）
        3. 双流对齐（三级降级策略）→ 分句 → 推送定稿

        Args:
            chunk: AudioChunk
            chunk_index: Chunk 索引

        Returns:
            ChunkProcessingResult: 处理结果
        """
        import time
        start_time = time.time()

        try:
            # 阶段 1: SenseVoice 快流推理 + 分句
            self.logger.debug(f"Chunk {chunk_index}: SenseVoice 快流推理")
            sv_result = await self._run_sensevoice(chunk)
            self._record_vram_usage('peak')  # 记录 SenseVoice 后的显存

            # 分句（Layer 1 + Layer 2）
            draft_sentences = self._split_sentences(
                sv_result,
                chunk,
                is_draft=True
            )

            # 立即推送草稿（使用 Chunk 级别的批量推送）
            self.subtitle_manager.add_draft_sentences(chunk_index, draft_sentences)

            self.logger.info(f"Chunk {chunk_index}: 草稿已推送 ({len(draft_sentences)} 个句子)")

            # 阶段 2: Whisper 慢流推理
            self.logger.debug(f"Chunk {chunk_index}: Whisper 慢流推理")
            whisper_prompt = self._build_whisper_prompt(draft_sentences, chunk_index)
            whisper_result = await self._run_whisper(chunk, whisper_prompt)
            self._record_vram_usage('peak')  # 记录 Whisper 后的显存

            # 阶段 3: 双流对齐（三级降级策略）+ 分句
            self.logger.debug(f"Chunk {chunk_index}: 双流对齐")
            final_sentences, alignment_level = await self._align_and_fallback(
                whisper_result,
                sv_result,
                chunk
            )

            # 更新 Prompt 缓存（使用定稿文本）
            self._update_prompt_cache(final_sentences)

            # 阶段 4: 推送定稿（使用 Chunk 级别的批量替换）
            self.subtitle_manager.replace_chunk(chunk_index, final_sentences)

            self.logger.info(
                f"Chunk {chunk_index}: 定稿已推送 ({len(final_sentences)} 个句子, "
                f"对齐级别={alignment_level.value})"
            )

            # 记录统计
            self.alignment_levels[alignment_level] += 1

            processing_time = time.time() - start_time

            return ChunkProcessingResult(
                chunk_index=chunk_index,
                sentences=final_sentences,
                alignment_level=alignment_level,
                processing_time=processing_time
            )

        except Exception as e:
            self.logger.error(f"Chunk {chunk_index} 处理失败: {e}", exc_info=True)
            processing_time = time.time() - start_time

            # 记录错误详情
            self.error_count += 1
            error_detail = {
                "chunk_index": chunk_index,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "processing_time": processing_time
            }
            self.error_details.append(error_detail)

            # 尝试使用 SenseVoice 草稿作为降级方案
            try:
                self.logger.warning(f"Chunk {chunk_index}: 尝试使用 SenseVoice 草稿作为降级方案")
                sv_result = await self._run_sensevoice(chunk)
                draft_sentences = self._split_sentences(sv_result, chunk, is_draft=False)

                # 设置为定稿状态（作为最终结果）
                for sentence in draft_sentences:
                    sentence.is_finalized = True
                    sentence.is_draft = False

                # 推送定稿
                self.subtitle_manager.replace_chunk(chunk_index, draft_sentences)

                self.logger.info(f"Chunk {chunk_index}: 降级方案成功，使用 SenseVoice 草稿")

                return ChunkProcessingResult(
                    chunk_index=chunk_index,
                    sentences=draft_sentences,
                    alignment_level=AlignmentLevel.SENSEVOICE_ONLY,
                    processing_time=processing_time,
                    error=f"降级到 SenseVoice: {str(e)}"
                )

            except Exception as e2:
                self.logger.error(f"Chunk {chunk_index}: 降级方案也失败: {e2}", exc_info=True)

                return ChunkProcessingResult(
                    chunk_index=chunk_index,
                    sentences=[],
                    alignment_level=AlignmentLevel.SENSEVOICE_ONLY,
                    processing_time=processing_time,
                    error=f"完全失败: {str(e)}, 降级失败: {str(e2)}"
                )

    async def _run_sensevoice(self, chunk: AudioChunk) -> Dict[str, Any]:
        """
        运行 SenseVoice 推理

        Args:
            chunk: AudioChunk

        Returns:
            Dict: SenseVoice 推理结果
        """
        result = await self.sensevoice_executor.execute(
            audio_array=chunk.audio,
            sample_rate=chunk.sample_rate,
            language=self.config.sensevoice_language,
            use_itn=True
        )
        return result

    async def _run_whisper(
        self,
        chunk: AudioChunk,
        initial_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行 Whisper 推理

        Args:
            chunk: AudioChunk
            initial_prompt: 初始提示词

        Returns:
            Dict: Whisper 推理结果
        """
        result = await self.whisper_executor.execute(
            audio=chunk.audio,
            start_time=chunk.start,
            end_time=chunk.end,
            language=self.config.whisper_language,
            initial_prompt=initial_prompt
        )
        return result

    def _split_sentences(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
        is_draft: bool = True
    ) -> List[SentenceSegment]:
        """
        对 SenseVoice 结果进行分句

        流程：
        1. Layer 1: SentenceSplitter 分句（快流/慢流使用不同策略）
        2. Layer 2: SemanticGrouper 语义分组（如果启用）
        3. 调整时间偏移

        快流策略：
        - Layer 1: 主要依赖 VAD 停顿（不依赖标点）
        - Layer 2: 依赖物理约束（时间间隔、句子长度）

        慢流策略：
        - Layer 1: 依赖 Whisper 的精准标点
        - Layer 2: 依赖语义完整性（续接词、从句判断）

        Args:
            sv_result: SenseVoice 推理结果
            chunk: AudioChunk
            is_draft: 是否为草稿

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

        # Layer 1: 分句（根据是否为草稿选择不同的分句器）
        if is_draft:
            # 快流：使用 draft_splitter（主要依赖 VAD 停顿）
            sentences = self.draft_splitter.split(words, text_clean)
            self.logger.debug(f"快流分句: {len(sentences)} 个句子（依赖 VAD 停顿）")
        else:
            # 慢流：使用 final_splitter（依赖标点）
            sentences = self.final_splitter.split(words, text_clean)
            self.logger.debug(f"慢流分句: {len(sentences)} 个句子（依赖标点）")

        # Layer 2: 语义分组（如果启用）
        if self.config.enable_semantic_grouping:
            if is_draft:
                # 快流：使用 draft_grouper（主要依赖物理约束）
                sentences = self.draft_grouper.group(sentences)
                self.logger.debug(f"快流语义分组: {len(sentences)} 个句子（物理约束）")
            else:
                # 慢流：使用 final_grouper（依赖语义完整性）
                sentences = self.final_grouper.group(sentences)
                self.logger.debug(f"慢流语义分组: {len(sentences)} 个句子（语义完整性）")

        # 调整时间偏移和设置状态
        for sentence in sentences:
            sentence.start += chunk.start
            sentence.end += chunk.start
            sentence.is_draft = is_draft
            sentence.is_finalized = not is_draft
            sentence.source = TextSource.SENSEVOICE

            # 调整 words 的时间偏移
            for word in sentence.words:
                word.start += chunk.start
                word.end += chunk.start

        self.logger.debug(f"分句完成: {len(sentences)} 个句子")

        return sentences

    def _build_whisper_prompt(
        self,
        draft_sentences: List[SentenceSegment],
        chunk_index: int
    ) -> str:
        """
        构建 Whisper Prompt（方案 B：关键词提取策略）

        新策略：
        1. 从上一个 Chunk 的定稿文本中提取关键词（专有名词）
        2. 添加用户词表
        3. 构建 "Glossary: word1, word2, ..." 格式的 prompt
        4. 避免使用完整句子，防止 Whisper 误判

        Args:
            draft_sentences: 当前 Chunk 的草稿句子列表
            chunk_index: Chunk 索引

        Returns:
            str: Whisper Prompt
        """
        # 获取 Prompt 构建器
        prompt_builder = get_prompt_builder()

        # 构建安全的 Prompt
        prompt = prompt_builder.build_prompt(
            previous_text=self.previous_whisper_text,
            user_glossary=self.config.user_glossary
        )

        if prompt:
            self.logger.info(f"Chunk {chunk_index}: 构建 Prompt = '{prompt}'")
        else:
            self.logger.debug(f"Chunk {chunk_index}: 无 Prompt（首个 Chunk 或无关键词）")

        return prompt

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

        try:
            # Level 1: 尝试双模态对齐
            whisper_text = whisper_result.get('text', '').strip()
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
            if aligned_subtitle.alignment_score < self.config.alignment_score_threshold:
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
            if self.config.enable_semantic_grouping:
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

            if not self.config.enable_fallback:
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
                if self.config.enable_semantic_grouping:
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
                sentences = self._split_sentences(sv_result, chunk, is_draft=False)

                # 设置为定稿状态（虽然是草稿，但作为最终结果）
                for sentence in sentences:
                    sentence.is_finalized = True
                    sentence.is_draft = False

                self.logger.info(f"使用 SenseVoice 草稿作为最终结果, 分句数={len(sentences)}")

                return sentences, AlignmentLevel.SENSEVOICE_ONLY

    def _update_prompt_cache(self, sentences: List[SentenceSegment]):
        """
        更新 Prompt 缓存

        将当前 Chunk 的定稿文本保存为下一个 Chunk 的 Prompt。

        Args:
            sentences: 句子列表
        """
        if not sentences:
            return

        # 拼接所有句子的文本
        texts = [s.text_clean or s.text for s in sentences]
        self.previous_whisper_text = ' '.join(texts)

        self.logger.debug(f"更新 Prompt 缓存: {len(self.previous_whisper_text)} 个字符")

    def _record_vram_usage(self, stage: str):
        """
        记录显存使用情况

        Args:
            stage: 阶段标识（initial/peak/final）
        """
        try:
            import torch
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                vram_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB

                if stage == 'initial':
                    self.vram_usage['initial_vram'] = vram_used
                elif stage == 'final':
                    self.vram_usage['final_vram'] = vram_used

                # 更新峰值
                if vram_used > self.vram_usage['peak_vram']:
                    self.vram_usage['peak_vram'] = vram_used

                self.logger.debug(
                    f"显存使用 ({stage}): {vram_used:.1f}MB (已分配), "
                    f"{vram_reserved:.1f}MB (已保留)"
                )
        except Exception as e:
            self.logger.debug(f"无法获取显存信息: {e}")

    def _log_statistics(self):
        """记录统计信息"""
        self.logger.info("=" * 60)
        self.logger.info("双流对齐流水线统计:")
        self.logger.info(f"  总 Chunk 数: {self.total_chunks}")
        self.logger.info(f"  已处理: {self.processed_chunks}")
        self.logger.info(f"  成功: {self.processed_chunks - self.error_count}")
        self.logger.info(f"  错误: {self.error_count}")

        if self.processed_chunks > 0:
            self.logger.info(f"  对齐级别分布:")
            for level, count in self.alignment_levels.items():
                percentage = (count / self.processed_chunks * 100)
                self.logger.info(f"    {level.value}: {count} ({percentage:.1f}%)")

        if self.error_details:
            self.logger.info(f"  错误详情:")
            for error in self.error_details:
                self.logger.info(
                    f"    Chunk {error['chunk_index']}: "
                    f"{error['error_type']} - {error['error_message']}"
                )

        # 显存使用统计
        if self.vram_usage['peak_vram'] > 0:
            self.logger.info(f"  显存使用:")
            self.logger.info(f"    初始: {self.vram_usage['initial_vram']:.1f}MB")
            self.logger.info(f"    峰值: {self.vram_usage['peak_vram']:.1f}MB")
            self.logger.info(f"    最终: {self.vram_usage['final_vram']:.1f}MB")

        self.logger.info("=" * 60)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "success_chunks": self.processed_chunks - self.error_count,
            "error_count": self.error_count,
            "alignment_levels": {
                level.value: count
                for level, count in self.alignment_levels.items()
            },
            "error_details": self.error_details,
            "vram_usage": self.vram_usage
        }


# 便捷函数
def get_dual_alignment_pipeline(
    job_id: str,
    config: Optional[DualAlignmentConfig] = None,
    logger: Optional[logging.Logger] = None
) -> DualAlignmentPipeline:
    """
    获取双流对齐流水线实例

    Args:
        job_id: 任务 ID
        config: 流水线配置
        logger: 日志记录器

    Returns:
        DualAlignmentPipeline 实例
    """
    return DualAlignmentPipeline(
        job_id=job_id,
        config=config,
        logger=logger
    )
