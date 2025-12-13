"""
SlowWorker - 慢流推理 Worker（GPU）

职责：
1. 基于 SenseVoice 结果构建安全的 Whisper Prompt
2. 执行 Whisper CUDA 推理（GPU）
3. 检测并防御 Whisper 幻觉（下划线、重复 prompt）
4. 填充 ProcessingContext.whisper_result

特点：
- 精度优先（~2秒/Chunk）
- Prompt 策略：关键词提取，避免完整句子
- 幻觉防御：内置检测逻辑，自动回退到 SenseVoice
"""
import logging
from typing import Dict, Optional, Any

from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.services.inference.whisper_executor import WhisperExecutor
from app.utils.prompt_builder import get_prompt_builder


class SlowWorker:
    """
    SlowWorker - 慢流推理 Worker（GPU）

    在三级流水线中负责：
    1. 构建安全的 Whisper Prompt
    2. Whisper 推理（GPU CUDA）
    3. 幻觉检测与防御
    """

    def __init__(
        self,
        whisper_language: str = "auto",
        user_glossary: Optional[list] = None,
        whisper_executor: Optional[WhisperExecutor] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 SlowWorker

        Args:
            whisper_language: Whisper 语言设置
            user_glossary: 用户词表
            whisper_executor: Whisper 执行器
            logger: 日志记录器
        """
        self.whisper_language = whisper_language
        self.user_glossary = user_glossary
        self.logger = logger or logging.getLogger(__name__)

        # 初始化执行器
        self.whisper_executor = whisper_executor or WhisperExecutor()

        # Prompt 缓存（上一个 Chunk 的定稿文本）
        self.previous_whisper_text: Optional[str] = None

    async def process(self, ctx: ProcessingContext):
        """
        处理单个 Chunk（慢流）

        流程：
        1. 构建安全的 Whisper Prompt
        2. Whisper 推理（带 Audio Overlap）
        3. 幻觉检测
        4. 填充 ctx.whisper_result

        Args:
            ctx: 处理上下文
        """
        chunk = ctx.audio_chunk

        # 阶段 1: 构建安全 Prompt（关键词提取策略）
        self.logger.debug(f"Chunk {ctx.chunk_index}: 构建 Whisper Prompt")
        prompt = self._build_safe_prompt()

        # 阶段 2: Whisper 推理（带 Audio Overlap）
        self.logger.debug(f"Chunk {ctx.chunk_index}: Whisper 慢流推理（带 Audio Overlap）")
        whisper_result = await self._run_whisper_with_overlap(ctx, prompt)

        # 阶段 3: 幻觉检测
        if self._is_hallucination(whisper_result, prompt):
            self.logger.warning(
                f"Chunk {ctx.chunk_index}: 检测到 Whisper 幻觉，回退到 SenseVoice"
            )
            # 回退：使用 SenseVoice 的 text_clean
            whisper_result['text'] = ctx.sv_result.get('text_clean', '')
            whisper_result['is_hallucination'] = True

        ctx.whisper_result = whisper_result

        self.logger.info(
            f"Chunk {ctx.chunk_index}: Whisper 推理完成 "
            f"(text_length={len(whisper_result.get('text', ''))})"
        )

    def _build_safe_prompt(self) -> str:
        """
        构建安全的 Whisper Prompt（关键词提取策略）

        新策略：
        1. 从上一个 Chunk 的定稿文本中提取关键词（专有名词）
        2. 添加用户词表
        3. 构建 "Glossary: word1, word2, ..." 格式的 prompt
        4. 避免使用完整句子，防止 Whisper 误判

        Returns:
            str: Whisper Prompt
        """
        prompt_builder = get_prompt_builder()

        # 构建安全的 Prompt
        prompt = prompt_builder.build_prompt(
            previous_text=self.previous_whisper_text,
            user_glossary=self.user_glossary
        )

        return prompt

    async def _run_whisper_with_overlap(
        self,
        ctx: ProcessingContext,
        initial_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行 Whisper 推理（带前向音频重叠）

        实现 Audio Overlap 策略：
        - 向前取 0.5 秒音频作为上下文预热
        - 避免 Whisper 在音频开头产生幻觉（冷启动问题）

        Args:
            ctx: ProcessingContext（包含 full_audio_array）
            initial_prompt: 初始提示词

        Returns:
            Dict: Whisper 推理结果
        """
        chunk = ctx.audio_chunk
        full_audio = ctx.full_audio_array
        sr = ctx.full_audio_sr

        # Audio Overlap 配置
        WHISPER_OVERLAP_SEC = 0.5  # 前向重叠 0.5 秒

        # 如果有完整音频数组，使用 Audio Overlap
        if full_audio is not None:
            # 计算重叠后的起始位置（不小于0）
            overlap_start = max(0.0, chunk.start - WHISPER_OVERLAP_SEC)
            start_sample = int(overlap_start * sr)
            end_sample = int(chunk.end * sr)

            # 确保不超出数组范围
            start_sample = max(0, start_sample)
            end_sample = min(len(full_audio), end_sample)

            # 从完整音频中提取重叠片段
            audio_with_overlap = full_audio[start_sample:end_sample]

            # 记录日志
            if overlap_start < chunk.start:
                self.logger.debug(
                    f"Whisper 添加 {chunk.start - overlap_start:.2f}s 前向重叠: "
                    f"[{overlap_start:.2f}s, {chunk.end:.2f}s]"
                )
        else:
            # 无完整音频，直接使用 chunk 音频
            audio_with_overlap = chunk.audio
            self.logger.debug("无完整音频数组，跳过 Audio Overlap")

        # 执行 Whisper 推理
        result = await self.whisper_executor.execute(
            audio=audio_with_overlap,
            start_time=chunk.start,
            end_time=chunk.end,
            language=self.whisper_language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False  # 防止双重提示词增益
        )

        return result

    def _is_hallucination(self, result: Dict[str, Any], prompt: Optional[str]) -> bool:
        """
        检测 Whisper 幻觉（四道检测防线）

        幻觉特征：
        1. 输出包含大量下划线（> 30%）
        2. 输出重复了 prompt 内容（相似度 > 80%）
        3. avg_logprob 过低（< -1.0，模型不确信）
        4. no_speech_prob 过高（> 0.6，静音段误识别）

        Args:
            result: Whisper 推理结果
            prompt: 使用的 prompt

        Returns:
            bool: True 表示检测到幻觉
        """
        text = result.get('text', '')
        if not text:
            return False

        # 检测 1：下划线占比 > 30%
        underscore_ratio = text.count('_') / max(len(text), 1)
        if underscore_ratio > 0.3:
            self.logger.warning(
                f"检测到下划线幻觉: 下划线占比 {underscore_ratio:.1%}, "
                f"text='{text[:50]}...'"
            )
            return True

        # 检测 2：重复 prompt 内容（相似度 > 80%）
        if prompt:
            prompt_words = set(prompt.split())
            text_words = set(text.split())
            if len(prompt_words) > 0:
                overlap_ratio = len(prompt_words & text_words) / len(prompt_words)
                # 如果 Whisper 输出与 prompt 重叠度超过 80%，且长度相近，可能是照抄
                if overlap_ratio > 0.8 and abs(len(text) - len(prompt)) < len(prompt) * 0.3:
                    self.logger.warning(
                        f"检测到提示词重复: 与 prompt 重叠度 {overlap_ratio:.1%}, "
                        f"prompt='{prompt[:30]}...', text='{text[:30]}...'"
                    )
                    return True

        # 检测 3 & 4：Dual-Stream Gating（置信度门控）
        raw_result = result.get('raw_result', {})
        segments = raw_result.get('segments', [])

        if segments:
            # 计算平均 logprob 和 no_speech_prob
            avg_logprob = sum(s.get('avg_logprob', -0.5) for s in segments) / len(segments)
            avg_no_speech = sum(s.get('no_speech_prob', 0.0) for s in segments) / len(segments)

            # 检测 3：avg_logprob 过低（模型在瞎猜）
            if avg_logprob < -1.0:
                self.logger.warning(
                    f"检测到低置信度幻觉: avg_logprob={avg_logprob:.2f} < -1.0, "
                    f"text='{text[:50]}...'"
                )
                return True

            # 检测 4：no_speech_prob 过高（静音段误识别）
            if avg_no_speech > 0.6 and text:
                self.logger.warning(
                    f"检测到静音段误识别: no_speech_prob={avg_no_speech:.2f} > 0.6, "
                    f"text='{text[:50]}...'"
                )
                return True

        return False

    def update_prompt_cache(self, whisper_text: str):
        """
        更新 Prompt 缓存

        将当前 Chunk 的定稿文本保存为下一个 Chunk 的 Prompt。

        Args:
            whisper_text: 定稿文本
        """
        self.previous_whisper_text = whisper_text
        self.logger.debug(f"更新 Prompt 缓存: {len(whisper_text)} 个字符")
