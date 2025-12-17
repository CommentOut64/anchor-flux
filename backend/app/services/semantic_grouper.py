"""
语义分组服务 (Layer 2 优化)

职责: 将物理上分开但语义上连续的句子标记为同一组

核心特性:
- 支持时间重叠检测 (VAD 切分不准时自动合并)
- 使用语言策略而非硬编码规则 (支持多语言扩展)
- 提供手动合并/拆分 API (供前端使用)

设计原则:
- 语义分组不改变时间戳，只添加元数据
- 分组信息用于 LLM 校对时提供上下文
- 前端可视化时用于分组显示
"""
import uuid
import logging
from typing import List, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from app.models.sensevoice_models import SentenceSegment
    from app.services.sentence_splitter import LanguageStrategy

# 延迟导入，避免相对导入问题
def _get_language_strategy(language: str):
    """延迟导入语言策略"""
    try:
        from app.services.sentence_splitter import get_language_strategy
        return get_language_strategy(language)
    except ImportError:
        # 测试环境下的导入
        import importlib.util
        from pathlib import Path

        splitter_path = Path(__file__).parent / "sentence_splitter.py"
        spec = importlib.util.spec_from_file_location("sentence_splitter", splitter_path)
        sentence_splitter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sentence_splitter)
        return sentence_splitter.get_language_strategy(language)

logger = logging.getLogger(__name__)


@dataclass
class GroupConfig:
    """语义分组配置"""
    # 分组触发条件
    max_group_gap: float = 2.0           # 组内最大间隔(秒)
    max_group_duration: float = 10.0     # 单组最大时长(秒)
    max_group_sentences: int = 5         # 单组最大句子数

    # 重叠检测配置
    enable_overlap_detection: bool = True  # 启用重叠检测
    overlap_tolerance: float = 0.05        # 重叠/接壤容差(秒)

    # 语言设置 (解决硬编码问题)
    language: str = "auto"                 # 语言: auto, zh, en, ja 等


class SemanticGrouper:
    """语义分组器"""

    def __init__(self, config: Optional[GroupConfig] = None):
        self.config = config or GroupConfig()
        # 根据语言获取对应的策略（延迟导入）
        self.language_strategy = _get_language_strategy(self.config.language)

    def group(self, sentences: List['SentenceSegment']) -> List['SentenceSegment']:
        """
        对句子列表进行语义分组

        Args:
            sentences: 已分句的句子列表

        Returns:
            添加了 group_id 和 is_soft_break 标记的句子列表
        """
        if not sentences:
            return sentences

        current_group_id = None
        current_group_start = 0
        current_group_count = 0

        for i, sentence in enumerate(sentences):
            should_start_new_group = self._should_start_new_group(
                sentences, i, current_group_id, current_group_start, current_group_count
            )

            if should_start_new_group:
                # 结束上一组
                if current_group_id and current_group_count > 1:
                    self._finalize_group(sentences, current_group_start, i - 1)

                # 开始新组
                current_group_id = str(uuid.uuid4())[:8]
                current_group_start = i
                current_group_count = 1
            else:
                current_group_count += 1

            sentence.group_id = current_group_id

            # 判断是否为软断点
            if i > 0:
                prev_sentence = sentences[i - 1]
                gap = sentence.start - prev_sentence.end
                if gap > 0.3 and sentence.group_id == prev_sentence.group_id:
                    sentence.is_soft_break = True

        # 处理最后一组
        if current_group_id and current_group_count > 1:
            self._finalize_group(sentences, current_group_start, len(sentences) - 1)

        # 设置组内位置
        self._set_group_positions(sentences)

        logger.debug(f"语义分组完成: {len(sentences)} 句 -> {len(set(s.group_id for s in sentences))} 组")

        return sentences

    def _check_time_overlap(
        self,
        prev_sentence: 'SentenceSegment',
        curr_sentence: 'SentenceSegment'
    ) -> bool:
        """
        检测时间重叠或接壤

        场景: VAD 切分不准时，两个 segment 时间上有微小重叠或完全接壤，
              这种情况 99% 属于同一句话

        Args:
            prev_sentence: 前一个句子
            curr_sentence: 当前句子

        Returns:
            True 表示存在重叠/接壤，应该合并到同一组
        """
        if not self.config.enable_overlap_detection:
            return False

        # 计算时间间隙
        gap = curr_sentence.start - prev_sentence.end

        # 重叠: gap < 0 (当前句子的开始时间早于上一句的结束时间)
        if gap < 0:
            logger.debug(f"检测到时间重叠: 前句结束={prev_sentence.end:.3f}s, 当前句开始={curr_sentence.start:.3f}s")
            return True

        # 接壤: gap 非常小 (在容差范围内)
        if gap <= self.config.overlap_tolerance:
            logger.debug(f"检测到时间接壤: 间隙={gap:.3f}s < 容差={self.config.overlap_tolerance:.3f}s")
            return True

        return False

    def _should_start_new_group(
        self,
        sentences: List['SentenceSegment'],
        current_idx: int,
        current_group_id: Optional[str],
        group_start_idx: int,
        group_count: int
    ) -> bool:
        """判断是否应该开始新的语义组"""

        # 第一个句子总是开始新组
        if current_idx == 0:
            return True

        prev_sentence = sentences[current_idx - 1]
        curr_sentence = sentences[current_idx]

        # 新增: 检测时间重叠/接壤 (强制合并到同一组)
        if self._check_time_overlap(prev_sentence, curr_sentence):
            logger.debug(f"时间重叠/接壤，合并到同一组")
            return False  # 不开始新组，合并到当前组

        # 检查时间间隔
        gap = curr_sentence.start - prev_sentence.end
        if gap > self.config.max_group_gap:
            logger.debug(f"间隔过大 ({gap:.3f}s > {self.config.max_group_gap:.3f}s)，开始新组")
            return True

        # 检查组时长
        group_start_time = sentences[group_start_idx].start
        group_duration = curr_sentence.end - group_start_time
        if group_duration > self.config.max_group_duration:
            logger.debug(f"组时长过大 ({group_duration:.3f}s > {self.config.max_group_duration:.3f}s)，开始新组")
            return True

        # 检查组内句子数
        if group_count >= self.config.max_group_sentences:
            logger.debug(f"组内句子数过多 ({group_count} >= {self.config.max_group_sentences})，开始新组")
            return True

        # 检查上一句是否为完整结尾 (使用语言策略)
        prev_text = prev_sentence.text.strip()
        sentence_end_chars = self.language_strategy.get_sentence_end_chars()

        if prev_text and prev_text[-1] in sentence_end_chars:
            # 完整结尾，检查下一句是否为续接 (使用语言策略)
            curr_text = curr_sentence.text.strip()
            is_continuation = self.language_strategy.is_continuation(curr_text)
            if not is_continuation:
                logger.debug(f"上一句完整结尾且下一句非续接，开始新组")
                return True

        return False

    def _finalize_group(self, sentences: List['SentenceSegment'], start_idx: int, end_idx: int):
        """完成一个语义组的标记（可扩展：添加组级别的后处理）"""
        group_id = sentences[start_idx].group_id
        logger.debug(f"完成语义组 {group_id}: 句子 {start_idx}-{end_idx} ({end_idx - start_idx + 1} 句)")

    def _set_group_positions(self, sentences: List['SentenceSegment']):
        """设置每个句子在组内的位置"""
        group_sentences = {}

        # 按组分类
        for i, s in enumerate(sentences):
            if s.group_id not in group_sentences:
                group_sentences[s.group_id] = []
            group_sentences[s.group_id].append(i)

        # 设置位置
        for group_id, indices in group_sentences.items():
            if len(indices) == 1:
                sentences[indices[0]].group_position = 'single'
            else:
                sentences[indices[0]].group_position = 'start'
                sentences[indices[-1]].group_position = 'end'
                for idx in indices[1:-1]:
                    sentences[idx].group_position = 'middle'

    # ============================================================================
    # API 方法：支持前端手动合并/拆分
    # ============================================================================

    def merge_groups(
        self,
        sentences: List['SentenceSegment'],
        group_id_1: str,
        group_id_2: str
    ) -> List['SentenceSegment']:
        """
        合并两个语义组

        Args:
            sentences: 句子列表
            group_id_1: 第一个组ID
            group_id_2: 第二个组ID (将合并到 group_id_1)

        Returns:
            更新后的句子列表
        """
        merge_count = 0
        for s in sentences:
            if s.group_id == group_id_2:
                s.group_id = group_id_1
                merge_count += 1

        # 重新计算组内位置
        self._set_group_positions(sentences)

        logger.info(f"合并语义组: {group_id_2} -> {group_id_1} ({merge_count} 句)")
        return sentences

    def split_group(
        self,
        sentences: List['SentenceSegment'],
        sentence_idx: int
    ) -> List['SentenceSegment']:
        """
        在指定位置拆分语义组

        Args:
            sentences: 句子列表
            sentence_idx: 拆分点 (从此句子开始创建新组)

        Returns:
            更新后的句子列表
        """
        if sentence_idx < 0 or sentence_idx >= len(sentences):
            logger.warning(f"拆分索引越界: {sentence_idx}")
            return sentences

        target = sentences[sentence_idx]
        old_group_id = target.group_id

        # 生成新的组ID
        new_group_id = str(uuid.uuid4())[:8]

        # 从拆分点开始，将后续同组句子分配到新组
        split_count = 0
        for i in range(sentence_idx, len(sentences)):
            if sentences[i].group_id == old_group_id:
                sentences[i].group_id = new_group_id
                split_count += 1
            else:
                break

        # 重新计算组内位置
        self._set_group_positions(sentences)

        logger.info(f"拆分语义组: {old_group_id} -> {new_group_id} (从索引 {sentence_idx} 拆分，{split_count} 句)")
        return sentences


# 单例访问
_grouper_instance = None


def get_semantic_grouper(config: GroupConfig = None) -> SemanticGrouper:
    """获取语义分组器单例"""
    global _grouper_instance
    if _grouper_instance is None:
        _grouper_instance = SemanticGrouper(config)
    return _grouper_instance
