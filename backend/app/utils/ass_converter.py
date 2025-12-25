"""
ASS (Advanced SubStation Alpha) 字幕格式转换器

支持将字幕数据转换为 ASS 格式，提供多种样式预设。
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ASSStyle:
    """ASS 字幕样式配置"""
    name: str = "Default"
    fontname: str = "Microsoft YaHei"
    fontsize: int = 24
    primary_colour: str = "&H00FFFFFF"  # 白色 (ABGR 格式)
    secondary_colour: str = "&H00FFFFFF"
    outline_colour: str = "&H00000000"  # 黑色描边
    back_colour: str = "&H00000000"
    bold: int = 0
    italic: int = 0
    underline: int = 0
    strike_out: int = 0
    scale_x: float = 100.0
    scale_y: float = 100.0
    spacing: float = 0.0
    angle: float = 0.0
    border_style: int = 1
    outline: float = 2.0  # 描边宽度
    shadow: float = 0.0
    alignment: int = 2  # 底部居中
    margin_l: int = 10
    margin_r: int = 10
    margin_v: int = 20
    encoding: int = 1


class ASSConverter:
    """ASS 格式转换器"""

    # 预定义样式预设
    STYLE_PRESETS = {
        "default": ASSStyle(
            name="Default",
            fontname="Microsoft YaHei",
            fontsize=48,
            outline=2.5,
            alignment=2
        ),
        "movie": ASSStyle(
            name="Movie",
            fontname="Source Han Sans CN",
            fontsize=52,
            outline=3.0,
            alignment=2
        ),
        "news": ASSStyle(
            name="News",
            fontname="SimHei",
            fontsize=42,
            primary_colour="&H0000FFFF",  # 黄色
            outline=0.0,
            alignment=1  # 底部左对齐
        ),
        "danmaku": ASSStyle(
            name="Danmaku",
            fontname="Microsoft YaHei",
            fontsize=20,
            outline=1.0,
            alignment=8  # 顶部居中
        )
    }

    @staticmethod
    def format_ass_timestamp(seconds: float) -> str:
        """
        格式化 ASS 时间戳

        Args:
            seconds: 秒数

        Returns:
            ASS 格式时间戳 (H:MM:SS.cc)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    @staticmethod
    def generate_script_info(title: str = "Untitled", video_width: int = 1920, video_height: int = 1080) -> str:
        """
        生成 ASS 文件的 Script Info 部分

        Args:
            title: 字幕标题
            video_width: 视频宽度
            video_height: 视频高度

        Returns:
            Script Info 文本
        """
        return f"""[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
"""

    @staticmethod
    def generate_style_section(style: ASSStyle) -> str:
        """
        生成 ASS 样式定义

        Args:
            style: ASS 样式对象

        Returns:
            样式定义文本
        """
        return (
            f"Style: {style.name},{style.fontname},{style.fontsize},"
            f"{style.primary_colour},{style.secondary_colour},{style.outline_colour},{style.back_colour},"
            f"{style.bold},{style.italic},{style.underline},{style.strike_out},"
            f"{style.scale_x},{style.scale_y},{style.spacing},{style.angle},"
            f"{style.border_style},{style.outline},{style.shadow},{style.alignment},"
            f"{style.margin_l},{style.margin_r},{style.margin_v},{style.encoding}"
        )

    @classmethod
    def convert_from_subtitles(
        cls,
        subtitles: List[Dict],
        output_path: Path,
        style_preset: str = "default",
        title: str = "Untitled",
        video_width: int = 1920,
        video_height: int = 1080
    ) -> None:
        """
        从字幕数据生成 ASS 文件

        Args:
            subtitles: 字幕数据列表，每项包含 start, end, text 字段
            output_path: 输出文件路径
            style_preset: 样式预设名称
            title: 字幕标题
            video_width: 视频宽度
            video_height: 视频高度
        """
        # 获取样式预设
        style = cls.STYLE_PRESETS.get(style_preset, cls.STYLE_PRESETS["default"])

        # 生成 ASS 文件内容
        content_parts = []

        # Script Info
        content_parts.append(cls.generate_script_info(title, video_width, video_height))

        # V4+ Styles
        content_parts.append("\n[V4+ Styles]")
        content_parts.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
        content_parts.append(cls.generate_style_section(style))

        # Events
        content_parts.append("\n[Events]")
        content_parts.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

        # 转换字幕事件
        for subtitle in subtitles:
            start = cls.format_ass_timestamp(subtitle.get("start", 0))
            end = cls.format_ass_timestamp(subtitle.get("end", 0))
            text = subtitle.get("text", "").replace("\n", "\\N")  # ASS 换行符

            event_line = f"Dialogue: 0,{start},{end},{style.name},,0,0,0,,{text}"
            content_parts.append(event_line)

        # 写入文件
        ass_content = "\n".join(content_parts)
        output_path.write_text(ass_content, encoding='utf-8-sig')  # 使用 UTF-8 BOM
