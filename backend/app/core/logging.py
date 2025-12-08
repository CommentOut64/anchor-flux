"""
统一的日志系统配置
支持毫秒精度的时间戳和统一的日志格式
"""

import logging
import sys
from pathlib import Path
from app.core.config import config


class MillisecondFormatter(logging.Formatter):
    """包含毫秒精度的日志格式化器"""

    def format(self, record):
        # 时间戳：精确到毫秒
        import datetime
        ct = datetime.datetime.fromtimestamp(record.created)
        timestamp = ct.strftime('%H:%M:%S') + '.%03d' % (record.msecs)

        # 提取日志来源（模块名）
        logger_name = record.name.split('.')[-1]

        # 统一格式：时间戳 [级别] [来源] 信息
        return f"{timestamp} [{record.levelname}] [{logger_name}] {record.getMessage()}"


class ThirdPartyFilter(logging.Filter):
    """过滤第三方库的多余日志"""

    # 需要完全禁止的日志
    BLOCKED_MESSAGES = [
        "No language specified, language will be first be detected",
        "Performing voice activity detection using Pyannote",
        "Lightning automatically upgraded your loaded checkpoint",
        "Model was trained with pyannote.audio",
        "Model was trained with torch",
        "TensorFloat-32 (TF32) has been disabled",
        "No active speech found in audio",
        "Warning: audio is shorter than 30s, language detection may be inaccurate",
        "Using `TRANSFORMERS_CACHE` is deprecated",
        "ReproducibilityWarning",
        "FutureWarning"
    ]

    def filter(self, record):
        msg = record.getMessage()
        # 检查是否是被阻止的消息
        for blocked in self.BLOCKED_MESSAGES:
            if blocked in msg:
                return False
        return True


def setup_logging():
    """配置日志系统"""

    # 创建根日志处理器
    root_logger = logging.getLogger()
    log_level = getattr(logging, config.LOG_LEVEL)
    root_logger.setLevel(logging.DEBUG)  # 设置根logger为DEBUG，让处理器来控制级别

    # 清除已有的处理器
    root_logger.handlers.clear()

    # 创建格式化器
    formatter = MillisecondFormatter()

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)  # 在处理器层面设置级别
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ThirdPartyFilter())
    root_logger.addHandler(console_handler)

    # 文件输出
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(log_level)  # 在处理器层面设置级别
    file_handler.setFormatter(formatter)
    file_handler.addFilter(ThirdPartyFilter())
    root_logger.addHandler(file_handler)

    # 设置第三方库日志级别为WARNING
    third_party_loggers = [
        'urllib3', 'multipart', 'transformers',
        'faster_whisper', 'ctranslate2',  # Faster-Whisper 及其底层库
        'silero', 'torch', 'pytorch_lightning', 'pyannote',
        'speechbrain', 'whisper'
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger(logger_name).addFilter(ThirdPartyFilter())

    # 禁用 uvicorn 的访问日志（INFO级别的请求日志）
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging system initialized - level: {config.LOG_LEVEL}")
    logger.info(f"Log file: {config.LOG_FILE}")

    return logger
