"""
简单的日志配置
"""

import logging
import sys
from pathlib import Path
from core.config import config


def setup_logging():
    """配置日志系统"""

    # 配置格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 控制台输出
            logging.StreamHandler(sys.stdout),
            # 文件输出
            logging.FileHandler(config.LOG_FILE, encoding='utf-8')
        ]
    )

    # 设置第三方库日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('multipart').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"📝 日志系统已初始化 - 级别: {config.LOG_LEVEL}")
    logger.info(f"📄 日志文件: {config.LOG_FILE}")

    return logger
