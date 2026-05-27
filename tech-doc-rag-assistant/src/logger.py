# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值（PEP 563）
# 好处：避免循环导入问题，减少运行时开销，支持前向引用
from __future__ import annotations

# logging：Python 标准库的日志模块，提供灵活的日志记录能力
import logging
# Path：pathlib 模块的路径操作类，提供面向对象的文件系统路径操作
from pathlib import Path


# 定义日志初始化函数
# name 参数：日志器的名称，默认值为 "tech_doc_rag"
# 返回值类型标注为 logging.Logger
def setup_logger(name: str = "tech_doc_rag") -> logging.Logger:
    """初始化统一日志器，输出到控制台与文件。"""

    # 获取或创建指定名称的 Logger 实例
    logger = logging.getLogger(name)
    # 设置日志级别为 INFO —— DEBUG < INFO < WARNING < ERROR < CRITICAL
    # INFO 级别意味着 DEBUG 级别的日志会被忽略
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler —— 如果 Logger 已经有 handler 了，说明之前被初始化过
    # 直接返回，避免日志重复输出
    if logger.handlers:
        return logger

    # 创建 logs 目录的 Path 对象
    log_dir = Path("logs")
    # 递归创建目录（如果不存在），parents=True 等价于 mkdir -p
    # exist_ok=True 表示目录已存在也不会报错
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志格式器
    # fmt 参数定义了每行日志的输出格式：
    #   %(asctime)s → 时间戳（格式由 datefmt 指定）
    #   %(levelname)s → 日志级别（INFO/WARNING/ERROR 等）
    #   %(name)s → Logger 的名称
    #   %(message)s → 日志消息正文
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # 时间显示为 年-月-日 时:分:秒
    )

    # 创建控制台输出 handler —— 日志会打印到终端/命令行
    stream_handler = logging.StreamHandler()
    # 给 handler 设置格式器
    stream_handler.setFormatter(formatter)

    # 创建文件输出 handler —— 日志会写入 logs/app.log 文件
    # encoding="utf-8" 确保中文日志不会乱码
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    # 给 handler 设置格式器
    file_handler.setFormatter(formatter)

    # 将两个 handler 添加到 Logger 上
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    # 返回配置好的 Logger 实例
    return logger


# 模块级别的 logger 变量 —— 在 import 这个模块时自动执行 setup_logger()
# 其他模块可以直接 from .logger import logger 来使用
# 不需要每次自己调用 setup_logger()
logger = setup_logger()
