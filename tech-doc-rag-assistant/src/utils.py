# 从 __future__ 导入 annotations，让所有类型注解使用延迟求值（PEP 563）
from __future__ import annotations

# Path：pathlib 模块的路径操作类
from pathlib import Path


# 工具函数：确保目录存在
# path 参数可以是字符串或 Path 对象，返回值是 Path 对象
# str | Path 是 Python 3.10+ 的联合类型语法（得益于 from __future__ import annotations）
def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则创建。"""

    # 将输入统一转为 Path 对象（如果已经是 Path 对象则保持不变）
    path_obj = Path(path)
    # 递归创建目录：parents=True 表示 mkdir -p，exist_ok=True 表示已存在不报错
    path_obj.mkdir(parents=True, exist_ok=True)
    # 返回 Path 对象，方便调用方链式操作
    return path_obj


# 工具函数：压缩并截断长文本
# text：原始文本，max_length：最大字符数（默认 180）
def truncate_text(text: str, max_length: int = 180) -> str:
    """压缩并截断长文本，便于前端展示引用摘要。"""

    # 压缩空白字符：split() 将文本按任意空白切分，再用单个空格 join 回去
    # 这会把多个连续空格、换行、制表符都压缩成单个空格
    compact = " ".join(text.split())
    # 如果压缩后的文本长度 ≤ 最大长度，直接返回
    if len(compact) <= max_length:
        return compact
    # 超长时截断：保留前 max_length-3 个字符，去掉末尾多余空白，再加 "..." 省略号
    # rstrip() 防止截断位置刚好在一个单词中间导致断词不美观
    return compact[: max_length - 3].rstrip() + "..."
