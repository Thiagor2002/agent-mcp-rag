#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块
提供项目通用的辅助功能，包括：
- 格式化日志输出
- 文件操作
- 环境变量安全管理
"""

import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def log_title(title: str, char: str = "=") -> None:
    """打印格式化的节标题到控制台"""
    print(f"\n{char * 20} {title} {char * 20}")


def log_section(title: str) -> None:
    """打印格式化的子标题到控制台"""
    print(f"\n--- {title} ---")


def safe_getenv(key: str, default: Optional[str] = None, mask: bool = True) -> Optional[str]:
    """
    安全地获取环境变量，支持敏感信息掩码输出

    Args:
        key: 环境变量名称
        default: 默认值
        mask: 是否在日志中对值进行掩码处理

    Returns:
        环境变量值或默认值
    """
    value = os.getenv(key, default)
    return value


def mask_secret(value: str, visible_chars: int = 4) -> str:
    """
    对敏感字符串进行掩码处理

    Args:
        value: 原始字符串
        visible_chars: 可见的前缀字符数

    Returns:
        掩码后的字符串，如 "sfd7****ve4g"
    """
    if not value or len(value) <= visible_chars * 2:
        return "*" * min(len(value), 8)
    return value[:visible_chars] + "*" * (len(value) - visible_chars * 2) + value[-visible_chars:]


def ensure_dir(path: Path) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        目录路径
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    截断文本到指定长度

    Args:
        text: 原始文本
        max_length: 最大长度

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"\n... [截断，原长度 {len(text)} 字符]"
