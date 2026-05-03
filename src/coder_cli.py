#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI Coder 工具模块

许多现代 Agent 框架正在从 MCP 转向 CLI 工具调用，因为：
1. MCP 工具定义（JSON Schema）占用大量上下文 token
2. CLI 命令更简洁，token 消耗更低
3. CLI 天然支持所有编程语言和工具链

本模块提供了一套 CLI 编码工具，可作为 Agent 的轻量级工具源，
替代或补充 MCP Server 的功能。

工具列表：
    - run_command:  执行 shell 命令并返回结果
    - read_file:    读取文件内容
    - write_file:   写入文件内容
    - list_dir:     列出目录内容
    - search_code:  在文件中搜索模式

设计原则：
    - 安全性：命令执行前进行白名单校验，防止注入攻击
    - 封装性：统一输入输出接口
    - 可扩展：易于添加新工具
"""

import os
import re
import subprocess
import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any

from .utils import log_title, truncate_text


# ---------------------------------------------------------------------------
# 安全配置
# ---------------------------------------------------------------------------

# 命令白名单 - 只允许执行这些命令（或前缀匹配）
ALLOWED_COMMANDS = {
    # 基础命令
    "ls", "dir", "pwd", "echo", "cat", "head", "tail", "wc",
    # 代码工具
    "python", "python3", "node", "npm", "npx", "go", "rustc", "cargo",
    "gcc", "g++", "make", "cmake", "javac", "java",
    # 版本控制
    "git", "diff", "patch",
    # 包管理
    "pip", "pip3", "uv", "poetry", "conda",
    # 文件操作（只读安全）
    "find", "grep", "rg", "tree",
    # 格式化/检查
    "black", "ruff", "mypy", "eslint", "prettier",
}

# 危险命令模式 - 即使在白名单中也禁止这些操作
DANGEROUS_PATTERNS = [
    r"\brm\s+(-rf?|--recursive)",    # 递归删除
    r"\bformat\s+/",                  # 格式化磁盘
    r">\s*/dev/",                     # 写入设备文件
    r"\bchmod\s+777",                # 危险的权限修改
    r"\bcurl\b.*\|\s*(ba)?sh",      # curl pipe to shell
    r"\bwget\b.*\|\s*(ba)?sh",      # wget pipe to shell
    r"\beval\b",                      # eval 执行
    r"\bexec\b",                      # exec 执行
    r"\bsudo\b",                      # sudo 提权
    r"\bdd\s+if=",                   # dd 磁盘操作
    r"\bmkfs\.",                      # 创建文件系统
]

# 允许的最大输出长度（字符）
MAX_OUTPUT_LENGTH = 10000

# 命令执行超时（秒）
COMMAND_TIMEOUT = 30


# ---------------------------------------------------------------------------
# 工具元数据
# ---------------------------------------------------------------------------

CODER_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "run_command",
        "description": "执行一个安全的 shell 命令并返回输出。仅限白名单中的命令。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令",
                },
                "working_dir": {
                    "type": "string",
                    "description": "工作目录路径，默认为当前目录",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "读取指定文件的内容",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对于项目根目录或绝对路径）",
                },
                "start_line": {
                    "type": "integer",
                    "description": "起始行号（1-based），默认从第 1 行开始",
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号（1-based，含），默认读到文件末尾",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "将内容写入文件，会覆盖已有文件",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径（相对于项目根目录或绝对路径）",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的内容",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "列出目录中的文件和子目录",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "目录路径，默认为当前目录",
                },
                "pattern": {
                    "type": "string",
                    "description": "可选的文件名通配符，如 *.py",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_code",
        "description": "在项目文件中搜索匹配的文本模式（使用 ripgrep）",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "要搜索的正则表达式或文本",
                },
                "path": {
                    "type": "string",
                    "description": "搜索路径，默认为当前目录",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "文件名过滤，如 *.py, *.ts",
                },
            },
            "required": ["pattern"],
        },
    },
]


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------

class CoderCLI:
    """
    CLI 编码工具集

    提供安全的命令行工具调用，作为 MCP 的轻量替代方案。
    每个方法对应一个 Agent 可调用的工具。

    Usage:
        coder = CoderCLI(workspace="/path/to/project")
        result = await coder.execute("run_command", command="ls -la")
    """

    def __init__(self, workspace: Optional[str] = None):
        """
        初始化 CLI 编码工具

        Args:
            workspace: 工作空间根目录，用于路径安全检查
        """
        self.workspace = Path(workspace or os.getcwd()).resolve()

    def get_tools(self) -> List[Dict[str, Any]]:
        """获取工具定义列表（OpenAI function calling 格式）"""
        return CODER_TOOLS

    async def execute(self, name: str, **kwargs) -> str:
        """
        执行指定的工具

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果字符串
        """
        handlers = {
            "run_command": self._run_command,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_dir": self._list_dir,
            "search_code": self._search_code,
        }

        handler = handlers.get(name)
        if not handler:
            return f"未知工具: {name}"

        log_title(f"CLI TOOL: {name}")
        try:
            result = await handler(**kwargs)
            return str(result) if result is not None else "命令执行完毕（无输出）"
        except Exception as e:
            return f"工具执行失败: {e}"

    # ------------------------------------------------------------------
    # 工具实现
    # ------------------------------------------------------------------

    async def _run_command(self, command: str, working_dir: str = "") -> str:
        """
        安全地执行 shell 命令

        安全检查流程：
        1. 解析命令获取基础命令名
        2. 检查是否在白名单中
        3. 检查是否包含危险模式
        4. 在子进程中执行并捕获输出
        """
        # 跳过空命令
        command = command.strip()
        if not command:
            return "错误: 命令为空"

        # 安全校验
        is_allowed, reason = self._validate_command(command)
        if not is_allowed:
            return f"安全限制: {reason}"

        # 确定工作目录
        cwd = self.workspace
        if working_dir:
            wd = Path(working_dir)
            cwd = wd if wd.is_absolute() else self.workspace / wd
            if not cwd.exists():
                return f"错误: 工作目录不存在: {cwd}"

        print(f"  $ {command}")

        try:
            proc = await __import__("asyncio").create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd),
            )
            stdout, stderr = await __import__("asyncio").wait_for(
                proc.communicate(), timeout=COMMAND_TIMEOUT
            )

            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n[STDERR]\n" + stderr.decode("utf-8", errors="replace")

            return truncate_text(output.strip() or "(无输出)", MAX_OUTPUT_LENGTH)

        except __import__("asyncio").TimeoutError:
            return f"错误: 命令执行超时 ({COMMAND_TIMEOUT}s)"
        except Exception as e:
            return f"错误: {e}"

    async def _read_file(
        self, path: str, start_line: int = 0, end_line: int = 0
    ) -> str:
        """读取文件内容，支持指定行范围"""
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return f"错误: 文件不存在: {file_path}"
        if not file_path.is_file():
            return f"错误: 不是文件: {file_path}"

        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()

            # 调整行范围
            s = max(0, start_line - 1) if start_line > 0 else 0
            e = min(len(lines), end_line) if end_line > 0 else len(lines)

            selected = lines[s:e]
            result = "\n".join(
                f"{i + s + 1:4d}| {line}"
                for i, line in enumerate(selected)
            )
            return result if result else "(空文件)"
        except UnicodeDecodeError:
            return "错误: 无法以 UTF-8 解码该文件（可能是二进制文件）"

    async def _write_file(self, path: str, content: str) -> str:
        """将内容写入文件"""
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"文件已写入: {file_path} ({len(content)} 字符)"

    async def _list_dir(self, path: str = "", pattern: str = "") -> str:
        """列出目录内容"""
        dir_path = self._resolve_path(path) if path else self.workspace
        if not dir_path.exists():
            return f"错误: 目录不存在: {dir_path}"
        if not dir_path.is_dir():
            return f"错误: 不是目录: {dir_path}"

        entries = sorted(dir_path.iterdir(), key=lambda e: (not e.is_dir(), e.name))

        if pattern:
            import fnmatch
            entries = [e for e in entries if fnmatch.fnmatch(e.name, pattern)]

        lines = []
        for entry in entries:
            prefix = "[DIR] " if entry.is_dir() else "[FILE]"
            size = ""
            if entry.is_file():
                try:
                    size = f" ({entry.stat().st_size:,} B)"
                except Exception:
                    pass
            lines.append(f"  {prefix} {entry.name}{size}")

        return "\n".join(lines) if lines else "(空目录)"

    async def _search_code(
        self, pattern: str, path: str = "", file_pattern: str = ""
    ) -> str:
        """在代码中搜索模式（使用 findstr 作为 Windows 回退）"""
        search_path = self._resolve_path(path) if path else self.workspace
        if not search_path.exists():
            return f"错误: 路径不存在: {search_path}"

        import asyncio

        # 优先使用 rg (ripgrep)
        cmd = ["rg", "--line-number", "--max-count=50", pattern, str(search_path)]
        if file_pattern:
            cmd.extend(["--glob", file_pattern])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            result = stdout.decode("utf-8", errors="replace").strip()
            return result if result else "未找到匹配结果"
        except (FileNotFoundError, Exception):
            # 回退到 grep / findstr
            return self._fallback_search(pattern, str(search_path), file_pattern)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> Path:
        """解析路径，防止路径遍历攻击"""
        p = Path(path)
        if p.is_absolute():
            # 检查绝对路径是否在工作空间内
            resolved = p.resolve()
            try:
                resolved.relative_to(self.workspace)
            except ValueError:
                raise PermissionError(
                    f"禁止访问工作空间外的路径: {path}"
                )
            return resolved
        else:
            return (self.workspace / p).resolve()

    @staticmethod
    def _validate_command(command: str) -> tuple:
        """
        验证命令安全性

        Returns:
            (是否允许, 原因说明)
        """
        # 1. 检查危险模式
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"命令包含危险操作（匹配模式: {pattern}）"

        # 2. 提取基础命令并检查白名单
        try:
            tokens = shlex.split(command)
            if not tokens:
                return False, "无法解析命令"
            base_cmd = os.path.basename(tokens[0])
        except ValueError:
            return False, "命令解析失败"

        if base_cmd not in ALLOWED_COMMANDS:
            return False, (
                f"命令 '{base_cmd}' 不在白名单中。"
                f"允许的命令: {', '.join(sorted(ALLOWED_COMMANDS))}"
            )

        return True, ""

    @staticmethod
    def _fallback_search(pattern: str, path: str, file_pattern: str) -> str:
        """使用 Python 内置功能进行代码搜索（无 ripgrep 时的回退）"""
        import fnmatch
        results = []
        search_path = Path(path)

        for f in search_path.rglob("*"):
            if not f.is_file():
                continue
            if file_pattern and not fnmatch.fnmatch(f.name, file_pattern):
                continue
            if f.stat().st_size > 1_000_000:  # 跳过 >1MB 的文件
                continue

            try:
                for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        results.append(f"{f}:{i}: {line.strip()}")
                        if len(results) >= 50:
                            break
            except Exception:
                continue

        return "\n".join(results) if results else "未找到匹配结果"
