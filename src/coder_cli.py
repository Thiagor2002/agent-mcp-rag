#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLI Coder 工具模块
==================

基于 LLM 直接推理的 CLI 编码工具。许多 LLM 对 Bash/CLI 代码
进行了偏好微调，可以直接理解任务并生成正确命令，无需复杂的
function calling 描述，从而大幅减少 token 消耗。

工具列表:
    - run_command:  安全执行 shell 命令
    - read_file:    读取文件内容
    - write_file:   写入文件内容
    - list_dir:     列出目录
    - search_code:  搜索代码模式

设计原则:
    - 安全性: 白名单 + 危险模式检测
    - 低 Token: 工具定义简洁，减少上下文消耗
    - 可扩展: 易于添加新工具
"""

import os
import re
import subprocess
import shlex
import asyncio
import fnmatch
from pathlib import Path
from typing import List, Optional, Dict, Any

from .utils import log_title, truncate_text


# ---------------------------------------------------------------------------
# 安全配置
# ---------------------------------------------------------------------------

ALLOWED_COMMANDS = {
    "ls", "dir", "pwd", "echo", "cat", "head", "tail", "wc", "sort", "uniq",
    "python", "python3", "node", "npm", "npx", "go", "rustc", "cargo",
    "gcc", "g++", "make", "cmake", "javac", "java",
    "git", "diff", "patch", "find", "grep", "rg", "tree",
    "pip", "pip3", "uv", "poetry", "conda",
    "black", "ruff", "mypy", "eslint", "prettier",
    "mkdir", "touch", "cp", "mv", "code", "cursor",
}

DANGEROUS_PATTERNS = [
    r"\brm\s+(-rf?|--recursive)", r">\s*/dev/",
    r"\bcurl\b.*\|\s*(ba)?sh", r"\bwget\b.*\|\s*(ba)?sh",
    r"\beval\b", r"\bexec\b", r"\bsudo\b", r"\bdd\s+if=",
    r"\bchmod\s+777", r"\bmkfs\.", r"\bformat\s+/",
]

MAX_OUTPUT_LENGTH = 10000
COMMAND_TIMEOUT = 30


# ---------------------------------------------------------------------------
# 工具元数据
# ---------------------------------------------------------------------------

CODER_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "run_command",
        "description": "Execute a safe shell command and return output",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command"},
                "working_dir": {"type": "string", "description": "Working directory"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read file contents with optional line range",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "start_line": {"type": "integer", "description": "Start line (1-based)"},
                "end_line": {"type": "integer", "description": "End line (inclusive)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List directory contents",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"},
                "pattern": {"type": "string", "description": "File pattern e.g. *.py"},
            },
        },
    },
    {
        "name": "search_code",
        "description": "Search for text pattern in project files",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex or text to search"},
                "path": {"type": "string", "description": "Search path"},
                "file_pattern": {"type": "string", "description": "Filter e.g. *.py"},
            },
            "required": ["pattern"],
        },
    },
]


# ---------------------------------------------------------------------------
# CoderCLI 主类
# ---------------------------------------------------------------------------

class CoderCLI:
    """
    CLI 编码工具集，作为 MCP 的轻量替代方案

    Usage:
        coder = CoderCLI()
        result = await coder.execute("run_command", command="ls -la")
        tools = coder.get_tools()
    """

    def __init__(self, workspace: str = ""):
        self.workspace = Path(workspace or os.getcwd()).resolve()

    def get_tools(self) -> List[Dict[str, Any]]:
        return CODER_TOOLS

    async def execute(self, name: str, **kwargs) -> str:
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

        log_title(f"CLI: {name}")
        try:
            result = await handler(**kwargs)
            return str(result) if result is not None else "OK"
        except Exception as e:
            return f"工具失败: {e}"

    # ------------------------------------------------------------------
    # 工具实现
    # ------------------------------------------------------------------

    async def _run_command(self, command: str, working_dir: str = "") -> str:
        command = command.strip()
        if not command:
            return "错误: 空命令"

        is_ok, reason = self._validate(command)
        if not is_ok:
            return f"安全限制: {reason}"

        cwd = self.workspace / working_dir if working_dir else self.workspace
        if not cwd.exists():
            return f"错误: 目录不存在 {cwd}"

        print(f"  $ {command}")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=str(cwd),
            )
            out, err = await asyncio.wait_for(proc.communicate(), COMMAND_TIMEOUT)
            output = out.decode("utf-8", errors="replace")
            if err:
                output += "\n[STDERR]\n" + err.decode("utf-8", errors="replace")
            return truncate_text(output.strip() or "(无输出)", MAX_OUTPUT_LENGTH)
        except asyncio.TimeoutError:
            return f"错误: 超时 ({COMMAND_TIMEOUT}s)"
        except Exception as e:
            return f"错误: {e}"

    async def _read_file(
        self, path: str, start_line: int = 0, end_line: int = 0
    ) -> str:
        fp = self._resolve(path)
        if not fp.is_file():
            return f"错误: {fp}"
        try:
            lines = fp.read_text("utf-8").splitlines()
            s = max(0, start_line - 1) if start_line > 0 else 0
            e = min(len(lines), end_line) if end_line > 0 else len(lines)
            return "\n".join(f"{i+s+1:4d}| {l}" for i, l in enumerate(lines[s:e]))
        except UnicodeDecodeError:
            return "错误: 无法解码"

    async def _write_file(self, path: str, content: str) -> str:
        fp = self._resolve(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, "utf-8")
        return f"已写入: {fp} ({len(content)} 字符)"

    async def _list_dir(self, path: str = "", pattern: str = "") -> str:
        d = self._resolve(path) if path else self.workspace
        if not d.is_dir():
            return f"错误: {d}"
        entries = sorted(d.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        if pattern:
            entries = [e for e in entries if fnmatch.fnmatch(e.name, pattern)]
        lines = []
        for e in entries:
            pfx = "[DIR]" if e.is_dir() else "[FILE]"
            sz = f" ({e.stat().st_size:,} B)" if e.is_file() else ""
            lines.append(f"  {pfx} {e.name}{sz}")
        return "\n".join(lines) or "(空)"

    async def _search_code(
        self, pattern: str, path: str = "", file_pattern: str = ""
    ) -> str:
        sp = self._resolve(path) if path else self.workspace
        if not sp.exists():
            return f"错误: {sp}"

        # 优先使用 ripgrep
        try:
            cmd = ["rg", "--line-number", "--max-count=50", pattern, str(sp)]
            if file_pattern:
                cmd.extend(["--glob", file_pattern])
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, _ = await asyncio.wait_for(proc.communicate(), 15.0)
            result = out.decode("utf-8", errors="replace").strip()
            return result or "未找到"
        except Exception:
            return self._fallback_search(pattern, sp, file_pattern)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
            try:
                resolved.relative_to(self.workspace)
            except ValueError:
                raise PermissionError(f"禁止访问工作空间外路径: {path}")
            return resolved
        return (self.workspace / p).resolve()

    @staticmethod
    def _validate(command: str) -> tuple:
        for pat in DANGEROUS_PATTERNS:
            if re.search(pat, command, re.IGNORECASE):
                return False, f"危险操作: {pat}"
        try:
            tokens = shlex.split(command)
            if not tokens:
                return False, "无法解析"
            base = os.path.basename(tokens[0])
        except ValueError:
            return False, "解析失败"
        if base not in ALLOWED_COMMANDS:
            return False, f"'{base}' 不在白名单"
        return True, ""

    @staticmethod
    def _fallback_search(pattern: str, path: Path, file_pattern: str) -> str:
        results = []
        for f in path.rglob("*"):
            if not f.is_file():
                continue
            if file_pattern and not fnmatch.fnmatch(f.name, file_pattern):
                continue
            if f.stat().st_size > 500_000:
                continue
            try:
                for i, line in enumerate(
                    f.read_text("utf-8", errors="replace").splitlines(), 1
                ):
                    if re.search(pattern, line, re.IGNORECASE):
                        results.append(f"{f}:{i}: {line.strip()}")
                        if len(results) >= 50:
                            break
            except Exception:
                continue
        return "\n".join(results) if results else "未找到"
