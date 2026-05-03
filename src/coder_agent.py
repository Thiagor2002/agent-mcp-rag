#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coder Agent - 专业代码处理 Agent
===============================

许多 LLM 对 CLI/Bash 代码进行了偏好微调，可以直接理解和生成
shell 命令而无需复杂的 function calling 描述。CoderAgent 利用
这一特性，使用纯文本 prompt 驱动代码任务。

功能:
- 代码生成: 根据需求生成 Python/Shell/其他语言代码
- 命令执行: 直接在本地执行生成的命令
- 文件操作: 读取/写入/搜索项目文件
- 代码审查: 对代码进行安全和质量检查

模型路由:
- 一般推理: 使用便宜模型 (CHEAP_MODEL)
- 代码任务: 使用专业代码模型 (CODER_MODEL)
"""

import os
import asyncio
import subprocess
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from .chat_openai import ChatOpenAI, ChatResponse
from .utils import log_title, log_section, truncate_text


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


class CoderAgent:
    """
    专业代码处理 Agent

    使用 LLM 直接生成并执行代码/命令，无需 function calling 格式。
    适用于代码生成、项目操作、文件管理等任务。

    Usage:
        agent = CoderAgent()
        result = await agent.generate_code("写一个快速排序函数")
        output = await agent.run_command("python -c 'print(1+1)'")
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        workspace: Optional[str] = None,
    ):
        self.workspace = Path(workspace or os.getcwd()).resolve()

        # 代码模型: 用于代码生成和命令推理
        self._coder = ChatOpenAI(
            api_key=api_key, base_url=base_url,
            system_prompt=self._coder_system_prompt(),
        )
        # 便宜模型: 用于一般任务
        self._cheap = ChatOpenAI(
            api_key=api_key, base_url=base_url,
            system_prompt="你是一个有用的编程助手。",
        )

    # ------------------------------------------------------------------
    # 代码生成
    # ------------------------------------------------------------------

    async def generate_code(
        self, task: str, language: str = "python"
    ) -> str:
        """根据任务描述生成代码"""
        prompt = self._code_gen_prompt(task, language)
        response = await self._coder.chat_code(prompt)
        return self._extract_code_block(response.content, language)

    async def explain_code(self, code: str) -> str:
        """解释代码功能"""
        response = await self._coder.chat_code(
            f"请详细解释以下代码的功能和逻辑:\n```\n{code}\n```"
        )
        return response.content

    async def review_code(self, code: str) -> str:
        """代码审查"""
        response = await self._coder.chat_code(
            f"请审查以下代码的安全性、性能和可读性，给出改进建议:\n```\n{code}\n```"
        )
        return response.content

    # ------------------------------------------------------------------
    # 命令执行
    # ------------------------------------------------------------------

    async def run_command(self, command: str, working_dir: str = "") -> str:
        """安全地执行 shell 命令"""
        command = command.strip()
        if not command:
            return "错误: 空命令"

        is_ok, reason = self._validate_command(command)
        if not is_ok:
            return f"安全限制: {reason}"

        cwd = self.workspace / working_dir if working_dir else self.workspace
        if not cwd.exists():
            return f"错误: 目录不存在 {cwd}"

        print(f"  $ {command}")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n[STDERR]\n" + stderr.decode("utf-8", errors="replace")
            return truncate_text(output.strip() or "(无输出)", 10000)
        except asyncio.TimeoutError:
            return "错误: 命令超时 (30s)"
        except Exception as e:
            return f"错误: {e}"

    async def smart_execute(self, task: str) -> str:
        """
        智能执行: 让 LLM 理解任务并生成命令，然后执行

        LLM 对 CLI 代码进行了微调，可以无需 function calling 描述
        直接生成正确的 shell 命令。
        """
        prompt = f"""你需要完成以下任务。请生成可直接执行的 shell 命令。

任务: {task}

工作目录: {self.workspace}
操作系统: {os.name}

请用 ```bash 代码块输出要执行的命令，一行一个命令。
注意: 只生成安全的命令，不要使用 rm -rf / sudo 等危险操作。
"""
        response = await self._coder.chat_code(prompt)
        commands = self._extract_commands(response.content)

        results = []
        for cmd in commands:
            result = await self.run_command(cmd)
            results.append(f"$ {cmd}\n{result}")

        return "\n\n".join(results)

    # ------------------------------------------------------------------
    # 文件操作
    # ------------------------------------------------------------------

    async def read_file(self, path: str, start: int = 0, end: int = 0) -> str:
        file_path = self._resolve_path(path)
        if not file_path.exists():
            return f"错误: 文件不存在 {file_path}"
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
            s = max(0, start - 1) if start > 0 else 0
            e = min(len(lines), end) if end > 0 else len(lines)
            return "\n".join(f"{i+s+1:4d}| {l}" for i, l in enumerate(lines[s:e]))
        except UnicodeDecodeError:
            return "错误: 无法解码（可能为二进制文件）"

    async def write_file(self, path: str, content: str) -> str:
        file_path = self._resolve_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"已写入: {file_path} ({len(content)} 字符)"

    async def list_dir(self, path: str = "", pattern: str = "") -> str:
        import fnmatch
        d = self._resolve_path(path) if path else self.workspace
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

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _coder_system_prompt() -> str:
        return """你是一个专业的编码助手，擅长:
1. 根据需求生成高质量代码
2. 分析、解释和审查代码
3. 生成安全的 shell 命令
4. 解答编程相关问题

请用清晰的代码示例回答，不要过度解释。"""

    @staticmethod
    def _code_gen_prompt(task: str, language: str) -> str:
        return f"""请根据以下需求生成 {language} 代码:

{task}

要求:
- 代码要有完善的注释
- 包含必要的错误处理
- 遵循 {language} 最佳实践

请用 ```{language} 代码块输出。"""

    @staticmethod
    def _extract_code_block(text: str, lang: str = "") -> str:
        """从 LLM 回复中提取代码块"""
        pattern = rf"```(?:{lang})?\s*\n(.*?)```" if lang else r"```\w*\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()

    @staticmethod
    def _extract_commands(text: str) -> List[str]:
        """从 LLM 回复中提取 shell 命令"""
        # 提取 bash 代码块
        pattern = r"```(?:bash|sh|shell)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            cmds = []
            for block in matches:
                for line in block.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("//"):
                        cmds.append(line)
            return cmds
        # 提取以 $ 开头的行
        dollar_cmds = re.findall(r"^\$\s+(.+)$", text, re.MULTILINE)
        return [c.strip() for c in dollar_cmds]

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            resolved = p.resolve()
            try:
                resolved.relative_to(self.workspace)
            except ValueError:
                raise PermissionError(f"禁止访问工作空间外的路径: {path}")
            return resolved
        return (self.workspace / p).resolve()

    @staticmethod
    def _validate_command(command: str) -> tuple:
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"危险操作匹配: {pattern}"
        try:
            import shlex
            tokens = shlex.split(command)
            if not tokens:
                return False, "无法解析命令"
            base = os.path.basename(tokens[0])
        except ValueError:
            return False, "命令解析失败"
        if base not in ALLOWED_COMMANDS:
            return False, f"'{base}' 不在白名单中"
        return True, ""
