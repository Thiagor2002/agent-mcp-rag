#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置模块 - OpenAI 兼容 API 多提供商配置

支持硅基流动 (SiliconFlow)、MiniMax 等 OpenAI 兼容 API 提供商。
配置优先级: 代码传入 > 环境变量 > 默认值

安全原则:
- API Key 仅从环境变量读取，不写入配置文件
- 敏感信息在日志中自动掩码
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# 加载项目根目录的 .env 文件
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# 全局基础配置
# ---------------------------------------------------------------------------

class GlobalConfig:
    """全局配置，从环境变量加载"""

    def __init__(self):
        # API 提供商配置
        self.llm_config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_api_base": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "openai_org_id": os.getenv("OPENAI_ORG_ID", ""),

            # 硅基流动
            "siliconflow_api_key": os.getenv("SILICONFLOW_API_KEY", ""),
            "siliconflow_api_base": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),

            # MiniMax
            "minimax_api_key": os.getenv("MINIMAX_API_KEY", ""),
            "minimax_api_base": os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"),

            # 模型配置
            "default_model_name": os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            "coder_model_name": os.getenv("CODER_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")),
            "cheap_model_name": os.getenv("CHEAP_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5"),

            # 请求参数
            "max_tokens": int(os.getenv("MAX_TOKENS", "8192")),
            "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            "top_p": float(os.getenv("TOP_P", "1.0")),
            "timeout": int(os.getenv("TIMEOUT", "60")),
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "retry_delay": int(os.getenv("RETRY_DELAY", "2")),
            "stream": os.getenv("STREAM", "True").lower() in ("true", "1", "yes"),
            "max_context_tokens": int(os.getenv("MAX_CONTEXT_TOKENS", "20000")),
        }

    def get_llm_config(self) -> Dict[str, Any]:
        return self.llm_config

    def get_value(self, key: str, default: Any = None) -> Any:
        return self.llm_config.get(key, os.getenv(key, default))


config = GlobalConfig()


# ---------------------------------------------------------------------------
# Pydantic 配置模型
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """LLM 配置模型"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 模型信息
    model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct")
    provider: str = Field(default="openai")
    model_type: str = Field(default="openai")

    # API 配置
    api_key: Optional[str] = Field(default=None)
    api_base: str = Field(default="https://api.openai.com/v1")
    organization_id: Optional[str] = Field(default=None)

    # 请求配置
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)
    timeout: float = Field(default=30.0)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=2.0)
    stream: bool = Field(default=True)

    # 上下文窗口
    max_context_tokens: int = Field(default=20000)

    # 工具配置
    tools: Optional[List[Dict[str, Any]]] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, str]]] = Field(default=None)

    # 系统提示
    system_message: str = Field(default="")

    @field_validator('api_key', mode='before')
    @classmethod
    def resolve_api_key(cls, v, info):
        """自动从环境变量解析 API Key"""
        if v:
            return v
        provider = info.data.get('provider', 'openai')
        # 尝试多个环境变量名
        env_keys = {
            "openai": ["OPENAI_API_KEY", "SILICONFLOW_API_KEY", "MINIMAX_API_KEY"],
            "siliconflow": ["SILICONFLOW_API_KEY", "OPENAI_API_KEY"],
            "minimax": ["MINIMAX_API_KEY", "OPENAI_API_KEY"],
        }
        for env_key in env_keys.get(provider, ["OPENAI_API_KEY"]):
            val = os.getenv(env_key, "")
            if val:
                return val
        return config.get_value('openai_api_key', '')

    @field_validator('temperature', 'top_p')
    @classmethod
    def validate_range(cls, v, info):
        field_name = info.field_name
        if field_name == 'temperature' and (v < 0 or v > 2):
            raise ValueError(f"temperature 必须在 0-2 之间，当前值: {v}")
        if field_name == 'top_p' and (v <= 0 or v > 1):
            raise ValueError(f"top_p 必须在 0-1 之间，当前值: {v}")
        return v

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        result = super().model_dump(**kwargs)
        result['openai_api_key'] = self.api_key
        result['openai_api_base'] = self.api_base
        result['openai_org_id'] = self.organization_id
        return result


class MemoryConfig(BaseModel):
    """内存配置"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage_dir: str = Field(default="data/memory")
    max_memory_items: int = Field(default=100)
    enable_compression: bool = Field(default=False)
    max_history_states: int = Field(default=10)
    auto_save_interval: int = Field(default=5)

    @field_validator('storage_dir')
    @classmethod
    def ensure_dir(cls, v):
        os.makedirs(v, exist_ok=True)
        return v


class ToolConfig(BaseModel):
    """工具执行配置"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    timeout: float = Field(default=30.0)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    allow_file_access: bool = Field(default=True)
    allowed_domains: List[str] = Field(default_factory=list)
    max_output_length: int = Field(default=10000)
    enabled_tools: List[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Agent 配置"""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="Augmented LLM Agent")
    description: str = Field(default="增强型 LLM 智能体 (Chat + MCP + CRAG)")
    version: str = Field(default="1.0.0")

    max_steps: int = Field(default=10)
    timeout: float = Field(default=300.0)
    stuck_timeout: float = Field(default=60.0)
    max_stuck_count: int = Field(default=3)

    log_level: int = Field(default=logging.INFO)
    log_file: Optional[str] = Field(default=None)

    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)
    tool_config: ToolConfig = Field(default_factory=ToolConfig)

    input_schema: Optional[Dict[str, Any]] = Field(default=None)
    output_schema: Optional[Dict[str, Any]] = Field(default=None)

    @model_validator(mode='after')
    def check_timeouts(self):
        if self.timeout > 0 and self.stuck_timeout > 0 and self.stuck_timeout > self.timeout / 2:
            self.stuck_timeout = self.timeout / 2
        return self

    @classmethod
    def from_json(cls, json_file: str) -> 'AgentConfig':
        with open(json_file, 'r', encoding='utf-8') as f:
            return cls.model_validate(json.load(f))

    def to_json(self, json_file: str) -> None:
        os.makedirs(os.path.dirname(json_file) or ".", exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2)

    def from_global_config(self) -> 'AgentConfig':
        """从全局环境变量配置加载值"""
        gc = config.get_llm_config()
        self.llm_config = LLMConfig(
            model_name=gc.get("default_model_name", self.llm_config.model_name),
            api_key=gc.get("openai_api_key", self.llm_config.api_key),
            api_base=gc.get("openai_api_base", self.llm_config.api_base),
            organization_id=gc.get("openai_org_id", self.llm_config.organization_id),
            max_tokens=gc.get("max_tokens", self.llm_config.max_tokens),
            temperature=gc.get("temperature", self.llm_config.temperature),
            top_p=gc.get("top_p", self.llm_config.top_p),
            timeout=gc.get("timeout", self.llm_config.timeout),
            max_retries=gc.get("max_retries", self.llm_config.max_retries),
            retry_delay=gc.get("retry_delay", self.llm_config.retry_delay),
            stream=gc.get("stream", self.llm_config.stream),
            max_context_tokens=gc.get("max_context_tokens", self.llm_config.max_context_tokens),
        )
        return self


default_agent_config = AgentConfig()


def load_agent_config(config_path: Optional[str] = None) -> AgentConfig:
    """加载 Agent 配置"""
    agent_config = AgentConfig.model_validate(default_agent_config.model_dump())
    agent_config.from_global_config()
    if config_path and os.path.exists(config_path):
        file_config = AgentConfig.from_json(config_path)
        for field in file_config.model_fields:
            if hasattr(file_config, field):
                setattr(agent_config, field, getattr(file_config, field))
    return agent_config
