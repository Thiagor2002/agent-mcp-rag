#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM接口实现
提供对多种大语言模型的统一访问接口和token计算功能
"""

import os
import json
import time
import logging
import requests
import tiktoken
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from .logger import logger
from .config import config
# 加载.env文件中的环境变量
load_dotenv()

class TokenCounter:
    """
    Token计算类
    负责计算文本的token数量，并检查是否超出最大限制
    """
    
    # 不同模型的编码器映射
    ENCODERS = {
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "claude-3-opus": "cl100k_base",  # 近似值
        "claude-3-sonnet": "cl100k_base",  # 近似值
        "claude-3-haiku": "cl100k_base",  # 近似值
        "gemini-pro": "cl100k_base",  # 近似值
        "llama-2": "cl100k_base",  # 近似值
        "mistral": "cl100k_base",  # 近似值
    }
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        初始化Token计算器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.encoder_name = self.ENCODERS.get(model_name, "cl100k_base")
        
        try:
            self.encoder = tiktoken.get_encoding(self.encoder_name)
        except Exception as e:
            logging.warning(f"无法加载编码器 {self.encoder_name}，使用默认编码器: {e}")
            self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        计算文本的token数量
        
        Args:
            text: 要计算的文本
            
        Returns:
            token数量
        """
        if not text:
            return 0
            
        return len(self.encoder.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        计算消息列表的token数量
        适用于OpenAI格式的消息列表
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "Hello"}, ...]
            
        Returns:
            token数量
        """
        if not messages:
            return 0
            
        # 基础token数（每条消息的开销）
        num_tokens = 0
        
        # 添加每条消息的token数
        for message in messages:
            # 每条消息有固定开销
            num_tokens += 4  # 消息格式开销
            
            # 添加每个字段的token数
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += self.count_tokens(value)
                    
                    # 字段名称的token数
                    num_tokens += 1  # 字段名称开销
            
            # 消息结束标记
            num_tokens += 2  # 消息结束开销
        
        # 整体开销
        num_tokens += 2  # 整体开销
        
        return num_tokens
    
    def check_tokens_limit(self, text: str, max_tokens: int) -> bool:
        """
        检查文本是否超出token限制
        
        Args:
            text: 要检查的文本
            max_tokens: 最大token数量
            
        Returns:
            是否在限制内
        """
        return self.count_tokens(text) <= max_tokens
    
    def truncate_text_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        将文本截断到token限制内
        
        Args:
            text: 要截断的文本
            max_tokens: 最大token数量
            
        Returns:
            截断后的文本
        """
        if not text:
            return ""
            
        tokens = self.encoder.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
            
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens)
    
    def truncate_messages_to_token_limit(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int,
        preserve_latest: bool = True
    ) -> List[Dict[str, str]]:
        """
        将消息列表截断到token限制内
        
        Args:
            messages: 消息列表
            max_tokens: 最大token数量
            preserve_latest: 是否优先保留最新的消息
            
        Returns:
            截断后的消息列表
        """
        if not messages:
            return []
            
        current_tokens = self.count_messages_tokens(messages)
        
        if current_tokens <= max_tokens:
            return messages
        
        # 需要截断
        result_messages = messages.copy()
        
        if preserve_latest:
            # 从前往后移除消息
            while result_messages and self.count_messages_tokens(result_messages) > max_tokens:
                result_messages.pop(0)
        else:
            # 从后往前移除消息
            while result_messages and self.count_messages_tokens(result_messages) > max_tokens:
                result_messages.pop()
        
        # 如果仍然超出限制，尝试截断最早的消息内容
        if result_messages and self.count_messages_tokens(result_messages) > max_tokens:
            first_message = result_messages[0]
            content = first_message.get("content", "")
            
            # 计算需要保留的token数
            other_messages_tokens = self.count_messages_tokens(result_messages[1:])
            remaining_tokens = max_tokens - other_messages_tokens - 5  # 5是消息格式开销
            
            if remaining_tokens > 0:
                truncated_content = self.truncate_text_to_token_limit(content, remaining_tokens)
                result_messages[0]["content"] = truncated_content
            else:
                # 如果其他消息已经超出限制，移除第一条消息
                result_messages.pop(0)
        
        return result_messages


class ModelType(str, Enum):
    """模型类型枚举"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """LLM配置数据类"""
    
    # 模型信息
    model_type: ModelType = ModelType.OPENAI
    model_name: str = None
    
    # API配置
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    org_id: Optional[str] = None
    
    # 请求配置
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 15.0
    
    # 重试配置
    max_retries: int = 3
    retry_delay: int = 2
    
    # 流式输出配置
    stream: bool = False
    
    # 上下文窗口配置
    max_context_tokens: int = 20000
    
    # 工具配置
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, str]]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量加载配置
        self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载配置"""
        # 导入配置模块
        try:
            llm_config = config.get_llm_config()
            
            # 根据模型类型加载API密钥
            if not self.api_key:
                if self.model_type == ModelType.OPENAI:
                    self.api_key = llm_config.get("openai_api_key")
                elif self.model_type == ModelType.AZURE_OPENAI:
                    self.api_key = llm_config.get("azure_openai_api_key")
                elif self.model_type == ModelType.ANTHROPIC:
                    self.api_key = llm_config.get("anthropic_api_key")
                elif self.model_type == ModelType.GOOGLE:
                    self.api_key = llm_config.get("google_api_key")
                elif self.model_type == ModelType.HUGGINGFACE:
                    self.api_key = llm_config.get("huggingface_api_key")
            
            # 加载基础URL
            if not self.base_url:
                if self.model_type == ModelType.OPENAI:
                    self.base_url = llm_config.get("openai_api_base")
                elif self.model_type == ModelType.AZURE_OPENAI:
                    self.base_url = llm_config.get("azure_openai_api_base")
                elif self.model_type == ModelType.ANTHROPIC:
                    self.base_url = llm_config.get("anthropic_api_base")
                elif self.model_type == ModelType.GOOGLE:
                    self.base_url = llm_config.get("google_api_base")
                elif self.model_type == ModelType.HUGGINGFACE:
                    self.base_url = llm_config.get("huggingface_api_base")
            
            # 加载组织ID
            if not self.org_id and self.model_type == ModelType.OPENAI:
                self.org_id = llm_config.get("openai_org_id")
                
            # 加载其他配置项
            if llm_config.get("max_tokens") is not None:
                self.max_tokens = llm_config.get("max_tokens")
            if llm_config.get("temperature") is not None:
                self.temperature = llm_config.get("temperature")
            if llm_config.get("top_p") is not None:
                self.top_p = llm_config.get("top_p")
            if llm_config.get("frequency_penalty") is not None:
                self.frequency_penalty = llm_config.get("frequency_penalty")
            if llm_config.get("presence_penalty") is not None:
                self.presence_penalty = llm_config.get("presence_penalty")
            if llm_config.get("timeout") is not None:
                self.timeout = llm_config.get("timeout")
            if llm_config.get("max_retries") is not None:
                self.max_retries = llm_config.get("max_retries")
            if llm_config.get("retry_delay") is not None:
                self.retry_delay = llm_config.get("retry_delay")
            if llm_config.get("stream") is not None:
                self.stream = llm_config.get("stream")
            if llm_config.get("max_context_tokens") is not None:
                self.max_context_tokens = llm_config.get("max_context_tokens")
            if llm_config.get("tools") is not None:
                self.tools = llm_config.get("tools")
            if llm_config.get("tool_choice") is not None:
                self.tool_choice = llm_config.get("tool_choice")
            if llm_config.get("default_model_name") is not None:
                self.model_name = llm_config.get("default_model_name")
                
        except ImportError:
            # 如果无法导入配置模块，则回退到从环境变量直接加载
            # 加载API密钥
            if not self.api_key:
                if self.model_type == ModelType.OPENAI:
                    self.api_key = os.environ.get("OPENAI_API_KEY")
                elif self.model_type == ModelType.AZURE_OPENAI:
                    self.api_key = os.environ.get("AZURE_OPENAI_API_KEY")
                elif self.model_type == ModelType.ANTHROPIC:
                    self.api_key = os.environ.get("ANTHROPIC_API_KEY")
                elif self.model_type == ModelType.GOOGLE:
                    self.api_key = os.environ.get("GOOGLE_API_KEY")
                elif self.model_type == ModelType.HUGGINGFACE:
                    self.api_key = os.environ.get("HUGGINGFACE_API_KEY")
            
            # 加载基础URL
            if not self.base_url:
                if self.model_type == ModelType.OPENAI:
                    self.base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
                elif self.model_type == ModelType.AZURE_OPENAI:
                    self.base_url = os.environ.get("AZURE_OPENAI_API_BASE")
                elif self.model_type == ModelType.ANTHROPIC:
                    self.base_url = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com")
                elif self.model_type == ModelType.GOOGLE:
                    self.base_url = os.environ.get("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com")
                elif self.model_type == ModelType.HUGGINGFACE:
                    self.base_url = os.environ.get("HUGGINGFACE_API_BASE", "https://api-inference.huggingface.co/models")
            
            # 加载组织ID
            if not self.org_id and self.model_type == ModelType.OPENAI:
                self.org_id = os.environ.get("OPENAI_ORG_ID")
                
            # 加载其他配置项
            if os.environ.get("OPENAI_MAX_TOKENS") is not None:
                self.max_tokens = int(os.environ.get("OPENAI_MAX_TOKENS"))
            if os.environ.get("OPENAI_TEMPERATURE") is not None:
                self.temperature = float(os.environ.get("OPENAI_TEMPERATURE"))
            if os.environ.get("OPENAI_TOP_P") is not None:
                self.top_p = float(os.environ.get("OPENAI_TOP_P"))
            if os.environ.get("OPENAI_FREQUENCY_PENALTY") is not None:
                self.frequency_penalty = float(os.environ.get("OPENAI_FREQUENCY_PENALTY"))
            if os.environ.get("OPENAI_PRESENCE_PENALTY") is not None:
                self.presence_penalty = float(os.environ.get("OPENAI_PRESENCE_PENALTY"))
            if os.environ.get("OPENAI_TIMEOUT") is not None:
                self.timeout = int(os.environ.get("OPENAI_TIMEOUT"))
            if os.environ.get("OPENAI_MAX_RETRIES") is not None:
                self.max_retries = int(os.environ.get("OPENAI_MAX_RETRIES"))
            if os.environ.get("OPENAI_RETRY_DELAY") is not None:
                self.retry_delay = int(os.environ.get("OPENAI_RETRY_DELAY"))
            if os.environ.get("OPENAI_STREAM") is not None:
                self.stream = os.environ.get("OPENAI_STREAM").lower() == "true"
            if os.environ.get("OPENAI_MAX_CONTEXT_TOKENS") is not None:
                self.max_context_tokens = int(os.environ.get("OPENAI_MAX_CONTEXT_TOKENS"))
            if os.environ.get("OPENAI_TOOLS") is not None:
                try:
                    self.tools = json.loads(os.environ.get("OPENAI_TOOLS"))
                except json.JSONDecodeError:
                    logger.warning("无法解析OPENAI_TOOLS环境变量，应为有效的JSON格式")
            if os.environ.get("OPENAI_TOOL_CHOICE") is not None:
                try:
                    tool_choice = os.environ.get("OPENAI_TOOL_CHOICE")
                    if tool_choice.startswith("{") and tool_choice.endswith("}"):
                        self.tool_choice = json.loads(tool_choice)
                    else:
                        self.tool_choice = tool_choice
                except json.JSONDecodeError:
                    logger.warning("无法解析OPENAI_TOOL_CHOICE环境变量，应为有效的JSON格式或字符串")


class LLMInterface:
    """
    LLM接口类
    提供对多种大语言模型的统一访问接口
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化LLM接口
        
        Args:
            config: LLM配置
        """
        self.config = config or LLMConfig()
        self.token_counter = TokenCounter(self.config.model_name)
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置"""
        if not self.config.api_key:
            logger.warning(f"未设置API密钥，模型类型: {self.config.model_type}")
        
        if not self.config.base_url:
            logger.warning(f"未设置基础URL，模型类型: {self.config.model_type}")
    
    def generate(
        self, 
        prompt: str, 
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成文本
        
        Args:
            prompt: 提示文本
            context: 上下文信息
                [{"role":xxx,"content":xxx}, {"role":xxx, "content":xxx}]
            **kwargs: 其他参数，可包括：
                - tools: 工具列表
                - tool_choice: 工具选择
                - 其他模型特定参数
            
        Returns:
            生成结果
        """

        messages = self._prepare_messages(prompt, context)
        # 检查token数量
        tokens_count = self.token_counter.count_messages_tokens(messages)
        max_context_tokens = self.config.max_context_tokens
        
        if tokens_count > max_context_tokens:
            logger.warning(
                f"消息token数量 ({tokens_count}) 超出最大上下文窗口 ({max_context_tokens})，将进行截断"
            )
            messages = self.token_counter.truncate_messages_to_token_limit(
                messages, max_context_tokens - self.config.max_tokens
            )
            
        # 处理工具参数
        if "tools" in kwargs and kwargs["tools"] is not None:
            self.config.tools = kwargs.pop("tools")
        if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
            self.config.tool_choice = kwargs.pop("tool_choice")
            
        # 根据模型类型选择不同的实现
        if self.config.model_type == ModelType.OPENAI:
            return self._generate_openai(messages, **kwargs)
        elif self.config.model_type == ModelType.AZURE_OPENAI:
            return self._generate_azure_openai(messages, **kwargs)
        elif self.config.model_type == ModelType.ANTHROPIC:
            return self._generate_anthropic(messages, **kwargs)
        elif self.config.model_type == ModelType.GOOGLE:
            return self._generate_google(messages, **kwargs)
        elif self.config.model_type == ModelType.HUGGINGFACE:
            return self._generate_huggingface(messages, **kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {self.config.model_type}")
    
    def _prepare_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        准备消息列表

        支持两种输入模式:
        1. context 为消息列表 (List[Dict]): 直接使用并追加当前 prompt
        2. context 为字典: 从 history / messages 等字段构建消息列表

        Args:
            prompt: 提示文本
            context: 上下文信息

        Returns:
            消息列表
        """
        # 模式1: context 已经是消息列表
        if isinstance(context, list):
            messages = list(context)
            if prompt:
                messages.append({"role": "user", "content": prompt})
            return messages

        # 模式2: context 是字典，包含 messages 或 history
        if context and isinstance(context, dict):
            if "messages" in context:
                messages = list(context["messages"])
                if prompt and messages and isinstance(messages[-1], dict) \
                   and messages[-1].get("role") == "user":
                    messages[-1]["content"] = prompt
                elif prompt:
                    messages.append({"role": "user", "content": prompt})
                return messages

            if "history" in context:
                system_message = context.get("system_message", "你是一个有用的AI助手。")
                messages = [{"role": "system", "content": system_message}]
                for entry in context["history"]:
                    if "user" in entry:
                        messages.append({"role": "user", "content": entry["user"]})
                    if "assistant" in entry:
                        messages.append({"role": "assistant", "content": entry["assistant"]})
                if prompt:
                    messages.append({"role": "user", "content": prompt})
                return messages

        # 模式3: 无上下文，构建最简消息列表
        messages = [
            {"role": "system", "content": "你是一个有用的AI助手。"},
        ]
        if prompt:
            messages.append({"role": "user", "content": prompt})
        return messages
    
    def _generate_openai(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用OpenAI生成文本
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数，可包括：
                - tools: 工具列表，例如：
                    [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "获取指定城市的天气信息",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {
                                            "type": "string",
                                            "description": "城市名称"
                                        },
                                        "unit": {
                                            "type": "string",
                                            "enum": ["celsius", "fahrenheit"],
                                            "description": "温度单位"
                                        }
                                    },
                                    "required": ["location"]
                                }
                            }
                        }
                    ]
                - tool_choice: 工具选择，可以是：
                    - "none": 不使用任何工具
                    - "auto": 自动选择工具
                    - 特定工具: {"type": "function", "function": {"name": "get_weather"}}
            
        Returns:
            生成结果，如果模型调用了工具，结果中会包含tool_calls字段
        """
        import openai
        
        # 设置API密钥和基础URL
        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            organization=self.config.org_id,
            timeout=self.config.timeout
        )
        
        # 合并配置和额外参数
        params = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stream": self.config.stream
        }
        
        # 添加工具配置（如果存在）
        if self.config.tools is not None:
            params["tools"] = self.config.tools
        if self.config.tool_choice is not None:
            params["tool_choice"] = self.config.tool_choice
        
        # 更新额外参数
        params.update({k: v for k, v in kwargs.items() if v is not None})
        
        # 记录请求
        logger.debug(f"OpenAI请求: {json.dumps(params, ensure_ascii=False)}")
        
        # 发送请求
        for retry in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                
                if params.get("stream", False):
                    # 流式响应
                    response_stream = client.chat.completions.create(**params)

                    # 收集流式响应
                    collected_content = ""
                    for chunk in response_stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            collected_content += chunk.choices[0].delta.content
                    
                    # 构造类似非流式响应的结构
                    response = {
                        "id": f"stream-{int(time.time())}",
                        "model": self.config.model_name,
                        "content": collected_content,
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": self.token_counter.count_messages_tokens(messages),
                            "completion_tokens": self.token_counter.count_tokens(collected_content),
                            "total_tokens": (
                                self.token_counter.count_messages_tokens(messages) + 
                                self.token_counter.count_tokens(collected_content)
                            )
                        }
                    }
                else:
                    # 非流式响应
                    response_obj = client.chat.completions.create(**params)
                    # 转换为字典
                    response = {
                        "id": response_obj.id,
                        "model": response_obj.model,
                        "content": response_obj.choices[0].message.content,
                        "finish_reason": response_obj.choices[0].finish_reason,
                        "usage": {
                            "prompt_tokens": response_obj.usage.prompt_tokens,
                            "completion_tokens": response_obj.usage.completion_tokens,
                            "total_tokens": response_obj.usage.total_tokens
                        }
                    }
                    
                    # 如果响应包含工具调用，添加到响应中
                    if hasattr(response_obj.choices[0].message, "tool_calls") and response_obj.choices[0].message.tool_calls:
                        response["tool_calls"] = [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                            for tool_call in response_obj.choices[0].message.tool_calls
                        ]
                
                elapsed = time.time() - start_time
                logger.debug(f"OpenAI响应耗时: {elapsed:.2f}秒")
                from pprint import pprint
                pprint(response)
                return response
                
            except Exception as e:
                if retry < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry)  # 指数退避
                    logger.warning(f"OpenAI请求失败 ({retry+1}/{self.config.max_retries}): {e}，将在 {delay} 秒后重试")
                    time.sleep(delay)
                else:
                    logger.error(f"OpenAI请求失败，已达到最大重试次数: {e}")
                    raise
    
    def _generate_azure_openai(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用Azure OpenAI生成文本
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            生成结果
        """
        import openai
        
        # 设置API密钥和基础URL
        client = openai.AzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self.config.base_url,
            api_version="2023-05-15",  # Azure OpenAI API版本
            timeout=self.config.timeout
        )
        
        # 合并配置和额外参数
        params = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stream": self.config.stream
        }
        
        # 更新额外参数
        params.update({k: v for k, v in kwargs.items() if v is not None})
        
        # 记录请求
        logger.debug(f"Azure OpenAI请求: {json.dumps(params, ensure_ascii=False)}")
        
        # 发送请求
        for retry in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                
                if params.get("stream", False):
                    # 流式响应
                    response_stream = client.chat.completions.create(**params)
                    
                    # 收集流式响应
                    collected_content = ""
                    for chunk in response_stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            collected_content += chunk.choices[0].delta.content
                    
                    # 构造类似非流式响应的结构
                    response = {
                        "id": f"stream-{int(time.time())}",
                        "model": self.config.model_name,
                        "content": collected_content,
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": self.token_counter.count_messages_tokens(messages),
                            "completion_tokens": self.token_counter.count_tokens(collected_content),
                            "total_tokens": (
                                self.token_counter.count_messages_tokens(messages) + 
                                self.token_counter.count_tokens(collected_content)
                            )
                        }
                    }
                else:
                    # 非流式响应
                    response_obj = client.chat.completions.create(**params)
                    
                    # 转换为字典
                    response = {
                        "id": response_obj.id,
                        "model": response_obj.model,
                        "content": response_obj.choices[0].message.content,
                        "finish_reason": response_obj.choices[0].finish_reason,
                        "usage": {
                            "prompt_tokens": response_obj.usage.prompt_tokens,
                            "completion_tokens": response_obj.usage.completion_tokens,
                            "total_tokens": response_obj.usage.total_tokens
                        }
                    }
                
                elapsed = time.time() - start_time
                logger.debug(f"Azure OpenAI响应耗时: {elapsed:.2f}秒")
                
                return response
                
            except Exception as e:
                if retry < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry)  # 指数退避
                    logger.warning(f"Azure OpenAI请求失败 ({retry+1}/{self.config.max_retries}): {e}，将在 {delay} 秒后重试")
                    time.sleep(delay)
                else:
                    logger.error(f"Azure OpenAI请求失败，已达到最大重试次数: {e}")
                    raise
    
    def _generate_anthropic(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用Anthropic生成文本
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            生成结果
        """
        import anthropic
        
        # 设置API密钥
        client = anthropic.Anthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url or "https://api.anthropic.com",
            timeout=self.config.timeout
        )
        
        # 转换OpenAI格式的消息为Anthropic格式
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})
        
        # 合并配置和额外参数
        params = {
            "model": self.config.model_name,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": self.config.stream
        }
        
        # 添加系统消息
        if system_message:
            params["system"] = system_message
        
        # 更新额外参数
        params.update({k: v for k, v in kwargs.items() if v is not None})
        
        # 记录请求
        logger.debug(f"Anthropic请求: {json.dumps(params, ensure_ascii=False)}")
        
        # 发送请求
        for retry in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                
                if params.get("stream", False):
                    # 流式响应
                    response_stream = client.messages.create(**params)
                    
                    # 收集流式响应
                    collected_content = ""
                    for chunk in response_stream:
                        if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                            collected_content += chunk.delta.text
                    
                    # 构造类似非流式响应的结构
                    response = {
                        "id": f"stream-{int(time.time())}",
                        "model": self.config.model_name,
                        "content": collected_content,
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": self.token_counter.count_messages_tokens(messages),
                            "completion_tokens": self.token_counter.count_tokens(collected_content),
                            "total_tokens": (
                                self.token_counter.count_messages_tokens(messages) + 
                                self.token_counter.count_tokens(collected_content)
                            )
                        }
                    }
                else:
                    # 非流式响应
                    response_obj = client.messages.create(**params)
                    
                    # 转换为字典
                    response = {
                        "id": response_obj.id,
                        "model": response_obj.model,
                        "content": response_obj.content[0].text,
                        "finish_reason": "stop",  # Anthropic不提供finish_reason
                        "usage": {
                            "prompt_tokens": self.token_counter.count_messages_tokens(messages),
                            "completion_tokens": self.token_counter.count_tokens(response_obj.content[0].text),
                            "total_tokens": (
                                self.token_counter.count_messages_tokens(messages) + 
                                self.token_counter.count_tokens(response_obj.content[0].text)
                            )
                        }
                    }
                
                elapsed = time.time() - start_time
                logger.debug(f"Anthropic响应耗时: {elapsed:.2f}秒")
                
                return response
                
            except Exception as e:
                if retry < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry)  # 指数退避
                    logger.warning(f"Anthropic请求失败 ({retry+1}/{self.config.max_retries}): {e}，将在 {delay} 秒后重试")
                    time.sleep(delay)
                else:
                    logger.error(f"Anthropic请求失败，已达到最大重试次数: {e}")
                    raise
    
    def _generate_google(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用Google Gemini生成文本
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            生成结果
        """
        import google.generativeai as genai
        
        # 设置API密钥
        genai.configure(api_key=self.config.api_key)
        
        # 转换OpenAI格式的消息为Google格式
        google_messages = []
        
        for msg in messages:
            if msg["role"] == "user":
                google_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                google_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "system":
                # Google API目前没有系统消息，将其作为用户消息添加
                google_messages.append({"role": "user", "parts": [{"text": f"System: {msg['content']}"}]})
        
        # 创建模型并生成
        model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_output_tokens": self.config.max_tokens,
            }
        )

        # 发送请求
        for retry in range(self.config.max_retries + 1):
            try:
                start_time = time.time()

                if self.config.stream:
                    response_stream = model.generate_content(
                        google_messages,
                        stream=True,
                    )
                    collected_content = ""
                    for chunk in response_stream:
                        if chunk.text:
                            collected_content += chunk.text
                    response = {
                        "id": f"gemini-{int(time.time())}",
                        "model": self.config.model_name,
                        "content": collected_content,
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": self.token_counter.count_messages_tokens(messages),
                            "completion_tokens": self.token_counter.count_tokens(collected_content),
                            "total_tokens": (
                                self.token_counter.count_messages_tokens(messages) +
                                self.token_counter.count_tokens(collected_content)
                            )
                        }
                    }
                else:
                    response_obj = model.generate_content(google_messages)
                    response = {
                        "id": f"gemini-{int(time.time())}",
                        "model": self.config.model_name,
                        "content": response_obj.text,
                        "finish_reason": "stop",
                        "usage": {
                            "prompt_tokens": self.token_counter.count_messages_tokens(messages),
                            "completion_tokens": self.token_counter.count_tokens(response_obj.text),
                            "total_tokens": (
                                self.token_counter.count_messages_tokens(messages) +
                                self.token_counter.count_tokens(response_obj.text)
                            )
                        }
                    }

                elapsed = time.time() - start_time
                logger.debug(f"Google 响应耗时: {elapsed:.2f}秒")
                return response

            except Exception as e:
                if retry < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry)
                    logger.warning(
                        f"Google 请求失败 ({retry+1}/{self.config.max_retries}): {e}，"
                        f"将在 {delay} 秒后重试"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Google 请求失败，已达到最大重试次数: {e}")
                    raise

    def _generate_huggingface(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用 HuggingFace 推理 API 生成文本

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            生成结果
        """
        url = f"{self.config.base_url}/{self.config.model_name}"

        # 将消息转换为 HuggingFace 格式的 prompt
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"Human: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
        }

        for retry in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                response_obj = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                )
                response_obj.raise_for_status()
                data = response_obj.json()

                content = data[0].get("generated_text", "") if isinstance(data, list) else str(data)
                elapsed = time.time() - start_time
                logger.debug(f"HuggingFace 响应耗时: {elapsed:.2f}秒")

                return {
                    "id": f"hf-{int(time.time())}",
                    "model": self.config.model_name,
                    "content": content,
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": self.token_counter.count_tokens(prompt),
                        "completion_tokens": self.token_counter.count_tokens(content),
                        "total_tokens": (
                            self.token_counter.count_tokens(prompt) +
                            self.token_counter.count_tokens(content)
                        )
                    }
                }

            except Exception as e:
                if retry < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry)
                    logger.warning(
                        f"HuggingFace 请求失败 ({retry+1}/{self.config.max_retries}): {e}，"
                        f"将在 {delay} 秒后重试"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"HuggingFace 请求失败，已达到最大重试次数: {e}")
                    raise
