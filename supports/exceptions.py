#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent系统异常模块
定义了Agent系统中使用的各种自定义异常
"""

from typing import Dict, List, Any, Optional


class BaseAgentError(Exception):
    """Agent基础异常类，所有Agent异常的父类"""
    
    def __init__(self, message: str, error_code: str = "AGENT_ERROR", 
                 details: Optional[Dict[str, Any]] = None):
        """
        初始化基础异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 错误详情
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """异常字符串表示"""
        return f"{self.error_code}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """将异常转换为字典表示，便于序列化"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class AgentTimeoutError(BaseAgentError):
    """
    Agent超时异常
    
    当Agent执行时间超过设定的超时限制时抛出
    """
    
    def __init__(self, message: str, timeout_seconds: float = 0, 
                 details: Optional[Dict[str, Any]] = None):
        """
        初始化超时异常
        
        Args:
            message: 错误消息
            timeout_seconds: 超时秒数
            details: 错误详情
        """
        details = details or {}
        details["timeout_seconds"] = timeout_seconds
        
        super().__init__(
            message=message,
            error_code="AGENT_TIMEOUT",
            details=details
        )
        self.timeout_seconds = timeout_seconds


class InvalidInputError(BaseAgentError):
    """
    无效输入异常
    
    当Agent接收到无效或不符合要求的输入数据时抛出
    """
    
    def __init__(self, message: str, errors: Optional[Dict[str, List[str]]] = None, 
                 input_data: Any = None):
        """
        初始化无效输入异常
        
        Args:
            message: 错误消息
            errors: 字段错误信息，格式为 {字段名: [错误消息列表]}
            input_data: 原始输入数据
        """
        details = {
            "errors": errors or {}
        }
        
        # 只在调试模式下包含输入数据
        if input_data is not None:
            try:
                # 尝试将输入数据转换为可序列化的格式
                import json
                json.dumps(input_data)
                details["input_data"] = input_data
            except (TypeError, ValueError):
                # 如果无法序列化，则只包含类型信息
                details["input_data_type"] = str(type(input_data))
        
        super().__init__(
            message=message,
            error_code="INVALID_INPUT",
            details=details
        )
        self.errors = errors or {}


class AgentStuckError(BaseAgentError):
    """
    Agent卡住异常
    
    当Agent在一段时间内没有取得任何进展时抛出
    
    注意: 此异常暂时不实现完整功能
    """
    
    def __init__(self, message: str = "Agent已卡住且无法继续"):
        """
        初始化卡住异常
        
        Args:
            message: 错误消息
        """
        super().__init__(
            message=message,
            error_code="AGENT_STUCK"
        )


# 以下是其他可能会用到的异常，根据需要扩展

class ToolExecutionError(BaseAgentError):
    """工具执行异常，当Agent执行工具时发生错误"""
    
    def __init__(self, message: str, tool_name: str, 
                 args: Optional[Dict[str, Any]] = None, 
                 cause: Optional[Exception] = None):
        """
        初始化工具执行异常
        
        Args:
            message: 错误消息
            tool_name: 工具名称
            args: 工具参数
            cause: 原始异常
        """
        details = {
            "tool_name": tool_name,
            "args": args or {}
        }
        
        if cause:
            details["cause"] = str(cause)
        
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            details=details
        )
        self.tool_name = tool_name
        self.args = args or {}
        self.cause = cause


class MemoryAccessError(BaseAgentError):
    """内存访问异常，当Agent访问内存时发生错误"""

    def __init__(self, message: str, operation: str,
                 agent_id: Optional[str] = None):
        """
        初始化内存访问异常

        Args:
            message: 错误消息
            operation: 执行的操作
            agent_id: Agent ID
        """
        details = {
            "operation": operation
        }

        if agent_id:
            details["agent_id"] = agent_id

        super().__init__(
            message=message,
            error_code="MEMORY_ACCESS_ERROR",
            details=details
        )
        self.operation = operation
        self.agent_id = agent_id


class ToolError(BaseAgentError):
    """工具相关异常，当工具注册、查找或执行过程中发生错误时抛出"""

    def __init__(self, message: str, tool_name: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        初始化工具异常

        Args:
            message: 错误消息
            tool_name: 工具名称
            details: 错误详情
        """
        details = details or {}
        if tool_name:
            details["tool_name"] = tool_name

        super().__init__(
            message=message,
            error_code="TOOL_ERROR",
            details=details
        )
        self.tool_name = tool_name