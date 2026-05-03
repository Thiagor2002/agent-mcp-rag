#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具基类模块
定义所有工具的基类和公共功能，包括：
- ToolResult: 工具执行结果封装
- BaseTool: 工具抽象基类
- Tool: 基础工具实现（带元数据验证）
- FunctionTool: 基于函数的工具包装器
"""

import inspect
from typing import Dict, List, Any, Optional, Callable, Union, Type, Set, Tuple
from pydantic import BaseModel, Field, create_model, ValidationError

from supports.logger import logger


class ToolResult(BaseModel):
    """工具执行结果封装，统一管理输出与错误信息"""

    output: Optional[str] = Field(default=None, description="工具执行的正常输出")
    error: Optional[str] = Field(default=None, description="工具执行的错误信息")

    @property
    def is_success(self) -> bool:
        return self.error is None

    @property
    def is_failure(self) -> bool:
        return self.error is not None

    def __str__(self) -> str:
        if self.error:
            return f"Error: {self.error}"
        return self.output or ""

    def __bool__(self) -> bool:
        return self.is_success


class ToolFailure(ToolResult):
    """明确的工具失败结果"""

    def __init__(self, error: str, **kwargs):
        super().__init__(error=error, **kwargs)


class ToolError(Exception):
    """工具执行异常"""

    def __init__(self, message: str, tool_name: Optional[str] = None):
        self.message = message
        self.tool_name = tool_name
        super().__init__(message)


class ToolMetadata(BaseModel):
    """工具元数据"""

    # 基本信息
    name: str
    description: str
    category: str = "general"
    version: str = "1.0.0"
    author: str = ""

    # 输入输出规范
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    # 使用示例
    examples: List[Dict[str, Any]] = Field(default_factory=list)

    # 安全信息
    requires_permissions: List[str] = Field(default_factory=list)
    is_dangerous: bool = False
    warning: Optional[str] = None


class BaseTool(BaseModel):
    """工具抽象基类，所有工具都应继承此类"""

    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> ToolResult:
        """执行工具调用"""
        return await self.execute(**kwargs)

    async def execute(self, **kwargs) -> ToolResult:
        """
        执行工具逻辑，子类必须实现此方法

        Args:
            **kwargs: 工具参数

        Returns:
            ToolResult: 执行结果
        """
        raise NotImplementedError("子类必须实现 execute 方法")

    def to_param(self) -> Dict[str, Any]:
        """将工具转换为 OpenAI function call 参数格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {},
            },
        }


class Tool(BaseTool):
    """基础工具实现，带元数据验证和动态输入模型生成"""

    metadata: ToolMetadata

    def __init__(self, metadata: Optional[ToolMetadata] = None, **kwargs):
        """
        初始化工具

        Args:
            metadata: 工具元数据，若未提供则自动生成
        """
        meta = metadata or self._get_default_metadata()
        self._validate_metadata(meta)
        super().__init__(
            name=meta.name,
            description=meta.description,
            parameters=meta.input_schema,
        )
        self.metadata = meta

    def _get_default_metadata(self) -> ToolMetadata:
        """获取默认元数据"""
        if hasattr(self.__class__, 'metadata'):
            return self.__class__.metadata

        return ToolMetadata(
            name=self.__class__.__name__.lower(),
            description=self.__doc__ or f"{self.__class__.__name__} tool"
        )

    @staticmethod
    def _validate_metadata(meta: ToolMetadata) -> None:
        """验证并补全元数据"""
        if not meta.name:
            meta.name = "unnamed_tool"
        if not meta.description:
            meta.description = "No description provided"

    async def execute(self, **kwargs) -> Any:
        """
        执行工具，子类覆盖此方法

        Args:
            **kwargs: 工具参数

        Returns:
            执行结果
        """
        raise NotImplementedError("子类必须实现 execute 方法")

    def validate_input(self, **kwargs) -> Dict[str, Any]:
        """
        根据 input_schema 验证输入参数

        Args:
            **kwargs: 工具参数

        Returns:
            验证后的参数字典

        Raises:
            ToolError: 参数验证失败时抛出
        """
        if not self.metadata.input_schema:
            return kwargs

        properties = self.metadata.input_schema.get("properties", {})
        required = self.metadata.input_schema.get("required", [])

        fields = {}
        for field_name, field_schema in properties.items():
            field_type = self._map_json_type_to_python(field_schema.get("type", "string"))
            field_default = ... if field_name in required else None
            field_description = field_schema.get("description", "")
            fields[field_name] = (field_type, Field(default=field_default, description=field_description))

        model_name = f"{self.metadata.name.capitalize()}Input"
        input_model = create_model(model_name, **fields)

        try:
            validated = input_model(**kwargs)
            return validated.model_dump()
        except ValidationError as e:
            errors = {}
            for error in e.errors():
                loc = ".".join(str(l) for l in error["loc"])
                if loc not in errors:
                    errors[loc] = []
                errors[loc].append(error["msg"])

            error_msg = f"工具 {self.metadata.name} 输入验证失败: "
            error_details = "; ".join(
                f"{field}: {', '.join(msgs)}" for field, msgs in errors.items()
            )
            raise ToolError(f"{error_msg}{error_details}", tool_name=self.metadata.name)

    @staticmethod
    def _map_json_type_to_python(json_type: str) -> Any:
        """将 JSON Schema 类型映射到 Python 类型"""
        type_mapping = {
            "string": Optional[str],
            "integer": Optional[int],
            "number": Optional[float],
            "boolean": Optional[bool],
            "array": Optional[List[Any]],
            "object": Optional[Dict[str, Any]],
            "null": type(None),
        }
        return type_mapping.get(json_type, Any)


class FunctionTool(Tool):
    """基于函数的工具，可以将普通函数包装为工具"""

    def __init__(self, func: Callable, metadata: Optional[ToolMetadata] = None):
        """
        初始化函数工具

        Args:
            func: 要包装的函数（支持同步和异步）
            metadata: 工具元数据，若未提供则从函数签名中提取
        """
        self.func = func
        meta = metadata or self._extract_metadata_from_func(func)
        super().__init__(metadata=meta)

    def _extract_metadata_from_func(self, func: Callable) -> ToolMetadata:
        """从函数签名和文档字符串中提取元数据"""
        name = func.__name__
        doc = func.__doc__ or f"{name} function"

        sig = inspect.signature(func)
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = Any
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation

            input_schema["properties"][param_name] = {
                "type": self._get_json_type_from_python(param_type),
                "description": f"{param_name} parameter"
            }

            if param.default == inspect.Parameter.empty:
                input_schema["required"].append(param_name)

        return ToolMetadata(
            name=name,
            description=doc,
            input_schema=input_schema
        )

    @staticmethod
    def _get_json_type_from_python(py_type: Any) -> str:
        """从 Python 类型推断 JSON Schema 类型"""
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null"
        }

        origin = getattr(py_type, "__origin__", py_type)

        if origin in (list, List, tuple, Tuple, set, Set):
            return "array"
        elif origin in (dict, Dict):
            return "object"
        elif origin is Union:
            args = getattr(py_type, "__args__", [])
            if len(args) == 2 and type(None) in args:
                other_type = next(arg for arg in args if arg is not type(None))
                return FunctionTool._get_json_type_from_python(other_type)

        return type_mapping.get(origin, "string")

    async def execute(self, **kwargs) -> Any:
        """
        执行工具函数

        Args:
            **kwargs: 工具参数

        Returns:
            函数执行结果
        """
        validated_kwargs = self.validate_input(**kwargs)

        try:
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**validated_kwargs)
            else:
                return self.func(**validated_kwargs)
        except Exception as e:
            logger.error(f"函数工具 '{self.metadata.name}' 执行失败: {e}")
            raise ToolError(str(e), tool_name=self.metadata.name)
