# agent-mcp-rag

**Augmented LLM Chatbot Agent (Chat + MCP + CRAG + CLI Coder) from scratch in Python**

不依赖任何框架（LangChain, LlamaIndex, CrewAI, AutoGen），纯 Python 实现的增强型 LLM Agent。

---

## 目录

- [项目结构](#项目结构)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [架构说明](#架构说明)
- [模块详解](#模块详解)
- [用户指南](#用户指南)
- [测试](#测试)
- [参考文档](#参考文档)
- [参考鸣谢](#参考鸣谢)

---

## 项目结构

```
agent-mcp-rag/
├── src/                          # 核心应用代码
│   ├── agent.py                  # Agent 核心循环 (CRAG + MCP + CLI)
│   ├── chat_openai.py            # LLM 接口 (多模型路由: cheap/coder)
│   ├── crag_retriever.py         # CRAG 检索器封装
│   ├── coder_agent.py            # 专业代码处理 Agent
│   ├── coder_cli.py              # CLI 编码工具 (替代 MCP 减少 token)
│   ├── mcp_client.py             # MCP 客户端 (JSON-RPC stdio)
│   ├── index.py                  # 主入口 (完整演示)
│   └── utils.py                  # 工具函数 & 安全模块
├── supports/                     # 基础库 (OpenManus 改进版)
│   ├── agent.py                  # BaseAgent 基类
│   ├── config.py                 # OpenAI 兼容多提供商配置
│   ├── crag.py                   # CRAG 引擎 (纯 Python)
│   ├── llm.py                    # LLM 统一接口
│   ├── schema.py                 # 消息/状态/内存数据模型
│   ├── memory_manager.py         # 会话内存管理
│   ├── exceptions.py             # 异常体系
│   ├── validators.py             # 输入/输出验证
│   ├── logger.py                 # 日志模块
│   └── tool/                     # 工具系统
│       ├── base.py               # BaseTool, ToolResult, ToolFailure
│       ├── mcp.py                # MCP 工具代理
│       ├── tool_collection.py    # 工具集合
│       └── tool_manager.py       # 工具管理器
├── tests/                        # 测试
│   ├── test_crag.py              # CRAG 引擎测试
│   ├── test_llm.py               # LLM 接口测试
│   ├── test_coder.py             # CLI Coder 测试
│   └── run_all.py                # 全部测试运行器
├── knowledge/                    # 知识库 (RAG 检索源)
│   └── user_*.md                 # 示例用户档案
├── output/                       # 输出目录
├── demo.py                       # 命令行演示脚本
├── demo.ipynb                    # Jupyter Notebook 演示
├── conf.yaml                     # YAML 配置文件
├── requirements.txt              # Python 依赖
├── .env.example                  # 环境变量模板
└── README.md
```

---

## 核心特性

### 1. CRAG (Corrective RAG) — 纯 Python 实现

```
User Query → Retrieve → Grade → Correct → Context → LLM Generate
                  ↓         ↓         ↓
              TF-IDF检索  相关性评估  知识补充/替换
```

- **Retrieve**: TF-IDF 稀疏向量检索（中英文混合分词）
- **Grade**: 关键词覆盖率 + 密度分析，分三级（高相关/模糊/不相关）
- **Correct**: 模糊文档补充搜索，不相关文档替换搜索
- **Generate**: 基于纠正后上下文进行 LLM 生成

### 2. MCP (Model Context Protocol) 客户端

- JSON-RPC stdio 协议实现
- 支持多 MCP Server (fetch, filesystem 等)
- 进程生命周期管理

### 3. CLI Coder — 轻量级工具替代 MCP

由于 LLM 对 CLI/Bash 代码进行了偏好微调，可直接理解命令意图，
无需复杂的 function calling 描述，大幅减少 token 消耗。

- `run_command`: 安全 shell 命令执行（白名单 + 危险模式检测）
- `read_file`: 文件读取（支持行范围）
- `write_file`: 文件写入
- `list_dir`: 目录列表
- `search_code`: 代码搜索（ripgrep / 回退方案）

### 4. 多模型路由

- **Cheap Model**: 一般推理任务（低成本、低延迟）
- **Coder Model**: 代码生成和审查（强推理能力）
- 测试时使用单一 API 即可

### 5. 信息安全

- API Key 从环境变量加载（不写入配置文件）
- 敏感信息日志掩码 `mask_secret()`
- 命令白名单 + 危险模式正则检测
- 路径遍历攻击防护

---

## 快速开始

### 前置要求

- Python 3.10+
- [uvx](https://github.com/astral-sh/uv) (用于运行 MCP Server)
- Node.js + npx (用于 filesystem MCP Server)

### 安装

```bash
git clone git@github.com:YongDeng715/agent-mcp-rag.git
cd agent-mcp-rag

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入真实的 API Key
```

### .env 配置

```bash
# 硅基流动 API
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_BASE_URL=https://api.siliconflow.cn/v1

# 模型配置
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct        # 默认模型
CODER_MODEL=deepseek-ai/DeepSeek-V3       # 代码模型
CHEAP_MODEL=Qwen/Qwen2.5-7B-Instruct      # 便宜模型
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5   # Embedding 模型
```

### 运行

```bash
# 完整流程演示 (CRAG + MCP + CLI)
python demo.py

# Agent 任务执行 (CRAG + Agent + MCP)
python -m src.index

# 运行测试
python tests/run_all.py
```

---

## 架构说明

### 数据流

```
┌─────────────────────────────────────────────────┐
│                    index.py                      │
│  ┌──────────┐   ┌─────────┐   ┌──────────────┐ │
│  │  CRAG    │   │  Agent  │   │   Output     │ │
│  │ Retriever│──▶│         │──▶│  antonette.md │ │
│  └──────────┘   └────┬────┘   └──────────────┘ │
│                      │                          │
│         ┌────────────┼────────────┐             │
│         ▼            ▼            ▼             │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│    │ MCP     │  │ MCP     │  │ CLI     │       │
│    │ Fetch   │  │ File    │  │ Coder   │       │
│    └─────────┘  └─────────┘  └─────────┘       │
└─────────────────────────────────────────────────┘
```

### Agent 循环

```
User Prompt
    │
    ▼
LLM Reasoning ──→ Tool Call? ──Yes──→ Execute Tool (MCP/CLI)
    │                                       │
    │            Tool Result ◀──────────────┘
    │                 │
    └─────────←───────┘
    │
    ▼
Final Response
```

---

## 模块详解

### src/chat_openai.py — LLM 接口

```python
from src.chat_openai import ChatOpenAI

llm = ChatOpenAI()

# 一般推理
response = await llm.chat("Hello!")

# 代码生成（使用 coder 模型）
response = await llm.chat_code("Write a Python sort function")

# 便宜模型
response = await llm.chat_cheap("Simple question")
```

### src/crag_retriever.py — CRAG 检索器

```python
from src.crag_retriever import CRAGRetriever

retriever = CRAGRetriever()
retriever.add_documents(["doc1", "doc2", ...])

# 简单 RAG (无纠正)
context = retriever.get_context(query, enable_correction=False)

# CRAG (带纠正)
context = retriever.get_context(query, enable_correction=True)
```

### src/coder_cli.py — CLI 工具

```python
from src.coder_cli import CoderCLI

coder = CoderCLI("/path/to/workspace")

# 执行命令
await coder.execute("run_command", command="ls -la")

# 读取文件
await coder.execute("read_file", path="README.md", start_line=1, end_line=10)

# 获取工具定义（给 Agent 使用）
tools = coder.get_tools()
```

### src/coder_agent.py — 代码 Agent

```python
from src.coder_agent import CoderAgent

agent = CoderAgent()

# 生成代码
code = await agent.generate_code("quick sort", "python")

# 审查代码
review = await agent.review_code("def div(a,b): return a/b")

# 智能执行
result = await agent.smart_execute("统计项目中的 Python 文件数")
```

### src/agent.py — 主 Agent

```python
from src.agent import Agent
from src.mcp_client import MCPClient
from src.coder_cli import CoderCLI

fetch_mcp = MCPClient("fetch", "uvx", ["mcp-server-fetch"])
coder = CoderCLI()

agent = Agent(
    model="Qwen/Qwen2.5-7B-Instruct",
    mcp_clients=[fetch_mcp],
    coder_cli=coder,
    context="RAG检索到的上下文...",
)
await agent.init()
result = await agent.invoke("帮我完成XX任务")
await agent.close()
```

---

## 用户指南

### 使用 SiliconFlow API

1. 注册 [硅基流动](https://cloud.siliconflow.cn)
2. 获取 API Key (以 `sk-` 开头)
3. 在 `.env` 中配置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`

### 使用 MiniMax API

1. 注册 [MiniMax](https://platform.minimaxi.com)
2. 获取 API Key
3. 在 `.env` 中设置 `OPENAI_API_KEY` 为新 Key，`OPENAI_BASE_URL` 为 `https://api.minimax.chat/v1`

### 推荐模型

| 用途 | 模型 | 提供商 |
|------|------|--------|
| 一般推理 | Qwen/Qwen2.5-7B-Instruct | SiliconFlow |
| 代码生成 | deepseek-ai/DeepSeek-V3 | SiliconFlow |
| Embedding | BAAI/bge-large-zh-v1.5 | SiliconFlow |

### 自定义知识库

在 `knowledge/` 目录下放置 `.md` 文件即可被 CRAG 检索器自动索引。

---

## 测试

```bash
# 运行全部测试
python tests/run_all.py

# 单独测试
python tests/test_crag.py      # CRAG 引擎
python tests/test_llm.py       # LLM 接口（需要 API）
python tests/test_coder.py     # CLI Coder
```

测试结果保存在 `output/` 目录中。

---

## 参考文档

### MCP (Model Context Protocol)
- [MCP Architecture](https://modelcontextprotocol.io/docs/concepts/architecture)
- [MCP Client Guide](https://modelcontextprotocol.io/quickstart/client)
- [Fetch MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch)
- [Filesystem MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)
- [FastMCP Tutorials](https://github.com/jlowin/fastmcp/tree/main/docs/tutorials)

### RAG & CRAG
- [Retrieval Augmented Generation Overview](https://scriv.ai/guides/retrieval-augmented-generation-overview/)
  - [RAG 中文译文](https://www.yuque.com/serviceup/misc/cn-retrieval-augmented-generation-overview)
- [Corrective RAG Paper](https://arxiv.org/pdf/2401.15884)
  - [CRAG from Scratch](https://github.com/FareedKhan-dev/all-rag-techniques/blob/main/20_crag.ipynb)
  - [CRAG from Scratch (中文)](https://github.com/liu673/rag-all-techniques/blob/master/src/full/20_crag.ipynb)
  - [CRAG by LangGraph](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/rag/langgraph_crag.ipynb)

### OpenAI 兼容 API
- [SiliconFlow API 文档](https://docs.siliconflow.cn)
- [MiniMax API 文档](https://platform.minimaxi.com/document)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

---

## 参考鸣谢

本项目深受以下开源项目启发，谨致诚挚感谢：

- [llm-mcp-rag](https://github.com/KelvinQiu802/llm-mcp-rag) — TypeScript 参考实现
- [OpenManus](https://github.com/mannaandpoem/OpenManus) — Agent 框架设计
- [all-rag-techniques](https://github.com/FareedKhan-dev/all-rag-techniques) — CRAG 算法实现
- [rag-all-techniques](https://github.com/liu673/rag-all-techniques) — RAG 技术合集
- [LangGraph CRAG](https://github.com/langchain-ai/langgraph) — LangGraph CRAG 教程
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Python MCP 实现
- [FastMCP](https://github.com/jlowin/fastmcp) — 简化 MCP 服务开发

---

## License

MIT License — 详见 [LICENSE](LICENSE)
