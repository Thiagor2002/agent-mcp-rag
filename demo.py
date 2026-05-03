#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo: Augmented LLM Agent (CRAG + MCP + CLI Coder)
===================================================

两个核心演示场景:
  场景1: CRAG 知识检索 + LLM 分析生成
         用户给出分析指令 → CRAG 从知识库检索相关内容
         → 可选结合网页搜索补充 → LLM 生成分析报告

  场景2: CLI 操作任务
         用户给出操作需求（移动文件、创建目录等）
         → CLI Coder 安全执行命令
"""

import asyncio
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_PATH))

from src.crag_retriever import CRAGRetriever
from src.chat_openai import ChatOpenAI
from src.coder_cli import CoderCLI
from src.utils import log_title, log_section, ensure_dir


# =========================================================================
# 准备：初始化知识库
# =========================================================================

def prepare_knowledge_base() -> CRAGRetriever:
    """
    初始化 CRAG 检索器并加载本地知识库

    知识库内容: knowledge/ 目录下的用户档案 (user_1.md ~ user_10.md)
    """
    retriever = CRAGRetriever()
    knowledge_dir = ROOT_PATH / "knowledge"
    loaded = retriever.load_knowledge_dir(str(knowledge_dir))
    if loaded > 0:
        print(f"  已从 knowledge/ 加载 {loaded} 个知识文档")
    # 额外添加一些分析相关文档
    retriever.add_documents([
        "2024年AI行业报告: 大语言模型(LLM)市场规模达$50B, 年增长率45%。"
        "主要玩家: OpenAI(GPT-4o), Anthropic(Claude 4), Google(Gemini)。"
        "企业级应用以RAG(检索增强生成)和Agent(智能体)为主要方向。",
        "2024年开发者调查报告: Python以67%使用率稳居第一, TypeScript(42%)超越JavaScript(38%)。"
        "Rust以83%满意度连续8年最受喜爱。AI/ML工具使用率达62%, 同比增长28%。",
        "数据工程最佳实践: 数据管道设计遵循ETL(Extract-Transform-Load)模式。"
        "Apache Spark处理批处理, Apache Kafka负责实时流, dbt用于数据转换。"
        "数据治理需关注质量、安全、元数据管理和血缘追踪。",
        "软件架构演进: 从单体→微服务→无服务器→AI原生架构。"
        "2024年趋势: Agent驱动架构(ADA)兴起, 多智能体协作系统成为主流。"
        "Retrieval-Augmented Generation(RAG)是连接LLM与企业数据的标准方案。",
    ])
    print(f"  总计 {retriever.doc_count} 个文档已索引")
    return retriever


# =========================================================================
# 场景1: CRAG 知识检索 + LLM 分析生成
# =========================================================================

async def scenario_1_crag_analysis(
    retriever: CRAGRetriever,
    user_query: str = "",
):
    """
    场景1: 用户给出分析指令 → CRAG 检索知识库 → 可选网页搜索 → LLM 生成回复

    流程:
      1. 用户输入分析需求
      2. CRAG 从知识库检索相关内容 (Retrieve → Grade → Correct)
      3. 如果检索结果不够 (模糊/不相关)，自动触发知识补充搜索
      4. 构建完整上下文 → LLM 生成分析报告
    """
    log_title("场景1: CRAG 知识检索 + LLM 分析生成")

    if not user_query:
        user_query = (
            "请分析当前AI行业的发展趋势和开发者生态，"
            "结合已知的数据工程和软件架构演进，"
            "给出2025年的技术展望和建议。"
        )

    print(f"\n  [用户指令]")
    print(f"  {user_query}")

    # Step 1: CRAG 检索
    print(f"\n  [CRAG 检索] 正在从知识库检索相关信息...")
    result = retriever.retrieve(user_query, top_k=5, enable_correction=True)

    doc_count = len(result.docs)
    print(f"  检索结果: {doc_count} 个文档片段")
    correct_count = sum(1 for d in result.docs if d.relevance.value == "correct")
    print(f"    高相关: {correct_count}, 模糊: {doc_count - correct_count}")

    context = result.corrected_context
    if not context:
        print("  [警告] CRAG 未找到相关内容，将仅依赖 LLM 知识")
        context = "知识库中暂无直接相关内容，请基于你的训练知识回答。"

    print(f"\n  [CRAG 操作日志]")
    for action in result.action_log:
        print(f"    → {action}")

    # Step 2: LLM 分析生成
    print(f"\n  [LLM 分析] 基于 CRAG 上下文生成回复...")
    try:
        llm = ChatOpenAI(
            system_prompt=(
                "你是一个专业的数据分析师和技术顾问。"
                "请基于提供的上下文信息，给出结构化的分析和建议。"
                "如果上下文信息不足，请明确说明并基于你的知识补充。"
                "输出格式: 使用 Markdown，包含标题、要点和总结。"
            ),
            context=context[:4000],  # 限制上下文长度
            temperature=0.7,
        )
        response = await llm.chat(user_query)
        print(f"\n  {'='*50}")
        print(f"  [LLM 分析报告]")
        print(f"  {'='*50}")
        print(f"  {response.content}")
        print(f"  {'='*50}")

    except Exception as e:
        print(f"\n  [LLM 调用失败] {e}")
        print(f"  [回退] 展示 CRAG 检索到的上下文:")
        print(f"  {'─'*40}")
        print(f"  {context[:1000]}...")

    # Step 3: 保存结果
    output_dir = ensure_dir(ROOT_PATH / "output")
    output_file = output_dir / "crag_analysis_result.md"
    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        output_content = f"""# CRAG 分析报告

> 生成时间: {timestamp}
> 查询: {user_query}

## CRAG 检索日志

{chr(10).join(f'- {log}' for log in result.action_log)}

## 检索到的文档 ({len(result.docs)} 个)

{context[:5000]}

## LLM 分析结果

{response.content if 'response' in dir() else 'LLM 未生成结果'}
"""
        output_file.write_text(output_content, encoding="utf-8")
        print(f"\n  [保存] 分析报告已保存到: {output_file}")
    except Exception as e:
        print(f"\n  [保存失败] {e}")

    return context, response.content if 'response' in dir() else ""


# =========================================================================
# 场景2: CLI 操作任务
# =========================================================================

async def scenario_2_cli_operations():
    """
    场景2: 用户给出操作需求 → CLI Coder 执行命令

    演示典型的文件/项目操作:
      - 创建目录结构
      - 移动/复制文件
      - 列出文件
      - 搜索代码
      - 统计信息
    """
    log_title("场景2: CLI 操作任务")

    coder = CoderCLI(str(ROOT_PATH))
    output_dir = ensure_dir(ROOT_PATH / "output" / "cli_demo")

    # ------------------------------------------------------------------
    # 任务列表
    # ------------------------------------------------------------------
    tasks = [
        {
            "desc": "创建项目目录结构",
            "tool": "run_command",
            "args": {
                "command": f"mkdir -p {output_dir}/data/raw {output_dir}/data/processed {output_dir}/models {output_dir}/reports"
            },
        },
        {
            "desc": "创建示例数据文件",
            "tool": "write_file",
            "args": {
                "path": str(output_dir / "data" / "raw" / "sample.csv"),
                "content": "id,name,score,category\n1,Alice,95,AI\n2,Bob,87,Web\n3,Carol,92,AI\n4,David,78,Web\n",
            },
        },
        {
            "desc": "查看创建的目录结构",
            "tool": "list_dir",
            "args": {"path": str(output_dir), "pattern": "*"},
        },
        {
            "desc": "统计项目中的 Python 文件",
            "tool": "run_command",
            "args": {"command": "find . -name '*.py' -not -path './.git/*' -not -path './.venv/*' | head -20"},
        },
        {
            "desc": "搜索包含 'CRAG' 的代码",
            "tool": "search_code",
            "args": {"pattern": "CRAG", "path": ".", "file_pattern": "*.py"},
        },
        {
            "desc": "查看文件内容 (sample.csv)",
            "tool": "read_file",
            "args": {"path": str(output_dir / "data" / "raw" / "sample.csv")},
        },
        {
            "desc": "统计项目代码行数",
            "tool": "run_command",
            "args": {"command": "find . -name '*.py' -not -path './.git/*' -exec cat {} + | wc -l"},
        },
    ]

    # ------------------------------------------------------------------
    # 执行任务
    # ------------------------------------------------------------------
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n  ┌─ 任务 {i}: {task['desc']}")
        print(f"  │  工具: {task['tool']}")

        result = await coder.execute(task["tool"], **task["args"])
        results.append({
            "task": task["desc"],
            "tool": task["tool"],
            "result": result,
        })

        # 格式化显示结果
        indent = "  │  "
        for line in result.split("\n")[:10]:
            print(f"{indent}{line}")
        if len(result.split("\n")) > 10:
            print(f"{indent}... (共 {len(result.splitlines())} 行)")

    # ------------------------------------------------------------------
    # 保存操作日志
    # ------------------------------------------------------------------
    log_file = output_dir / "cli_operations_log.md"
    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"""# CLI 操作日志

> 时间: {timestamp}
> 工作目录: {ROOT_PATH}

## 执行的任务

"""
    for i, r in enumerate(results, 1):
        log_content += f"""
### 任务 {i}: {r['task']}

- **工具**: `{r['tool']}`
- **结果**:
```
{r['result'][:500]}
```
"""
    log_file.write_text(log_content, encoding="utf-8")
    print(f"\n  [保存] 操作日志已保存到: {log_file}")

    return results


# =========================================================================
# 主流程
# =========================================================================

async def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  Augmented LLM Agent — 场景演示                          ║")
    print("║  CRAG 知识检索 + CLI 命令操作                            ║")
    print("╚" + "═" * 58 + "╝")

    # ---- 初始化 ----
    print("\n[初始化] 加载知识库...")
    retriever = prepare_knowledge_base()

    # ---- 场景1: CRAG 检索 + LLM 分析 ----
    await scenario_1_crag_analysis(retriever)

    # ---- 场景2: CLI 操作任务 ----
    await scenario_2_cli_operations()

    # ---- 完成 ----
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  演示完成! 结果保存在 output/ 目录                        ║")
    print("╚" + "═" * 58 + "╝")
    print()


if __name__ == "__main__":
    asyncio.run(main())
