"""
使用示例
演示如何将视觉工具集成到 LangChain Agent 中。
"""
import os
import sys

# 确保可以导入同目录的模块
sys.path.insert(0, os.path.dirname(__file__))

from tools import ALL_VISION_TOOLS, CV_TOOLS, VLM_TOOLS


def example_1_direct_tool_call():
    """示例 1：直接调用工具（不通过 Agent）"""
    print("=" * 60)
    print("示例 1：直接调用 OCR 工具")
    print("=" * 60)

    from .ocr_tool import capture_and_ocr

    # 方式 A：从摄像头捕获并 OCR
    # result = capture_and_ocr.invoke({})

    # 方式 B：从本地图片 OCR（测试用）
    result = capture_and_ocr.invoke({"image_path": "/tmp/test_image.jpg"})
    print(result)


def example_2_langchain_agent():
    """示例 2：集成到 LangChain AgentExecutor"""
    print("=" * 60)
    print("示例 2：LangChain AgentExecutor 集成")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate

    # 初始化 LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 创建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个具备视觉感知能力的 AI 助手。
你可以使用以下工具来"看"世界：

1. capture_and_ocr - 从摄像头捕获画面并识别文字（快速，适合文档/屏幕文字）
2. capture_and_ocr_detail - OCR 详细版（包含文字坐标位置）
3. capture_and_analyze - 用多模态大模型深度分析画面（适合复杂场景理解）
4. capture_and_answer - 针对画面回答具体问题

选择建议：
- 如果只需要识别文字内容 → 用 capture_and_ocr
- 如果需要理解画面内容、分析场景 → 用 capture_and_analyze
- 如果有具体问题要问 → 用 capture_and_answer
"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 创建 Agent
    agent = create_tool_calling_agent(llm, ALL_VISION_TOOLS, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=ALL_VISION_TOOLS, verbose=True)

    # 测试对话
    result = agent_executor.invoke({
        "input": "请看一下摄像头，告诉我屏幕上有什么文字？"
    })
    print(f"\nAgent 回答: {result['output']}")


def example_3_langgraph():
    """示例 3：集成到 LangGraph"""
    print("=" * 60)
    print("示例 3：LangGraph 集成")
    print("=" * 60)

    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # LangGraph 的 create_react_agent 直接接受 tools 列表
    agent = create_react_agent(llm, ALL_VISION_TOOLS)

    # 调用
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "请看一下摄像头，描述你看到了什么？"}
        ]
    })

    # 打印最后一条消息（Agent 的回复）
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.type == "ai":
            print(f"Agent: {msg.content}")


def example_4_selective_tools():
    """示例 4：只给 Agent 部分工具（按需组合）"""
    print("=" * 60)
    print("示例 4：按需选择工具组合")
    print("=" * 60)

    from langgraph.prebuilt import create_react_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # 场景：只需要 OCR 能力的轻量 Agent
    ocr_only_agent = create_react_agent(llm, CV_TOOLS)

    # 场景：只需要 VLM 的深度分析 Agent
    vlm_only_agent = create_react_agent(llm, VLM_TOOLS)

    print("✓ OCR 专用 Agent 已创建（工具：capture_and_ocr, capture_and_ocr_detail）")
    print("✓ VLM 专用 Agent 已创建（工具：capture_and_analyze, capture_and_answer）")


if __name__ == "__main__":
    # 快速验证工具是否正常加载
    print("视觉工具加载验证：")
    print(f"  全部工具: {[t.name for t in ALL_VISION_TOOLS]}")
    print(f"  CV 工具:  {[t.name for t in CV_TOOLS]}")
    print(f"  VLM 工具: {[t.name for t in VLM_TOOLS]}")
    print()

    # 取消注释以运行具体示例
    # example_1_direct_tool_call()
    # example_2_langchain_agent()
    # example_3_langgraph()
    # example_4_selective_tools()
