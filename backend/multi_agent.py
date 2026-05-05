import os
os.environ["LC_OUTPUT_VERSION"] = "v0"
from dotenv import load_dotenv
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
import re
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent

import langchain_openai.chat_models.base as _openai_base
_original_convert_msg = _openai_base._convert_message_to_dict
def _patched_convert_message_to_dict(message, api="chat/completions"):
    msg_dict = _original_convert_msg(message, api=api)
    if hasattr(message, "additional_kwargs") and "reasoning_content" in message.additional_kwargs:
        rc = message.additional_kwargs["reasoning_content"]
        if rc is not None:
            msg_dict["reasoning_content"] = rc
    return msg_dict
_openai_base._convert_message_to_dict = _patched_convert_message_to_dict


# 尝试相对导入，如果失败则使用绝对导入
try:
    from .tools import *
    from .Conversation_storage import *
    from .vision_tools.tools import ALL_VISION_TOOLS
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from backend.tools import *
    from backend.Conversation_storage import *
    from backend.vision_tools.tools import ALL_VISION_TOOLS

import json

# 确保加载项目根目录的.env文件
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("OPENAI_API_KEY")
CODING_API_KEY = os.getenv("CODING_API_KEY")
CODING_BASE_URL = os.getenv("CODING_BASE_URL")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

# ============================================================
# 0. 多 Agent 可视化追踪
# ============================================================

storage = ConversationStorage()

exec_visualizer = AgentExecutionVisualizer()

from langchain_core.callbacks import BaseCallbackHandler

class ToolEventCallbackHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        from .tools import _fire_tool_event, get_current_step_id
        tool_name = serialized.get("name", "") or kwargs.get("name", "")
        if tool_name:
            step_id = get_current_step_id()
            _fire_tool_event({
                "type": "tool_start",
                "tool_name": tool_name,
                "step_id": step_id,
            })

    def on_tool_end(self, output, **kwargs):
        from .tools import _fire_tool_event, get_current_step_id
        tool_name = kwargs.get("name", "")
        if tool_name:
            step_id = get_current_step_id()
            _fire_tool_event({
                "type": "tool_end",
                "tool_name": tool_name,
                "step_id": step_id,
            })

tool_event_handler = ToolEventCallbackHandler()


def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要"""
    # 提取旧对话
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 生成摘要
    summary_prompt = f"""请总结以下对话的关键信息：
                        {old_conversation}
                        总结（包含用户信息、重要事实、待办事项）："""

    summary = model.invoke(summary_prompt).content
    return summary


# ============================================================
# 1. 定义共享状态
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    supervisor_instruction: str  # SUPERVISOR给子agent的执行指令
    execution_history: list  # 执行历史，用于重新思考
    retry_count: int = 0  # 重试次数，防止无限循环
    last_agent: str = ""  # 上一个执行的 Agent
    last_output: str = ""  # 上一个 Agent 的输出
    pending_video_url: str = ""  # 待下载的视频URL（从搜索结果中提取）
    awaiting_download_confirmation: bool = False  # 是否等待用户确认下载
    current_plan: list = []  # 当前执行中的任务计划
    plan_active: bool = False  # 是否有正在执行的计划



# ============================================================
# 3. 创建 LLM 实例
# ============================================================

def _make_model():
    if "deepseek" in MODEL.lower():
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(
            model=MODEL,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0.3,
            timeout=60,
            max_retries=3,
        )
    return init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        extra_body={"enable_thinking": False},
        temperature=0.3,
        timeout=60,
        max_retries=3,
    )

single_model = _make_model()

vision_model = init_chat_model(
    model=os.getenv("VISION_MODEL", "qwen3.6-plus"),
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.3,
    timeout=60,
    max_retries=3,
    extra_body={"enable_thinking": False},
)

coding_model = init_chat_model(
    model=os.getenv("CODING_MODEL", "qwen-coder-max"),
    model_provider="openai",
    api_key=CODING_API_KEY,
    base_url=CODING_BASE_URL,
    temperature=0.3,
    timeout=60,
    max_retries=3,
    extra_body={"enable_thinking": False},
)
# ============================================================
# 4. 创建子 Agent
# ============================================================

knowledge_agent = create_agent(
    model=_make_model(),
    tools=[search_knowledge_base, web_search, bilibili_video_search, bilibili_download],
    system_prompt=(
        "你是知识库助手，专门处理文档检索、知识问答、联网搜索和视频搜索。\n\n"
        "【重要规则】你的训练数据有时效性，可能不包含最新信息。因此：\n"
        "- 当用户询问任何关于价格、新闻、时事、产品信息、最新动态等可能随时间变化的内容时，"
        "你必须使用 web_search 工具搜索网络获取最新信息，绝不能仅凭自身知识回答。\n"
        "- 即使你认为某个产品不存在或某个事件未发生，也必须先使用 web_search 搜索验证，"
        "因为你的知识可能已过时。\n"
        "- 只有在确认是常识性问题（如数学、历史等不变的事实）时，才可以不使用工具。\n\n"
        "工具使用指南：\n"
        "- 当用户询问关于某些可能存在于知识库里的专业知识的内容时，使用 search_knowledge_base 工具检索信息。\n"
        "- 当用户询问当前信息、新闻、价格、产品详情或任何可能不在知识库中的事实时，使用 web_search 工具搜索网络。\n"
        "- 当用户想要查找、搜索B站视频时，使用 bilibili_video_search 工具，但请注意输入的关键词不能包含空格。"
        "你需要先分析用户的自然语言描述，提取出最核心的搜索关键词，然后将这些关键词作为参数传给工具。\n"
        "例如：用户输入'我想下载《爱情怎么翻译》来看一看'，你应该提取关键词'爱情怎么翻译'并调用工具。\n"
        "- 当用户想要下载指定的B站视频时，使用 bilibili_download 工具。"
        "你需要提供视频链接和可选的下载目录。\n"
        "例如：用户输入'帮我下载这个视频 https://www.bilibili.com/video/BV18SxTzpEc4'或者bilibili_video_search已经寻找出相应的视频链接后，你应该调用 bilibili_download 工具，若未提供链接，请提醒用户或询问是否需要帮忙从bilibili寻找相关视频。\n"
        "返回视频链接和下载方式。\n\n"
        "基于工具返回的检索结果回答问题，如果信息不足请诚实说明。\n"
        "回答时请简洁明了。"
    ),
)

reminder_agent = create_agent(
    model=_make_model(),
    tools=[set_reminder],
    system_prompt=(
        "你是提醒助手，专门处理定时提醒和日程管理。\n"
        "当用户需要设置提醒时，使用 set_reminder 工具。\n"
        "你需要从用户消息中提取提醒内容和时间。\n"
        "时间格式必须是 YYYY-MM-DD HH:MM，如果用户说的是相对时间（如'明天下午3点'），请计算为绝对时间。\n"
        "设置成功后告知用户提醒详情。"
    ),
)

city_info_agent = create_agent(
    model=_make_model(),
    tools=[get_city_info],
    system_prompt=(
        "你是城市信息查询助手，专门处理当前城市的信息查询问题。\n"
        "当用户询问当前城市的详细信息时例如日期、时间、温度、天气描述等，使用 get_city_info 工具查询。\n"
        "用自然友好的语言总结城市信息。"
    ),
)

download_agent = create_agent(
    model=_make_model(),
    tools=[bilibili_download],
    system_prompt=(
        "你是下载助手，专门处理所有资源的下载问题。\n"
        "当用户需要下载视频时，使用 bilibili_download 工具。\n"
        "你需要从用户消息中提取视频链接。\n"
        "返回视频下载链接。"
    ),
)

vision_agent = create_agent(
        model=_make_model(),
        tools=ALL_VISION_TOOLS,
        system_prompt=(
            "你是一个视觉处理助手，擅长使用摄像头捕获图像并进行分析。当用户需要：\n" \
                   "1. 识别图像中的文字（OCR）\n" \
                   "2. 分析图像内容（VLM）\n" \
                   "3. 回答关于图像的问题时\n" \
                   "请使用相应的视觉工具来处理。\n\n" \
                   "可用工具：\n" \
                   "- capture_and_ocr: 捕获图像并识别文字\n" \
                   "- capture_and_ocr_detail: 捕获图像并详细识别文字\n" \
                   "- capture_and_analyze: 捕获图像并分析内容\n" \
                   "- capture_and_answer: 捕获图像并回答问题\n\n" \
                   "当用户要求打开摄像头或分析图像时，请使用上述工具。"
    ),
)

# 创建代码处理 Agent
coding_agent = create_agent(
    model=_make_model(),
    tools=[create_file, modify_file, read_file, read_file_summary, add_file_summary, run_command, execute_confirmed_command],
    system_prompt=(
        "你是代码处理助手，专门处理文件操作和代码相关问题。\n\n"
        "工具使用指南：\n"
        "- 当用户需要创建文件时，使用 create_file 工具（如果文件已存在，设置 overwrite=True）。\n"
        "  建议在创建文件时提供 summary 参数，简要描述文件功能，方便后续快速了解文件用途。\n"
        "- 当用户需要修改文件时，使用 modify_file 工具：\n"
        "  - 方式1（推荐）：提供 new_content 参数，完全替换文件内容\n"
        "  - 方式2：使用 modifications 参数进行行级精确修改（type/line/old_text/new_text）\n"
        "- 当用户需要查看文件内容时，根据需求选择：\n"
        "  - read_file_summary：只读取文件头部的 @summary 摘要注释，快速了解文件功能（节省token）\n"
        "  - read_file：读取文件完整内容（需要了解代码细节时使用）\n"
        "- 当需要为已有文件添加或更新摘要时，使用 add_file_summary 工具。\n"
        "- 当需要执行终端命令时，使用 run_command 工具。\n\n"
        "【重要】命令确认流程：\n"
        "run_command 工具需要用户确认后才会执行命令。流程如下：\n"
        "1. 调用 run_command 时，工具会返回一个确认请求消息（包含 [COMMAND_CONFIRM_REQUIRED] 标记）\n"
        "2. 你必须将命令内容告知用户，并询问用户是否确认执行\n"
        "3. 当用户回复确认（如'确认'、'执行'、'好的'等）后，使用 execute_confirmed_command 工具执行命令\n"
        "4. execute_confirmed_command 的参数必须与之前 run_command 的参数完全一致\n"
        "5. 如果用户拒绝，告知用户命令已取消，不要再次尝试执行\n\n"
        "【重要】文件摘要系统 - 节省 Token 的关键：\n"
        "每个代码文件头部应有 @summary 注释块，格式如下：\n"
        "  # @summary 本文件实现了用户认证功能，包括登录、注册和密码重置\n"
        "  # 提供了 AuthManager 类和相关的 API 路由\n"
        "  # @end\n\n"
        "使用规则：\n"
        "1. 当你只需要知道某个文件是做什么的（不需要看代码细节）→ 使用 read_file_summary\n"
        "2. 当你需要浏览多个文件来找到特定功能在哪个文件中 → 使用 read_file_summary\n"
        "3. 当你需要理解代码的具体实现、修改代码 → 使用 read_file 读取完整内容\n"
        "4. 创建新文件时，务必提供 summary 参数添加功能描述\n"
        "5. 修改文件后，如果文件没有 @summary 块，使用 add_file_summary 添加\n\n"
        "【重要】代码修改与测试工作流：\n"
        "当你修改或创建代码后，必须按以下流程验证：\n"
        "1. 使用 create_file 或 modify_file 完成代码修改\n"
        "2. 立即使用 run_command 运行测试或执行代码来验证修改是否正确\n"
        "3. 分析运行结果：\n"
        "   - 如果测试通过/代码运行成功 → 报告任务完成，说明测试结果\n"
        "   - 如果测试失败/代码报错 → 分析错误原因，使用 modify_file 修复代码，然后重新运行测试\n"
        "4. 重复步骤2-3，直到代码运行成功或已尝试3次\n\n"
        "测试命令示例：\n"
        "- Python文件：run_command(command='python 文件路径')\n"
        "- 运行测试：run_command(command='pytest 测试文件路径')\n"
        "- 检查语法：run_command(command='python -m py_compile 文件路径')\n"
        "- 安装依赖：run_command(command='pip install 包名')\n"
        "- 查看目录：run_command(command='ls 目录路径')\n\n"
        "【重要】错误修复循环：\n"
        "- 当 run_command 返回失败结果时，仔细阅读 stderr 中的错误信息\n"
        "- 根据错误类型（语法错误、导入错误、逻辑错误等）制定修复策略\n"
        "- 修复后必须重新运行测试验证\n"
        "- 如果尝试3次仍无法修复，报告当前状态和遇到的错误，让调度中心决定下一步\n\n"
        "当用户询问代码相关问题时，你可以直接回答，不需要使用工具。\n"
        "回答时请简洁明了，提供准确的代码示例和解释。"
    ),
)

# 创建任务调度 Agent
task_scheduler_agent = create_agent(
    model=_make_model(),
    tools=[plan_task, get_task_list],
    system_prompt=(
        "你是任务调度助手，专门负责将复杂任务分解为可执行的步骤并创建任务计划。\n\n"
        "当收到一个复杂任务时，你需要：\n"
        "1. 仔细分析任务需求，识别所有需要完成的子任务\n"
        "2. 将任务分解为具体、可执行的步骤\n"
        "3. 使用 plan_task 工具创建任务列表\n"
        "4. 每个任务项必须包含三个字段：\n"
        "   - content: 祈使句描述，如 \"创建项目目录结构\"\n"
        "   - status: 第一个任务设为 \"in_progress\"，其余设为 \"pending\"\n"
        "   - activeForm: 进行时描述，如 \"正在创建项目目录结构\"\n"
        "5. 创建完成后，使用 get_task_list 查看并确认任务列表\n\n"
        "任务分解原则：\n"
        "- 每个任务应该能由单个agent在一次执行中完成\n"
        "- 任务之间要有合理的先后顺序和依赖关系\n"
        "- 先创建基础结构，再实现功能，最后测试验证\n"
        "- 同一时间只能有一个任务处于 in_progress 状态\n"
        "- 不要创建过于细碎的任务（少于3步不需要规划）\n"
        "- 每个任务的 content 应该清晰明确，让执行agent知道具体要做什么\n\n"
        "输出格式：\n"
        "创建完任务列表后，请简要总结计划内容，包括总步骤数和各步骤概要。"
    ),
)

# ============================================================
# 5. Supervisor 节点 —— 分析意图并路由
# ============================================================

SUPERVISOR_PROMPT = (
    "你是一个智能任务调度与指挥中心，负责分析用户意图、选择最合适的助手，并给出明确的执行指令。\n\n"
    "可选助手及其能力：\n"
    "- scheduler: 任务调度助手\n"
    "  可用工具：plan_task（规划任务列表）、get_task_list（查看任务进度）\n"
    "  适用场景：当任务较复杂，需要分解为多个步骤来联合多个agent或工具完成时，先使用scheduler设计todo list\n"
    "  工作流程：scheduler创建计划后返回给你审核，你审核通过后根据计划逐步分派任务给其他agent\n\n"
    "- knowledge: 知识库及联网搜索助手\n"
    "  可用工具：search_knowledge_base（知识库检索）、web_search（联网搜索）、bilibili_video_search（B站视频搜索）、bilibili_download（B站视频下载）\n"
    "  适用场景：文档检索、知识问答、联网搜索实时信息、B站视频搜索\n\n"
    "- coding: 代码处理助手\n"
    "  可用工具：create_file（创建文件）、modify_file（修改文件）、read_file（读取文件）、run_command（请求执行命令，需用户确认）、execute_confirmed_command（执行已确认的命令）\n"
    "  适用场景：文件操作、代码编写与修改、运行测试验证代码、终端命令执行\n"
    "  特别说明：coding助手在修改代码后会自动运行测试验证，如果测试失败会自动修复并重试\n"
    "  命令确认流程：run_command需要用户确认，确认后coding助手会使用execute_confirmed_command执行命令\n\n"
    "- download: 下载确认助手\n"
    "  工作流程：当用户请求下载视频时，路由到 download 助手，系统会向用户发送确认请求\n"
    "  用户确认后，系统会自动路由到 knowledge 助手执行下载\n"
    "  你不需要处理用户对下载的确认回复，系统会自动处理\n\n"
    "- reminder: 提醒助手\n"
    "  可用工具：set_reminder（设置提醒）\n"
    "  适用场景：定时提醒、闹钟、日程安排\n\n"
    "- cityinfo: 城市信息助手\n"
    "  可用工具：get_city_info（查询城市天气/时间等信息）\n"
    "  适用场景：城市天气、日期时间、温度等查询\n\n"
    "- vision: 视觉处理助手\n"
    "  可用工具：capture_and_ocr（捕获图像并识别文字）、capture_and_ocr_detail（捕获图像并详细识别文字）、capture_and_analyze（捕获图像并分析内容）、capture_and_answer（捕获图像并回答问题）\n"
    "  适用场景：打开摄像头、摄像头操作、图像分析、文字识别、基于图像的问答\n\n"
    "- chat: 闲聊助手\n"
    "  适用场景：问候、闲聊等无需工具的场景\n\n"
    "- FINISH: 完成（当工具执行完毕，用户问题已解决时选择）\n\n"
    "你的职责：\n"
    "1. 分析用户意图，选择最合适的助手\n"
    "2. 给出明确的执行指令，告诉助手应该做什么、使用什么工具、如何处理问题\n"
    "3. 如果有执行历史，评估结果并决定是否需要重新执行\n\n"
    "【重要】任务计划工作流：\n"
    "当检测到复杂任务（需要3步以上、涉及多种操作）时：\n"
    "1. 首先路由到 scheduler 助手创建任务计划\n"
    "2. scheduler 返回计划后，你审核计划是否合理\n"
    "3. 如果合理，根据计划中的第一个任务分派给对应agent执行\n"
    "4. agent执行完成后返回结果，你评估是否完成当前任务\n"
    "5. 如果完成，标记 task_completed=true，继续分派下一个任务\n"
    "6. 如果失败，重新分派或调整策略\n"
    "7. 所有任务完成后选择 FINISH\n\n"
    "指令编写规则：\n"
    "- 指令要具体明确，指出应该使用哪个工具以及搜索/查询的关键词\n"
    "- 对于knowledge助手，如果涉及实时信息（价格、新闻、产品等），明确要求使用web_search工具\n"
    "- 对于knowledge助手，如果涉及视频搜索但用户未提供具体链接，应使用bilibili_video_search工具搜索视频\n"
    "- 对于视频下载请求（用户已提供链接），路由到 download 助手\n"
    "- 对于coding助手，指令应包含具体的文件操作内容和测试验证要求\n"
    "- 改写用户问题为更精准的搜索关键词\n"
    "- 如果有执行历史且上次结果不理想，在指令中说明问题并给出新的策略\n\n"
    "- 对于vision助手，如果用户请求分析上传的图像，则使用capture_and_analyze工具\n"
    "- 对于vision助手，如果用户需要调用摄像头并对摄像头拍摄的图片进行分析，明确要求使用capture_and_answer工具\n"
    "- 对于vision助手，如果用户需要调用摄像头并详细识别文字，明确要求使用capture_and_ocr_detail工具\n"
    "重新思考逻辑：\n"
    "- 如果看到执行历史，评估上一个助手的执行结果是否完成了任务\n"
    "- 如果结果不匹配用户问题或失败，重新选择助手并在指令中说明新的执行策略\n"
    "- 如果重试次数超过2次，选择 FINISH 并在指令中说明失败原因\n\n"
    "请严格按以下JSON格式输出，不要输出其他内容：\n"
    "```json\n"
    '{"decision": "助手名称", "instruction": "具体执行指令", "task_completed": true或false}\n'
    "```\n\n"
    "字段说明：\n"
    "- decision: 选择哪个助手（scheduler/knowledge/coding/download/reminder/cityinfo/vision/chat/FINISH）\n"
    "- instruction: 给助手的执行指令\n"
    "- task_completed: 当有正在执行的任务计划时，判断上一个任务是否已完成（true=完成可进入下一任务，false=未完成需要继续）\n"
    "  如果没有正在执行的任务计划，设为 false\n\n"
    "示例：\n"
    '用户问"帮我创建一个Python项目，包含用户登录和数据管理" → {"decision": "scheduler", "instruction": "规划一个Python项目的创建任务，包含：1.创建项目目录结构 2.实现用户登录模块 3.实现数据管理模块 4.编写测试验证"}\n'
    '用户问"iphone17 promax 1TB多少钱" → {"decision": "knowledge", "instruction": "使用web_search工具搜索\"iPhone 17 Pro Max 1TB 价格\"", "task_completed": false}\n'
    '用户问"帮我找一下Python教程视频" → {"decision": "knowledge", "instruction": "使用bilibili_video_search工具搜索\"Python教程\"", "task_completed": false}\n'
    '用户问"帮我下载这个视频 https://www.bilibili.com/video/BV18SxTzpEc4" → {"decision": "download", "instruction": "生成下载确认请求", "task_completed": false}\n'
    '用户问"北京天气" → {"decision": "cityinfo", "instruction": "使用get_city_info工具查询北京的天气信息", "task_completed": false}\n'
    '用户问"你好" → {"decision": "chat", "instruction": "友好地回应用户的问候", "task_completed": false}\n'
    '执行历史显示coding完成了计划中的第一个任务 → {"decision": "coding", "instruction": "根据计划执行下一个任务：xxx", "task_completed": true}\n'
    '执行历史显示上次搜索失败 → {"decision": "knowledge", "instruction": "上次web_search未返回有效结果，请使用不同的关键词重新搜索", "task_completed": false}'
)


def supervisor_node(state: AgentState) -> dict:
    exec_visualizer.on_node_start("supervisor", state)

    messages = state["messages"]
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    execution_history = state.get("execution_history", [])
    retry_count = state.get("retry_count", 0)
    plan_active = state.get("plan_active", False)
    current_plan = state.get("current_plan", [])

    # 优先检查：对话历史中是否有待确认的下载请求
    pending_download = _extract_pending_download(messages)
    if pending_download:
        user_response = last_user_msg.lower().strip()
        if any(kw in user_response for kw in ["确认", "是", "好", "下载", "yes", "ok", "确定"]):
            video_url = pending_download["video_url"]
            output_dir = pending_download["output_dir"]
            output = {
                "next_agent": "knowledge",
                "supervisor_instruction": f"用户已确认下载，请使用bilibili_download工具下载视频\"{video_url}\"，下载到\"{output_dir}\"",
                "execution_history": execution_history,
                "retry_count": 0,
                "pending_video_url": "",
                "awaiting_download_confirmation": False,
                "current_plan": current_plan,
                "plan_active": plan_active,
            }
            exec_visualizer.on_node_end("supervisor", state, output)
            return output
        elif any(kw in user_response for kw in ["取消", "否", "不", "no", "cancel"]):
            output = {
                "next_agent": "FINISH",
                "supervisor_instruction": "用户取消了下载操作",
                "execution_history": execution_history,
                "retry_count": 0,
                "pending_video_url": "",
                "awaiting_download_confirmation": False,
                "current_plan": current_plan,
                "plan_active": plan_active,
            }
            exec_visualizer.on_node_end("supervisor", state, output)
            return output

    # 检查：knowledge搜索完成后，是否需要自动路由到download进行确认
    pending_video_url = state.get("pending_video_url", "")
    awaiting_confirmation = state.get("awaiting_download_confirmation", False)
    
    if pending_video_url and awaiting_confirmation:
        output = {
            "next_agent": "download",
            "supervisor_instruction": f"用户请求下载视频，请生成确认请求。视频URL：{pending_video_url}",
            "execution_history": execution_history,
            "retry_count": 0,
            "pending_video_url": pending_video_url,
            "awaiting_download_confirmation": False,
            "current_plan": current_plan,
            "plan_active": plan_active,
        }
        exec_visualizer.on_node_end("supervisor", state, output)
        return output

    # 关键词预检查：视觉/摄像头相关请求直接路由到 vision 节点
    vision_already_executed = any(step.get("agent") == "vision" for step in execution_history)
    if not vision_already_executed:
        vision_keywords = ["摄像头", "拍照", "截图", "捕获", "图像", "画面", "视觉", "ocr", "识别文字",
                           "看", "看到", "看到什么", "前面有什么", "拍", "拍到的", "拍到的内容",
                           "capture", "camera", "vision", "webcam", "photo", "screenshot"]
        if any(kw in last_user_msg.lower() for kw in vision_keywords):
            output = {
                "next_agent": "vision",
                "supervisor_instruction": f"用户请求涉及摄像头或图像操作，请使用视觉工具处理。用户原始消息：{last_user_msg}",
                "execution_history": execution_history,
                "retry_count": 0,
                "pending_video_url": state.get("pending_video_url", ""),
                "awaiting_download_confirmation": state.get("awaiting_download_confirmation", False),
                "current_plan": current_plan,
                "plan_active": plan_active,
            }
            exec_visualizer.on_node_end("supervisor", state, output)
            return output

    # 构建路由提示
    routing_content = f"用户消息：{last_user_msg}\n\n"
    
    if execution_history:
        routing_content += "执行历史：\n"
        for i, step in enumerate(execution_history):
            routing_content += f"{i+1}. Agent: {step['agent']}\n"
            if "tools_used" in step and step["tools_used"]:
                routing_content += f"   使用的工具: {', '.join(step['tools_used'])}\n"
            routing_content += f"   输出: {step.get('output', '')[:300]}...\n"
        routing_content += f"\n重试次数：{retry_count}\n\n"

    # 如果有正在执行的任务计划，加入计划信息
    if plan_active and current_plan:
        plan_summary = "当前任务计划：\n"
        for i, task in enumerate(current_plan):
            status_icon = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(task.get("status", "pending"), "⏳")
            plan_summary += f"  {status_icon} 任务{i+1}: {task.get('content', '')} [{task.get('status', 'pending')}]\n"
        routing_content += plan_summary + "\n"
        routing_content += "请根据当前任务计划决定下一步：如果上一个任务已完成，设置task_completed=true并分派下一个任务；如果未完成，重新分派或调整策略。\n\n"

    routing_content += "请分析并输出决策和指令："

    routing_messages = [
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=routing_content),
    ]

    model = _make_model()
    response = model.invoke(routing_messages)
    response_text = response.content.strip()

    # 解析JSON响应
    decision = "FINISH"
    instruction = ""
    task_completed = False
    try:
        json_str = response_text
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(json_str)
        decision = parsed.get("decision", "FINISH").strip().lower()
        instruction = parsed.get("instruction", "")
        task_completed = parsed.get("task_completed", False)
    except (json.JSONDecodeError, KeyError, IndexError):
        text_lower = response_text.lower()
        if "scheduler" in text_lower:
            decision = "scheduler"
        elif "knowledge" in text_lower:
            decision = "knowledge"
        elif "coding" in text_lower:
            decision = "coding"
        elif "reminder" in text_lower:
            decision = "reminder"
        elif "download" in text_lower:
            decision = "download"
        elif "cityinfo" in text_lower:
            decision = "cityinfo"
        elif "vision" in text_lower:
            decision = "vision"
        elif "chat" in text_lower:
            decision = "chat"
        else:
            decision = "FINISH"
        instruction = ""

    # 处理任务计划的状态更新
    new_plan = current_plan
    new_plan_active = plan_active
    if plan_active and task_completed and current_plan:
        result = advance_to_next_task()
        if result["all_completed"]:
            new_plan = []
            new_plan_active = False
            decision = "FINISH"
            instruction = "所有任务已完成"
        elif result["has_next"]:
            new_plan = get_current_plan()
            next_task = result["next_task"]
            decision = "coding"
            instruction = next_task.get("content", instruction)

    # 确定下一个节点
    if "scheduler" in decision:
        next_agent = "scheduler"
    elif "knowledge" in decision:
        next_agent = "knowledge"
    elif "coding" in decision:
        next_agent = "coding"
    elif "reminder" in decision:
        next_agent = "reminder"
    elif "download" in decision:
        next_agent = "download"
    elif "cityinfo" in decision:
        next_agent = "cityinfo"
    elif "vision" in decision:
        next_agent = "vision"
    elif "chat" in decision:
        next_agent = "chat"
    else:
        next_agent = "FINISH"

    if retry_count >= 2:
        next_agent = "FINISH"

    # 检测下载意图
    should_set_download_intent = (
        next_agent == "knowledge" and
        "bilibili_video_search" in instruction and
        any(kw in last_user_msg.lower() for kw in ["下载", "下载视频", "下載"]) and
        not any(url in instruction for url in ["https://", "BV", "bilibili.com"])
    )

    output = {
        "next_agent": next_agent,
        "supervisor_instruction": instruction,
        "execution_history": execution_history,
        "retry_count": retry_count + 1 if next_agent != "FINISH" else 0,
        "pending_video_url": state.get("pending_video_url", ""),
        "awaiting_download_confirmation": should_set_download_intent or state.get("awaiting_download_confirmation", False),
        "current_plan": new_plan,
        "plan_active": new_plan_active,
    }

    exec_visualizer.on_node_end("supervisor", state, output)
    return output


# ============================================================
# 6. 子 Agent 节点
# ============================================================

def _extract_tools_used(result: dict) -> list:
    """从 Agent 执行结果中提取使用的工具列表"""
    tools_used = []
    for msg in result["messages"]:
        # 检查是否是 ToolMessage（表明工具被调用）
        if hasattr(msg, "type") and msg.type == "tool":
            if hasattr(msg, "name") and msg.name:
                tools_used.append(msg.name)
        # 检查 Additional_kwargs 中是否包含工具信息
        if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
            if "tool_calls" in msg.additional_kwargs:
                for tc in msg.additional_kwargs["tool_calls"]:
                    if "name" in tc:
                        tools_used.append(tc["name"])
    # 去重并保持顺序
    return list(dict.fromkeys(tools_used))


def _build_agent_messages(state: AgentState) -> list:
    """构建子agent的输入消息，注入SUPERVISOR指令"""
    messages = list(state["messages"])
    instruction = state.get("supervisor_instruction", "")
    if instruction:
        messages.append(SystemMessage(content=f"[调度指令] {instruction}"))
    return messages


def knowledge_node(state: AgentState) -> dict:
    exec_visualizer.on_node_start("knowledge", state)

    agent_messages = _build_agent_messages(state)
    result = knowledge_agent.invoke({"messages": agent_messages}, config={"callbacks": [tool_event_handler]})
    ai_msg = result["messages"][-1]
    
    tools_used = _extract_tools_used(result)
    
    # 检查是否调用了bilibili_video_search，如果是则提取第一个视频URL
    pending_video_url = ""
    if "bilibili_video_search" in tools_used:
        
        # 从搜索结果中提取第一个B站视频链接
        url_match = re.search(r'https?://www\.bilibili\.com/video/\S+', ai_msg.content)
        if url_match:
            pending_video_url = url_match.group(0).rstrip(')').rstrip(']').rstrip('。').rstrip('.')
        else:
            # 尝试匹配BV号
            bv_match = re.search(r'BV[a-zA-Z0-9]+', ai_msg.content)
            if bv_match:
                pending_video_url = f"https://www.bilibili.com/video/{bv_match.group(0)}"
    
    execution_history = state.get("execution_history", [])
    execution_history.append({
        "agent": "knowledge",
        "output": ai_msg.content,
        "tools_used": tools_used if tools_used else [],
        "timestamp": exec_visualizer.steps[-1].timestamp if exec_visualizer.steps else ""
    })
    
    output = {
        "messages": [ai_msg], 
        "next_agent": "FINISH",
        "supervisor_instruction": "",
        "execution_history": execution_history,
        "last_agent": "knowledge",
        "last_output": ai_msg.content,
        "pending_video_url": pending_video_url or state.get("pending_video_url", ""),
        "awaiting_download_confirmation": state.get("awaiting_download_confirmation", False),
        "current_plan": state.get("current_plan", []),
        "plan_active": state.get("plan_active", False),
    }

    exec_visualizer.on_node_end("knowledge", state, output, tools_used)
    return output


def reminder_node(state: AgentState) -> dict:
    exec_visualizer.on_node_start("reminder", state)

    agent_messages = _build_agent_messages(state)
    result = reminder_agent.invoke({"messages": agent_messages}, config={"callbacks": [tool_event_handler]})
    ai_msg = result["messages"][-1]
    
    tools_used = _extract_tools_used(result)
    
    execution_history = state.get("execution_history", [])
    execution_history.append({
        "agent": "reminder",
        "output": ai_msg.content,
        "tools_used": tools_used if tools_used else [],
        "timestamp": exec_visualizer.steps[-1].timestamp if exec_visualizer.steps else ""
    })
    
    output = {
        "messages": [ai_msg], 
        "next_agent": "FINISH",
        "supervisor_instruction": "",
        "execution_history": execution_history,
        "last_agent": "reminder",
        "last_output": ai_msg.content,
        "pending_video_url": state.get("pending_video_url", ""),
        "awaiting_download_confirmation": state.get("awaiting_download_confirmation", False),
        "current_plan": state.get("current_plan", []),
        "plan_active": state.get("plan_active", False),
    }

    exec_visualizer.on_node_end("reminder", state, output, tools_used)
    return output


def cityinfo_node(state: AgentState) -> dict:
    exec_visualizer.on_node_start("cityinfo", state)

    agent_messages = _build_agent_messages(state)
    result = city_info_agent.invoke({"messages": agent_messages}, config={"callbacks": [tool_event_handler]})
    ai_msg = result["messages"][-1]
    
    tools_used = _extract_tools_used(result)
    
    execution_history = state.get("execution_history", [])
    execution_history.append({
        "agent": "cityinfo",
        "output": ai_msg.content,
        "tools_used": tools_used if tools_used else [],
        "timestamp": exec_visualizer.steps[-1].timestamp if exec_visualizer.steps else ""
    })
    
    output = {
        "messages": [ai_msg], 
        "next_agent": "FINISH",
        "supervisor_instruction": "",
        "execution_history": execution_history,
        "last_agent": "cityinfo",
        "last_output": ai_msg.content,
        "pending_video_url": state.get("pending_video_url", ""),
        "awaiting_download_confirmation": state.get("awaiting_download_confirmation", False),
        "current_plan": state.get("current_plan", []),
        "plan_active": state.get("plan_active", False),
    }

    exec_visualizer.on_node_end("cityinfo", state, output, tools_used)
    return output


def _extract_pending_download(messages: list) -> dict | None:
    """从对话历史中检测待确认的下载请求（只检查最近一条AI消息）"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if "[PENDING_DOWNLOAD]" in msg.content and "[/PENDING_DOWNLOAD]" in msg.content:
                try:
                    content = msg.content
                    url_start = content.index("url=") + 4
                    url_end = content.index("\n", url_start) if "\n" in content[url_start:] else content.index("[/PENDING_DOWNLOAD]", url_start)
                    video_url = content[url_start:url_end].strip()
                    
                    dir_str = ""
                    if "dir=" in content:
                        dir_start = content.index("dir=") + 4
                        dir_end = content.index("\n", dir_start) if "\n" in content[dir_start:] else content.index("[/PENDING_DOWNLOAD]", dir_start)
                        dir_str = content[dir_start:dir_end].strip()
                    
                    return {"video_url": video_url, "output_dir": dir_str or "."}
                except (ValueError, IndexError):
                    return None
            # 只检查最近一条AI消息，如果不是待确认下载就停止
            break
    return None


def download_node(state: AgentState) -> dict:
    """下载确认节点 - 生成确认消息后结束，等待用户下次输入确认"""
    exec_visualizer.on_node_start("download", state)

    instruction = state.get("supervisor_instruction", "")
    
    # 优先从状态中读取待下载URL（由knowledge搜索后设置）
    video_url = state.get("pending_video_url", "")
    
    # 如果状态中没有URL，才从指令中提取
    if not video_url:
        url_match = re.search(r'https?://www\.bilibili\.com/video/\S+', instruction)
        if url_match:
            video_url = url_match.group(0).rstrip('"').rstrip("'").rstrip("，").rstrip(",")
        
        if not video_url:
            url_match2 = re.search(r'BV[a-zA-Z0-9]+', instruction)
            if url_match2:
                video_url = f"https://www.bilibili.com/video/{url_match2.group(0)}"
    
    output_dir = "."
    dir_match = re.search(r'下载到[：:]?\s*(\S+)', instruction)
    if dir_match:
        output_dir = dir_match.group(1).strip('"').strip("'").rstrip("，").rstrip(",")

    # 如果仍然没有URL，返回错误提示
    if not video_url:
        no_url_msg = (
            "无法获取视频链接。\n"
            "请先使用 bilibili_video_search 工具搜索视频，找到具体链接后再请求下载。\n"
            "或者直接提供要下载的视频链接。"
        )
        execution_history = state.get("execution_history", [])
        execution_history.append({
            "agent": "download",
            "output": "无法获取视频链接",
            "tools_used": [],
            "timestamp": exec_visualizer.steps[-1].timestamp if exec_visualizer.steps else ""
        })
        output = {
            "messages": [AIMessage(content=no_url_msg)],
            "next_agent": "FINISH",
            "supervisor_instruction": "",
            "execution_history": execution_history,
            "last_agent": "download",
            "last_output": no_url_msg,
            "pending_video_url": "",
            "current_plan": state.get("current_plan", []),
            "plan_active": state.get("plan_active", False),
        }
        exec_visualizer.on_node_end("download", state, output)
        return output

    confirmation_msg = (
        "我准备为你下载视频，需要你确认以下信息：\n\n"
        f"📺 视频链接：{video_url}\n"
        f"📁 下载目录：{output_dir}\n\n"
        "请确认是否要继续下载？\n"
        "回复「确认」或「是」开始下载\n"
        "回复「取消」或「否」取消下载\n\n"
        f"[PENDING_DOWNLOAD]\nurl={video_url}\ndir={output_dir}\n[/PENDING_DOWNLOAD]"
    )

    execution_history = state.get("execution_history", [])
    execution_history.append({
        "agent": "download",
        "output": f"等待用户确认下载：{video_url}",
        "tools_used": [],
        "timestamp": exec_visualizer.steps[-1].timestamp if exec_visualizer.steps else ""
    })

    output = {
        "messages": [AIMessage(content=confirmation_msg)],
        "next_agent": "FINISH",
        "supervisor_instruction": "",
        "execution_history": execution_history,
        "last_agent": "download",
        "last_output": confirmation_msg,
        "pending_video_url": "",
        "current_plan": state.get("current_plan", []),
        "plan_active": state.get("plan_active", False),
    }

    exec_visualizer.on_node_end("download", state, output)
    return output


# ============================================================
# 7. 直接回答节点（闲聊 / 无需工具的场景）
# ============================================================

def direct_answer_node(state: AgentState) -> dict:
    exec_visualizer.on_node_start("direct_answer", state)

    model = _make_model()
    instruction = state.get("supervisor_instruction", "")
    system_content = (
        "你是 Sixty-Three，一个友好且智能的AI助手。\n"
        "你像 Tony Stark 的 Friday 一样，用简洁自然的方式回答用户的问题。"
    )
    if instruction:
        system_content += f"\n\n[调度指令] {instruction}"
    system_msg = SystemMessage(content=system_content)
    all_msgs = [system_msg] + state["messages"]
    response = model.invoke(all_msgs)
    
    tools_used = []
    
    execution_history = state.get("execution_history", [])
    execution_history.append({
        "agent": "direct_answer",
        "output": response.content,
        "tools_used": tools_used if tools_used else [],
        "timestamp": exec_visualizer.steps[-1].timestamp if exec_visualizer.steps else ""
    })
    
    output = {
        "messages": [response], 
        "next_agent": "FINISH",
        "supervisor_instruction": "",
        "execution_history": execution_history,
        "last_agent": "direct_answer",
        "last_output": response.content,
        "current_plan": state.get("current_plan", []),
        "plan_active": state.get("plan_active", False),
    }

    exec_visualizer.on_node_end("direct_answer", state, output)
    return output


def vision_node(state: AgentState) -> AgentState:

    """视觉处理节点"""
    exec_visualizer.on_node_start("vision", state)

    messages = state["messages"]
    execution_history = state.get("execution_history", [])

    # 执行 Agent
    response = vision_agent.invoke({"messages": messages}, config={"callbacks": [tool_event_handler]})
    output_message = response["messages"][-1]

    # 提取使用的工具
    tools_used = _extract_tools_used(response)

    # 记录执行历史
    execution_history.append({
        "agent": "vision",
        "output": output_message.content,
        "tools_used": tools_used if tools_used else [],
        "timestamp": exec_visualizer.steps[-1].timestamp if exec_visualizer.steps else ""
    })

    output = {
        "messages": [output_message], 
        "next_agent": "FINISH",
        "supervisor_instruction": "",
        "execution_history": execution_history,
        "last_agent": "vision",
        "last_output": output_message.content,
        "current_plan": state.get("current_plan", []),
        "plan_active": state.get("plan_active", False),
    }
    
    exec_visualizer.on_node_end("vision", state, output, tools_used)
    return output


def coding_node(state: AgentState) -> AgentState:
    """代码处理节点 - 包含代码修改和测试验证的反馈循环"""
    exec_visualizer.on_node_start("coding", state)

    agent_messages = _build_agent_messages(state)
    execution_history = state.get("execution_history", [])

    response = coding_agent.invoke({"messages": agent_messages}, config={"callbacks": [tool_event_handler]})
    output_message = response["messages"][-1]

    tools_used = _extract_tools_used(response)

    execution_history.append({
        "agent": "coding",
        "output": output_message.content,
        "tools_used": tools_used if tools_used else [],
        "timestamp": exec_visualizer.steps[-1].timestamp if exec_visualizer.steps else ""
    })

    output = {
        "messages": [output_message], 
        "next_agent": "supervisor",
        "supervisor_instruction": "",
        "execution_history": execution_history,
        "last_agent": "coding",
        "last_output": output_message.content,
        "current_plan": state.get("current_plan", []),
        "plan_active": state.get("plan_active", False),
    }

    exec_visualizer.on_node_end("coding", state, output, tools_used)
    return output


def scheduler_node(state: AgentState) -> dict:
    """任务调度节点 - 创建任务计划并返回给supervisor审核"""
    exec_visualizer.on_node_start("scheduler", state)

    agent_messages = _build_agent_messages(state)
    result = task_scheduler_agent.invoke({"messages": agent_messages}, config={"callbacks": [tool_event_handler]})
    ai_msg = result["messages"][-1]

    tools_used = _extract_tools_used(result)

    plan = get_current_plan()

    execution_history = state.get("execution_history", [])
    execution_history.append({
        "agent": "scheduler",
        "output": ai_msg.content,
        "tools_used": tools_used if tools_used else [],
        "timestamp": exec_visualizer.steps[-1].timestamp if exec_visualizer.steps else ""
    })

    output = {
        "messages": [ai_msg],
        "next_agent": "supervisor",
        "supervisor_instruction": "",
        "execution_history": execution_history,
        "last_agent": "scheduler",
        "last_output": ai_msg.content,
        "current_plan": plan,
        "plan_active": bool(plan),
        "pending_video_url": state.get("pending_video_url", ""),
        "awaiting_download_confirmation": state.get("awaiting_download_confirmation", False),
    }

    exec_visualizer.on_node_end("scheduler", state, output, tools_used)
    return output


# ============================================================
# 8. 路由函数
# ============================================================

def route_from_supervisor(state: AgentState) -> str:
    return state["next_agent"]


def _route_from_sub_agent(state: AgentState) -> str:
    """子 Agent 执行后的路由：根据 next_agent 决定返回 supervisor 还是结束"""
    next_agent = state.get("next_agent", "supervisor")
    if next_agent == "FINISH":
        return "FINISH"
    return "supervisor"


# ============================================================
# 9. 构建 LangGraph 工作流
# ============================================================

def build_multi_agent_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("download", download_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("reminder", reminder_node)
    builder.add_node("cityinfo", cityinfo_node)
    builder.add_node("direct_answer", direct_answer_node)
    builder.add_node("vision", vision_node)
    builder.add_node("coding", coding_node)
    builder.add_node("scheduler", scheduler_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "knowledge": "knowledge",
            "reminder": "reminder",
            "download": "download",
            "cityinfo": "cityinfo",
            "chat": "direct_answer",
            "vision": "vision",
            "coding": "coding",
            "scheduler": "scheduler",
            "FINISH": END,
        },
    )

    # 子 Agent 执行后返回到 SUPERVISOR 进行重新思考
    builder.add_edge("knowledge", "supervisor")
    builder.add_edge("reminder", "supervisor")
    builder.add_edge("scheduler", "supervisor")
    builder.add_conditional_edges(
        "vision",
        _route_from_sub_agent,
        {"supervisor": "supervisor", "FINISH": END},
    )
    builder.add_conditional_edges(
        "coding",
        _route_from_sub_agent,
        {"supervisor": "supervisor", "FINISH": END},
    )
    builder.add_edge("download", END)
    builder.add_edge("cityinfo", "supervisor")
    builder.add_edge("direct_answer", END)

    return builder.compile()


# ============================================================
# 10. 对外接口 —— 替代原有 chat_with_agent
# ============================================================

multi_agent_graph = build_multi_agent_graph()


def chat_with_multi_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session", verbose: bool = True):
    """使用多 Agent 协作处理用户消息"""
    from .tools import set_current_user_id, get_last_rag_context, reset_tool_call_guards

    set_current_user_id(user_id)
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    exec_visualizer.reset()

    messages = storage.load(user_id, session_id)

    if len(messages) > 50:
        summary = summarize_old_messages(single_model, messages[:40])
        messages = [SystemMessage(content=f"之前的对话摘要：\n{summary}")] + messages[40:]

    messages.append(HumanMessage(content=user_text))

    result = multi_agent_graph.invoke(
        {
            "messages": messages, 
            "next_agent": "",
            "supervisor_instruction": "",
            "execution_history": [],
            "retry_count": 0,
            "last_agent": "",
            "last_output": "",
            "current_plan": get_current_plan(),
            "plan_active": bool(get_current_plan()),
        },
        config={"recursion_limit": 15},
    )

    response_content = ""
    raw_content = ""
    if result.get("messages"):
        last_msg = result["messages"][-1]
        raw_content = getattr(last_msg, "content", str(last_msg))
        # 过滤掉结构化标记，不暴露给用户
        if "[PENDING_DOWNLOAD]" in raw_content:
            response_content = re.sub(r'\[PENDING_DOWNLOAD\].*?\[/PENDING_DOWNLOAD\]', '', raw_content, flags=re.DOTALL).strip()
        else:
            response_content = raw_content

    # 保存到对话历史时保留原始内容（含标记），以便下次SUPERVISOR检测
    messages.append(AIMessage(content=raw_content))

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None
    
    # 生成 agent 工具调用记录
    execution_history = result.get("execution_history", [])
    agent_tool_trace = []
    for step in execution_history:
        agent_name = step.get("agent", "")
        tools_used = step.get("tools_used", [])
        if agent_name and tools_used:
            tool_names = ",".join(tools_used)
            agent_tool_trace.append(f"{agent_name}({tool_names})")
        elif agent_name:
            agent_tool_trace.append(agent_name)
    
    # 转换为 "xxxnode→xxxnode→..." 格式
    agent_tool_path = "→".join(agent_tool_trace)
    
    extra_message_data = [None] * (len(messages) - 1) + [{
        "rag_trace": rag_trace,
        "agent_tool_path": agent_tool_path
    }]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    if verbose:
        print("\n" + "=" * 60)
        print(" 多 Agent 执行追踪")
        print("=" * 60)
        exec_visualizer.print_trace()

    return {
        "response": response_content,
        "rag_trace": rag_trace,
        "execution_trace": exec_visualizer.get_execution_trace(),
    }


# ============================================================
# 11. 独立运行测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangGraph 多 Agent 协作 Demo")
    print("=" * 60)
    print()
    print("可用指令示例：")
    print("  - 天气查询：北京今天天气怎么样？")
    print("  - 设置提醒：明天下午3点提醒我开会")
    print("  - 知识检索：文档中关于XXX的内容是什么？")
    print("  - 闲聊：你好，你是谁？")
    print("  - 退出：quit")
    print()

    test_messages = []
    while True:
        user_input = input("\n🧑 你：").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        result = chat_with_multi_agent(
            user_text=user_input,
            user_id="demo_user",
            session_id="demo_session",
            verbose=True,
        )

        if result.get("response"):
            print(f"\n🤖 Sixty-Three：{result['response']}")
