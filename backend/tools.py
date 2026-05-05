from typing import Optional, List
import os
import requests
import datetime
from zoneinfo import ZoneInfo
import threading
import json
import re
import time
import urllib.parse
from functools import reduce
from hashlib import md5
from dotenv import load_dotenv
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

load_dotenv()

AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None
_RAG_STEP_LOOP = None
_CURRENT_USER_ID = None

_DOWNLOAD_PROGRESS_CALLBACK = None
_DOWNLOAD_PROGRESS_LOCK = threading.Lock()

def set_download_progress_callback(callback):
    global _DOWNLOAD_PROGRESS_CALLBACK
    with _DOWNLOAD_PROGRESS_LOCK:
        _DOWNLOAD_PROGRESS_CALLBACK = callback

def get_download_progress_callback():
    global _DOWNLOAD_PROGRESS_CALLBACK
    with _DOWNLOAD_PROGRESS_LOCK:
        return _DOWNLOAD_PROGRESS_CALLBACK

def clear_download_progress_callback():
    global _DOWNLOAD_PROGRESS_CALLBACK
    with _DOWNLOAD_PROGRESS_LOCK:
        _DOWNLOAD_PROGRESS_CALLBACK = None

_TOOL_EVENT_CALLBACK = None
_TOOL_EVENT_LOCK = threading.Lock()

def set_tool_event_callback(callback):
    global _TOOL_EVENT_CALLBACK
    with _TOOL_EVENT_LOCK:
        _TOOL_EVENT_CALLBACK = callback

def clear_tool_event_callback():
    global _TOOL_EVENT_CALLBACK
    with _TOOL_EVENT_LOCK:
        _TOOL_EVENT_CALLBACK = None

def _fire_tool_event(event_data: dict):
    with _TOOL_EVENT_LOCK:
        callback = _TOOL_EVENT_CALLBACK
    if callback:
        callback(event_data)

_CURRENT_STEP_ID = None

def set_current_step_id(step_id):
    global _CURRENT_STEP_ID
    _CURRENT_STEP_ID = step_id

def get_current_step_id():
    return _CURRENT_STEP_ID


def _set_last_rag_context(context: dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0


def set_current_user_id(user_id: Optional[str]):
    """设置当前用户ID"""
    global _CURRENT_USER_ID
    _CURRENT_USER_ID = user_id


def get_current_user_id() -> Optional[str]:
    """获取当前用户ID"""
    return _CURRENT_USER_ID


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP = asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        except Exception:
            pass

def get_city_timezone(city: str) -> str:
    """根据城市名称获取对应的时区
    
    Args:
        city: 城市名称
    
    Returns:
        时区字符串，如 "Asia/Shanghai"
    """
    city_timezone_map = {
        "北京": "Asia/Shanghai",
        "上海": "Asia/Shanghai",
        "广州": "Asia/Shanghai",
        "深圳": "Asia/Shanghai",
        "杭州": "Asia/Shanghai",
        "成都": "Asia/Shanghai",
        "纽约": "America/New_York",
        "洛杉矶": "America/Los_Angeles",
        "伦敦": "Europe/London",
        "巴黎": "Europe/Paris",
        "东京": "Asia/Tokyo",
        "首尔": "Asia/Seoul",
        "悉尼": "Australia/Sydney"
    }
    
    # 尝试匹配拼音或英文名称
    city_aliases = {
        "beijing": "Asia/Shanghai",
        "shanghai": "Asia/Shanghai",
        "guangzhou": "Asia/Shanghai",
        "shenzhen": "Asia/Shanghai",
        "hangzhou": "Asia/Shanghai",
        "chengdu": "Asia/Shanghai",
        "new york": "America/New_York",
        "los angeles": "America/Los_Angeles",
        "london": "Europe/London",
        "paris": "Europe/Paris",
        "tokyo": "Asia/Tokyo",
        "seoul": "Asia/Seoul",
        "sydney": "Australia/Sydney"
    }
    
    lowercase_city = city.lower()

        # 尝试直接匹配城市名
    if city in city_timezone_map:
        return city_timezone_map[city]

    elif lowercase_city in city_aliases:
        return city_aliases[lowercase_city]
    
    else:
        return "Asia/Shanghai"


_USER_OPENID_MAP = None
_USER_OPENID_MAP_LOCK = threading.Lock()


def _get_user_openid_map() -> dict:
    global _USER_OPENID_MAP
    if _USER_OPENID_MAP is not None:
        return _USER_OPENID_MAP
    with _USER_OPENID_MAP_LOCK:
        if _USER_OPENID_MAP is not None:
            return _USER_OPENID_MAP
        mapping_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_openid_map.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                _USER_OPENID_MAP = json.load(f)
        else:
            _USER_OPENID_MAP = {}
    return _USER_OPENID_MAP


def _save_user_openid_map():
    global _USER_OPENID_MAP
    if _USER_OPENID_MAP is None:
        return
    with _USER_OPENID_MAP_LOCK:
        mapping_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_openid_map.json')
        os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(_USER_OPENID_MAP, f, ensure_ascii=False, indent=2)


def _get_wechat_client():
    from .wechat_reminder import WeChatOfficialAccount
    appid = os.getenv("WECHAT_APP_ID", "")
    appsecret = os.getenv("WECHAT_APP_SECRET", "")
    if not appid or not appsecret:
        return None
    return WeChatOfficialAccount(appid=appid, appsecret=appsecret)


def _get_user_openid(user_id: str) -> str:
    default_openid = os.getenv("WECHAT_DEFAULT_OPENID", "")
    mapping = _get_user_openid_map()
    return mapping.get(user_id, default_openid)


def _get_global_scheduler():
    """获取全局调度器实例"""
    from .reminder_scheduler import get_global_scheduler
    scheduler = get_global_scheduler()
    client = _get_wechat_client()
    if client:
        scheduler.set_wechat_client(client)
    return scheduler


@tool("set_reminder")
def set_reminder(content: str, remind_time: str) -> str:
    """Set a timed reminder that will send a WeChat notification at the specified time.
    Use this when the user asks to set a reminder, alarm, or scheduled notification.
    
    Args:
        content: The reminder content, e.g. "吃药", "开会", "提交报告"
        remind_time: The reminder time in format "YYYY-MM-DD HH:MM", e.g. "2026-04-15 14:30"
    """
    try:
        reminder_dt = datetime.datetime.strptime(remind_time.strip(), "%Y-%m-%d %H:%M")

        now = datetime.datetime.now()
        if reminder_dt <= now:
            return f"提醒时间已过，请设置未来的时间。当前时间：{now.strftime('%Y-%m-%d %H:%M')}"

        user_id = _CURRENT_USER_ID or "default_user"
        openid = _get_user_openid(user_id)

        if not openid:
            return (
                f"⚠️ 未配置微信推送，提醒仅在服务器端记录。\n"
                f"提醒内容：{content.strip()}\n"
                f"提醒时间：{remind_time.strip()}\n"
                f"请在 .env 中设置 WECHAT_DEFAULT_OPENID，或在微信公众号中发送「绑定」完成账号关联。"
            )

        client = _get_wechat_client()
        if client is None:
            return (
                f"⚠️ 微信公众号未配置（缺少 WECHAT_APP_ID 或 WECHAT_APP_SECRET），提醒仅在服务器端记录。\n"
                f"提醒内容：{content.strip()}\n"
                f"提醒时间：{remind_time.strip()}"
            )

        # 使用全局调度器
        scheduler = _get_global_scheduler()
        
        # 生成唯一任务ID
        import uuid
        task_id = f"reminder_{user_id}_{uuid.uuid4().hex[:8]}"
        
        title = f"{content.strip()}提醒"
        result = scheduler.add_reminder(
            task_id=task_id,
            remind_time=reminder_dt,
            title=title,
            content=content.strip(),
            openid=openid
        )

        if result.get("success"):
            wait_text = result.get("message", "")
            return (
                f"✅ 已设置微信提醒：{content.strip()}\n"
                f"⏰ 提醒时间：{remind_time.strip()}\n"
                f"📱 将通过微信公众号推送通知\n"
                f"{wait_text}\n"
                f"⚠️ 请确保48小时内与公众号有过互动，否则消息无法送达。"
            )
        else:
            return f"设置提醒失败：{result.get('message', '未知错误')}"
    except ValueError:
        return f"时间格式错误，请使用格式：YYYY-MM-DD HH:MM（例如：2026-04-15 14:30）"
    except Exception as e:
        return f"设置提醒失败：{str(e)}"


# def get_current_city_info(location: str, extensions: Optional[str] = "base") -> str:
#     """获取城市信息"""
#     if not location:
#         return "location参数不能为空"
#     if extensions not in ("base", "all"):
#         return "extensions参数错误，请输入base或all"

#     if not AMAP_CITY_INFO_API or not AMAP_API_KEY:
#         return "城市信息服务未配置（缺少 AMAP_CITY_INFO_API 或 AMAP_API_KEY）"

#     params = {
#         "key": AMAP_API_KEY,
#         "city": location,
#         "extensions": extensions,
#         "output": "json",
#     }

#     try:
#         resp = requests.get(AMAP_CITY_INFO_API, params=params, timeout=10)
#         resp.raise_for_status()
#         data = resp.json()
#         if data.get("status") != "1":
#             return f"查询失败：{data.get('info', '未知错误')}"

#         if extensions == "base":
#             lives = data.get("lives", [])
#             if not lives:
#                 return f"未查询到 {location} 的城市信息数据"
#             w = lives[0]
#             return (
#                 f"【{w.get('city', location)} 实时城市信息】\n"
#                 f"天气状况：{w.get('city_info', '未知')}\n"
#                 f"温度：{w.get('temperature', '未知')}℃\n"
#                 f"湿度：{w.get('humidity', '未知')}%\n"
#                 f"风向：{w.get('winddirection', '未知')}\n"
#                 f"风力：{w.get('windpower', '未知')}级\n"
#                 f"更新时间：{w.get('reporttime', '未知')}"
#             )

#         forecasts = data.get("forecasts", [])
#         if not forecasts:
#             return f"未查询到 {location} 的城市信息数据"
#         f0 = forecasts[0]
#         out = [f"【{f0.get('city', location)} 城市信息】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
#         today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
#         out += [
#             "今日城市信息：",
#             f"  白天：{today.get('dayweather','未知')}",
#             f"  夜间：{today.get('nightweather','未知')}",
#             f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
#         ]
#         return "\n".join(out)

#     except requests.exceptions.Timeout:
#         return "错误：请求天气服务超时"
#     except requests.exceptions.RequestException as e:
#         return f"错误：天气服务请求失败 - {e}"
#     except Exception as e:
#         return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1



@tool("web_search")
def web_search(query: str, num_results: int = 10) -> str:
    """Search the web for information on a given query.
    Use this when the user asks for current information, news, or facts that may not be in the knowledge base.
    
    Args:
        query: The search query, e.g. "2026年巴黎奥运会开幕时间", "最新的人工智能技术进展"
        num_results: The number of search results to return (default: 10)
    """
    try:
        from dotenv import load_dotenv
        import os
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path=env_path)
        # 使用百度 API 进行搜索（需要设置 BAIDU_WEB_SEARCH_API_KEY 环境变量）
        baidu_api_key = os.getenv("BAIDU_WEB_SEARCH_API_KEY")
        if baidu_api_key:
            import requests
            import json
            
            # 百度搜索 API 地址（根据实际 API 文档调整）
            url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
            
            # 构建请求参数
            headers = {
                "Content-Type": "application/json",
                "X-Appbuilder-Authorization": f"Bearer {baidu_api_key}"
            }
            
            payload = {
                "messages": [
                        {
                            "content": query,
                            "role": "user"
                        }
                    ],
                "search_source": "baidu_search_v2",
                "resource_type_filter": [{"type": "web","top_k": num_results}]
            }
            
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            data = response.json()
            
            if "references" in data:
                results = []
                for i, item in enumerate(data["references"]):
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("url", "")
                    results.append(f"{i+1}. {title}\n{snippet}\n{link}")
                return "\n\n".join(results)
            elif "result" in data and "items" in data["result"]:
                results = []
                for i, item in enumerate(data["result"]["items"]):
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("url", "")
                    results.append(f"{i+1}. {title}\n{snippet}\n{link}")
                return "\n\n".join(results)
            else:
                return f"搜索失败，未找到结果{data}"
        else:
            return "搜索失败，未找到结果"
    except Exception as e:
        return f"搜索失败：{str(e)}"

    # from .rag_pipeline import run_rag_graph

    # user_id = get_current_user_id()
    # rag_result = run_rag_graph(query, user_id=user_id)

    # docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    # rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}    
    # if rag_trace:
    #     _set_last_rag_context({"rag_trace": rag_trace})

    # if not docs:
    #     return "No relevant documents found in the knowledge base."

    # formatted = []
    # for i, result in enumerate(docs, 1):
    #     source = result.get("filename", "Unknown")
    #     page = result.get("page_number", "N/A")
    #     text = result.get("text", "")
    #     formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    # return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)


@tool("list_directory")
def list_directory(path: str = ".") -> str:
    """列出指定目录下的文件和文件夹。
    
    Args:
        path: 目录路径，默认为当前目录（.）
        
    Returns:
        目录内容的格式化列表
    """
    try:
        # 确保路径是绝对路径
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        
        # 安全检查：防止访问敏感目录
        sensitive_dirs = [
            os.path.expanduser("~"),  # 用户目录
            "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
            "C:\\Users", "C:\\ProgramData"
        ]
        
        for sensitive_dir in sensitive_dirs:
            if path.lower().startswith(sensitive_dir.lower()):
                return f"安全限制：禁止访问敏感目录 {sensitive_dir}"
        
        # 检查目录是否存在
        if not os.path.exists(path):
            return f"错误：目录 '{path}' 不存在"
        
        if not os.path.isdir(path):
            return f"错误：'{path}' 不是一个目录"
        
        # 获取目录内容
        items = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                items.append({"name": item, "type": "directory", "path": item_path})
            else:
                items.append({"name": item, "type": "file", "path": item_path})
        
        # 格式化输出
        result = f"目录内容 ({path}):\n\n"
        for item in items:
            icon = "📁" if item["type"] == "directory" else "📄"
            result += f"{icon} {item['name']}\n"
        
        return result
        
    except PermissionError:
        return f"权限错误：无法访问目录 '{path}'"
    except Exception as e:
        return f"错误：{str(e)}"


_FILE_SUMMARY_MARKERS = {
    ".py": ("# ", ""),
    ".js": ("// ", ""),
    ".ts": ("// ", ""),
    ".tsx": ("// ", ""),
    ".jsx": ("// ", ""),
    ".java": ("// ", ""),
    ".c": ("// ", ""),
    ".cpp": ("// ", ""),
    ".h": ("// ", ""),
    ".go": ("// ", ""),
    ".rs": ("// ", ""),
    ".rb": ("# ", ""),
    ".sh": ("# ", ""),
    ".bat": ("REM ", ""),
    ".ps1": ("# ", ""),
    ".html": ("<!-- ", " -->"),
    ".css": ("/* ", " */"),
    ".sql": ("-- ", ""),
    ".lua": ("-- ", ""),
    ".swift": ("// ", ""),
    ".kt": ("// ", ""),
    ".vue": ("<!-- ", " -->"),
    ".yaml": ("# ", ""),
    ".yml": ("# ", ""),
    ".toml": ("# ", ""),
    ".ini": ("; ", ""),
    ".md": ("", ""),
}

_SUMMARY_TAG_START = "@summary "
_SUMMARY_TAG_END = "@end"


def _get_comment_style(file_path: str) -> tuple:
    ext = os.path.splitext(file_path)[1].lower()
    return _FILE_SUMMARY_MARKERS.get(ext, ("# ", "# "))


def _has_summary_header(content: str, line_prefix: str) -> bool:
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith(line_prefix.strip()):
            after = stripped[len(line_prefix.strip()):].strip()
            if after.startswith(_SUMMARY_TAG_START):
                return True
        if stripped and not stripped.startswith(line_prefix.strip()) and not stripped.startswith("#!") and not stripped.startswith("//!") and not stripped.startswith("<?"):
            break
    return False


def _extract_summary_block(content: str, line_prefix: str, line_suffix: str = "") -> str | None:
    lines = content.split("\n")
    summary_lines = []
    in_summary = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(line_prefix.strip()):
            after = stripped[len(line_prefix.strip()):].strip()
            if after.endswith(line_suffix.strip()):
                after = after[: -len(line_suffix.strip())].strip() if line_suffix.strip() else after
            if after.startswith(_SUMMARY_TAG_START):
                in_summary = True
                first_line = after[len(_SUMMARY_TAG_START):].strip()
                if first_line:
                    summary_lines.append(first_line)
                continue
            if in_summary:
                if after.startswith(_SUMMARY_TAG_END):
                    break
                summary_lines.append(after)
        elif in_summary:
            break
        elif stripped and not stripped.startswith("#!") and not stripped.startswith("//!") and not stripped.startswith("<?"):
            break
    return "\n".join(summary_lines) if summary_lines else None


def _build_summary_header(summary_text: str, line_prefix: str, line_suffix: str = "") -> str:
    lines = summary_text.strip().split("\n")
    header_lines = []
    header_lines.append(f"{line_prefix}{_SUMMARY_TAG_START}{lines[0].strip()}{line_suffix}")
    for line in lines[1:]:
        header_lines.append(f"{line_prefix}{line.strip()}{line_suffix}")
    header_lines.append(f"{line_prefix}{_SUMMARY_TAG_END}{line_suffix}")
    return "\n".join(header_lines)


def _ensure_summary_in_content(content: str, summary_text: str, file_path: str) -> str:
    line_prefix, line_suffix = _get_comment_style(file_path)
    if not line_prefix:
        return content
    if _has_summary_header(content, line_prefix):
        return content
    header = _build_summary_header(summary_text, line_prefix, line_suffix)
    shebang = ""
    rest = content
    if content.startswith("#!") or content.startswith("//!") or content.startswith("<?xml") or content.startswith("<?php"):
        first_newline = content.find("\n")
        if first_newline != -1:
            shebang = content[:first_newline + 1]
            rest = content[first_newline + 1:]
    return shebang + header + "\n\n" + rest


@tool("read_file_summary")
def read_file_summary(file_path: str, encoding: str = "utf-8") -> str:
    """读取文件的摘要描述（文件头部的 @summary 注释块），用于快速了解文件功能而无需读取全部内容。

    工作原理：
    - 每个代码文件头部应该有一个 @summary 注释块，简要描述该文件的功能
    - 本工具只读取这个摘要块，而不是整个文件，节省大量 token
    - 如果文件没有 @summary 块，则自动回退到读取全部内容，并提示你可以添加摘要

    使用场景：
    - 当你只需要知道某个文件是做什么的，不需要了解代码细节时
    - 当你需要浏览多个文件来找到特定功能在哪个文件中时
    - 当你需要快速了解项目结构时

    如果你需要查看代码的具体实现细节，请使用 read_file 工具读取完整内容。

    Args:
        file_path: 文件路径
        encoding: 文件编码，默认为utf-8

    Returns:
        文件摘要描述，或完整内容（如果没有摘要）
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return f"错误：文件 '{file_path}' 不存在"

        if not os.path.isfile(file_path):
            return f"错误：'{file_path}' 不是一个文件"

        line_prefix, line_suffix = _get_comment_style(file_path)
        if not line_prefix:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                content = f.read(5000)
            return f"文件类型不支持摘要注释，完整内容：\n{content}"

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            head_lines = []
            for i, line in enumerate(f):
                if i >= 30:
                    break
                head_lines.append(line)
            head_content = "".join(head_lines)

        summary = _extract_summary_block(head_content, line_prefix, line_suffix)
        if summary:
            return f"📄 文件摘要 ({file_path}):\n{summary}"

        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            full_content = f.read()
        max_chars = 8000
        if len(full_content) > max_chars:
            full_content = full_content[:max_chars] + "\n\n[文件内容过长，已截断]"
        return f"⚠️ 文件没有 @summary 摘要注释，回退读取完整内容 ({file_path}):\n\n{full_content}\n\n💡 建议使用 add_file_summary 工具为该文件添加摘要描述，以便后续快速读取。"

    except PermissionError:
        return f"权限错误：无法读取文件 '{file_path}'"
    except Exception as e:
        return f"错误：{str(e)}"


@tool("add_file_summary")
def add_file_summary(file_path: str, summary: str) -> str:
    """为文件添加 @summary 摘要注释块。如果文件已有摘要则更新。

    摘要注释会添加在文件头部（shebang 之后），格式为：
    # @summary 这里是文件功能描述
    # 具体说明第二行
    # @end

    Args:
        file_path: 文件路径
        summary: 文件功能的简要描述，1-3句话即可

    Returns:
        操作结果
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return f"错误：文件 '{file_path}' 不存在"

        if not os.path.isfile(file_path):
            return f"错误：'{file_path}' 不是一个文件"

        line_prefix, line_suffix = _get_comment_style(file_path)
        if not line_prefix:
            return f"错误：该文件类型不支持摘要注释"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if _has_summary_header(content, line_prefix):
            lines = content.split("\n")
            new_lines = []
            skip = False
            found_summary = False
            for line in lines:
                stripped = line.strip()
                after_prefix = stripped[len(line_prefix.strip()):].strip() if stripped.startswith(line_prefix.strip()) else stripped
                if after_prefix.startswith(_SUMMARY_TAG_START):
                    skip = True
                    found_summary = True
                    header = _build_summary_header(summary, line_prefix, line_suffix)
                    new_lines.extend(header.split("\n"))
                    continue
                if skip:
                    if after_prefix.startswith(_SUMMARY_TAG_END):
                        skip = False
                    continue
                new_lines.append(line)
            content = "\n".join(new_lines)
        else:
            content = _ensure_summary_in_content(content, summary, file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"✅ 摘要已添加/更新：{file_path}"

    except Exception as e:
        return f"错误：{str(e)}"


@tool("read_file")
def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """读取指定文件的内容。读取完成后会检查文件是否有 @summary 摘要注释，如果没有会提示你添加。

    Args:
        file_path: 文件路径
        encoding: 文件编码，默认为utf-8

    Returns:
        文件内容或错误信息
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        sensitive_dirs = [
            os.path.expanduser("~"),
            "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
            "C:\\Users", "C:\\ProgramData"
        ]

        for sensitive_dir in sensitive_dirs:
            if file_path.lower().startswith(sensitive_dir.lower()):
                return f"安全限制：禁止访问敏感目录中的文件"

        if not os.path.exists(file_path):
            return f"错误：文件 '{file_path}' 不存在"

        if not os.path.isfile(file_path):
            return f"错误：'{file_path}' 不是一个文件"

        file_size = os.path.getsize(file_path)
        max_size = 10 * 1024 * 1024
        if file_size > max_size:
            return f"错误：文件过大（{file_size / 1024 / 1024:.2f}MB），最大支持10MB"

        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()

        max_chars = 10000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[文件内容过长，已截断]"

        result = f"文件内容 ({file_path}):\n\n{content}"

        line_prefix, line_suffix = _get_comment_style(file_path)
        if line_prefix and not _has_summary_header(content, line_prefix):
            result += "\n\n⚠️ 该文件没有 @summary 摘要注释。请使用 add_file_summary 工具为该文件添加摘要描述（1-3句话概括文件功能），以便后续快速读取。格式示例：add_file_summary(file_path='...', summary='本文件实现了XXX功能')"

        return result

    except PermissionError:
        return f"权限错误：无法读取文件 '{file_path}'"
    except UnicodeDecodeError:
        return f"编码错误：无法使用 {encoding} 编码读取文件，尝试其他编码"
    except Exception as e:
        return f"错误：{str(e)}"

# @tool("send_message")
# def send_message(message: str) -> str:
#     """发送消息到公众号。
    
#     Args:
#         message: 要发送的消息
        
#     Returns:
#         发送结果或错误信息
#     """
#     try:
#         # 调用聊天机器人API发送消息
#         response = send_to_public_account(message)
#         return f"消息已发送：{message}"
#     except Exception as e:
#         return f"错误：{str(e)}"

@tool("get_city_info")
def get_city_info(city: str):
    """获取当前城市的城市信息。

    Returns:
        城市信息或错误信息
    """
    resp = requests.get(f"https://wttr.in/{city}?format=j1")
    data = resp.json()
    
    # 天气描述和温度
    weather = data["current_condition"][0]["weatherDesc"][0]["value"]
    temp_c = data["current_condition"][0]["temp_C"]
    
    # 使用 datetime 获取指定城市的时间信息
    timezone_str = get_city_timezone(city)
    city_timezone = ZoneInfo(timezone_str)
    city_datetime = datetime.datetime.now(city_timezone)
    
    # 获取时间、日期、星期几
    current_time = city_datetime.strftime("%H:%M:%S")
    current_date = city_datetime.strftime("%Y-%m-%d")
    weekday = city_datetime.strftime("%A")
    weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][city_datetime.weekday()]
    
    return {
        "城市": city,
        "天气": weather,
        "温度": f"{temp_c}°C",
        "当前时间": current_time,
        "当前日期": current_date+weekday_cn,
        "时区": timezone_str
    }


@tool
def create_file(file_path: str, content: str, overwrite: bool = False, summary: str = "") -> str:
    """创建指定文件并写入内容。

    Args:
        file_path: 文件路径
        content: 要写入文件的内容
        overwrite: 如果文件已存在，是否覆盖（默认为 False）
        summary: 文件功能摘要描述（可选）。如果提供，会在文件头部自动添加 @summary 注释块，
                 方便后续通过 read_file_summary 快速了解文件功能而无需读取全部内容。
                 建议为每个新建文件提供1-3句功能描述。

    Returns:
        文件创建结果或错误信息
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        if os.path.exists(file_path) and not overwrite:
            return f"错误：文件 '{file_path}' 已存在，请设置 overwrite=True 来覆盖"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if summary:
            content = _ensure_summary_in_content(content, summary, file_path)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return f"文件创建成功：{file_path}"
    except Exception as e:
        return f"错误：{str(e)}"


@tool
def modify_file(file_path: str, new_content: str = None, modifications: list[dict] = None) -> str:
    """修改指定文件内容。
    
    方式1（推荐）：直接替换整个文件内容
    方式2：基于行的精确修改
    
    Args:
        file_path: 文件路径
        new_content: 新的完整文件内容（如果提供，将完全替换文件内容）
        modifications: 修改操作列表（仅在不使用 new_content 时使用），每个操作包含：
            - type: 操作类型 (replace, insert, delete)
            - line: 行号（对于 insert 是插入位置，从1开始）
            - old_text: 要替换的旧文本（仅用于 replace 操作）
            - new_text: 新文本（用于 replace 和 insert 操作）
        
    Returns:
        文件修改结果或错误信息
    """
    try:
        # 确保路径是绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"错误：文件 '{file_path}' 不存在"
        
        if not os.path.isfile(file_path):
            return f"错误：'{file_path}' 不是一个文件"
        
        # 方式1：直接替换整个文件内容
        if new_content is not None:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return f"文件修改成功（完全替换）：{file_path}"
        
        # 方式2：基于行的精确修改
        if modifications is None or len(modifications) == 0:
            return "错误：请提供 new_content（完整替换）或 modifications（行级修改）"
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 应用修改（从后往前操作，避免行号偏移）
        modifications.sort(key=lambda x: x.get('line', 0), reverse=True)
        
        for mod in modifications:
            mod_type = mod.get('type')
            line = mod.get('line', 1) - 1  # 转换为 0-based 索引
            
            if mod_type == 'replace':
                old_text = mod.get('old_text')
                new_text = mod.get('new_text')
                if 0 <= line < len(lines):
                    # 如果提供了 old_text，尝试精确匹配
                    if old_text:
                        if old_text in lines[line]:
                            lines[line] = lines[line].replace(old_text, new_text)
                        else:
                            return f"警告：第 {line+1} 行未找到匹配的文本 '{old_text}'，跳过此修改"
                    else:
                        # 如果没有提供 old_text，直接替换整行
                        lines[line] = new_text + '\n'
            elif mod_type == 'insert':
                new_text = mod.get('new_text')
                lines.insert(line, new_text + '\n')
            elif mod_type == 'delete':
                if 0 <= line < len(lines):
                    del lines[line]
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        return f"文件修改成功（行级修改）：{file_path}"
    except Exception as e:
        return f"错误：{str(e)}"

@tool
def coding_plan(file_path:str) -> str:
    """生成代码执行计划。

    Args:
        file_path: 文件路径

    Returns:
        代码执行计划
    """
    command_plan = ""
    return command_plan


# ============================================================
# B站视频搜索工具
# ============================================================

MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52
]

BILIBILI_DEFAULT_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://www.bilibili.com',
}

_bilibili_client = None
_bilibili_client_lock = threading.Lock()


def _get_mixin_key(orig: str) -> str:
    return reduce(lambda s, i: s + orig[i], MIXIN_KEY_ENC_TAB, '')[:32]


def _enc_wbi(params: dict, img_key: str, sub_key: str) -> dict:
    mixin_key = _get_mixin_key(img_key + sub_key)
    params['wts'] = round(time.time())
    params = dict(sorted(params.items()))
    params = {
        k: ''.join(filter(lambda c: c not in "!'()*", str(v)))
        for k, v in params.items()
    }
    query = urllib.parse.urlencode(params)
    params['w_rid'] = md5((query + mixin_key).encode()).hexdigest()
    return params





class _BilibiliSearchClient:
    SEARCH_API_URL = "https://api.bilibili.com/x/web-interface/wbi/search/type"
    NAV_API_URL = "https://api.bilibili.com/x/web-interface/nav"

    def __init__(self, sessdata=None, proxy=None, timeout=15):
        self.session = requests.Session()
        self.session.headers.update(BILIBILI_DEFAULT_HEADERS)
        self.timeout = timeout
        if proxy:
            self.session.proxies = {'http': proxy, 'https': proxy}
        if sessdata:
            self.session.cookies.set('SESSDATA', sessdata, domain='.bilibili.com')
        self._img_key = None
        self._sub_key = None
        self._keys_expire = 0

    def _ensure_keys(self):
        if self._img_key and time.time() < self._keys_expire:
            return
        try:
            self.session.get('https://www.bilibili.com/', timeout=self.timeout)
        except requests.RequestException:
            pass
        try:
            resp = self.session.get(self.NAV_API_URL, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"无法连接到 Bilibili API: {e}") from e

        if data.get('code') != 0:
            wbi_img = data.get('data', {}).get('wbi_img')
            if wbi_img and wbi_img.get('img_url') and wbi_img.get('sub_url'):
                self._img_key = wbi_img['img_url'].rsplit('/', 1)[1].split('.')[0]
                self._sub_key = wbi_img['sub_url'].rsplit('/', 1)[1].split('.')[0]
                self._keys_expire = time.time() + 3600
                return
            raise RuntimeError(
                f"获取 WBI 密钥失败: code={data.get('code')}, "
                f"message={data.get('message', '未知错误')}"
            )

        wbi_img = data['data']['wbi_img']
        self._img_key = wbi_img['img_url'].rsplit('/', 1)[1].split('.')[0]
        self._sub_key = wbi_img['sub_url'].rsplit('/', 1)[1].split('.')[0]
        self._keys_expire = time.time() + 3600

    def search_videos(self, keyword, page=1, page_size=10, order="totalrank", duration=0):
        self._ensure_keys()
        params = {
            'search_type': 'video',
            'keyword': keyword,
            'page': page,
            'page_size': min(page_size, 50),
            'order': order,
            'duration': duration,
        }
        signed_params = _enc_wbi(params, self._img_key, self._sub_key)
        resp = self.session.get(self.SEARCH_API_URL, params=signed_params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        if data.get('code') != 0:
            raise RuntimeError(f"搜索失败: code={data.get('code')}, message={data.get('message')}")

        raw_results = data.get('data', {}).get('result', [])
        parsed_results = []
        for item in raw_results:
            clean_title = re.sub(r'<[^>]+>', '', item.get('title', ''))
            bvid = item.get('bvid', '')
            url = f"https://www.bilibili.com/video/{bvid}" if bvid else item.get('arcurl', '')
            pic = item.get('pic', '')
            if pic.startswith('//'):
                pic = 'https:' + pic
            parsed_results.append({
                'title': clean_title,
                'bvid': bvid,
                'url': url,
                'author': item.get('author', ''),
                'play': item.get('play', 0),
                'danmaku': item.get('video_review', 0),
                'favorites': item.get('favorites', 0),
                'duration': item.get('duration', ''),
                'description': item.get('description', ''),
                'tag': item.get('tag', ''),
                'typename': item.get('typename', ''),
                'is_pay': item.get('is_pay', 0),
            })

        return {
            'num_results': data.get('data', {}).get('numResults', 0),
            'page': data.get('data', {}).get('page', 1),
            'results': parsed_results,
        }


def _get_bilibili_client() -> _BilibiliSearchClient:
    global _bilibili_client
    with _bilibili_client_lock:
        if _bilibili_client is None:
            env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
            load_dotenv(dotenv_path=env_path)
            sessdata = os.getenv("BILIBILI_SESSDATA")
            proxy = os.getenv("BILIBILI_PROXY")
            _bilibili_client = _BilibiliSearchClient(sessdata=sessdata, proxy=proxy)
        return _bilibili_client


@tool("bilibili_video_search")
def bilibili_video_search(query: str, top_k: int = 5) -> str:
    """在 Bilibili（B站）上搜索视频。
    当用户想要查找、搜索或下载视频时使用此工具。
    输入搜索关键词，返回最匹配的视频列表，包含视频链接和下载方式。

    Args:
        query: 搜索关键词，例如："爱情怎么翻译"、"Python 教程"、"三体解说"
        top_k: 返回的搜索结果数量，默认为5
    """
    keyword = query

    try:
        client = _get_bilibili_client()
        search_result = client.search_videos(keyword=keyword, page=1, page_size=top_k)
    except Exception as e:
        return f"搜索失败：{str(e)}\n\n请稍后重试，或检查网络连接。"

    results = search_result.get('results', [])

    if not results:
        return (
            f"未找到与「{keyword}」相关的视频。\n\n"
            f"建议：\n"
            f"1. 尝试更换关键词\n"
            f"2. 检查是否有错别字\n"
            f"3. 使用更简短的关键词重新搜索"
        )

    output_parts = []
    total = search_result.get('num_results', 0)
    output_parts.append(
        f"在 Bilibili 上搜索「{keyword}」，共找到约 {total} 个结果，以下是最匹配的 {len(results)} 个：\n"
    )

    for i, video in enumerate(results, 1):
        play_count = video['play']
        play_str = f"{play_count / 10000:.1f}万" if play_count >= 10000 else str(play_count)
        fav_count = video['favorites']
        fav_str = f"{fav_count / 10000:.1f}万" if fav_count >= 10000 else str(fav_count)
        pay_tag = " [付费]" if video.get('is_pay') else ""

        output_parts.append(
            f"[{i}] {video['title']}{pay_tag}\n"
            f"  链接：{video['url']}\n"
            f"  UP主：{video['author']}  |  分区：{video['typename']}\n"
            f"  播放：{play_str}  |  收藏：{fav_str}  |  弹幕：{video['danmaku']}\n"
            f"  时长：{video['duration']}"
        )
        if video.get('tag'):
            output_parts.append(f"  标签：{video['tag']}")

    output_parts.append(f"\n下载方式：")
    output_parts.append("  方式1 (推荐)：yt-dlp 命令行工具")
    output_parts.append("    安装：pip install yt-dlp")
    if results:
        example_url = results[0]['url']
        output_parts.append(
            f'    下载：yt-dlp -f "bv[ext=mp4]+ba[ext=m4a]" --merge-output-format mp4 "{example_url}"'
        )
    output_parts.append("  方式2：you-get 工具")
    output_parts.append("    安装：pip install you-get")
    output_parts.append("  方式3：在线解析网站（无需安装）")
    output_parts.append("    https://xbeibeix.com/api/bilibili/")
    output_parts.append("    https://snapany.com/")

    return '\n'.join(output_parts)


@tool("bilibili_download")
def bilibili_download(video_url: str, output_dir: str = ".") -> str:
    """在 Bilibili（B站）上下载视频。
    当用户想要下载视频时使用此工具。
    输入视频链接，返回下载结果。

    Args:
        video_url: 视频链接，例如："https://www.bilibili.com/video/BV18SxTzpEc4"
        output_dir: 下载目录，默认为当前目录（.）
    """
    import os
    try:
        import yt_dlp
    except ImportError:
        return f"❌ 未找到 yt-dlp 库，请先安装：pip install yt-dlp"
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取进度回调函数
    progress_callback = get_download_progress_callback()
    
    # 下载状态
    download_status = {
        "progress": 0,
        "speed": "",
        "eta": "",
        "filename": "",
        "error": None,
        "complete": False
    }
    
    class ProgressLogger:
        def __init__(self, status_dict):
            self.status = status_dict
            self.last_callback_time = 0
            self.callback_interval = 0.5  # 每0.5秒调用一次回调
        
        def __call__(self, d):
            import time
            current_time = time.time()
            
            if d['status'] == 'downloading':
                self.status['progress'] = d.get('_percent_str', '0%').replace('%', '')
                self.status['speed'] = d.get('_speed_str', '')
                self.status['eta'] = d.get('_eta_str', '')
                self.status['filename'] = d.get('filename', '')
                
                # 限制回调频率
                if current_time - self.last_callback_time >= self.callback_interval and progress_callback:
                    progress_callback({
                        "type": "download_progress",
                        "progress": float(self.status['progress']) if self.status['progress'].replace('.', '').isdigit() else 0,
                        "speed": self.status['speed'],
                        "eta": self.status['eta'],
                        "filename": os.path.basename(self.status['filename'])
                    })
                    self.last_callback_time = current_time
            
            elif d['status'] == 'finished':
                self.status['complete'] = True
                if progress_callback:
                    progress_callback({
                        "type": "download_progress",
                        "progress": 100,
                        "speed": "完成",
                        "eta": "",
                        "filename": os.path.basename(self.status['filename'])
                    })
            
            elif d['status'] == 'error':
                self.status['error'] = d.get('error', 'Unknown error')
                if progress_callback:
                    progress_callback({
                        "type": "download_error",
                        "error": self.status['error']
                    })
    
    # 配置 yt-dlp
    ydl_opts = {
        'format': 'bv[ext=mp4]+ba[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'progress_hooks': [ProgressLogger(download_status)],
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 提取视频信息
            info = ydl.extract_info(video_url, download=False)
            video_title = info.get('title', '未知视频')
            
            # 开始下载
            ydl.download([video_url])
            
            if download_status['error']:
                return f"❌ 下载失败：{download_status['error']}\n\n视频链接：{video_url}"
            
            return f"✅ 视频下载成功！\n\n视频标题：{video_title}\n视频链接：{video_url}\n下载目录：{os.path.abspath(output_dir)}"
    
    except Exception as e:
        error_msg = str(e)
        if progress_callback:
            progress_callback({
                "type": "download_error",
                "error": error_msg
            })
        return f"❌ 下载失败：{error_msg}\n\n视频链接：{video_url}\n下载目录：{os.path.abspath(output_dir)}"


# ============================================================
# 任务规划工具 (Task Planning Tools)
# 参考 hongmeng-llm-main 的 TodoWriteTool 设计
# ============================================================

_SESSION_TASKS = {}
_SESSION_TASKS_LOCK = threading.Lock()


def _get_session_tasks(session_id: str = "default") -> list:
    with _SESSION_TASKS_LOCK:
        return _SESSION_TASKS.get(session_id, [])


def _set_session_tasks(session_id: str, tasks: list):
    with _SESSION_TASKS_LOCK:
        _SESSION_TASKS[session_id] = tasks


def _reset_session_tasks(session_id: str = "default"):
    with _SESSION_TASKS_LOCK:
        _SESSION_TASKS.pop(session_id, None)


def _render_task_list(tasks: list) -> str:
    if not tasks:
        return "📋 当前没有任务"
    lines = ["📋 任务列表："]
    for i, t in enumerate(tasks, 1):
        status = t.get("status", "pending")
        content = t.get("content", "")
        if status == "completed":
            icon = "✅"
        elif status == "in_progress":
            icon = "🔄"
        else:
            icon = "⬜"
        lines.append(f"  {icon} {i}. {content} [{status}]")
    completed = sum(1 for t in tasks if t.get("status") == "completed")
    pending = sum(1 for t in tasks if t.get("status") == "pending")
    in_progress = sum(1 for t in tasks if t.get("status") == "in_progress")
    lines.append(f"\n进度：{completed} 完成 / {in_progress} 进行中 / {pending} 待办 / 共 {len(tasks)} 项")
    return "\n".join(lines)


@tool("plan_task")
def plan_task(todos: list[dict]) -> str:
    """Create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Use this tool proactively in these scenarios:
1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests a task plan - When the user directly asks you to plan the work
4. User provides multiple tasks - When users provide a list of things to be done
5. After receiving new instructions - Immediately capture user requirements as tasks
6. When you start working on a task - Mark it as in_progress BEFORE beginning work
7. After completing a task - Mark it as completed and add any new follow-up tasks

When NOT to use this tool:
- There is only a single, straightforward task
- The task is trivial and can be completed in less than 3 trivial steps
- The task is purely conversational or informational

IMPORTANT: Each task item must have:
- content: The imperative form describing what needs to be done (e.g., "Fix authentication bug")
- status: One of "pending", "in_progress", "completed"
- activeForm: The present continuous form shown during execution (e.g., "Fixing authentication bug")

Task Management Rules:
- Update task status in real-time as you work
- Mark tasks completed IMMEDIATELY after finishing
- Only ONE task should be in_progress at any time
- Complete current tasks before starting new ones
- Remove tasks that are no longer relevant

    Args:
        todos: The updated task list. Each item is a dict with keys:
            - content (str, required): Task description in imperative form, e.g. "Create user model"
            - status (str, required): One of "pending", "in_progress", "completed"
            - activeForm (str, required): Task description in present continuous form, e.g. "Creating user model"
    """
    session_id = _CURRENT_USER_ID or "default"

    for i, item in enumerate(todos):
        if "content" not in item or not item["content"]:
            return f"错误：第 {i+1} 个任务缺少 content 字段"
        if "status" not in item:
            return f"错误：第 {i+1} 个任务缺少 status 字段"
        if item["status"] not in ("pending", "in_progress", "completed"):
            return f"错误：第 {i+1} 个任务的 status 必须是 pending/in_progress/completed 之一"
        if "activeForm" not in item or not item["activeForm"]:
            return f"错误：第 {i+1} 个任务缺少 activeForm 字段"

    in_progress_count = sum(1 for t in todos if t.get("status") == "in_progress")
    if in_progress_count > 1:
        return "错误：同一时间只能有一个任务处于 in_progress 状态，请调整后再提交"

    old_tasks = _get_session_tasks(session_id)
    _set_session_tasks(session_id, todos)

    all_done = all(t.get("status") == "completed" for t in todos) if todos else False

    if all_done:
        _reset_session_tasks(session_id)
        return "✅ 所有任务已完成！任务列表已清空。"

    result = _render_task_list(todos)

    if old_tasks:
        old_pending = sum(1 for t in old_tasks if t.get("status") == "pending")
        old_in_progress = sum(1 for t in old_tasks if t.get("status") == "in_progress")
        old_completed = sum(1 for t in old_tasks if t.get("status") == "completed")
        new_completed = sum(1 for t in todos if t.get("status") == "completed")
        if new_completed > old_completed:
            result += "\n\n🎉 有任务完成了！请继续执行下一个 in_progress 任务。"

    return result


@tool("get_task_list")
def get_task_list() -> str:
    """Get the current task list for the session. Use this to review the current progress and decide what to do next.

    Returns:
        The formatted task list with status information
    """
    session_id = _CURRENT_USER_ID or "default"
    tasks = _get_session_tasks(session_id)
    if not tasks:
        return "📋 当前没有任务列表。如果需要规划任务，请使用 plan_task 工具创建。"
    return _render_task_list(tasks)


def get_current_plan(session_id: str = "") -> list:
    sid = session_id or _CURRENT_USER_ID or "default"
    return _get_session_tasks(sid)


def advance_to_next_task(session_id: str = "") -> dict:
    sid = session_id or _CURRENT_USER_ID or "default"
    tasks = _get_session_tasks(sid)
    if not tasks:
        return {"has_next": False, "next_task": None, "all_completed": True}

    current_index = None
    for i, task in enumerate(tasks):
        if task.get("status") == "in_progress":
            current_index = i
            break

    if current_index is None:
        return {"has_next": False, "next_task": None, "all_completed": True}

    tasks[current_index]["status"] = "completed"

    next_index = current_index + 1
    if next_index < len(tasks):
        tasks[next_index]["status"] = "in_progress"
        _set_session_tasks(sid, tasks)
        return {
            "has_next": True,
            "next_task": tasks[next_index],
            "all_completed": False,
            "completed_task": tasks[current_index]["content"],
        }
    else:
        _reset_session_tasks(sid)
        return {
            "has_next": False,
            "next_task": None,
            "all_completed": True,
            "completed_task": tasks[current_index]["content"],
        }


def reset_current_plan(session_id: str = ""):
    sid = session_id or _CURRENT_USER_ID or "default"
    _reset_session_tasks(sid)


# ============================================================
# 终端命令工具 (Terminal Command Tool)
# 参考 hongmeng-llm-main 的 BashTool/PowerShellTool 设计
# ============================================================

_DANGEROUS_COMMANDS = [
    "rm -rf /", "rm -rf /*", "del /s /q C:\\", "format C:",
    "shutdown", "restart", "taskkill /f", "reg delete",
    "rd /s /q C:\\", "rmdir /s /q C:\\",
]


def _validate_command(command: str) -> str:
    command_lower = command.lower().strip()
    for dangerous in _DANGEROUS_COMMANDS:
        if dangerous.lower() in command_lower:
            return f"安全限制：禁止执行危险命令 '{dangerous}'"
    return ""


@tool("run_command")
def run_command(command: str, timeout: int = 30, working_dir: str = "") -> str:
    """Execute a terminal command and return its output. Use this tool to:
1. Run code/tests to verify modifications work correctly
2. Get system information (e.g., directory listing, environment variables)
3. Install dependencies or run build commands
4. Execute any shell/PowerShell command needed for development

IMPORTANT workflow for code modification and testing:
- After modifying code with create_file or modify_file, use this tool to run tests
- Analyze the output: if tests pass, mark the task as completed
- If tests fail, analyze the error message, fix the code, and re-run tests
- Repeat until the code works correctly

IMPORTANT: This tool requires user confirmation before execution. When you call this tool, it will return a confirmation request message. You MUST tell the user about the command and wait for their confirmation. When the user confirms, use the execute_confirmed_command tool with the SAME parameters to execute the command.

Safety rules:
- Do NOT run destructive commands (rm -rf /, del /s /q, format, etc.)
- Do NOT run commands that modify system configuration
- Always use absolute paths when possible
- For long-running commands, set an appropriate timeout

    Args:
        command: The terminal command to execute (PowerShell on Windows)
        timeout: Maximum execution time in seconds (default: 30, max: 300)
        working_dir: Working directory for the command (default: current directory)
    """
    safety_error = _validate_command(command)
    if safety_error:
        return safety_error

    _fire_tool_event({
        "type": "command_confirm_request",
        "command": command,
        "timeout": timeout,
        "working_dir": working_dir or os.getcwd(),
    })

    return (
        f"[COMMAND_CONFIRM_REQUIRED]\n"
        f"command={command}\n"
        f"timeout={timeout}\n"
        f"working_dir={working_dir or os.getcwd()}\n"
        f"[/COMMAND_CONFIRM_REQUIRED]\n\n"
        f"此命令需要用户确认后才能执行。请告知用户即将执行的命令内容，并等待用户确认。"
        f"当用户确认后，请使用 execute_confirmed_command 工具执行此命令，参数必须完全一致：\n"
        f'execute_confirmed_command(command="{command}", timeout={timeout}, working_dir="{working_dir or os.getcwd()}")'
    )


@tool("execute_confirmed_command")
def execute_confirmed_command(command: str, timeout: int = 30, working_dir: str = "") -> str:
    """Execute a terminal command that has already been confirmed by the user.
This tool should ONLY be called after the user has explicitly confirmed they want to run the command.
Do NOT call this tool without user confirmation.

    Args:
        command: The terminal command to execute (must be the same as the one shown to the user)
        timeout: Maximum execution time in seconds (default: 30, max: 300)
        working_dir: Working directory for the command (default: current directory)
    """
    import subprocess

    safety_error = _validate_command(command)
    if safety_error:
        return safety_error

    timeout = min(max(timeout, 1), 300)

    cwd = working_dir if working_dir else os.getcwd()
    if not os.path.isabs(cwd):
        cwd = os.path.abspath(cwd)

    try:
        is_windows = os.name == "nt"
        if is_windows:
            cmd_args = ["powershell", "-NoProfile", "-Command", command]
        else:
            cmd_args = ["bash", "-c", command]

        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        exit_code = result.returncode

        output_parts = []

        if stdout.strip():
            max_stdout = 8000
            if len(stdout) > max_stdout:
                output_parts.append(f"stdout (truncated):\n{stdout[:max_stdout]}\n... [输出过长，已截断]")
            else:
                output_parts.append(f"stdout:\n{stdout.strip()}")

        if stderr.strip():
            max_stderr = 4000
            if len(stderr) > max_stderr:
                output_parts.append(f"stderr (truncated):\n{stderr[:max_stderr]}\n... [错误输出过长，已截断]")
            else:
                output_parts.append(f"stderr:\n{stderr.strip()}")

        if not output_parts:
            output_parts.append("(无输出)")

        status = "✅ 成功" if exit_code == 0 else f"❌ 失败 (exit code: {exit_code})"
        output_parts.append(f"\n状态: {status}")

        return "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return f"⏰ 命令执行超时（{timeout}秒），请增加 timeout 参数或优化命令"
    except FileNotFoundError:
        return f"❌ 命令未找到，请检查命令是否正确：{command}"
    except Exception as e:
        return f"❌ 命令执行失败：{str(e)}"


