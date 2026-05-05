"""
WeChatReminderTool - 微信公众号提醒工具
=========================================
通过微信公众号（订阅号）的客服消息接口，实现定时提醒功能。

功能：
1. LangChain Tool：自然语言 → 结构化提醒数据（时间 + 内容）
2. 微信公众号推送模块：通过客服消息接口发送提醒
3. 一次性调度脚本：创建提醒 → 等待 → 到点推送

依赖：
    pip install requests langchain-core langchain-openai pydantic

重要限制（订阅号）：
    - 客服消息需要在用户 48 小时内与公众号有过互动才能发送
    - 每次用户交互后可发送 5 条消息
    - 因此使用前，用户需要先向公众号发送一条消息来激活窗口

使用示例：
    from wechat_reminder import WeChatReminderTool

    tool = WeChatReminderTool(
        wechat_appid="your_appid",
        wechat_appsecret="your_appsecret",
        wechat_openid="user_openid",
        llm_api_key="your-llm-key",       # 用于解析自然语言
        llm_base_url="https://api.openai.com/v1",
        llm_model="gpt-4o-mini",
    )
    result = tool.invoke({"content": "今天下午五点提醒我喝水"})
    print(result)
"""

import re
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Optional, Type

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ============================================================
# 微信公众号 API 客户端
# ============================================================

class WeChatOfficialAccount:
    """微信公众号 API 客户端"""

    TOKEN_URL = "https://api.weixin.qq.com/cgi-bin/token"
    CUSTOM_MSG_URL = "https://api.weixin.qq.com/cgi-bin/message/custom/send"

    def __init__(self, appid: str, appsecret: str):
        """
        初始化微信公众号客户端。

        Args:
            appid: 公众号 AppID
            appsecret: 公众号 AppSecret
        """
        self.appid = appid
        self.appsecret = appsecret
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

    def get_access_token(self) -> str:
        """
        获取 access_token（带缓存，提前 5 分钟刷新）。

        Returns:
            access_token 字符串

        Raises:
            RuntimeError: 获取失败时抛出
        """
        if self._access_token and time.time() < self._token_expires_at - 300:
            return self._access_token

        url = (
            f"{self.TOKEN_URL}"
            f"?grant_type=client_credential"
            f"&appid={self.appid}"
            f"&secret={self.appsecret}"
        )
        resp = requests.get(url, timeout=15)
        data = resp.json()

        if "access_token" in data:
            self._access_token = data["access_token"]
            self._token_expires_at = time.time() + data.get("expires_in", 7200)
            return self._access_token

        raise RuntimeError(
            f"获取 access_token 失败: {data.get('errcode')} - {data.get('errmsg')}\n"
            f"请检查 AppID/AppSecret 是否正确，以及服务器 IP 是否已加入白名单。"
        )

    def send_text_message(self, openid: str, content: str) -> dict:
        """
        发送客服文本消息。

        Args:
            openid: 用户的 OpenID
            content: 文本消息内容

        Returns:
            API 返回的 JSON 数据
        """
        access_token = self.get_access_token()
        url = f"{self.CUSTOM_MSG_URL}?access_token={access_token}"

        payload = {
            "touser": openid,
            "msgtype": "text",
            "text": {
                "content": content
            }
        }

        resp = requests.post(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            timeout=15,
        )
        return resp.json()

    def send_reminder(self, openid: str, title: str, content: str, remind_time: str) -> dict:
        """
        发送提醒消息（格式化的客服文本消息）。

        Args:
            openid: 用户的 OpenID
            title: 提醒标题
            content: 提醒内容
            remind_time: 提醒时间（格式化字符串）

        Returns:
            API 返回的 JSON 数据
        """
        message = (
            f"⏰ 提醒通知\n"
            f"{'─' * 20}\n"
            f"📌 {title}\n"
            f"📝 {content}\n"
            f"🕐 设定时间：{remind_time}\n"
            f"{'─' * 20}\n"
            f"来自你的智能助手 💙"
        )
        return self.send_text_message(openid, message)


# ============================================================
# 自然语言 → 结构化提醒数据（LLM 解析）
# ============================================================

REMINDER_PARSE_PROMPT = """\
你是一个提醒时间解析专家。用户会用自然语言描述他们想设置的提醒，
你需要从中提取出精确的提醒时间和提醒内容。

## 当前时间
{current_time}

## 解析规则

1. **时间解析**：
   - "今天下午五点" → 今天的 17:00
   - "明天早上8点半" → 明天的 08:30
   - "后天晚上9点" → 后天的 21:00
   - "30分钟后" → 当前时间 + 30分钟
   - "2小时后提醒我" → 当前时间 + 2小时
   - "下周一上午10点" → 下周一的 10:00
   - "2026年5月1日下午3点" → 2026-05-01 15:00
   - 如果只说了时间没说日期，默认为今天（如果时间已过则为明天）

2. **内容提取**：
   - "提醒我喝水" → 喝水
   - "下午三点开会" → 开会
   - "记得拿快递" → 拿快递
   - "该吃药了" → 吃药

3. **标题生成**：
   - 根据内容自动生成简短的提醒标题（2-8个字）
   - 例如："喝水" → "喝水提醒"，"开会" → "会议提醒"

## 输出格式

严格按以下 JSON 格式输出，不要输出任何其他内容：
{{"remind_time": "YYYY-MM-DD HH:MM", "title": "提醒标题", "content": "提醒内容"}}

示例：
{{"remind_time": "2026-04-23 17:00", "title": "喝水提醒", "content": "喝水"}}"""


class ReminderParser:
    """
    提醒内容解析器（LLM 驱动）。

    从自然语言中解析出提醒时间和提醒内容。
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        timeout: int = 15,
    ):
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url
        self._llm_model = llm_model
        self._timeout = timeout
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "使用 LLM 解析需要安装 langchain-openai：pip install langchain-openai"
            )

        init_kwargs = {
            "model": self._llm_model,
            "temperature": 0,
            "max_tokens": 200,
            "timeout": self._timeout,
        }
        if self._llm_api_key:
            init_kwargs["api_key"] = self._llm_api_key
        if self._llm_base_url:
            init_kwargs["base_url"] = self._llm_base_url

        self._llm = ChatOpenAI(**init_kwargs)
        return self._llm

    def parse(self, user_input: str) -> dict:
        """
        从自然语言中解析提醒时间和内容。

        Args:
            user_input: 用户的自然语言输入

        Returns:
            包含 remind_time, title, content 的字典

        Raises:
            ValueError: 解析失败时抛出
        """
        llm = self._get_llm()

        from langchain_core.messages import HumanMessage, SystemMessage

        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

        messages = [
            SystemMessage(content=REMINDER_PARSE_PROMPT.format(current_time=current_time)),
            HumanMessage(content=user_input),
        ]

        response = llm.invoke(messages)
        text = response.content.strip()

        # 清理可能的 markdown 代码块包裹
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        # 解析 JSON
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"LLM 返回的内容无法解析为 JSON: {text}")

        # 验证必要字段
        required_fields = ["remind_time", "title", "content"]
        for field in required_fields:
            if field not in data or not data[field]:
                raise ValueError(f"解析结果缺少必要字段: {field}")

        # 验证时间格式
        try:
            remind_dt = datetime.strptime(data["remind_time"], "%Y-%m-%d %H:%M")
            # 如果时间已过，自动推迟到明天
            if remind_dt < datetime.now():
                remind_dt += timedelta(days=1)
                data["remind_time"] = remind_dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError(f"时间格式错误: {data['remind_time']}，应为 YYYY-MM-DD HH:MM")

        return data


# ============================================================
# 正则 Fallback 解析器
# ============================================================

def parse_reminder_regex(user_input: str) -> Optional[dict]:
    """
    用正则从自然语言中解析提醒（简易 fallback）。

    Returns:
        解析成功返回 dict，失败返回 None
    """
    text = user_input.strip()
    now = datetime.now()

    # 提取提醒内容
    content = text
    remind_time = None

    # 匹配 "X点" / "X点半" / "X点X分"
    remind_time = None

    # 中文数字映射
    cn_num_map = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
                  '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                  '十一': 11, '十二': 12}

    def _cn_to_num(s: str) -> Optional[int]:
        """将中文数字转换为阿拉伯数字"""
        if not s:
            return None
        if s.isdigit():
            return int(s)
        return cn_num_map.get(s)

    # 模式1: "X小时后" / "X分钟后"
    rel_match = re.search(r'(\d+)\s*(小时|分钟|个小时|分)后', text)
    if rel_match:
        amount = int(rel_match.group(1))
        unit = rel_match.group(2)
        if '小时' in unit:
            remind_time = now + timedelta(hours=amount)
        else:
            remind_time = now + timedelta(minutes=amount)

    # 模式2: "今天/明天/后天 + 上午/下午/晚上 + X点X分"
    if remind_time is None:
        abs_match = re.search(
            r'(今天|明天|后天)?\s*(上午|下午|晚上|早上|中午|夜里|凌晨)?\s*'
            r'([零一二两三四五六七八九十百\d]+)\s*[点时:：]\s*'
            r'([零一二两三四五六七八九十百\d半]*)\s*[分分钟]?',
            text
        )
        if abs_match:
            day_word = abs_match.group(1)
            period = abs_match.group(2) or ''
            hour_str = abs_match.group(3)
            min_str = abs_match.group(4)

            hour = _cn_to_num(hour_str)
            if hour is None:
                remind_time = None
            else:
                # 处理分钟
                if min_str == '半':
                    minute = 30
                elif min_str:
                    minute = _cn_to_num(min_str)
                    if minute is None:
                        minute = 0
                else:
                    minute = 0

                day_offset = 0
                if day_word == '明天':
                    day_offset = 1
                elif day_word == '后天':
                    day_offset = 2

                # 上午/下午/晚上调整
                if '下午' in period or '晚上' in period:
                    if hour < 12:
                        hour += 12
                elif '凌晨' in period:
                    if hour == 12:
                        hour = 0

                target_date = now.date() + timedelta(days=day_offset)
                remind_time = datetime(target_date.year, target_date.month, target_date.day, hour, minute)

    if remind_time is None:
        return None

    # 如果时间已过，推迟到明天
    if remind_time < now:
        remind_time += timedelta(days=1)

    # 提取内容：去掉时间相关的词
    content = re.sub(
        r'(今天|明天|后天)?\s*(上午|下午|晚上|早上|中午|夜里|凌晨)?\s*'
        r'[零一二两三四五六七八九十百\d]+\s*[点时:：]\s*[零一二两三四五六七八九十百\d半]*\s*[分分钟]?',
        '', text
    )
    content = re.sub(r'\d+\s*(小时|分钟|个小时|分)后', '', content)
    content = re.sub(r'^(提醒我|提醒|记得|别忘了|通知我|叫我|到时候)\s*', '', content)
    content = re.sub(r'(吧|啊|呢|呀|哦|噢|哈|嘿)*$', '', content)
    content = content.strip()

    if not content:
        content = "自定义提醒"

    return {
        "remind_time": remind_time.strftime("%Y-%m-%d %H:%M"),
        "title": f"{content}提醒",
        "content": content,
    }


# ============================================================
# 提醒调度器
# ============================================================

class ReminderScheduler:
    """
    一次性提醒调度器。

    创建提醒后，在后台线程中等待到指定时间，然后通过微信公众号发送提醒。
    """

    def __init__(self, wechat_client: WeChatOfficialAccount, openid: str):
        """
        Args:
            wechat_client: 微信公众号客户端
            openid: 接收提醒的用户 OpenID
        """
        self.wechat = wechat_client
        self.openid = openid
        self._reminders = []

    def schedule(
        self,
        remind_time: str,
        title: str,
        content: str,
        callback=None,
    ) -> dict:
        """
        创建一个定时提醒。

        Args:
            remind_time: 提醒时间，格式 "YYYY-MM-DD HH:MM"
            title: 提醒标题
            content: 提醒内容
            callback: 提醒触发后的回调函数 callback(success, message)

        Returns:
            提醒信息字典
        """
        try:
            target_dt = datetime.strptime(remind_time, "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError(f"时间格式错误: {remind_time}，应为 YYYY-MM-DD HH:MM")

        now = datetime.now()
        if target_dt <= now:
            # 如果时间刚好是当前分钟，给至少 10 秒的缓冲
            if (now - target_dt).total_seconds() < 60:
                target_dt = now + timedelta(seconds=10)
            else:
                return {
                    "success": False,
                    "message": f"提醒时间 {remind_time} 已过，无法创建提醒",
                }

        wait_seconds = (target_dt - now).total_seconds()

        reminder_info = {
            "remind_time": remind_time,
            "title": title,
            "content": content,
            "wait_seconds": wait_seconds,
            "status": "scheduled",
        }
        self._reminders.append(reminder_info)

        # 在后台线程中等待并发送
        thread = threading.Thread(
            target=self._wait_and_send,
            args=(target_dt, title, content, callback),
            daemon=True,
        )
        thread.start()

        return {
            "success": True,
            "message": f"提醒已创建！将在 {remind_time}（约 {self._format_wait(wait_seconds)} 后）发送提醒",
            "remind_time": remind_time,
            "title": title,
            "content": content,
            "wait_seconds": wait_seconds,
        }

    def _wait_and_send(
        self,
        target_dt: datetime,
        title: str,
        content: str,
        callback=None,
    ):
        """后台等待并发送提醒"""
        now = datetime.now()
        wait_seconds = (target_dt - now).total_seconds()

        # 打印等待信息
        print(f"[提醒调度] 等待中... {title} 将在 {target_dt.strftime('%Y-%m-%d %H:%M')} 触发"
              f"（还需等待 {self._format_wait(wait_seconds)}）")

        # 等待到指定时间（每 10 秒检查一次，避免系统休眠导致偏差过大）
        while datetime.now() < target_dt:
            remaining = (target_dt - datetime.now()).total_seconds()
            if remaining > 60:
                time.sleep(min(60, remaining - 30))  # 提前 30 秒醒来做精确等待
            elif remaining > 1:
                time.sleep(1)
            else:
                time.sleep(0.1)

        # 发送提醒
        try:
            result = self.wechat.send_reminder(
                openid=self.openid,
                title=title,
                content=content,
                remind_time=target_dt.strftime("%Y-%m-%d %H:%M"),
            )

            if result.get("errcode") == 0:
                msg = f"✅ 提醒「{title}」已成功发送到微信！"
                print(f"[提醒调度] {msg}")
                if callback:
                    callback(True, msg)
            else:
                errcode = result.get("errcode")
                errmsg = result.get("errmsg", "未知错误")
                msg = f"❌ 提醒「{title}」发送失败: [{errcode}] {errmsg}"
                if errcode == 45047:
                    msg += "\n💡 提示：48小时交互窗口已过期，请先向公众号发送一条消息来激活窗口。"
                print(f"[提醒调度] {msg}")
                if callback:
                    callback(False, msg)

        except Exception as e:
            msg = f"❌ 提醒「{title}」发送异常: {e}"
            print(f"[提醒调度] {msg}")
            if callback:
                callback(False, msg)

    @staticmethod
    def _format_wait(seconds: float) -> str:
        """格式化等待时间"""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            return f"{int(seconds / 60)}分钟"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}小时{minutes}分钟"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}天{hours}小时"

    def get_pending_reminders(self) -> list:
        """获取所有待执行的提醒"""
        return [r for r in self._reminders if r["status"] == "scheduled"]


# ============================================================
# LangChain 工具
# ============================================================

class WeChatReminderInput(BaseModel):
    """微信提醒工具的输入 schema"""
    content: str = Field(
        description=(
            "用户的自然语言提醒内容，例如：'今天下午五点提醒我喝水'、"
            "'明天早上8点提醒我开会'、'30分钟后提醒我拿快递'"
        )
    )


class WeChatReminderTool(BaseTool):
    """
    微信公众号提醒工具 - LangChain Tool

    根据用户的自然语言描述，解析出提醒时间和内容，
    通过微信公众号的客服消息接口在指定时间发送提醒。

    功能：
    1. LLM 智能解析自然语言 → 精确的提醒时间 + 内容
    2. 通过微信公众号客服消息接口推送提醒
    3. 支持后台定时调度

    注意：
    - 订阅号需要在用户 48 小时内有过互动才能发送客服消息
    - 使用前请确保用户已向公众号发送过消息
    """

    name: str = "wechat_reminder"
    description: str = (
        "设置微信提醒。当用户想要设置定时提醒、闹钟、备忘时使用此工具。"
        "输入自然语言描述（如'今天下午五点提醒我喝水'），"
        "工具会自动解析时间和内容，在指定时间通过微信公众号发送提醒。"
    )
    args_schema: Type[BaseModel] = WeChatReminderInput

    # 微信公众号配置
    wechat_appid: str = ""
    wechat_appsecret: str = ""
    wechat_openid: str = ""

    # LLM 配置
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"

    _wechat_client: Optional[WeChatOfficialAccount] = None
    _scheduler: Optional[ReminderScheduler] = None
    _parser: Optional[ReminderParser] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.wechat_appid or not self.wechat_appsecret:
            raise ValueError("必须提供 wechat_appid 和 wechat_appsecret")
        if not self.wechat_openid:
            raise ValueError("必须提供 wechat_openid（接收提醒的用户 OpenID）")

        self._wechat_client = WeChatOfficialAccount(
            appid=self.wechat_appid,
            appsecret=self.wechat_appsecret,
        )
        self._scheduler = ReminderScheduler(
            wechat_client=self._wechat_client,
            openid=self.wechat_openid,
        )

        # 如果配置了 LLM，初始化解析器
        if self.llm_api_key or self.llm_base_url:
            self._parser = ReminderParser(
                llm_api_key=self.llm_api_key,
                llm_base_url=self.llm_base_url,
                llm_model=self.llm_model,
            )

    def _parse_reminder(self, user_input: str) -> dict:
        """解析提醒内容（LLM 优先，正则 fallback）"""
        # 优先使用 LLM
        if self._parser is not None:
            try:
                return self._parser.parse(user_input)
            except Exception as e:
                print(f"[提醒工具] LLM 解析失败，尝试正则 fallback: {e}")

        # Fallback：正则
        result = parse_reminder_regex(user_input)
        if result:
            return result

        raise ValueError(
            f"无法解析提醒内容: {user_input}\n"
            f"请使用类似 '今天下午五点提醒我喝水' 的格式"
        )

    def _run(
        self,
        content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        创建提醒并调度发送。

        Args:
            content: 用户的自然语言提醒内容
            run_manager: LangChain 回调管理器

        Returns:
            提醒创建结果的格式化字符串
        """
        # 1. 解析提醒内容
        try:
            reminder = self._parse_reminder(content)
        except ValueError as e:
            return f"❌ {str(e)}"

        # 2. 创建定时提醒
        try:
            result = self._scheduler.schedule(
                remind_time=reminder["remind_time"],
                title=reminder["title"],
                content=reminder["content"],
            )
        except Exception as e:
            return f"❌ 创建提醒失败: {e}"

        if not result["success"]:
            return result["message"]

        # 3. 返回格式化结果
        return (
            f"✅ 提醒已创建！\n\n"
            f"📌 标题：{reminder['title']}\n"
            f"📝 内容：{reminder['content']}\n"
            f"🕐 提醒时间：{reminder['remind_time']}\n"
            f"⏳ 约 {result.get('wait_seconds_text', '')} 后触发\n\n"
            f"提醒将通过微信公众号发送到你的微信。"
            f"\n⚠️ 注意：请确保 48 小时内与公众号有过互动，否则消息无法送达。"
        )

    async def _arun(
        self,
        content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """异步执行"""
        return self._run(content=content, run_manager=run_manager)

    @property
    def scheduler(self) -> ReminderScheduler:
        """获取调度器实例"""
        return self._scheduler

    @property
    def wechat_client(self) -> WeChatOfficialAccount:
        """获取微信客户端实例"""
        return self._wechat_client


# ============================================================
# 便捷函数
# ============================================================

def create_reminder(
    content: str,
    appid: str,
    appsecret: str,
    openid: str,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    blocking: bool = False,
) -> str:
    """
    便捷函数：创建一个微信提醒。

    Args:
        content: 用户的自然语言提醒内容
        appid: 公众号 AppID
        appsecret: 公众号 AppSecret
        openid: 用户 OpenID
        llm_api_key: LLM API Key（可选）
        llm_base_url: LLM API base_url（可选）
        llm_model: LLM 模型名称
        blocking: 是否阻塞等待提醒执行完毕

    Returns:
        提醒创建结果字符串
    """
    tool = WeChatReminderTool(
        wechat_appid=appid,
        wechat_appsecret=appsecret,
        wechat_openid=openid,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
    )
    result = tool.invoke({"content": content})

    if blocking:
        # 阻塞等待所有提醒完成
        while tool.scheduler.get_pending_reminders():
            time.sleep(10)

    return result


def send_message_now(
    content: str,
    appid: str,
    appsecret: str,
    openid: str,
    title: str = "消息通知",
) -> dict:
    """
    便捷函数：立即发送一条消息到微信（不经过定时调度）。

    Args:
        content: 消息内容
        appid: 公众号 AppID
        appsecret: 公众号 AppSecret
        openid: 用户 OpenID
        title: 消息标题

    Returns:
        API 返回结果
    """
    wechat = WeChatOfficialAccount(appid=appid, appsecret=appsecret)
    return wechat.send_reminder(
        openid=openid,
        title=title,
        content=content,
        remind_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
