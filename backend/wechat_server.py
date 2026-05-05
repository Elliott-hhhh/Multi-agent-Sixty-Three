"""
微信消息接收服务器
==================
用于接收微信公众号推送的消息，提取用户 OpenID。

功能：
1. 处理微信服务器验证（GET 请求）
2. 接收用户消息（POST 请求），提取 OpenID
3. 自动回复（可选）
4. 将 OpenID 持久化保存到文件

依赖：
    pip install flask

运行：
    python wechat_server.py

    # 如果需要指定端口
    python wechat_server.py --port 8080
"""

import hashlib
import time
import os
import json
import argparse
from functools import wraps
from datetime import datetime

try:
    from flask import Flask, request, make_response
except ImportError:
    raise ImportError("请先安装 Flask：pip install flask")


# ============================================================
# 配置（请修改为你自己的）
# ============================================================

# 必须和公众号后台「服务器配置」中的 Token 一致
WECHAT_TOKEN = "my_wechat_token_2026"

# OpenID 存储文件路径
OPENID_FILE = "wechat_openids.json"

# 是否开启自动回复（设为 True 时，收到消息会自动回复一条文字消息）
AUTO_REPLY = True

# 自动回复内容
AUTO_REPLY_TEXT = "收到！你的消息已记录，智能助手正在为你服务 🤖"


# ============================================================
# Flask 应用
# ============================================================

app = Flask(__name__)


# ============================================================
# OpenID 存储管理
# ============================================================

class OpenIDStore:
    """简单的 OpenID 存储管理（基于 JSON 文件）"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._data = self._load()

    def _load(self) -> dict:
        """从文件加载数据"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"openids": [], "messages": []}

    def _save(self):
        """保存数据到文件"""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def add_openid(self, openid: str, nickname: str = ""):
        """添加一个 OpenID（去重）"""
        if openid not in self._data["openids"]:
            self._data["openids"].append(openid)
            self._save()
            return True  # 新用户
        return False  # 已存在

    def add_message(self, openid: str, content: str, msg_type: str = "text"):
        """记录一条消息"""
        self._data["messages"].append({
            "openid": openid,
            "content": content,
            "msg_type": msg_type,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        # 只保留最近 100 条消息
        if len(self._data["messages"]) > 100:
            self._data["messages"] = self._data["messages"][-100:]
        self._save()

    def get_all_openids(self) -> list:
        """获取所有 OpenID"""
        return self._data["openids"]

    def get_recent_messages(self, limit: int = 10) -> list:
        """获取最近的消息"""
        return self._data["messages"][-limit:]


store = OpenIDStore(OPENID_FILE)


# ============================================================
# 微信签名验证
# ============================================================

def check_signature(signature: str, timestamp: str, nonce: str) -> bool:
    """
    验证微信服务器签名。

    Args:
        signature: 微信加密签名
        timestamp: 时间戳
        nonce: 随机数

    Returns:
        签名是否有效
    """
    params = [WECHAT_TOKEN, timestamp, nonce]
    params.sort()
    sha1 = hashlib.sha1()
    sha1.update("".join(params).encode('utf-8'))
    return sha1.hexdigest() == signature


# ============================================================
# XML 解析工具
# ============================================================

def parse_xml(xml_str: str) -> dict:
    """
    简易 XML 解析（不依赖 lxml，用正则提取字段）。

    微信推送的 XML 格式：
    <xml>
        <ToUserName><![CDATA[公众号原始ID]]></ToUserName>
        <FromUserName><![CDATA[用户OpenID]]></FromUserName>
        <CreateTime>1348831860</CreateTime>
        <MsgType><![CDATA[text]]></MsgType>
        <Content><![CDATA[消息内容]]></Content>
        <MsgId>1234567890123456</MsgId>
    </xml>
    """
    import re

    result = {}
    # 匹配 <TagName><![CDATA[value]]></TagName>
    for match in re.finditer(r'<(\w+)><!\[CDATA\[(.*?)\]\]></\1>', xml_str, re.DOTALL):
        result[match.group(1)] = match.group(2)

    # 匹配 <TagName>value</TagName>（非 CDATA 字段，如 CreateTime、MsgId）
    for match in re.finditer(r'<(\w+)>([^<]+)</\1>', xml_str):
        tag = match.group(1)
        if tag not in result:  # CDATA 优先
            result[tag] = match.group(2)

    return result


def build_text_reply(to_user: str, from_user: str, content: str) -> str:
    """
    构建文本消息回复的 XML。

    Args:
        to_user: 接收方 OpenID（用户）
        from_user: 发送方（公众号原始 ID）
        content: 回复内容
    """
    return f"""<xml>
<ToUserName><![CDATA[{to_user}]]></ToUserName>
<FromUserName><![CDATA[{from_user}]]></FromUserName>
<CreateTime>{int(time.time())}</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[{content}]]></Content>
</xml>"""


# ============================================================
# 路由
# ============================================================

@app.route('/wechat', methods=['GET', 'POST'])
def wechat_handler():
    """微信消息处理入口"""

    # --- GET 请求：服务器验证 ---
    if request.method == 'GET':
        signature = request.args.get('signature', '')
        timestamp = request.args.get('timestamp', '')
        nonce = request.args.get('nonce', '')
        echostr = request.args.get('echostr', '')

        # 调试日志：打印微信发来的参数
        print(f"[验证] 收到验证请求:")
        print(f"        signature : {signature}")
        print(f"        timestamp : {timestamp}")
        print(f"        nonce     : {nonce}")
        print(f"        echostr   : {echostr}")
        print(f"        本地 Token: {WECHAT_TOKEN}")

        if check_signature(signature, timestamp, nonce):
            print("[验证] ✅ 微信服务器验证通过")
            return make_response(echostr)
        else:
            print("[验证] ❌ 签名验证失败")
            print(f"        请检查公众号后台的 Token 是否和本地 Token 一致")
            return make_response("Invalid signature", 403)

    # --- POST 请求：接收消息 ---
    if request.method == 'POST':
        xml_data = request.data.decode('utf-8')
        msg = parse_xml(xml_data)

        openid = msg.get('FromUserName', '')
        msg_type = msg.get('MsgType', '')
        to_user = msg.get('ToUserName', '')  # 公众号原始 ID
        content = msg.get('Content', '')
        event = msg.get('Event', '')

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- 处理关注事件 ---
        if msg_type == 'event' and event == 'subscribe':
            is_new = store.add_openid(openid)
            print(f"[事件] 🎉 新用户关注！OpenID: {openid} (新用户: {is_new})")

            # 回复欢迎消息
            reply = build_text_reply(
                to_user=openid,
                from_user=to_user,
                content="欢迎关注！发送任意消息即可激活智能提醒服务 🤖\n\n"
                        "你的 OpenID 已记录，现在可以使用了。"
            )
            return make_response(reply)

        # --- 处理取消关注 ---
        if msg_type == 'event' and event == 'unsubscribe':
            print(f"[事件] 用户取消关注: {openid}")
            return make_response("", 200)

        # --- 处理文本消息 ---
        if msg_type == 'text':
            is_new = store.add_openid(openid)
            store.add_message(openid, content, msg_type)

            print(f"[消息] [{timestamp}] OpenID: {openid} (新用户: {is_new})")
            print(f"        内容: {content}")

            # 显示所有已收集的 OpenID
            all_openids = store.get_all_openids()
            print(f"[状态] 已收集 {len(all_openids)} 个 OpenID:")
            for oid in all_openids:
                print(f"        - {oid}")

            # 自动回复
            if AUTO_REPLY:
                reply = build_text_reply(
                    to_user=openid,
                    from_user=to_user,
                    content=AUTO_REPLY_TEXT,
                )
                return make_response(reply)

            return make_response("", 200)

        # --- 处理其他消息类型（图片、语音等） ---
        is_new = store.add_openid(openid)
        print(f"[消息] [{timestamp}] OpenID: {openid}, 类型: {msg_type} (新用户: {is_new})")

        if AUTO_REPLY:
            reply = build_text_reply(
                to_user=openid,
                from_user=to_user,
                content=AUTO_REPLY_TEXT,
            )
            return make_response(reply)

        return make_response("", 200)


@app.route('/openids', methods=['GET'])
def get_openids():
    """查看已收集的所有 OpenID（调试用）"""
    openids = store.get_all_openids()
    messages = store.get_recent_messages(10)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>微信 OpenID 管理</title>
<style>
body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
h1 {{ color: #333; }}
h2 {{ color: #07c160; }}  /* 微信绿 */
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background: #07c160; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 14px; }}
.copy-btn {{ background: #07c160; color: white; border: none; padding: 6px 16px;
             border-radius: 4px; cursor: pointer; margin-left: 8px; }}
.copy-btn:hover {{ background: #06ad56; }}
</style></head><body>
<h1>📱 微信 OpenID 管理面板</h1>

<h2>已收集的用户 ({len(openids)} 人)</h2>
<table>
<tr><th>#</th><th>OpenID</th><th>操作</th></tr>"""

    for i, oid in enumerate(openids, 1):
        html += f"""<tr><td>{i}</td><td><code>{oid}</code></td>
<td><button class="copy-btn" onclick="navigator.clipboard.writeText('{oid}')">复制</button></td></tr>"""

    html += """</table>

<h2>最近消息</h2>
<table>
<tr><th>时间</th><th>OpenID</th><th>内容</th></tr>"""

    for msg in reversed(messages):
        html += f"""<tr><td>{msg['time']}</td><td><code>{msg['openid'][:20]}...</code></td>
<td>{msg['content'][:50]}</td></tr>"""

    html += """</table>
<p style="color: #999; margin-top: 30px;">将 OpenID 复制到 wechat_reminder.py 的 wechat_openid 配置中即可使用提醒功能。</p>
</body></html>"""

    return html


# ============================================================
# 启动
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='微信消息接收服务器')
    parser.add_argument('--port', type=int, default=8080, help='监听端口（默认 8080）')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址（默认 0.0.0.0）')
    parser.add_argument('--token', type=str, default=None, help=f'微信 Token（默认: {WECHAT_TOKEN}）')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    args = parser.parse_args()

    if args.token:
        WECHAT_TOKEN = args.token

    print("=" * 50)
    print("📱 微信消息接收服务器")
    print("=" * 50)
    print(f"  Token: {WECHAT_TOKEN}")
    print(f"  地址: http://{args.host}:{args.port}/wechat")
    print(f"  管理面板: http://{args.host}:{args.port}/openids")
    print(f"  OpenID 存储: {os.path.abspath(OPENID_FILE)}")
    print(f"  自动回复: {'开启' if AUTO_REPLY else '关闭'}")
    print()
    print("⚠️  请确保：")
    print("  1. 公众号后台的 Token 和此处一致")
    print("  2. 服务器 IP 已加入公众号白名单")
    print("  3. 公众号后台的服务器 URL 指向此地址")
    print()
    print("🚀 服务器启动中...")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)
