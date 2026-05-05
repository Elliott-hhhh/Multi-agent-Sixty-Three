"""
BilibiliVideoSearchTool - LangChain 工具
=========================================
根据用户自然语言需求，自动在 B 站搜索视频并返回链接。

功能：
1. 从用户自然语言中提取搜索关键词
2. 调用 B 站搜索 API（带 WBI 签名）获取视频结果
3. 返回最匹配的视频链接 + 下载方式指引

依赖：
    pip install requests langchain-core

使用示例：
    from bilibili_tool import BilibiliVideoSearchTool

    tool = BilibiliVideoSearchTool()
    result = tool.invoke({"query": "我想下载《爱情怎么翻译》来看一看"})
    print(result)
"""

import re
import time
import json
from functools import reduce
from hashlib import md5
from typing import Optional, Type

import requests
import urllib.parse

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ============================================================
# WBI 签名相关
# ============================================================

# 固定的字符打乱编码表（B站 WBI 签名用）
MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52
]

# 默认请求头
DEFAULT_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Referer': 'https://www.bilibili.com',
}


def get_mixin_key(orig: str) -> str:
    """对 imgKey 和 subKey 进行字符顺序打乱编码"""
    return reduce(lambda s, i: s + orig[i], MIXIN_KEY_ENC_TAB, '')[:32]


def enc_wbi(params: dict, img_key: str, sub_key: str) -> dict:
    """为请求参数进行 WBI 签名"""
    mixin_key = get_mixin_key(img_key + sub_key)
    params['wts'] = round(time.time())
    params = dict(sorted(params.items()))
    params = {
        k: ''.join(filter(lambda c: c not in "!'()*", str(v)))
        for k, v in params.items()
    }
    query = urllib.parse.urlencode(params)
    params['w_rid'] = md5((query + mixin_key).encode()).hexdigest()
    return params


# ============================================================
# 关键词提取
# ============================================================

def extract_search_keyword(user_query: str) -> str:
    """
    从用户自然语言中提取搜索关键词。

    支持的模式：
    - "我想下载《XXX》" → XXX
    - "帮我搜索 XXX" → XXX
    - "找一下 XXX 视频" → XXX
    - "有没有 XXX" → XXX
    - 直接输入关键词 → 原样返回
    """
    text = user_query.strip()

    # 模式1：提取书名号《》中的内容
    book_match = re.search(r'《(.+?)》', text)
    if book_match:
        keyword = book_match.group(1)
        # 检查书名号后面是否还有额外信息
        after = text[book_match.end():].strip()
        # 去掉常见的动词后缀
        after = re.sub(r'^(来看一看|来看|看一下|吧|啊|呢|呀|嘛|的|了|？|\?|！|\!|。|\.)*$', '', after)
        if after:
            keyword = keyword + ' ' + after
        return keyword.strip()

    # 模式2：提取引号中的内容
    quote_match = re.search(r'[""「」](.+?)[""「」]', text)
    if quote_match:
        return quote_match.group(1).strip()

    # 模式3：去掉常见的意图动词前缀（长前缀优先匹配）
    prefix_patterns = [
        # 先尝试匹配更长的前缀模式
        r'^能不能帮我搜[索一下]*\s*(.+)',
        r'^能不能帮我找[一下]*\s*(.+)',
        r'^请帮我搜一下\s*(.+)',
        r'^请帮我找一下\s*(.+)',
        r'^帮我搜一下\s*(.+)',
        r'^帮我找一下\s*(.+)',
        r'^我想下载\s*(.+)',
        r'^帮我下载\s*(.+)',
        r'^我想看[一看]*\s*(.+)',
        r'^帮我搜[索]*\s*(.+)',
        r'^帮我找[一下]*\s*(.+)',
        r'^搜一下\s*(.+)',
        r'^找一下\s*(.+)',
        r'^搜一搜\s*(.+)',
        r'^找一找\s*(.+)',
        r'^能不能帮我\s*(.+)',
        r'^能不能\s*(.+)',
        r'^能否\s*(.+)',
        r'^推荐一下?\s*(.+)',
        r'^给我\s*(.+)',
        r'^能给我\s*(.+)',
        r'^有没有\s*(.+)',
        r'^请帮我\s*(.+)',
        r'^帮我\s*(.+)',
        r'^搜索\s*(.+)',
        r'^查找\s*(.+)',
        r'^请\s*(.+)',
        r'^麻烦\s*(.+)',
    ]
    for pattern in prefix_patterns:
        match = re.match(pattern, text)
        if match:
            keyword = match.group(1).strip()
            # 去掉尾部语气词和标点（保留有搜索价值的词如"番剧""动漫"等）
            keyword = re.sub(
                r'(来看一看|来看|看一下|吧|啊|呢|呀|嘛|？|\?|！|\!|。|\.)$',
                '', keyword
            )
            return keyword.strip() if keyword.strip() else match.group(1).strip()

    # 模式4：直接返回（去掉尾部标点和语气词）
    keyword = re.sub(
        r'(来看一看|来看|看一下|吧|啊|呢|呀|嘛|的|了|？|\?|！|\!|。|\.|视频)$',
        '', text
    )
    return keyword.strip() if keyword.strip() else text


# ============================================================
# B站搜索客户端
# ============================================================

class BilibiliSearchClient:
    """B站搜索 API 客户端（带 WBI 签名）"""

    SEARCH_API_URL = "https://api.bilibili.com/x/web-interface/wbi/search/type"
    NAV_API_URL = "https://api.bilibili.com/x/web-interface/nav"

    def __init__(
        self,
        sessdata: Optional[str] = None,
        proxy: Optional[str] = None,
        timeout: int = 15,
    ):
        """
        初始化 B站搜索客户端。

        Args:
            sessdata: B站登录 Cookie 中的 SESSDATA（可选，用于获取更高清晰度）
            proxy: 代理地址，如 "socks5://127.0.0.1:10808"（可选）
            timeout: 请求超时时间（秒）
        """
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.timeout = timeout

        if proxy:
            self.session.proxies = {
                'http': proxy,
                'https': proxy,
            }

        if sessdata:
            self.session.cookies.set('SESSDATA', sessdata, domain='.bilibili.com')

        self._img_key: Optional[str] = None
        self._sub_key: Optional[str] = None
        self._keys_expire: float = 0

    def _ensure_keys(self):
        """确保 WBI 密钥可用（带缓存，1小时过期）"""
        if self._img_key and time.time() < self._keys_expire:
            return

        # 先访问首页获取 buvid3 cookie
        try:
            self.session.get('https://www.bilibili.com/', timeout=self.timeout)
        except requests.RequestException:
            pass

        # 获取 WBI 密钥
        try:
            resp = self.session.get(self.NAV_API_URL, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"无法连接到 Bilibili API: {e}") from e

        if data.get('code') != 0:
            # 尝试从 wbi_img 字段获取（即使 code 非零，有时该字段仍可用）
            wbi_img = data.get('data', {}).get('wbi_img')
            if wbi_img and wbi_img.get('img_url') and wbi_img.get('sub_url'):
                self._img_key = wbi_img['img_url'].rsplit('/', 1)[1].split('.')[0]
                self._sub_key = wbi_img['sub_url'].rsplit('/', 1)[1].split('.')[0]
                self._keys_expire = time.time() + 3600
                return
            raise RuntimeError(
                f"获取 WBI 密钥失败: code={data.get('code')}, "
                f"message={data.get('message', '未知错误')}。"
                f"请检查网络连接，或尝试设置 SESSDATA。"
            )

        wbi_img = data['data']['wbi_img']
        self._img_key = wbi_img['img_url'].rsplit('/', 1)[1].split('.')[0]
        self._sub_key = wbi_img['sub_url'].rsplit('/', 1)[1].split('.')[0]
        self._keys_expire = time.time() + 3600  # 1小时过期

    def search_videos(
        self,
        keyword: str,
        page: int = 1,
        page_size: int = 10,
        order: str = "totalrank",
        duration: int = 0,
    ) -> dict:
        """
        搜索 B站视频。

        Args:
            keyword: 搜索关键词
            page: 页码（从1开始）
            page_size: 每页结果数量（最大50）
            order: 排序方式
                - totalrank: 综合排序（默认）
                - click: 最多点击
                - pubdate: 最新发布
                - dm: 最多弹幕
                - stow: 最多收藏
                - scores: 最多评论
            duration: 时长筛选
                - 0: 全部（默认）
                - 1: 10分钟以下
                - 2: 10-30分钟
                - 3: 30-60分钟
                - 4: 60分钟以上

        Returns:
            包含搜索结果的字典，格式：
            {
                "code": 0,
                "num_results": 1000,
                "page": 1,
                "results": [
                    {
                        "title": "视频标题",
                        "bvid": "BV1xxx",
                        "url": "https://www.bilibili.com/video/BV1xxx",
                        "author": "UP主名",
                        "play": 123456,
                        "danmaku": 7890,
                        "favorites": 1234,
                        "duration": "10:30",
                        "description": "视频简介",
                        "pubdate": 1234567890,
                        "cover": "https://...",
                        "tag": "标签1,标签2",
                        "typename": "分区名",
                    },
                    ...
                ]
            }
        """
        self._ensure_keys()

        params = {
            'search_type': 'video',
            'keyword': keyword,
            'page': page,
            'page_size': min(page_size, 50),
            'order': order,
            'duration': duration,
        }

        signed_params = enc_wbi(params, self._img_key, self._sub_key)

        resp = self.session.get(
            self.SEARCH_API_URL,
            params=signed_params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get('code') != 0:
            raise RuntimeError(f"搜索失败: code={data.get('code')}, message={data.get('message')}")

        raw_results = data.get('data', {}).get('result', [])
        parsed_results = []

        for item in raw_results:
            # 清理标题中的 HTML 标签（B站搜索结果中标题可能包含 <em> 高亮标签）
            clean_title = re.sub(r'<[^>]+>', '', item.get('title', ''))

            # 构造标准 URL（优先使用 bvid）
            bvid = item.get('bvid', '')
            if bvid:
                url = f"https://www.bilibili.com/video/{bvid}"
            else:
                url = item.get('arcurl', '')

            # 封面图 URL 补全协议
            pic = item.get('pic', '')
            if pic.startswith('//'):
                pic = 'https:' + pic

            parsed_results.append({
                'title': clean_title,
                'bvid': bvid,
                'url': url,
                'author': item.get('author', ''),
                'mid': item.get('mid', 0),
                'play': item.get('play', 0),
                'danmaku': item.get('video_review', 0),
                'favorites': item.get('favorites', 0),
                'duration': item.get('duration', ''),
                'description': item.get('description', ''),
                'pubdate': item.get('pubdate', 0),
                'cover': pic,
                'tag': item.get('tag', ''),
                'typename': item.get('typename', ''),
                'is_pay': item.get('is_pay', 0),
            })

        return {
            'code': 0,
            'num_results': data.get('data', {}).get('numResults', 0),
            'page': data.get('data', {}).get('page', 1),
            'results': parsed_results,
        }


# ============================================================
# 下载链接生成
# ============================================================

def generate_download_info(bvid: str, title: str) -> dict:
    """
    为视频生成下载相关信息。

    由于 B站视频流 URL 有效期仅 120 分钟且需要鉴权，
    这里提供多种实用的下载方式供用户选择。

    Args:
        bvid: 视频 BV 号
        title: 视频标题

    Returns:
        下载信息字典
    """
    video_url = f"https://www.bilibili.com/video/{bvid}"

    return {
        'bilibili_url': video_url,
        'download_methods': {
            'yt-dlp (推荐)': {
                'description': '命令行工具，支持批量下载，自动选择最佳画质',
                'install': 'pip install yt-dlp',
                'commands': [
                    f'yt-dlp -f "bv[ext=mp4]+ba[ext=m4a]" --merge-output-format mp4 "{video_url}"',
                    f'# 仅下载音频: yt-dlp -f "ba" -x --audio-format mp3 "{video_url}"',
                ],
            },
            'you-get': {
                'description': '轻量级 Python 下载工具',
                'install': 'pip install you-get',
                'commands': [
                    f'you-get "{video_url}"',
                ],
            },
            '在线解析网站': {
                'description': '无需安装，粘贴链接即可下载',
                'urls': [
                    f'https://xbeibeix.com/api/bilibili/?url={video_url}',
                    f'https://snapany.com/',
                ],
            },
            'bilibili-api-python': {
                'description': 'Python 库，适合集成到项目中（需要登录凭据）',
                'install': 'pip install bilibili-api-python',
                'note': '需要 SESSDATA 凭据才能获取高清视频流',
            },
        },
    }


# ============================================================
# LangChain 工具
# ============================================================

class BilibiliVideoSearchInput(BaseModel):
    """B站视频搜索工具的输入 schema"""
    query: str = Field(
        description=(
            "用户的自然语言查询，例如：'我想下载《爱情怎么翻译》来看一看'、"
            "'帮我找一下 Python 教程'、'有没有好看的科幻电影'"
        )
    )
    top_k: int = Field(
        default=5,
        description="返回的搜索结果数量，默认为5",
        ge=1,
        le=20,
    )


class BilibiliVideoSearchTool(BaseTool):
    """
    B站视频搜索工具 - LangChain Tool

    根据用户的自然语言需求，自动在 Bilibili 上搜索视频，
    返回最匹配的视频链接和下载方式。

    功能：
    1. 从自然语言中智能提取搜索关键词
    2. 调用 B站搜索 API（自动处理 WBI 签名）
    3. 返回格式化的搜索结果（标题、链接、UP主、播放量等）
    4. 提供多种下载方式指引（yt-dlp、you-get、在线解析等）
    """

    name: str = "bilibili_video_search"
    description: str = (
        "在 Bilibili（B站）上搜索视频。"
        "当用户想要查找、搜索或下载视频时使用此工具。"
        "输入用户的自然语言描述，工具会自动提取关键词并搜索。"
        "返回最匹配的视频列表，包含视频链接和下载方式。"
    )
    args_schema: Type[BaseModel] = BilibiliVideoSearchInput

    # 可配置参数
    sessdata: Optional[str] = None
    proxy: Optional[str] = None
    timeout: int = 15
    _client: Optional[BilibiliSearchClient] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = BilibiliSearchClient(
            sessdata=self.sessdata,
            proxy=self.proxy,
            timeout=self.timeout,
        )

    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        执行搜索并返回格式化结果。

        Args:
            query: 用户的自然语言查询
            top_k: 返回结果数量
            run_manager: LangChain 回调管理器

        Returns:
            格式化的搜索结果字符串
        """
        # 1. 提取搜索关键词
        keyword = extract_search_keyword(query)

        # 2. 搜索视频
        try:
            search_result = self._client.search_videos(
                keyword=keyword,
                page=1,
                page_size=top_k,
            )
        except Exception as e:
            return f"❌ 搜索失败：{str(e)}\n\n请稍后重试，或检查网络连接。"

        results = search_result.get('results', [])

        if not results:
            return (
                f"🔍 未找到与「{keyword}」相关的视频。\n\n"
                f"建议：\n"
                f"1. 尝试更换关键词\n"
                f"2. 检查是否有错别字\n"
                f"3. 使用更简短的关键词重新搜索"
            )

        # 3. 格式化输出
        output_parts = []

        # 搜索摘要
        total = search_result.get('num_results', 0)
        output_parts.append(
            f"🎬 在 Bilibili 上搜索「{keyword}」，"
            f"共找到约 {total} 个结果，以下是最匹配的 {len(results)} 个：\n"
        )

        # 视频列表
        for i, video in enumerate(results, 1):
            # 格式化播放量
            play_count = video['play']
            if play_count >= 10000:
                play_str = f"{play_count / 10000:.1f}万"
            else:
                play_str = str(play_count)

            # 格式化收藏量
            fav_count = video['favorites']
            if fav_count >= 10000:
                fav_str = f"{fav_count / 10000:.1f}万"
            else:
                fav_str = str(fav_count)

            # 标记付费视频
            pay_tag = " 💰付费" if video.get('is_pay') else ""

            output_parts.append(
                f"{'─' * 50}\n"
                f"📌 [{i}] {video['title']}{pay_tag}\n"
                f"   🔗 链接：{video['url']}\n"
                f"   👤 UP主：{video['author']}  |  📂 分区：{video['typename']}\n"
                f"   ▶️ 播放：{play_str}  |  💾 收藏：{fav_str}  |  💬 弹幕：{video['danmaku']}\n"
                f"   ⏱️ 时长：{video['duration']}"
            )

            if video.get('tag'):
                output_parts.append(f"   🏷️ 标签：{video['tag']}")

        # 下载指引
        output_parts.append(f"\n{'─' * 50}")
        output_parts.append("📥 下载方式：")
        output_parts.append("   方式1 (推荐)：yt-dlp 命令行工具")
        output_parts.append("     安装：pip install yt-dlp")
        if results:
            example_url = results[0]['url']
            output_parts.append(
                f"     下载：yt-dlp -f \"bv[ext=mp4]+ba[ext=m4a]\" "
                f"--merge-output-format mp4 \"{example_url}\""
            )
        output_parts.append("   方式2：you-get 工具")
        output_parts.append("     安装：pip install you-get")
        output_parts.append("   方式3：在线解析网站（无需安装）")
        output_parts.append("     https://xbeibeix.com/api/bilibili/")
        output_parts.append("     https://snapany.com/")

        return '\n'.join(output_parts)

    async def _arun(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """异步执行（当前为同步实现）"""
        return self._run(query=query, top_k=top_k, run_manager=run_manager)


# ============================================================
# 便捷函数（非 LangChain 场景下也可直接使用）
# ============================================================

def search_bilibili_videos(
    query: str,
    top_k: int = 5,
    sessdata: Optional[str] = None,
    proxy: Optional[str] = None,
) -> str:
    """
    便捷函数：搜索 B站视频并返回格式化结果。

    Args:
        query: 用户的自然语言查询
        top_k: 返回结果数量
        sessdata: B站 SESSDATA（可选）
        proxy: 代理地址（可选）

    Returns:
        格式化的搜索结果字符串
    """
    tool = BilibiliVideoSearchTool(
        sessdata=sessdata,
        proxy=proxy,
    )
    return tool.invoke({"query": query, "top_k": top_k})


def search_bilibili_videos_raw(
    keyword: str,
    top_k: int = 10,
    sessdata: Optional[str] = None,
    proxy: Optional[str] = None,
) -> dict:
    """
    便捷函数：搜索 B站视频并返回原始结构化数据。

    Args:
        keyword: 搜索关键词（直接关键词，非自然语言）
        top_k: 返回结果数量
        sessdata: B站 SESSDATA（可选）
        proxy: 代理地址（可选）

    Returns:
        包含搜索结果的字典
    """
    client = BilibiliSearchClient(
        sessdata=sessdata,
        proxy=proxy,
    )
    return client.search_videos(keyword=keyword, page=1, page_size=top_k)
