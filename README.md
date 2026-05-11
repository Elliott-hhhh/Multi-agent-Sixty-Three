# Sixty-Three — 多 Agent 智能助手

基于 LangGraph 构建的多 Agent 协作系统，通过 Supervisor 统一调度多个专业 Agent，实现知识检索、联网搜索、定时提醒、城市信息查询、视觉分析、代码处理等能力。
演示视频：【一个多agent项目，逐渐掌控你的需求】 https://www.bilibili.com/video/BV1wrRkBoETM/?share_source=copy_web&vd_source=4d4c31d4f92e85c72344a3f3174d414f
## 项目架构

```
muti_agent/
├── backend/
│   ├── multi_agent.py          # 核心工作流：Agent 定义、StateGraph 构建、路由逻辑
│   ├── app.py                  # FastAPI 服务入口，SSE 流式输出
│   ├── tools.py                # 工具定义：文件操作、命令执行、知识库检索等
│   ├── bilibili_tool.py        # B站视频搜索与下载工具
│   ├── reminder_scheduler.py   # 定时提醒调度器
│   ├── wechat_reminder.py      # 微信公众号提醒推送
│   ├── wechat_server.py        # 微信公众号服务端
│   ├── Conversation_storage.py # 对话历史持久化存储
│   └── vision_tools/           # 视觉模块
│       ├── camera_manager.py   # 摄像头管理（打开/捕获/关闭）
│       ├── vlm_tool.py         # 多模态大模型视觉分析工具
│       ├── ocr_tool.py         # PaddleOCR 文字识别工具
│       └── config.py           # 视觉模块配置
├── frontend/
│   ├── index.html              # 前端界面（对话、执行流程可视化、历史记录）
│   └── style.css               # 样式表
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
└── .gitignore
```

## 多 Agent 工作流设计

### 整体架构

系统采用 **Supervisor（调度中心）+ 多专业 Agent** 的架构模式：

```
                        ┌─────────────┐
                        │   用户请求   │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │  Supervisor  │  ← 智能路由决策
                        │  (调度中心)  │
                        └──────┬──────┘
                               │
          ┌────────┬───────┬───┴───┬────────┬────────┬────────┐
          ▼        ▼       ▼       ▼        ▼        ▼        ▼
     ┌────────┐┌───────┐┌──────┐┌──────┐┌───────┐┌──────┐┌──────┐
     │knowledge││reminder││cityinfo││vision││coding ││download││ chat │
     │ 知识检索 ││ 定时提醒││城市信息││视觉分析││代码处理││ 视频下载││闲聊  │
     └────────┘└───────┘└──────┘└──────┘└───────┘└──────┘└──────┘
          │        │       │       │        │        │        │
          └────────┴───┬───┘       │        │        │        │
                       │           │        │        │        │
                  ┌────▼────┐      │        │        │    ┌───▼───┐
                  │Supervisor│◄─────┘        │        │    │ END  │
                  │(重新思考) │◄──────────────┘        │    └───────┘
                  └────┬────┘                           │
                       │                          ┌────▼────┐
                  ┌────▼────┐                      │  END   │
                  │  END    │                      └─────────┘
                  └─────────┘
```

### 状态流转机制

系统通过 `AgentState` 在所有节点间传递状态：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # 对话消息（自动累积）
    next_agent: str                            # 下一个要执行的 Agent
    supervisor_instruction: str                # Supervisor 给子 Agent 的调度指令
    execution_history: list                    # 执行历史（避免重复执行）
    retry_count: int                           # 重试次数
    last_agent: str                            # 上一个执行的 Agent
    last_output: str                           # 上一个 Agent 的输出
    pending_video_url: str                     # 待下载的视频 URL
    awaiting_download_confirmation: bool       # 是否等待用户确认下载
```

### 路由策略

Supervisor 采用**关键词预检查 + LLM 智能决策**的双重路由策略：

1. **关键词预检查**：对视觉/摄像头等关键词直接路由，同时检查 `execution_history` 避免重复
2. **LLM 智能决策**：对复杂意图，由 LLM 分析用户消息和执行历史后输出 JSON 格式的路由决策
3. **状态驱动路由**：如检测到 `pending_video_url`，自动路由到下载确认流程

### 条件边设计

子 Agent 执行完成后，通过条件边决定下一步：

| 节点 | next_agent=FINISH | next_agent=supervisor |
|------|:-:|:-:|
| knowledge | ✅ 直接结束 | ✅ 返回 Supervisor |
| reminder | — | ✅ 返回 Supervisor |
| cityinfo | — | ✅ 返回 Supervisor |
| vision | ✅ 直接结束 | ✅ 返回 Supervisor |
| coding | — | ✅ 返回 Supervisor |
| download | ✅ 直接结束 | — |
| direct_answer | ✅ 直接结束 | — |

## 各节点功能说明

### Supervisor（调度中心）

**职责**：分析用户意图，决定将任务路由到哪个专业 Agent

**路由逻辑**：
- 关键词预检查 → 直接路由（如视觉请求）
- 下载确认状态检查 → 自动路由到下载流程
- LLM 智能决策 → 输出 `{"decision": "agent_name", "instruction": "..."}`

**防重复机制**：检查 `execution_history`，避免已执行的节点被重复调度

---

### Knowledge Node（知识检索）

**职责**：文档检索、知识问答、联网搜索、B站视频搜索

**工具**：
- `search_knowledge_base` — 向量知识库检索
- `web_search` — 百度联网搜索
- `bilibili_video_search` — B站视频搜索
- `bilibili_download` — B站视频下载

**特殊逻辑**：搜索视频后自动提取 URL，保存到 `pending_video_url` 状态中

---

### Reminder Node（定时提醒）

**职责**：设置定时提醒，通过微信公众号推送

**工具**：
- `set_reminder` — 设置提醒（解析自然语言时间）

**推送渠道**：微信公众号模板消息

---

### CityInfo Node（城市信息）

**职责**：查询城市天气、日期时间等信息

**工具**：
- `get_city_info` — 查询城市天气和信息

---

### Vision Node（视觉分析）

**职责**：摄像头捕获、图像分析、文字识别

**工具**：
- `capture_and_analyze` — 捕获图像并分析内容（多模态大模型）
- `capture_and_answer` — 捕获图像并回答问题
- `capture_and_ocr` — 捕获图像并识别文字（PaddleOCR）
- `capture_and_ocr_detail` — 捕获图像并详细识别文字

**模型**：Qwen-VL 多模态模型

**资源管理**：使用上下文管理器（`with CameraManager() as camera`）确保摄像头资源正确释放

---

### Coding Node（代码处理）

**职责**：文件创建/修改/读取、命令执行

**工具**：
- `create_file` — 创建文件
- `modify_file` — 修改文件（行级精确修改）
- `read_file` — 读取文件完整内容
- `read_file_summary` — 读取文件摘要（节省 Token）
- `add_file_summary` — 为文件添加摘要注释
- `run_command` — 执行终端命令（需用户确认）
- `execute_confirmed_command` — 执行已确认的命令

**安全机制**：命令执行需用户确认，防止误操作

---

### Download Node（视频下载）

**职责**：B站视频下载确认与执行

**流程**：
1. 从 `pending_video_url` 读取待下载 URL
2. 生成确认请求，等待用户回复
3. 用户确认后执行下载

---

### Direct Answer Node（闲聊）

**职责**：处理通用闲聊、简单问答

**特点**：无需工具调用，直接由 LLM 生成回答

## Memory 机制

### 对话持久化

- 通过 `Conversation_storage` 模块将对话历史保存到本地文件
- 每个会话通过 `session_id` 隔离
- 支持跨会话恢复对话上下文

### 自动摘要

当对话消息超过 50 条时，自动将前 40 条总结为摘要：

```python
if len(messages) > 50:
    summary = summarize_old_messages(model, messages[:40])
    messages = [SystemMessage(content=f"之前的对话摘要：\n{summary}")] + messages[40:]
```

### 多 Agent 消息共享

所有 Agent 共享同一个 `AgentState`，通过状态传递实现消息共享：

- **消息级共享**：`messages` 列表在所有节点间自动累积
- **状态级共享**：`pending_video_url` 等字段跨节点传递
- **历史级共享**：`execution_history` 让所有 Agent 知道之前执行了什么
- **指令注入**：Supervisor 通过 `supervisor_instruction` 向子 Agent 传递调度指令

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/your-username/muti_agent.git
cd muti_agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key（详见下方 [环境变量说明](#环境变量说明)）

### 4. 启动服务

在项目根目录执行：

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

浏览器访问 `http://localhost:8000`

## 环境变量说明


### 必需配置

| 变量名 | 说明 | 获取方式 |
|--------|------|----------|
| `OPENAI_API_KEY` | 主模型 API Key | [DeepSeek 开放平台](https://platform.deepseek.com/) 或 [OpenAI](https://platform.openai.com/) 注册获取 |
| `BASE_URL` | 主模型 API 地址 | DeepSeek: `https://api.deepseek.com`；OpenAI: `https://api.openai.com/v1` |
| `MODEL` | 主模型名称 | 如 `deepseek-chat`、`gpt-4o` 等 |
| `CODING_API_KEY` | 代码模型 API Key | [阿里云 DashScope](https://dashscope.console.aliyun.com/) 开通并获取 |
| `CODING_BASE_URL` | 代码模型 API 地址 | 阿里云: `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `WEATHER_API_KEY` | 天气查询 API Key | [和风天气](https://dev.qweather.com/) 注册获取 |
| `BAIDU_WEB_SEARCH_API_KEY` | 百度搜索 API Key | [百度智能云](https://cloud.baidu.com/) 开通搜索 API 获取 |

### 可选配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `VISION_MODEL` | 视觉模型名称 | `qwen3.6-plus` |
| `CODING_MODEL` | 代码模型名称 | `qwen-coder-max` |
| `GRADE_MODEL` | 评分模型名称 | `qwen3-max` |
| `CAMERA_INDEX` | 摄像头设备编号 | `0` |
| `CAMERA_WIDTH` | 摄像头分辨率宽度 | `1280` |
| `CAMERA_HEIGHT` | 摄像头分辨率高度 | `720` |

### 微信公众号配置（可选）

如需定时提醒推送功能，需配置以下变量：

| 变量名 | 说明 | 获取方式 |
|--------|------|----------|
| `WECHAT_APP_ID` | 公众号 AppID | [微信公众平台](https://mp.weixin.qq.com/) 开发者设置 |
| `WECHAT_APP_SECRET` | 公众号 AppSecret | 同上 |
| `WECHAT_TOKEN` | 公众号 Token | 自定义，需与公众平台配置一致 |
| `WECHAT_TEMPLATE_ID` | 模板消息 ID | 公众平台 → 模板消息 中创建 |
| `WECHAT_DEFAULT_OPENID` | 默认推送用户 OpenID | 用户关注公众号后获取 |

### Embedding 模型配置（可选）

| 变量名 | 说明 | 获取方式 |
|--------|------|----------|
| `HF_EMBEDDING_MODEL_PATH` | 本地 Embedding 模型路径 | 下载 [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base) 到本地 |


## 技术栈

- **框架**：LangChain + LangGraph（Agent 构建 & 工作流编排）
- **模型**：DeepSeek / Qwen / OpenAI（通过 OpenAI 兼容接口统一调用）
- **视觉**：Qwen-VL（多模态分析）+ PaddleOCR（文字识别）+ OpenCV（摄像头）
- **后端**：FastAPI + Uvicorn（SSE 流式输出）
- **前端**：原生 HTML/CSS/JS + Marked.js（Markdown 渲染）
- **存储**：本地文件系统（对话历史持久化）
- **推送**：微信公众号模板消息

## License

MIT
