from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pathlib import Path
from pydantic import BaseModel
import os
import json
import asyncio
import threading
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


class ChatRequest(BaseModel):
    message: str
    user_id: str = "web_user"
    session_id: str = "default_session"


def create_app() -> FastAPI:
    app = FastAPI(title="Sixty-Three Multi-Agent API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.post("/api/chat")
    async def chat_endpoint(req: ChatRequest):
        return StreamingResponse(
            _run_agent_stream(req.message, req.user_id, req.session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/history")
    async def get_history(user_id: str = "web_user"):
        from .Conversation_storage import ConversationStorage
        storage = ConversationStorage()
        data = storage._load()
        if user_id not in data:
            return {"sessions": []}
        sessions = []
        for session_id, session_data in data[user_id].items():
            messages = session_data.get("messages", [])
            first_human = next((m for m in messages if m.get("type") == "human"), None)
            round_count = sum(1 for m in messages if m.get("type") == "human")
            sessions.append({
                "sessionId": session_id,
                "title": first_human["content"][:50] if first_human else "空对话",
                "roundCount": round_count,
                "messageCount": len(messages),
                "updatedAt": session_data.get("updated_at", ""),
            })
        sessions.sort(key=lambda s: s.get("updatedAt", ""), reverse=True)
        return {"sessions": sessions}

    @app.get("/api/history/{session_id}")
    async def get_session_history(session_id: str, user_id: str = "web_user"):
        from .Conversation_storage import ConversationStorage
        storage = ConversationStorage()
        data = storage._load()
        if user_id not in data or session_id not in data[user_id]:
            return {"session": None}
        session_data = data[user_id][session_id]
        return {
            "session": {
                "sessionId": session_id,
                "messages": session_data.get("messages", []),
                "updatedAt": session_data.get("updated_at", ""),
            }
        }

    @app.delete("/api/history/{session_id}")
    async def delete_session_history(session_id: str, user_id: str = "web_user"):
        from .Conversation_storage import ConversationStorage
        storage = ConversationStorage()
        deleted = storage.delete_session(user_id, session_id)
        return {"deleted": deleted}

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

    return app


def _run_agent_stream(message: str, user_id: str, session_id: str):
    """通过SSE实时推送agent执行状态"""
    from .multi_agent import (
        chat_with_multi_agent,
        download_agent,
        download_agent,
        exec_visualizer,
        knowledge_agent,
        reminder_agent,
        city_info_agent,
        _make_model,
        AgentState,
    )
    from .Conversation_storage import AgentExecutionVisualizer
    from .tools import set_current_user_id, get_last_rag_context, reset_tool_call_guards, set_download_progress_callback, clear_download_progress_callback, set_tool_event_callback, clear_tool_event_callback, set_current_step_id
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph, START, END

    # 创建一个自定义的visualizer来捕获事件
    step_counter = [0]
    captured_events = []

    class StreamVisualizer(AgentExecutionVisualizer):
        def on_node_start(self, node_name: str, input_data: dict):
            super().on_node_start(node_name, input_data)
            step_counter[0] += 1
            instruction = ""
            if isinstance(input_data, dict):
                instruction = input_data.get("supervisor_instruction", "")
            set_current_step_id(step_counter[0])
            captured_events.append({
                "type": "agent_start",
                "node": node_name,
                "step_id": step_counter[0],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "instruction": instruction,
            })

        def on_node_end(self, node_name: str, input_data: dict, output_data: dict, tools_used: list = None):
            super().on_node_end(node_name, input_data, output_data, tools_used)
            output_text = ""
            if isinstance(output_data, dict):
                if "messages" in output_data and output_data["messages"]:
                    last_msg = output_data["messages"][-1]
                    output_text = getattr(last_msg, "content", str(last_msg))[:200]
                elif "next_agent" in output_data:
                    output_text = f"路由: {output_data['next_agent']}"

            captured_events.append({
                "type": "agent_end",
                "node": node_name,
                "step_id": step_counter[0],
                "output": output_text,
                "tools_used": tools_used or [],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })

    # 替换全局visualizer
    import backend.multi_agent as ma_module
    original_visualizer = ma_module.exec_visualizer
    stream_viz = StreamVisualizer()
    ma_module.exec_visualizer = stream_viz

    result_holder = {}
    error_holder = {}

    def run_agent():
        try:
            set_current_user_id(user_id)
            get_last_rag_context(clear=True)
            reset_tool_call_guards()
            
            # 设置下载进度回调
            def download_progress_callback(progress_data):
                captured_events.append({
                    "type": progress_data["type"],
                    **progress_data
                })
            
            set_download_progress_callback(download_progress_callback)

            def tool_event_callback(event_data):
                captured_events.append(event_data)

            set_tool_event_callback(tool_event_callback)

            result = chat_with_multi_agent(
                user_text=message,
                user_id=user_id,
                session_id=session_id,
                verbose=False,
            )
            result_holder["result"] = result
        except Exception as e:
            error_holder["error"] = str(e)
        finally:
            ma_module.exec_visualizer = original_visualizer
            clear_download_progress_callback()
            clear_tool_event_callback()

    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()

    async def generate():
        sent_events = 0
        while agent_thread.is_alive() or sent_events < len(captured_events):
            while sent_events < len(captured_events):
                event = captured_events[sent_events]
                print(f"[SSE] Sending event: {event.get('type', 'unknown')}")
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                sent_events += 1
            await asyncio.sleep(0.1)

        agent_thread.join(timeout=120)

        while sent_events < len(captured_events):
            event = captured_events[sent_events]
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            sent_events += 1

        if "error" in error_holder:
            response_event = {
                "type": "response",
                "content": f"错误: {error_holder['error']}",
                "agent_tool_path": "",
            }
        elif "result" in result_holder:
            r = result_holder["result"]
            response_event = {
                "type": "response",
                "content": r.get("response", ""),
                "agent_tool_path": "",
            }
            execution_history = r.get("execution_history", [])
            if execution_history:
                traces = []
                for step in execution_history:
                    agent_name = step.get("agent", "")
                    tools = step.get("tools_used", [])
                    if agent_name and tools:
                        traces.append(f"{agent_name}({','.join(tools)})")
                    elif agent_name:
                        traces.append(agent_name)
                response_event["agent_tool_path"] = "→".join(traces)
        else:
            response_event = {
                "type": "response",
                "content": "处理超时，请重试",
                "agent_tool_path": "",
            }

        yield f"data: {json.dumps(response_event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return generate()


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8000)))
