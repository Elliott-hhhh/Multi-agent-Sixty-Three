from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

@dataclass
class ExecutionStep:
    step_id: int
    node_name: str
    input_data: dict
    output_data: dict
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    duration_ms: float = 0
    tools_used: list = field(default_factory=list)

    def to_display(self) -> str:
        lines = [
            f"\n{'=' * 60}",
            f" Step {self.step_id} | {self.node_name.upper()}",
            f" {self.timestamp}",
            f"{'=' * 60}",
            # f"\n 输入:",
        ]

        # if "messages" in self.input_data:
        #     for i, msg in enumerate(self.input_data["messages"]):
        #         msg_type = type(msg).__name__
        #         content = msg.content if hasattr(msg, "content") else str(msg)
        #         lines.append(f"   [{i}] {msg_type}: {content[:200]}{'...' if len(content) > 200 else ''}")

        if "next_agent" in self.input_data and self.input_data["next_agent"]:
            lines.append(f"   路由目标: {self.input_data['next_agent']}")

        # 显示使用的工具
        if self.tools_used:
            lines.append(f"   使用的工具: {', '.join(self.tools_used)}")
        
        lines.append(f"\n 输出:")
        if "messages" in self.output_data and self.output_data["messages"]:
            for msg in self.output_data["messages"]:
                msg_type = type(msg).__name__
                content = msg.content if hasattr(msg, "content") else str(msg)
                lines.append(f"   {msg_type}: {content[:300]}{'...' if len(content) > 300 else ''}")
        if "next_agent" in self.output_data:
            lines.append(f"   下一节点: {self.output_data['next_agent']}")

        return "\n".join(lines)


class AgentExecutionVisualizer:
    def __init__(self):
        self.steps: list[ExecutionStep] = []
        self._step_counter = 0
        self._node_start_time: dict[str, float] = {}

    def reset(self):
        self.steps = []
        self._step_counter = 0
        self._node_start_time = {}

    def on_node_start(self, node_name: str, input_data: dict):
        import time
        self._node_start_time[node_name] = time.perf_counter() * 1000

    def on_node_end(self, node_name: str, input_data: dict, output_data: dict, tools_used: list = None):
        import time
        self._step_counter += 1
        duration = time.perf_counter() * 1000 - self._node_start_time.get(node_name, 0)

        step = ExecutionStep(
            step_id=self._step_counter,
            node_name=node_name,
            input_data=self._serialize(input_data),
            output_data=self._serialize(output_data),
            duration_ms=duration,
            tools_used=tools_used if tools_used else []
        )
        self.steps.append(step)

    def _serialize(self, data: Any) -> dict:
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if k == "messages":
                    result[k] = v
                elif isinstance(v, (str, int, float, bool, type(None))):
                    result[k] = v
                else:
                    result[k] = str(v)[:100]
            return result
        return {"value": str(data)[:200]}

    def get_execution_trace(self) -> list[dict]:
        return [
            {
                "step_id": s.step_id,
                "node_name": s.node_name,
                "input_summary": self._summarize_input(s.input_data),
                "output_summary": self._summarize_output(s.output_data),
                "timestamp": s.timestamp,
                "duration_ms": s.duration_ms,
            }
            for s in self.steps
        ]

    def _summarize_input(self, data: dict) -> str:
        if "messages" in data:
            msgs = data["messages"]
            last_msg = msgs[-1] if msgs else None
            if last_msg and hasattr(last_msg, "content"):
                content = last_msg.content
                return f"messages[{len(msgs)}]: {content[:80]}..."
        if "next_agent" in data:
            return f"路由: {data['next_agent']}"
        return str(data)[:80]

    def _summarize_output(self, data: dict) -> str:
        if "messages" in data and data["messages"]:
            last = data["messages"][-1]
            content = last.content if hasattr(last, "content") else str(last)
            return f"{type(last).__name__}: {content[:80]}..."
        if "next_agent" in data:
            return f"路由: {data['next_agent']}"
        return str(data)[:80]

    def print_trace(self):
        for step in self.steps:
            print(step.to_display())
        print(f"\n{'=' * 60}")
        print(f" 共执行 {len(self.steps)} 个步骤")

class ConversationStorage:
    """对话存储"""

    def __init__(self, storage_file: str = None):
        if storage_file:
            storage_path = os.path.abspath(storage_file)
        else:
            package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_dir = os.path.join(package_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            storage_path = os.path.join(data_dir, "customer_service_history.json")

        self.storage_file = storage_path

    def save(self, user_id: str, session_id: str, messages: list, metadata: dict = None, extra_message_data: list = None):
        """保存对话"""
        data = self._load()

        if user_id not in data:
            data[user_id] = {}

        serialized = []
        for idx, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                msg_type = "human"
            elif isinstance(msg, AIMessage):
                msg_type = "ai"
            elif isinstance(msg, SystemMessage):
                msg_type = "system"
            else:
                msg_type = str(type(msg).__name__).lower()
            record = {
                "type": msg_type,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            }
            if extra_message_data and idx < len(extra_message_data):
                extra = extra_message_data[idx] or {}
                if "rag_trace" in extra:
                    record["rag_trace"] = extra["rag_trace"]
                if "agent_tool_path" in extra:
                    record["agent_tool_path"] = extra["agent_tool_path"]
            serialized.append(record)

        data[user_id][session_id] = {
            "messages": serialized,
            "metadata": metadata or {},
            "updated_at": datetime.now().isoformat()
        }

        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, user_id: str, session_id: str) -> list:
        """加载对话"""
        data = self._load()

        if user_id not in data or session_id not in data[user_id]:
            return []

        messages = []
        for msg_data in data[user_id][session_id]["messages"]:
            if msg_data["type"] == "human":
                messages.append(HumanMessage(content=msg_data["content"]))
            elif msg_data["type"] == "ai":
                messages.append(AIMessage(content=msg_data["content"]))
            elif msg_data["type"] == "system":
                messages.append(SystemMessage(content=msg_data["content"]))

        return messages

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        data = self._load()
        if user_id not in data:
            return []
        return list(data[user_id].keys())

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        data = self._load()
        if user_id not in data or session_id not in data[user_id]:
            return False

        del data[user_id][session_id]
        if not data[user_id]:
            del data[user_id]

        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    def _load(self) -> dict:
        """加载数据"""
        if not os.path.exists(self.storage_file):
            return {}
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}