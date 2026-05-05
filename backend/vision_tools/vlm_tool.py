"""
VLM 工具（多模态大模型方案）
使用 Qwen 系列模型进行复杂视觉理解，适合需要语义推理的场景。
设计为 LangChain @tool，可被 Agent 直接调用。
"""
import base64
import logging
from typing import Optional

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from .config import DEFAULT_PROMPT, API_KEY, BASE_URL, VLM_MODEL, VLM_MAX_TOKENS, VLM_TEMPERATURE
from .camera_manager import CameraManager

logger = logging.getLogger(__name__)

# ==================== VLM 客户端单例 ====================
_vlm_client = None


def get_vlm_client():
    """懒加载 Qwen 视觉客户端（单例模式）。"""
    global _vlm_client
    if _vlm_client is None:
        _vlm_client = init_chat_model(
            model=VLM_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            max_tokens=VLM_MAX_TOKENS,
            temperature=VLM_TEMPERATURE,
            timeout=60,
            max_retries=3,
            extra_body={"enable_thinking": False},
        )
        logger.info(f"VLM 客户端已初始化 (model={VLM_MODEL})")
    return _vlm_client


def _analyze_frame_with_vlm(frame, prompt: str) -> str:
    """
    使用 Qwen 视觉模型分析一帧图像。
    Args:
        frame: numpy BGR 数组
        prompt: 给模型的提示词
    Returns:
        模型的分析结果文本
    """
    import cv2
    client = get_vlm_client()

    _, buffer = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": prompt}
        ]
    )

    response = client.invoke([message])
    return response.content


# ==================== LangChain Tool 定义 ====================

@tool
def capture_and_analyze(
    prompt: str = "",
    image_path: Optional[str] = None,
    show_preview: bool = False,
) -> str:
    """
    从摄像头捕获画面，使用多模态大模型（GPT-4o）进行深度视觉分析。
    适用于复杂场景理解、物体识别、场景描述、UI 分析等需要语义推理的任务。
    如果只需要简单的文字识别，建议使用 capture_and_ocr 工具（更快更便宜）。

    Args:
        prompt: 对画面的分析要求，例如：
            - "描述画面中的主要内容和场景"
            - "画面中有哪些人？他们在做什么？"
            - "分析这个 UI 界面的布局，列出所有可交互元素"
            - "这张图片中是否存在安全隐患？"
            如果为空，则使用默认提示词进行通用描述。
        image_path: 可选，本地图片文件路径。提供时直接读取文件，不使用摄像头。
        show_preview: 是否显示摄像头预览窗口，按空格键捕获，按ESC键退出。

    Returns:
        GPT-4o 的视觉分析结果文本。
    """
    try:
        if not prompt:
            prompt = DEFAULT_PROMPT

        if image_path:
            frame = CameraManager.load_image(image_path)
            if frame is None:
                return f"错误：无法加载图片文件 {image_path}"
        else:
            with CameraManager() as camera:
                if show_preview:
                    frame = camera.show_preview()
                else:
                    frame = camera.capture_frame()
            if frame is None:
                return "错误：无法从摄像头捕获画面，请检查摄像头是否连接。"

        result = _analyze_frame_with_vlm(frame, prompt)
        return result

    except Exception as e:
        logger.error(f"VLM 分析失败: {e}", exc_info=True)
        return f"视觉分析出错: {str(e)}"


@tool
def capture_and_answer(
    question: str,
    image_path: Optional[str] = None,
    show_preview: bool = False,
) -> str:
    """
    从摄像头捕获画面，然后针对画面回答一个具体问题。
    适用于"看到什么就回答什么"的问答场景。

    Args:
        question: 关于画面的具体问题，例如：
            - "屏幕上显示的错误代码是什么？"
            - "这个仪表盘的读数是多少？"
            - "图片中的人穿着什么颜色的衣服？"
        image_path: 可选，本地图片文件路径。提供时直接读取文件，不使用摄像头。
        show_preview: 是否显示摄像头预览窗口，按空格键捕获，按ESC键退出。

    Returns:
        针对问题的回答。
    """
    try:
        if not question:
            return "错误：请提供一个具体的问题。"

        if image_path:
            frame = CameraManager.load_image(image_path)
            if frame is None:
                return f"错误：无法加载图片文件 {image_path}"
        else:
            with CameraManager() as camera:
                if show_preview:
                    frame = camera.show_preview()
                else:
                    frame = camera.capture_frame()
            if frame is None:
                return "错误：无法从摄像头捕获画面。"

        prompt = (
            f"请仔细观察这张图片，回答以下问题：\n\n"
            f"问题：{question}\n\n"
            f"请直接给出简洁准确的回答，不需要额外解释。"
        )
        result = _analyze_frame_with_vlm(frame, prompt)
        return result

    except Exception as e:
        logger.error(f"VLM 问答失败: {e}", exc_info=True)
        return f"视觉问答出错: {str(e)}"
