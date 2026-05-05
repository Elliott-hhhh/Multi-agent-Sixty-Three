"""
OCR 工具（CV 方案）
使用 PaddleOCR 进行文字识别，适合文档、屏幕文字等场景。
设计为 LangChain @tool，可被 Agent 直接调用。
"""
import logging
from typing import Optional

from langchain_core.tools import tool

from .config import OCR_LANG, OCR_USE_GPU
from .camera_manager import CameraManager

logger = logging.getLogger(__name__)

# ==================== PaddleOCR 单例 ====================
_ocr_engine = None


def get_ocr_engine():
    """懒加载 PaddleOCR 引擎（单例模式，避免重复初始化）。"""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,       # 启用文字方向分类
            lang=OCR_LANG,     # 语言
            use_gpu=OCR_USE_GPU,
            show_log=False,
        )
        logger.info(f"PaddleOCR 引擎已初始化 (lang={OCR_LANG}, gpu={OCR_USE_GPU})")
    return _ocr_engine


def _run_ocr_on_frame(frame) -> str:
    """
    对一帧图像执行 OCR，返回格式化的识别结果。
    Args:
        frame: numpy BGR 数组
    Returns:
        格式化的 OCR 文本结果
    """
    engine = get_ocr_engine()

    # PaddleOCR 接受 numpy 数组（BGR）或文件路径
    result = engine.ocr(frame, cls=True)

    if not result or result[0] is None:
        return "未检测到任何文字。"

    lines = []
    for idx, line in enumerate(result[0]):
        bbox, (text, confidence) = line[0], line[1]
        # bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        lines.append(f"[{idx + 1}] (置信度: {confidence:.2%}) {text}")

    output = f"OCR 识别结果（共 {len(lines)} 行）：\n" + "\n".join(lines)
    return output


# ==================== LangChain Tool 定义 ====================

@tool
def capture_and_ocr(image_path: Optional[str] = None, show_preview: bool = False) -> str:
    """
    从摄像头捕获画面并执行 OCR 文字识别。
    适用于文档、屏幕截图、证件、纸质文字等场景。
    如果提供了 image_path，则直接从文件读取而不使用摄像头。

    Args:
        image_path: 可选，本地图片文件路径。提供时直接读取文件，不使用摄像头。
        show_preview: 是否显示摄像头预览窗口，按空格键捕获，按ESC键退出。

    Returns:
        OCR 识别出的文字内容，包含行号和置信度。
    """
    try:
        if image_path:
            # 从文件加载
            frame = CameraManager.load_image(image_path)
            if frame is None:
                return f"错误：无法加载图片文件 {image_path}"
        else:
            # 从摄像头捕获
            with CameraManager() as camera:
                if show_preview:
                    frame = camera.show_preview()
                else:
                    frame = camera.capture_frame()

            if frame is None:
                return "错误：无法从摄像头捕获画面，请检查摄像头是否连接。"

        result = _run_ocr_on_frame(frame)
        return result

    except Exception as e:
        logger.error(f"OCR 处理失败: {e}", exc_info=True)
        return f"OCR 处理出错: {str(e)}"


@tool
def capture_and_ocr_detail(image_path: Optional[str] = None, show_preview: bool = False) -> str:
    """
    从摄像头捕获画面并执行 OCR 文字识别（详细版）。
    返回每行文字的坐标位置、置信度等详细信息。
    适用于需要精确定位文字位置的场景。

    Args:
        image_path: 可选，本地图片文件路径。提供时直接读取文件，不使用摄像头。
        show_preview: 是否显示摄像头预览窗口，按空格键捕获，按ESC键退出。

    Returns:
        包含坐标和置信度的详细 OCR 结果。
    """
    try:
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

        engine = get_ocr_engine()
        result = engine.ocr(frame, cls=True)

        if not result or result[0] is None:
            return "未检测到任何文字。"

        lines = []
        for idx, line in enumerate(result[0]):
            bbox, (text, confidence) = line[0], line[1]
            # 取四个角点的坐标
            coords = " → ".join([f"({int(p[0])},{int(p[1])})" for p in bbox])
            lines.append(
                f"[{idx + 1}] 文字: {text}\n"
                f"     置信度: {confidence:.2%}\n"
                f"     位置: {coords}"
            )

        output = f"OCR 详细结果（共 {len(lines)} 行）：\n\n" + "\n\n".join(lines)
        return output

    except Exception as e:
        logger.error(f"OCR 详细处理失败: {e}", exc_info=True)
        return f"OCR 处理出错: {str(e)}"
