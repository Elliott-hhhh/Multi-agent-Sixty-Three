"""
统一导出所有视觉工具
在你的 Agent 框架中，直接 import 本文件即可获取所有工具。
"""
from .ocr_tool import capture_and_ocr, capture_and_ocr_detail
from .vlm_tool import capture_and_analyze, capture_and_answer

# 所有工具列表 —— 可直接传给 Agent 的 tools 参数
ALL_VISION_TOOLS = [
    capture_and_ocr,
    capture_and_ocr_detail,
    capture_and_analyze,
    capture_and_answer,
]

# 快速工具（CV 方案，低延迟）
CV_TOOLS = [
    capture_and_ocr,
    capture_and_ocr_detail,
]

# 深度工具（VLM 方案，高理解力）
VLM_TOOLS = [
    capture_and_analyze,
    capture_and_answer,
]

__all__ = [
    "capture_and_ocr",
    "capture_and_ocr_detail",
    "capture_and_analyze",
    "capture_and_answer",
    "ALL_VISION_TOOLS",
    "CV_TOOLS",
    "VLM_TOOLS",
]
