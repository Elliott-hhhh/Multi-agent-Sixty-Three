"""
配置文件
集中管理摄像头参数、VLM 模型参数等
"""
import os

# ==================== 摄像头配置 ====================
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))          # 摄像头设备编号
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))       # 分辨率宽度
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))      # 分辨率高度
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "30"))             # 帧率
FRAME_SAVE_DIR = os.getenv("FRAME_SAVE_DIR", "/tmp/vision_frames")  # 帧缓存目录

# ==================== VLM 配置（通用）====================
API_KEY = os.getenv("OPENAI_API_KEY", "")  # 使用与其他模型相同的 API 密钥
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")        # 使用与其他模型相同的基础 URL
VLM_MODEL = os.getenv("VISION_MODEL", "qwen3.6-plus")               # 模型名称
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "1024"))   # 最大输出 token
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.1")) # 低温度，更稳定

# ==================== OCR 配置 ====================
OCR_LANG = os.getenv("OCR_LANG", "ch")  # PaddleOCR 语言: ch(中文), en(英文)
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() == "true"

# ==================== 通用配置 ====================
DEFAULT_PROMPT = "请详细描述这张图片中的内容，包括文字、物体、布局等关键信息。"
