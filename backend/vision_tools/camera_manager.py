"""
摄像头管理模块
提供摄像头的打开、关闭、帧捕获功能，支持上下文管理器。
同时支持从本地图片文件读取（用于测试或非实时场景）。
"""
import os
import base64
import time
import logging
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, FRAME_SAVE_DIR

logger = logging.getLogger(__name__)


class CameraManager:
    """摄像头管理器，支持上下文管理器自动释放资源。"""

    def __init__(
        self,
        camera_index: int = None,
        width: int = None,
        height: int = None,
        fps: int = None,
    ):
        self.camera_index = camera_index or CAMERA_INDEX
        self.width = width or CAMERA_WIDTH
        self.height = height or CAMERA_HEIGHT
        self.fps = fps or CAMERA_FPS
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False

        # 确保帧缓存目录存在
        os.makedirs(FRAME_SAVE_DIR, exist_ok=True)

    def open(self) -> bool:
        """打开摄像头并设置参数。"""
        if self._is_open:
            return True

        # 尝试不同的后端
        backends = [
            (cv2.CAP_ANY, "CAP_ANY"),
            (cv2.CAP_DSHOW, "CAP_DSHOW"),
            (cv2.CAP_MSMF, "CAP_MSMF"),
        ]

        for backend, backend_name in backends:
            logger.info(f"尝试使用 {backend_name} 后端打开摄像头")
            self._cap = cv2.VideoCapture(self.camera_index, backend)
            if self._cap.isOpened():
                logger.info(f"成功使用 {backend_name} 后端打开摄像头")
                break
        
        if not self._cap or not self._cap.isOpened():
            logger.error(f"无法打开摄像头 (index={self.camera_index})")
            return False

        # 使用更保守的参数设置
        try:
            # 先获取摄像头支持的实际分辨率
            actual_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"摄像头实际参数: 分辨率={actual_width}x{actual_height}, fps={actual_fps}")
            
            # 尝试设置参数，但不强制要求成功
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 读取几帧预热（某些摄像头需要预热）
            for i in range(5):
                ret, _ = self._cap.read()
                if ret:
                    break
                import time
                time.sleep(0.1)
        except Exception as e:
            logger.warning(f"设置摄像头参数时出错: {e}")

        self._is_open = True
        logger.info(
            f"摄像头已打开: index={self.camera_index}, "
            f"目标分辨率={self.width}x{self.height}, 目标fps={self.fps}"
        )
        return True

    def close(self):
        """关闭摄像头，释放资源。"""
        if self._cap is not None:
            try:
                # 先尝试读取一帧，确保摄像头处于可操作状态
                ret, _ = self._cap.read()
            except Exception:
                pass
            
            # 释放摄像头资源
            self._cap.release()
            self._cap = None
            
            # 强制垃圾回收，确保资源完全释放
            import gc
            gc.collect()
        
        # 关闭所有 OpenCV 窗口
        try:
            cv2.destroyAllWindows()
            # 等待窗口完全关闭
            import time
            time.sleep(0.1)
        except Exception:
            pass
        
        self._is_open = False
        logger.info("摄像头已关闭")

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        从摄像头捕获一帧。
        Returns:
            BGR 格式的 numpy 数组，失败返回 None。
        """
        if not self._is_open and not self.open():
            return None

        # 添加重试机制，最多尝试3次
        for i in range(3):
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame
            logger.warning(f"帧捕获失败 (尝试 {i+1}/3)")
            # 短暂延迟后重试
            import time
            time.sleep(0.1)
        
        logger.error("多次尝试后仍无法捕获帧")
        return None

    def show_preview(self, window_name: str = "Camera Preview") -> Optional[np.ndarray]:
        """
        显示摄像头实时预览，按空格键捕获图像，按ESC键退出。
        Args:
            window_name: 预览窗口名称
        Returns:
            用户捕获的帧（BGR格式），如果用户未捕获则返回None
        """
        if not self._is_open and not self.open():
            return None

        logger.info("开始摄像头预览，按空格键捕获图像，按ESC键退出")
        captured_frame = None

        while True:
            frame = self.capture_frame()
            if frame is None:
                logger.warning("无法获取帧，退出预览")
                break

            # 显示帧
            cv2.imshow(window_name, frame)

            # 等待按键
            key = cv2.waitKey(1) & 0xFF

            # 按空格键捕获图像
            if key == ord(' '):
                captured_frame = frame.copy()
                logger.info("图像已捕获")
                # 显示捕获成功提示
                cv2.putText(frame, "Captured!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(window_name, frame)
                cv2.waitKey(500)  # 显示500ms
                break

            # 按ESC键退出
            elif key == 27:
                logger.info("用户退出预览")
                break

        # 清理窗口
        cv2.destroyAllWindows()
        return captured_frame

    def capture_and_save(self, filename: str = None) -> Optional[str]:
        """
        捕获一帧并保存为文件。
        Args:
            filename: 文件名（不含路径），默认用时间戳命名。
        Returns:
            保存的文件绝对路径，失败返回 None。
        """
        frame = self.capture_frame()
        if frame is None:
            return None

        if filename is None:
            filename = f"frame_{int(time.time() * 1000)}.jpg"

        filepath = os.path.join(FRAME_SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        logger.debug(f"帧已保存: {filepath}")
        return filepath

    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        从文件加载图片（用于测试或非实时场景）。
        支持 jpg/png/bmp 等常见格式。
        """
        if not os.path.exists(image_path):
            logger.error(f"图片文件不存在: {image_path}")
            return None
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图片: {image_path}")
            return None
        return img

    @staticmethod
    def frame_to_base64(frame: np.ndarray, format: str = ".jpg") -> str:
        """将 BGR 帧转为 base64 编码字符串（用于 API 调用）。"""
        _, buffer = cv2.imencode(format, frame)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def frame_to_pil(frame: np.ndarray) -> Image.Image:
        """将 BGR 帧转为 PIL Image（用于 PaddleOCR 等需要 PIL 输入的库）。"""
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def frame_to_file(frame: np.ndarray, filepath: str) -> str:
        """将帧保存到指定路径，返回文件路径。"""
        cv2.imwrite(filepath, frame)
        return filepath

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
