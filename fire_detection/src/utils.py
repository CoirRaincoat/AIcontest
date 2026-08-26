"""
工具函数模块 — 可视化、日志、结果导出等通用功能
"""
import cv2
import json
import csv
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np

# ============================================================
# 日志配置
# ============================================================

def setup_logger(name: str = "fire_detection",
                 log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """配置并返回 logger 实例"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(console_fmt)
        logger.addHandler(file_handler)

    return logger


# 默认全局 logger
logger = setup_logger()


# ============================================================
# 检测结果可视化
# ============================================================

# 类别颜色映射（火焰用红色系，烟雾用灰色系）
CLASS_COLORS = {
    0: (0, 0, 255),      # fire - 红色
    1: (128, 128, 128),  # smoke - 灰色
    # 后备颜色
    "default": (0, 255, 0),
}

# 类别中文名
CLASS_NAMES = {
    0: "fire",
    1: "smoke",
}


def get_color(class_id: int) -> Tuple[int, int, int]:
    """获取类别对应的 BGR 颜色"""
    return CLASS_COLORS.get(class_id, CLASS_COLORS["default"])


def draw_detections(image: np.ndarray,
                    boxes: List[List[float]],
                    scores: List[float],
                    class_ids: List[int],
                    conf_threshold: float = 0.25,
                    line_thickness: int = 2,
                    font_scale: float = 0.6) -> np.ndarray:
    """
    在图像上绘制检测框和标签。

    Args:
        image: BGR 图像 (H, W, 3)
        boxes: 检测框列表 [x1, y1, x2, y2] (像素坐标)
        scores: 置信度列表
        class_ids: 类别 ID 列表
        conf_threshold: 置信度阈值，低于此值不绘制
        line_thickness: 边框线宽
        font_scale: 标签字体大小

    Returns:
        绘制后的图像
    """
    result = image.copy()
    h, w = result.shape[:2]

    for box, score, cls_id in zip(boxes, scores, class_ids):
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        # 边界裁剪
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = get_color(cls_id)
        class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        label = f"{class_name} {score:.2f}"

        # 绘制检测框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, line_thickness)

        # 绘制标签背景和文字
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(result, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

    return result


def draw_heatmap(image: np.ndarray,
                 detection_map: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    """
    在图像上叠加检测热力图。

    Args:
        image: 原始 BGR 图像
        detection_map: 检测热力图 (H, W) 或 (H, W, 1)，值域 [0, 1]
        alpha: 热力图透明度

    Returns:
        叠加后的图像
    """
    if detection_map.ndim == 2:
        detection_map = detection_map[:, :, np.newaxis]

    heatmap = cv2.applyColorMap(
        (detection_map * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    result = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return result


# ============================================================
# 结果导出
# ============================================================

def export_to_json(results: List[Dict],
                   output_path: str,
                   image_size: Optional[Tuple[int, int]] = None) -> None:
    """
    将检测结果导出为 JSON 文件。

    Args:
        results: 检测结果列表，每个元素格式:
            {
                "image": str,           # 图像文件名
                "detections": [
                    {
                        "bbox": [x1, y1, x2, y2],  # 像素坐标
                        "score": float,
                        "class_id": int,
                        "class_name": str
                    },
                    ...
                ]
            }
        output_path: 输出 JSON 文件路径
        image_size: 图像尺寸 (W, H)，可选
    """
    output = {
        "generated_at": datetime.now().isoformat(),
        "total_images": len(results),
        "image_size": {"width": image_size[0], "height": image_size[1]} if image_size else None,
        "results": results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"检测结果已导出至: {output_path}")


def export_to_csv(results: List[Dict], output_path: str) -> None:
    """
    将检测结果导出为 CSV 文件（扁平化）。

    Args:
        results: 同 export_to_json 的 results 格式
        output_path: 输出 CSV 文件路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        image_name = r["image"]
        for det in r["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            rows.append({
                "image": image_name,
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": round(det["score"], 4),
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
            })

    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"检测结果已导出至: {output_path} ({len(rows)} 个检测)")
    else:
        logger.warning("没有检测结果可导出")


# ============================================================
# 图像处理辅助
# ============================================================

def letterbox_resize(image: np.ndarray,
                     target_size: int = 640,
                     stride: int = 32) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Letterbox resize — 保持宽高比缩放并填充至目标尺寸。

    Args:
        image: 输入图像 (H, W, 3)
        target_size: 目标边长
        stride: 步长（确保尺寸可被 stride 整除）

    Returns:
        (resized_image, scale_ratio, padding)
        - resized_image: 缩放填充后的图像
        - scale_ratio: 缩放比例
        - padding: (pad_left, pad_top)
    """
    h, w = image.shape[:2]
    # 计算缩放比例
    r = min(target_size / h, target_size / w)
    new_h, new_w = int(h * r), int(w * r)

    # 缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 填充至目标尺寸
    dw = target_size - new_w
    dh = target_size - new_h
    pad_left = dw // 2
    pad_top = dh // 2

    padded = cv2.copyMakeBorder(
        resized, pad_top, dh - pad_top, pad_left, dw - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    return padded, r, (pad_left, pad_top)


def scale_boxes(boxes: np.ndarray,
                original_shape: Tuple[int, int],
                resized_shape: Tuple[int, int]) -> np.ndarray:
    """
    将 letterbox 坐标系下的检测框还原到原始图像坐标。

    Args:
        boxes: 检测框 [N, 4] (x1, y1, x2, y2)
        original_shape: 原始图像 (H, W)
        resized_shape: letterbox 后图像尺寸 (H, W)

    Returns:
        原始坐标下的检测框
    """
    oh, ow = original_shape
    rh, rw = resized_shape
    gain = min(rh / oh, rw / ow)
    pad_w = (rw - ow * gain) / 2
    pad_h = (rh - oh * gain) / 2

    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h
    boxes[:, :4] /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)
    return boxes


# ============================================================
# 时间工具
# ============================================================

class FPSCounter:
    """帧率计数器"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.times = []
        self._prev_time = datetime.now()

    def tick(self) -> float:
        """记录一帧并返回当前 FPS"""
        now = datetime.now()
        elapsed = (now - self._prev_time).total_seconds()
        self._prev_time = now

        self.times.append(elapsed)
        if len(self.times) > self.window_size:
            self.times.pop(0)

        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def reset(self):
        self.times.clear()
        self._prev_time = datetime.now()
