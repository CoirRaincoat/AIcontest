"""
数据处理模块 — 数据加载、预处理与分析
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils import logger


# ============================================================
# 图像级标注数据处理
# ============================================================

def load_image_annotations(json_path: str) -> Dict:
    """
    加载图像级标注文件 (train_image.json)。

    预期格式:
    {
        "image_name_1.jpg": {"label": 0, ...},
        "image_name_2.jpg": {"label": 1, ...},
        ...
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"加载图像级标注: {json_path} ({len(data)} 条记录)")
    return data


def analyze_image_labels(data: Dict) -> Dict:
    """
    分析图像级标注的分布情况。

    Returns:
        {"total": N, "labels": {label: count, ...}, "label_names": {...}}
    """
    label_counts = {}
    for img_name, info in data.items():
        label = info.get("label", info.get("category", -1))
        label_counts[label] = label_counts.get(label, 0) + 1

    result = {
        "total": len(data),
        "labels": label_counts,
        "label_distribution": {
            k: round(v / len(data) * 100, 2) for k, v in label_counts.items()
        }
    }
    logger.info(f"图像级标注分析: {result}")
    return result


# ============================================================
# 数据集统计与分析
# ============================================================

def compute_dataset_stats(image_dir: str, annotation_path: str = None) -> Dict:
    """
    计算数据集统计信息。

    Args:
        image_dir: 图像目录
        annotation_path: COCO 标注文件路径 (可选)

    Returns:
        统计数据字典
    """
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*"))
    # 过滤图像文件
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = [f for f in image_files if f.suffix.lower() in valid_exts]

    stats = {
        "total_images": len(image_files),
        "image_sizes": [],
        "aspect_ratios": [],
    }

    for img_path in image_files:
        try:
            img = Image.open(img_path)
            w, h = img.size
            stats["image_sizes"].append((w, h))
            stats["aspect_ratios"].append(w / h)
        except Exception as e:
            logger.warning(f"无法读取图像 {img_path.name}: {e}")

    if stats["image_sizes"]:
        sizes = np.array(stats["image_sizes"])
        stats["avg_width"] = float(np.mean(sizes[:, 0]))
        stats["avg_height"] = float(np.mean(sizes[:, 1]))
        stats["min_width"] = int(np.min(sizes[:, 0]))
        stats["min_height"] = int(np.min(sizes[:, 1]))
        stats["max_width"] = int(np.max(sizes[:, 0]))
        stats["max_height"] = int(np.max(sizes[:, 1]))

    logger.info(f"数据集统计: {stats['total_images']} 张图像, "
                f"平均尺寸: {stats.get('avg_width', 0):.0f}x{stats.get('avg_height', 0):.0f}")

    return stats


# ============================================================
# 数据增强工具
# ============================================================

class FireAugmentation:
    """
    火灾检测专用数据增强。

    针对监控场景特点设计:
    - 光照模拟：随机亮度/对比度调整
    - 烟雾模拟：随机噪声/模糊
    - 部分遮挡：随机擦除
    """

    def __init__(self,
                 brightness_range: Tuple[float, float] = (0.6, 1.6),
                 contrast_range: Tuple[float, float] = (0.6, 1.6),
                 blur_kernel_range: Tuple[int, int] = (0, 5),
                 noise_prob: float = 0.3,
                 erase_prob: float = 0.2):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_kernel_range = blur_kernel_range
        self.noise_prob = noise_prob
        self.erase_prob = erase_prob

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """对图像应用随机增强"""
        img = image.copy()

        # 1. 随机亮度调整
        if random.random() < 0.5:
            alpha = random.uniform(*self.brightness_range)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

        # 2. 随机对比度调整
        if random.random() < 0.5:
            alpha = random.uniform(*self.contrast_range)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

        # 3. HSV 扰动
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] += random.randint(-10, 10)       # Hue
            hsv[:, :, 1] *= random.uniform(0.7, 1.3)       # Saturation
            hsv[:, :, 2] *= random.uniform(0.7, 1.3)       # Value
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 4. 随机高斯模糊（模拟烟雾模糊效果）
        if random.random() < 0.3:
            ksize = random.randint(*self.blur_kernel_range)
            if ksize > 0 and ksize % 2 == 1:
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # 5. 随机噪声（模拟低光照传感器噪声）
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, random.uniform(5, 15), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        # 6. 随机擦除（模拟部分遮挡）
        if random.random() < self.erase_prob:
            h, w = img.shape[:2]
            erase_w = random.randint(w // 8, w // 4)
            erase_h = random.randint(h // 8, h // 4)
            ex = random.randint(0, w - erase_w)
            ey = random.randint(0, h - erase_h)
            img[ey:ey + erase_h, ex:ex + erase_w] = random.randint(0, 128)

        return img


# ============================================================
# 目标检测数据集 (用于 Ultralytics 之外的自定义训练)
# ============================================================

class FireDetectionDataset(Dataset):
    """
    火灾检测 PyTorch Dataset。

    用于自定义训练流程（非 Ultralytics 内置训练时使用）。
    支持加载 YOLO 格式标注和图像级标注。
    """

    def __init__(self,
                 image_dir: str,
                 label_dir: str = None,
                 image_ann_path: str = None,
                 img_size: int = 640,
                 augment: bool = False,
                 transform: Optional[Callable] = None):
        """
        Args:
            image_dir: 图像目录
            label_dir: YOLO 标注目录 (可选，用于目标检测)
            image_ann_path: 图像级标注 JSON (可选，用于图像分类)
            img_size: 目标图像尺寸
            augment: 是否启用数据增强
            transform: 自定义 transform
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.img_size = img_size
        self.augment = augment
        self.custom_transform = transform

        self.fire_aug = FireAugmentation() if augment else None

        # 加载图像文件列表
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        self.images = sorted([
            f for f in self.image_dir.glob("*")
            if f.suffix.lower() in valid_exts
        ])

        # 加载图像级标注
        self.image_labels = None
        if image_ann_path:
            self.image_labels = load_image_annotations(image_ann_path)

        logger.info(f"FireDetectionDataset: {len(self.images)} 张图像")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.images[idx]

        # 加载图像
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"无法读取图像: {img_path}")
            # 返回空占位符
            return {"image": np.zeros((self.img_size, self.img_size, 3), dtype=np.float32),
                    "path": str(img_path), "shape": (self.img_size, self.img_size)}

        original_shape = image.shape[:2]

        # 数据增强
        if self.augment and self.fire_aug:
            image = self.fire_aug(image)

        # Resize + letterbox
        image, scale, pad = self._letterbox(image, self.img_size)

        # 归一化
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC → CHW

        result = {
            "image": torch.from_numpy(image),
            "path": str(img_path),
            "shape": original_shape,
            "scale": scale,
            "pad": pad,
        }

        # 加载标注
        if self.label_dir:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            labels = self._load_labels(label_path)
            result["labels"] = labels

        # 图像级标签
        if self.image_labels and img_path.name in self.image_labels:
            result["image_label"] = self.image_labels[img_path.name]

        return result

    def _letterbox(self, image: np.ndarray, target_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize"""
        h, w = image.shape[:2]
        r = min(target_size / h, target_size / w)
        new_h, new_w = int(h * r), int(w * r)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dw = target_size - new_w
        dh = target_size - new_h
        pad_left = dw // 2
        pad_top = dh // 2

        padded = cv2.copyMakeBorder(
            resized, pad_top, dh - pad_top, pad_left, dw - pad_left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return padded, r, (pad_left, pad_top)

    @staticmethod
    def _load_labels(label_path: Path) -> torch.Tensor:
        """加载 YOLO 格式标注 [class_id, cx, cy, w, h]"""
        if not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append([float(x) for x in parts[:5]])

        return torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)


def create_dataloader(dataset: FireDetectionDataset,
                      batch_size: int = 16,
                      shuffle: bool = True,
                      num_workers: int = 4) -> DataLoader:
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: batch,  # 保持字典格式
    )
