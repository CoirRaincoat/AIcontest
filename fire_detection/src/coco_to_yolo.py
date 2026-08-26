"""
COCO 格式标注 → YOLO 格式转换工具

COCO bbox: [x, y, width, height] (绝对像素值)
YOLO bbox: [class_id, cx, cy, w, h] (归一化到 [0, 1])
YOLO 目录结构:
    data/
    ├── train/
    │   ├── images/   (*.jpg)
    │   └── labels/   (*.txt)
    ├── val/
    │   ├── images/   (*.jpg)
    │   └── labels/   (*.txt)
    └── data.yaml
"""
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
import random

from utils import logger


def load_coco_annotations(json_path: str) -> Dict:
    """加载 COCO 格式标注文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    logger.info(f"加载 COCO 标注: {json_path}")
    logger.info(f"  - 图像数: {len(coco.get('images', []))}")
    logger.info(f"  - 标注数: {len(coco.get('annotations', []))}")
    logger.info(f"  - 类别数: {len(coco.get('categories', []))}")
    return coco


def build_coco_maps(coco: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    构建 COCO 映射表。

    Returns:
        (image_id_to_info, image_id_to_anns, cat_id_to_name)
    """
    # 图像 ID → 图像信息
    image_id_to_info = {
        img["id"]: img for img in coco["images"]
    }

    # 图像 ID → 标注列表
    image_id_to_anns: Dict[int, List] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        image_id_to_anns[img_id].append(ann)

    # 类别 ID → 类别名
    cat_id_to_name = {
        cat["id"]: cat["name"] for cat in coco.get("categories", [])
    }

    return image_id_to_info, image_id_to_anns, cat_id_to_name


def coco_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    转换单个 bbox: COCO [x, y, w, h] → YOLO [cx, cy, w, h] (归一化)

    Args:
        bbox: COCO 格式 [x, y, width, height] (像素值)
        img_w: 图像宽度
        img_h: 图像高度

    Returns:
        YOLO 格式 (cx, cy, w, h) 归一化到 [0, 1]
    """
    x, y, bw, bh = bbox
    # 中心点
    cx = (x + bw / 2) / img_w
    cy = (y + bh / 2) / img_h
    # 宽高归一化
    nw = bw / img_w
    nh = bh / img_h
    return cx, cy, nw, nh


def convert_annotations(coco: Dict,
                        image_dir: str,
                        output_dir: str,
                        val_ratio: float = 0.2,
                        seed: int = 42) -> Dict[int, str]:
    """
    转换 COCO 标注到 YOLO 格式，并划分训练/验证集。

    Args:
        coco: COCO 标注字典
        image_dir: 原始图像目录
        output_dir: 输出根目录
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        类别 ID → 类别名 映射
    """
    random.seed(seed)

    image_id_to_info, image_id_to_anns, cat_id_to_name = build_coco_maps(coco)
    all_image_ids = list(image_id_to_info.keys())

    if not all_image_ids:
        logger.error("COCO 标注中没有图像数据！")
        return cat_id_to_name

    # 随机划分训练/验证集
    random.shuffle(all_image_ids)
    n_val = max(1, int(len(all_image_ids) * val_ratio))
    val_ids: Set[int] = set(all_image_ids[:n_val])
    train_ids: Set[int] = set(all_image_ids[n_val:])

    logger.info(f"训练集图像: {len(train_ids)}, 验证集图像: {len(val_ids)}")

    output_dir = Path(output_dir)
    image_dir = Path(image_dir)

    # 创建 YOLO 目录结构
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_out = output_dir / split / "images"
        lbl_out = output_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        skipped = 0
        for img_id in ids:
            img_info = image_id_to_info[img_id]
            img_filename = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            # 复制/链接图像
            src_img = image_dir / img_filename
            dst_img = img_out / img_filename

            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                skipped += 1
                if skipped <= 5:
                    logger.warning(f"图像不存在: {src_img}")
                continue

            # 生成 YOLO 标注文件
            anns = image_id_to_anns.get(img_id, [])
            label_lines = []
            for ann in anns:
                cat_id = ann["category_id"]
                bbox = ann["bbox"]  # COCO: [x, y, w, h]

                # COCO cat_id → YOLO 0-based class index
                # 构建排序后的映射: {cat_id: yolo_index}
                sorted_cat_ids = sorted(cat_id_to_name.keys())
                cat_id_to_yolo = {cid: i for i, cid in enumerate(sorted_cat_ids)}
                yolo_cls = cat_id_to_yolo[cat_id]

                cx, cy, nw, nh = coco_bbox_to_yolo(bbox, img_w, img_h)

                # 边界检查
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                label_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # 写入标注文件
            stem = Path(img_filename).stem
            label_path = lbl_out / f"{stem}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

        if skipped > 0:
            logger.warning(f"{split}: {skipped} 张图像未找到，已跳过")

    logger.info("COCO → YOLO 转换完成！")
    return cat_id_to_name


def generate_yaml(output_dir: str,
                  cat_id_to_name: Dict[int, str],
                  nc: int = None) -> None:
    """
    生成 YOLOv8 训练用的 data.yaml。

    Args:
        output_dir: 输出目录
        cat_id_to_name: 类别 ID → 类别名
        nc: 类别数（可选，默认为 cat_id_to_name 的长度）
    """
    output_dir = Path(output_dir).resolve()

    if nc is None:
        nc = len(cat_id_to_name)

    # 构建类别映射
    # 将任意 cat_id 映射到 0-based 连续索引
    sorted_cats = sorted(cat_id_to_name.items())
    names = {i: name for i, (cat_id, name) in enumerate(sorted_cats)}

    yaml_content = f"""# YOLOv8 Dataset Configuration - Auto-generated
# Fire Detection Dataset

path: {output_dir.as_posix()}
train: train/images
val: val/images

# Number of classes
nc: {nc}

# Class names
names:
"""
    for idx, name in names.items():
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    logger.info(f"data.yaml 已生成: {yaml_path}")
    logger.info(f"类别 ({nc}): {names}")


def main():
    parser = argparse.ArgumentParser(description="COCO 标注 → YOLO 格式转换")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="数据集目录 (包含 images/ 和 train_coco.json)")
    parser.add_argument("--coco_json", type=str, default=None,
                        help="COCO 标注 JSON 路径 (默认: data_dir/train_coco.json)")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="图像目录 (默认: data_dir/images)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: data_dir)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="验证集比例 (默认: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    coco_json = Path(args.coco_json) if args.coco_json else data_dir / "train_coco.json"
    image_dir = Path(args.image_dir) if args.image_dir else data_dir / "images"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    # 验证路径
    if not coco_json.exists():
        logger.error(f"COCO 标注文件不存在: {coco_json}")
        logger.info("请将 train_coco.json 放入 data/ 目录，或通过 --coco_json 指定路径")
        return

    if not image_dir.exists():
        logger.error(f"图像目录不存在: {image_dir}")
        logger.info("请将训练图像放入 data/images/ 目录，或通过 --image_dir 指定路径")
        return

    # 执行转换
    coco = load_coco_annotations(str(coco_json))
    cat_id_to_name = convert_annotations(coco, str(image_dir), str(output_dir), args.val_ratio, args.seed)
    generate_yaml(str(output_dir), cat_id_to_name)


if __name__ == "__main__":
    main()
