"""
模型评估脚本 — 计算 mAP、Precision、Recall，绘制 PR 曲线和混淆矩阵
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import logger, CLASS_NAMES


# ============================================================
# 设置中文字体
# ============================================================

def setup_chinese_font():
    """尝试设置 matplotlib 中文字体"""
    try:
        # 尝试常见中文字体
        for font in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
            try:
                matplotlib.font_manager.findfont(font, fallback_to_default=False)
                plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue
    except Exception:
        pass
    # 回退：忽略 Unicode 警告
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


setup_chinese_font()


# ============================================================
# 评估函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="🔥 Fire Detection - Model Evaluation")

    parser.add_argument("--weights", type=str, required=True,
                        help="训练好的模型权重路径 (.pt)")
    parser.add_argument("--data", type=str, default=None,
                        help="data.yaml 路径 (默认自动查找)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小 (默认: 16)")
    parser.add_argument("--device", type=str, default="0",
                        help="设备 (默认: 0)")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="置信度阈值 (默认: 0.001)")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="IoU 阈值 (默认: 0.6)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="评估数据划分 (默认: val)")
    parser.add_argument("--save_json", action="store_true",
                        help="保存评估结果为 JSON")
    parser.add_argument("--save_plots", action="store_true", default=True,
                        help="保存评估图表 (默认: True)")
    parser.add_argument("--output_dir", type=str, default="../outputs",
                        help="输出目录 (默认: ../outputs)")

    return parser.parse_args()


def evaluate_model(args) -> Dict:
    """执行模型评估"""
    from ultralytics import YOLO

    # 验证模型路径
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"模型文件不存在: {args.weights}")
        sys.exit(1)

    # 验证数据配置
    if args.data:
        data_path = Path(args.data)
    else:
        # 自动查找
        candidates = [
            Path("../data/data.yaml"),
            Path("./data/data.yaml"),
            Path("../configs/fire_config.yaml"),
        ]
        data_path = None
        for p in candidates:
            if p.resolve().exists():
                data_path = p.resolve()
                break
        if data_path is None:
            logger.error("未找到 data.yaml！请通过 --data 指定路径")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("📊 模型评估")
    logger.info(f"权重: {args.weights}")
    logger.info(f"数据: {data_path}")
    logger.info(f"设备: {args.device}")
    logger.info("=" * 60)

    # 加载模型
    model = YOLO(str(weights_path))

    # 运行验证
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        plots=args.save_plots,
        save_json=args.save_json,
        project=str(output_dir),
        name="evaluation",
    )

    # 提取关键指标
    results = {
        "model": str(weights_path),
        "data": str(data_path),
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "f1_score": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)),
        },
    }

    # 打印结果
    logger.info("-" * 40)
    logger.info("📈 评估结果:")
    logger.info(f"  mAP@50:      {results['metrics']['mAP50']:.4f}")
    logger.info(f"  mAP@50-95:   {results['metrics']['mAP50_95']:.4f}")
    logger.info(f"  Precision:   {results['metrics']['precision']:.4f}")
    logger.info(f"  Recall:      {results['metrics']['recall']:.4f}")
    logger.info(f"  F1-Score:    {results['metrics']['f1_score']:.4f}")
    logger.info("-" * 40)

    # 保存评估结果
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"评估结果已保存至: {results_path}")

    return results


# ============================================================
# PR 曲线绘制
# ============================================================

def plot_pr_curve(precision_curve: List[float],
                  recall_curve: List[float],
                  ap: float,
                  class_name: str = "fire",
                  output_path: Optional[str] = None):
    """绘制单个类别的 PR 曲线"""
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, "b-", linewidth=2,
             label=f"{class_name} (AP={ap:.3f})")
    plt.fill_between(recall_curve, 0, precision_curve, alpha=0.1, color="blue")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {class_name}")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"PR 曲线已保存: {output_path}")
    else:
        plt.show()


def plot_confusion_from_eval(confusion_matrix: np.ndarray,
                              class_names: List[str],
                              output_path: Optional[str] = None,
                              normalize: bool = True):
    """绘制混淆矩阵"""
    if normalize and confusion_matrix.sum() > 0:
        cm = confusion_matrix.astype(float)
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
    else:
        cm = confusion_matrix

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(n_classes + 4, n_classes + 3))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel="Predicted",
           ylabel="True",
           title="Confusion Matrix")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格中标注数值
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"混淆矩阵已保存: {output_path}")
    else:
        plt.show()


# ============================================================
# 批量图像测试
# ============================================================

def run_batch_evaluation(model,
                         image_dir: str,
                         conf_threshold: float = 0.25,
                         iou_threshold: float = 0.45) -> Dict:
    """
    对一组图像运行推理并统计结果。

    Returns:
        {image_name: [{bbox, score, class}, ...], ...}
    """
    image_dir = Path(image_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([f for f in image_dir.iterdir() if f.suffix.lower() in valid_exts])

    results = {}
    for img_path in images:
        preds = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        detections = []
        for pred in preds:
            boxes = pred.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    detections.append({
                        "bbox": boxes.xyxy[i].cpu().tolist(),
                        "score": float(boxes.conf[i]),
                        "class": int(boxes.cls[i]),
                        "class_name": CLASS_NAMES.get(int(boxes.cls[i]), f"class_{int(boxes.cls[i])}"),
                    })
        results[img_path.name] = detections

    # 统计
    total_detections = sum(len(v) for v in results.values())
    images_with_fire = sum(1 for v in results.values() if len(v) > 0)

    logger.info(f"批量评估: {len(results)} 张图像, "
                f"{images_with_fire} 张检测到目标, "
                f"共 {total_detections} 个检测框")

    return results


def main():
    args = parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
