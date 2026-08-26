"""
模型训练脚本 — 基于 Ultralytics YOLOv8

支持:
- 命令行参数灵活配置
- YOLOv8 预训练权重迁移学习
- 断点续训
- 自动日志和模型保存
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import logger


def parse_args():
    parser = argparse.ArgumentParser(description="🔥 Fire Detection - YOLOv8 Training")

    # 模型参数
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
                                 "yolov8n.yaml", "yolov8s.yaml", "yolov8m.yaml"],
                        help="YOLOv8 模型 (预训练权重或配置文件)")

    # 数据参数
    parser.add_argument("--data", type=str, default=None,
                        help="data.yaml 路径 (默认: ../data/data.yaml 或 ../configs/fire_config.yaml)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸 (默认: 640)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小 (默认: 16)")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备: 0,1,2,... 或 'cpu' (默认: 0)")
    parser.add_argument("--workers", type=int, default=4,
                        help="数据加载线程数 (默认: 4)")

    # 优化参数
    parser.add_argument("--lr", type=float, default=0.001,
                        help="初始学习率 (默认: 0.001)")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["AdamW", "SGD", "Adam"],
                        help="优化器 (默认: AdamW)")
    parser.add_argument("--patience", type=int, default=20,
                        help="早停耐心值 (默认: 20)")

    # 保存参数
    parser.add_argument("--project", type=str, default="../runs/train",
                        help="项目保存目录 (默认: ../runs/train)")
    parser.add_argument("--name", type=str, default="fire_detection",
                        help="实验名称 (默认: fire_detection)")
    parser.add_argument("--exist_ok", action="store_true",
                        help="允许覆盖已有的实验目录")

    # 续训参数
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点续训 (指定 .pt 文件路径或 True 自动查找)")
    parser.add_argument("--resume_best", action="store_true",
                        help="从最佳模型续训 (last.pt)")

    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--cos_lr", action="store_true", default=True,
                        help="使用余弦退火学习率调度")
    parser.add_argument("--close_mosaic", type=int, default=10,
                        help="最后 N 轮关闭 Mosaic 增强 (默认: 10)")
    parser.add_argument("--multi_scale", action="store_true", default=True,
                        help="启用多尺度训练")

    return parser.parse_args()


def get_data_config(args) -> str:
    """确定 data.yaml 路径"""
    if args.data:
        data_path = Path(args.data)
        if data_path.exists():
            return str(data_path.resolve())

    # 尝试默认路径
    candidates = [
        Path("../data/data.yaml"),
        Path("./data/data.yaml"),
        Path("../configs/fire_config.yaml"),
    ]
    for p in candidates:
        if p.resolve().exists():
            return str(p.resolve())

    logger.error("未找到 data.yaml！请先运行 coco_to_yolo.py 转换数据，或通过 --data 指定路径")
    logger.info("示例: python src/coco_to_yolo.py --data_dir ./data")
    sys.exit(1)


def main():
    args = parse_args()

    # 延迟导入 Ultralytics（加快 --help 速度）
    try:
        from ultralytics import YOLO, settings
    except ImportError:
        logger.error("未安装 ultralytics！请运行: pip install ultralytics")
        sys.exit(1)

    # 禁用 Ultralytics 的在线检查
    settings.update({"sync": False, "hub": False})

    data_yaml = get_data_config(args)
    logger.info(f"使用数据配置: {data_yaml}")

    # ---- 确定模型 ----
    model_path = args.model
    if args.resume:
        model_path = args.resume
        logger.info(f"🔄 从检查点续训: {model_path}")
    elif args.resume_best:
        project_dir = Path(args.project) / args.name
        best_pt = project_dir / "weights" / "last.pt"
        if best_pt.exists():
            model_path = str(best_pt)
            logger.info(f"🔄 从最佳模型续训: {model_path}")
        else:
            logger.warning(f"未找到 last.pt，使用初始模型: {args.model}")

    logger.info(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # ---- 训练配置 ----
    train_args = {
        "data": data_yaml,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "lr0": args.lr,
        "patience": args.patience,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "cos_lr": args.cos_lr,
        "close_mosaic": args.close_mosaic,
        "multi_scale": args.multi_scale,
        "pretrained": True,
        "verbose": True,
        "save": True,
        "save_period": 10,
        "val": True,
        "plots": True,
    }

    logger.info("=" * 60)
    logger.info("🔥 开始训练火灾检测模型")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model}")
    logger.info(f"数据: {data_yaml}")
    logger.info(f"Epochs: {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")
    logger.info(f"设备: {args.device} | 优化器: {args.optimizer} | LR: {args.lr}")
    logger.info("=" * 60)

    try:
        results = model.train(**train_args)
        logger.info("=" * 60)
        logger.info("✅ 训练完成！")
        logger.info(f"最佳模型保存至: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
        logger.info("=" * 60)
        return results

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练出错: {e}")
        raise


if __name__ == "__main__":
    main()
