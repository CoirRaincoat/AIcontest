"""
推理模块 — 单张/批量图像火灾检测推理
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    logger, draw_detections, CLASS_NAMES,
    export_to_json, export_to_csv, FPSCounter
)


def parse_args():
    parser = argparse.ArgumentParser(description="🔥 Fire Detection - Inference")

    parser.add_argument("--weights", type=str, required=True,
                        help="模型权重路径 (.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="输入源: 图像文件 / 图像目录 / 视频文件")
    parser.add_argument("--output", type=str, default="../outputs",
                        help="输出目录 (默认: ../outputs)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU 阈值 NMS (默认: 0.45)")
    parser.add_argument("--device", type=str, default="0",
                        help="设备: 0 或 'cpu' (默认: 0)")
    parser.add_argument("--save_json", action="store_true",
                        help="保存结果为 JSON")
    parser.add_argument("--save_csv", action="store_true",
                        help="保存结果为 CSV")
    parser.add_argument("--no_save_images", action="store_true",
                        help="不保存标注后的图像")
    parser.add_argument("--show", action="store_true",
                        help="显示检测结果窗口")

    return parser.parse_args()


def process_image(model, image_path: str, args) -> Optional[dict]:
    """处理单张图像"""
    image = cv2.imread(image_path)
    if image is None:
        logger.warning(f"无法读取图像: {image_path}")
        return None

    # 推理
    results = model.predict(
        source=image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                detections.append({
                    "bbox": boxes.xyxy[i].cpu().tolist(),
                    "score": float(boxes.conf[i]),
                    "class_id": int(boxes.cls[i]),
                    "class_name": CLASS_NAMES.get(int(boxes.cls[i]), f"class_{int(boxes.cls[i])}"),
                })

    # 可视化
    if not args.no_save_images and detections:
        boxes_list = [d["bbox"] for d in detections]
        scores_list = [d["score"] for d in detections]
        cls_list = [d["class_id"] for d in detections]

        annotated = draw_detections(image, boxes_list, scores_list, cls_list, args.conf)
        output_dir = Path(args.output) / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / Path(image_path).name
        cv2.imwrite(str(output_path), annotated)

        # 对于 fire 检测，也用红色框标注
        fire_count = sum(1 for d in detections if d["class_id"] == 0)
        smoke_count = sum(1 for d in detections if d["class_id"] == 1)
        logger.info(f"{Path(image_path).name}: "
                    f"🔥 {fire_count} 处火焰 | 🌫 {smoke_count} 处烟雾 → {output_path}")

    return {
        "image": Path(image_path).name,
        "detections": detections,
    }


def process_video(model, video_path: str, args):
    """处理视频文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return

    # 视频写入器
    output_dir = Path(args.output) / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = output_dir / f"detected_{Path(video_path).stem}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    logger.info(f"处理视频: {video_path} ({total_frames} 帧, {fps} FPS)")

    fps_counter = FPSCounter()
    frame_idx = 0
    fire_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 每 N 帧推理一次（跳帧加速）
        if frame_idx % 3 == 1 or frame_idx == 1:
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            current_detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    boxes_list = boxes.xyxy.cpu().tolist()
                    scores_list = boxes.conf.cpu().tolist()
                    cls_list = boxes.cls.cpu().tolist()
                    current_detections = (boxes_list, scores_list, cls_list)
                    fire_frames += 1

        # 绘制最近一次检测结果
        if current_detections:
            boxes_list, scores_list, cls_list = current_detections
            frame = draw_detections(frame, boxes_list, scores_list, cls_list, args.conf)

        # 显示 FPS
        current_fps = fps_counter.tick()
        cv2.putText(frame, f"FPS: {current_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if args.show:
            cv2.imshow("Fire Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        writer.write(frame)

        if frame_idx % 100 == 0:
            logger.info(f"进度: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    logger.info(f"视频处理完成: {output_path}")
    logger.info(f"总帧数: {frame_idx}, 检测到火灾的帧: {fire_frames} ({100*fire_frames/frame_idx:.1f}%)")


def main():
    args = parse_args()

    from ultralytics import YOLO

    # 加载模型
    if not Path(args.weights).exists():
        logger.error(f"模型文件不存在: {args.weights}")
        sys.exit(1)

    logger.info(f"加载模型: {args.weights}")
    model = YOLO(args.weights)

    source_path = Path(args.source)

    # 判断输入类型
    if not source_path.exists():
        logger.error(f"输入源不存在: {args.source}")
        sys.exit(1)

    # 视频文件
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    if source_path.suffix.lower() in video_exts:
        process_video(model, str(source_path), args)
        return

    # 单张图像
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if source_path.is_file() and source_path.suffix.lower() in image_exts:
        result = process_image(model, str(source_path), args)
        if result and (args.save_json or args.save_csv):
            if args.save_json:
                export_to_json([result], str(Path(args.output) / "results.json"))
            if args.save_csv:
                export_to_csv([result], str(Path(args.output) / "results.csv"))
        return

    # 图像目录（批量）
    if source_path.is_dir():
        image_files = sorted([
            f for f in source_path.glob("*") if f.suffix.lower() in image_exts
        ])
        logger.info(f"批量推理: {len(image_files)} 张图像")

        all_results = []
        total_fire = 0
        total_smoke = 0

        for img_path in image_files:
            result = process_image(model, str(img_path), args)
            if result:
                all_results.append(result)
                total_fire += sum(1 for d in result["detections"] if d["class_id"] == 0)
                total_smoke += sum(1 for d in result["detections"] if d["class_id"] == 1)

        logger.info("=" * 50)
        logger.info(f"批量推理完成: {len(all_results)}/{len(image_files)} 张图像")
        logger.info(f"🔥 火焰检测: {total_fire} 处 | 🌫 烟雾检测: {total_smoke} 处")
        logger.info("=" * 50)

        # 导出结果
        if args.save_json:
            export_to_json(all_results, str(Path(args.output) / "results.json"))
        if args.save_csv:
            export_to_csv(all_results, str(Path(args.output) / "results.csv"))

        return

    logger.error(f"不支持的输入格式: {args.source}")


if __name__ == "__main__":
    main()
