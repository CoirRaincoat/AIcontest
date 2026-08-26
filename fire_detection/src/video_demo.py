"""
实时视频火灾检测演示 — 支持摄像头实时检测和视频文件检测

使用方式:
    # 摄像头实时检测
    python src/video_demo.py --weights models/best.pt

    # 视频文件检测
    python src/video_demo.py --weights models/best.pt --source test.mp4

    # RTSP 网络摄像头
    python src/video_demo.py --weights models/best.pt --source rtsp://xxx
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    logger, draw_detections, CLASS_NAMES, CLASS_COLORS,
    FPSCounter, export_to_json
)


def parse_args():
    parser = argparse.ArgumentParser(description="🔥 Fire Detection - Real-time Video Demo")

    parser.add_argument("--weights", type=str, required=True,
                        help="模型权重路径 (.pt)")
    parser.add_argument("--source", type=str, default="0",
                        help="输入源: 摄像头索引 (0,1,2) / 视频文件路径 / RTSP URL")
    parser.add_argument("--output", type=str, default=None,
                        help="保存输出视频路径 (可选)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU NMS 阈值 (默认: 0.45)")
    parser.add_argument("--device", type=str, default="0",
                        help="设备 (默认: 0)")
    parser.add_argument("--skip_frames", type=int, default=1,
                        help="跳帧间隔 (1=每帧检测, 2=隔一帧, 默认: 1)")
    parser.add_argument("--alert_conf", type=float, default=0.5,
                        help="报警置信度阈值 (默认: 0.5)")
    parser.add_argument("--no_display", action="store_true",
                        help="不显示实时画面")
    parser.add_argument("--save_log", action="store_true",
                        help="保存检测日志")

    return parser.parse_args()


def draw_alert_overlay(frame: np.ndarray, alert_level: str, alert_count: int) -> np.ndarray:
    """
    在帧上绘制报警覆盖层。

    Args:
        frame: 当前帧
        alert_level: 报警级别 "safe" / "warning" / "danger"
        alert_count: 连续检测到火灾的帧数

    Returns:
        叠加后的帧
    """
    h, w = frame.shape[:2]

    # 顶部状态栏
    if alert_level == "danger":
        bar_color = (0, 0, 255)  # 红色
        status_text = f"!!! FIRE ALERT !!!"
    elif alert_level == "warning":
        bar_color = (0, 165, 255)  # 橙色
        status_text = f"WARNING: Possible Fire Detected"
    else:
        bar_color = (0, 128, 0)  # 绿色
        status_text = "SAFE: No Fire Detected"

    # 绘制顶部状态栏
    bar_height = 40
    cv2.rectangle(frame, (0, 0), (w, bar_height), bar_color, -1)
    cv2.putText(frame, status_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 绘制报警计数
    if alert_level != "safe":
        cv2.putText(frame, f"Alert frames: {alert_count}",
                    (w - 250, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def draw_info_panel(frame: np.ndarray, fps: float, detections: list,
                    alert_conf: float, frame_idx: int) -> np.ndarray:
    """在帧底部绘制信息面板"""
    h, w = frame.shape[:2]

    # 半透明底部面板
    panel_h = 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    y_offset = h - panel_h + 20

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 检测统计
    fire_count = sum(1 for d in detections if (d[2] == 0 if isinstance(d, tuple) else d.get("class_id") == 0))
    smoke_count = len(detections) - fire_count

    cv2.putText(frame, f"Fire: {fire_count}", (150, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Smoke: {smoke_count}", (300, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

    # 帧号和时间
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Frame: {frame_idx} | {time_str}",
                (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 置信度阈值
    cv2.putText(frame, f"Threshold: {alert_conf}", (10, y_offset + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def main():
    args = parse_args()

    from ultralytics import YOLO

    # 加载模型
    if not Path(args.weights).exists():
        logger.error(f"模型文件不存在: {args.weights}")
        sys.exit(1)

    logger.info(f"加载模型: {args.weights}")
    model = YOLO(args.weights)

    # 打开视频源
    source = args.source
    if source.isdigit():
        source = int(source)
        logger.info(f"打开摄像头: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"无法打开视频源: {source}")
        sys.exit(1)

    # 获取视频信息
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"分辨率: {width}x{height}, 视频FPS: {fps_video}")

    # 视频写入器
    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_video if fps_video > 0 else 30, (width, height))
        logger.info(f"输出视频: {output_path}")

    # 检测日志
    detection_log = [] if args.save_log else None

    # 状态跟踪
    fps_counter = FPSCounter()
    frame_idx = 0
    alert_count = 0
    alert_cooldown = 0
    ALERT_THRESHOLD = 5       # 连续 N 帧触发高危报警
    COOLDOWN_FRAMES = 30      # 报警冷却帧数

    current_detections = []   # 保存最近一次检测结果
    alert_level = "safe"

    logger.info("=" * 60)
    logger.info("🔥 实时火灾检测已启动")
    logger.info(f"   置信度阈值: {args.conf} | 报警阈值: {args.alert_conf}")
    logger.info(f"   按 'q' 退出 | 按 's' 截图 | 按 'p' 暂停")
    logger.info("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("视频流结束")
                break

            frame_idx += 1

            # 跳帧处理
            do_detect = (frame_idx % args.skip_frames == 1) or (frame_idx == 1)

            if do_detect:
                # 模型推理
                results = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                )

                # 提取检测结果
                current_detections = []
                has_high_conf_fire = False

                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for i in range(len(boxes)):
                            score = float(boxes.conf[i])
                            cls_id = int(boxes.cls[i])
                            bbox = boxes.xyxy[i].cpu().tolist()

                            current_detections.append({
                                "bbox": bbox,
                                "score": score,
                                "class_id": cls_id,
                                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                            })

                            # 检查高置信度火焰
                            if cls_id == 0 and score >= args.alert_conf:
                                has_high_conf_fire = True

                # 报警逻辑
                if has_high_conf_fire:
                    if alert_cooldown == 0:
                        alert_count += 1
                    else:
                        alert_cooldown -= 1

                    if alert_count >= ALERT_THRESHOLD:
                        alert_level = "danger"
                else:
                    if alert_count > 0:
                        alert_cooldown = COOLDOWN_FRAMES
                    alert_count = max(0, alert_count - 1)
                    if alert_count == 0:
                        alert_level = "safe"
                    elif alert_count >= 2:
                        alert_level = "warning"

            # 绘制检测框
            if current_detections:
                boxes_list = [d["bbox"] for d in current_detections]
                scores_list = [d["score"] for d in current_detections]
                cls_list = [d["class_id"] for d in current_detections]
                frame = draw_detections(frame, boxes_list, scores_list, cls_list, args.conf)

            # 绘制报警覆盖层
            frame = draw_alert_overlay(frame, alert_level, alert_count)

            # 绘制信息面板
            fps = fps_counter.tick()
            frame = draw_info_panel(frame, fps, current_detections, args.conf, frame_idx)

            # 保存日志
            if args.save_log and do_detect:
                log_entry = {
                    "frame": frame_idx,
                    "timestamp": datetime.now().isoformat(),
                    "alert_level": alert_level,
                    "fps": round(fps, 1),
                    "detections": [
                        {"class": d["class_name"], "score": round(d["score"], 3)}
                        for d in current_detections
                    ],
                }
                detection_log.append(log_entry)

            # 显示
            if not args.no_display:
                cv2.imshow("Fire Detection System", frame)

            # 写入输出
            if writer:
                writer.write(frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("用户退出")
                break
            elif key == ord("s"):
                # 截图
                screenshot_dir = Path("../outputs/screenshots")
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshot_dir / f"fire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(screenshot_path), frame)
                logger.info(f"截图已保存: {screenshot_path}")
            elif key == ord("p"):
                logger.info("暂停 — 按任意键继续")
                cv2.waitKey(0)

    except KeyboardInterrupt:
        logger.info("检测被用户中断")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # 保存日志
        if args.save_log and detection_log:
            log_path = Path("../outputs") / f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            export_to_json(
                [{"video_source": str(source), "log": detection_log}],
                str(log_path)
            )

        logger.info(f"总计处理帧数: {frame_idx}")
        logger.info("🔥 火灾检测系统已停止")


if __name__ == "__main__":
    main()
