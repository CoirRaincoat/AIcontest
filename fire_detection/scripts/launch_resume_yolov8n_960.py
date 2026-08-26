"""Launch YOLOv8n resume training as a detached Windows process."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["POLARS_SKIP_CPU_CHECK"] = "1"
    env["YOLO_CONFIG_DIR"] = str(root / ".ultralytics")
    env["MPLCONFIGDIR"] = str(root / ".matplotlib")
    env.setdefault("WINDIR", r"C:\Windows")
    env.setdefault("SystemRoot", r"C:\Windows")
    return env


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    out_log = log_dir / "resume_yolov8n_960.out.log"
    err_log = log_dir / "resume_yolov8n_960.err.log"
    pid_file = log_dir / "resume_yolov8n_960.pid"

    code = (
        "from ultralytics import YOLO; "
        "YOLO('runs/detect/runs/train/fire_yolov8n_960/weights/last.pt').train(resume=True)"
    )
    cmd = [sys.executable, "-c", code]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    with out_log.open("w", encoding="utf-8", errors="replace") as stdout, err_log.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        process = subprocess.Popen(
            cmd,
            cwd=root,
            env=clean_env(),
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )

    pid_file.write_text(str(process.pid), encoding="utf-8")
    print(f"started pid={process.pid}")
    print(f"stdout={out_log}")
    print(f"stderr={err_log}")


if __name__ == "__main__":
    main()
