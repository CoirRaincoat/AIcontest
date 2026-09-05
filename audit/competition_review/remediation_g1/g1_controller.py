"""G1 formal-training controller (recoverable, self-contained, detached).

Runs the three production models m -> s -> s_aug in the frozen order, ONE formal training each.
Designed to be launched as a DETACHED process (Start-Process) so it survives the tool call; it is
resumable: on restart it reads g1_controller_status.json, skips completed models, and resumes an
incomplete model from its last.pt checkpoint (bounded by MAX_RESUME).

Hard rules enforced here:
  - cumulative formal-training wall clock <= BUDGET_H (12h)
  - one formal seed / one formal training per model; no rerun on poor metrics
  - holdout labels/metrics are NEVER read here
  - monitor aborts on temp>=90C once, temp>=87C x3, C:<50GB, NaN/Inf, 20min no-progress
  - heartbeat to g1_controller_status.json every 60s (PID/stage/cumulative-GPU-time/temp/VRAM/disk/ckpt)
  - only this process's own training child is ever terminated
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FD = ROOT / "fire_detection"
CFG_DIR = HERE / "ultralytics_config"
STATUS = ROOT / "audit/competition_review/machine/g1_controller_status.json"
SPLIT_FREEZE = HERE / "split/SPLIT_FREEZE.json"

MODELS = [
    {"key": "m", "hist": "fire_yolo26m_960", "init": "yolo26m.pt"},
    {"key": "s", "hist": "fire_yolo26s_960", "init": "yolo26s.pt"},
    {"key": "s_aug", "hist": "fire_yolo26s_aug_960", "init": "yolo26s.pt"},
]
BUDGET_H = 12.0
MAX_RESUME = 2
HEARTBEAT_S = 60


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def nvidia():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        mem, temp, util = r.stdout.strip().splitlines()[0].split(", ")
        return int(mem), int(temp), int(util)
    except Exception:
        return None, None, None


def disk_free_gb():
    import shutil
    return shutil.disk_usage("C:\\").free / 2**30


class Controller:
    def __init__(self):
        self.status = self._load()
        self.heartbeat_stop = threading.Event()
        self.abort = threading.Event()
        self.abort_reason = None
        self.child = None
        self.hot87 = 0
        self.cur = {"model": None, "run_dir": None}
        self.start_ts = time.time()

    def _load(self):
        if STATUS.exists():
            try:
                return json.loads(STATUS.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"models": {}, "cumulative_train_s": 0.0, "resume_ledger": {},
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def _save_status(self, stage):
        self.status["stage"] = stage
        self.status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.status["cumulative_train_s"] = self.status.get("cumulative_train_s", 0.0)
        STATUS.write_text(json.dumps(self.status, ensure_ascii=False, indent=2), encoding="utf-8")

    def _heartbeat(self):
        while not self.heartbeat_stop.wait(HEARTBEAT_S):
            mem, temp, util = nvidia()
            self.status["heartbeat"] = {
                "t": time.strftime("%H:%M:%S"),
                "pid": os.getpid(),
                "model": self.cur["model"],
                "run_dir": str(self.cur["run_dir"]) if self.cur["run_dir"] else None,
                "mem_MiB": mem, "temp_C": temp, "util_pct": util,
                "disk_free_GB": round(disk_free_gb(), 1),
                "cumulative_train_s": round(self.status.get("cumulative_train_s", 0.0), 1),
                "latest_checkpoint": self._latest_ckpt(),
            }
            self._save_status(self.status.get("stage", "running"))

    def _latest_ckpt(self):
        rd = self.cur.get("run_dir")
        if not rd:
            return None
        w = rd / "weights"
        if not w.exists():
            return None
        pts = sorted(w.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(pts[0]) if pts else None

    def _monitor(self):
        while not self.abort.wait(30):
            mem, temp, util = nvidia()
            free = disk_free_gb()
            if temp is not None:
                if temp >= 90:
                    self._do_abort(f"temp {temp}C >= 90C")
                    return
                if temp >= 87:
                    self.hot87 += 1
                    if self.hot87 >= 3:
                        self._do_abort("temp >=87C x3")
                        return
            if free < 50.0:
                self._do_abort(f"C: free {free:.1f}GB < 50GB")
                return
            rd = self.cur.get("run_dir")
            if rd and rd.exists():
                if self._results_nan(rd):
                    self._do_abort("NaN/Inf in results.csv")
                    return
                newest = max((p.stat().st_mtime for p in rd.rglob("*") if p.is_file()),
                             default=None)
                if newest and time.time() - newest > 20 * 60:
                    self._do_abort("20min no progress")
                    return

    def _results_nan(self, rd):
        csv = rd / "results.csv"
        if not csv.exists():
            return False
        try:
            lines = csv.read_text(encoding="utf-8").strip().splitlines()
            return bool(lines) and any(t in lines[-1].lower() for t in ("nan", "inf"))
        except Exception:
            return False

    def _do_abort(self, reason):
        self.abort_reason = reason
        self.abort.set()
        (HERE / "TRAINING_ABORTED.txt").write_text(reason, encoding="utf-8")
        if self.child and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(30)
            except subprocess.TimeoutExpired:
                self.child.kill()

    def run_all(self):
        # verify the frozen split is intact before any training (no re-split allowed)
        if not SPLIT_FREEZE.exists():
            self._do_abort("SPLIT_FREEZE.json missing - re-split forbidden")
            self._save_status("ABORTED:split_freeze_missing")
            print("STOP::SPLIT_FREEZE.json missing")
            return 2
        fr = json.loads(SPLIT_FREEZE.read_text(encoding="utf-8"))
        if not fr.get("frozen") or fr.get("phase_A_verification", {}).get("cross_pairs_le8_after_regroup") != 0:
            self._do_abort("split freeze drifted - re-split forbidden")
            self._save_status("ABORTED:split_freeze_drift")
            print("STOP::split freeze drifted")
            return 2
        threading.Thread(target=self._heartbeat, daemon=True).start()
        threading.Thread(target=self._monitor, daemon=True).start()
        self._save_status("started")
        for spec in MODELS:
            if self.abort.is_set():
                break
            self._save_status(f"before_{spec['key']}")
            if not self._budget_ok(spec):
                self.status["stopped_budget"] = True
                self._save_status("STOPPED_BUDGET")
                print(f"STOPPED_BUDGET before {spec['key']}")
                break
            done = self.status["models"].get(spec["key"], {}).get("best_sha256")
            if done:
                print(f"SKIP {spec['key']} (already complete)")
                continue
            self._train_one(spec)
        self.heartbeat_stop.set()
        self._save_status("finished" if not self.abort.is_set() else f"aborted:{self.abort_reason}")
        print("CONTROLLER_DONE")
        return 0

    def _budget_ok(self, spec):
        spent = self.status.get("cumulative_train_s", 0.0)
        remaining = self._project_remaining()
        return (spent + remaining) <= BUDGET_H * 3600

    def _project_remaining(self):
        # estimate remaining models from the most recent model's actual per-epoch time
        last_ep = self.status.get("last_measured_per_epoch_s")
        if not last_ep:
            # conservative: use dry-run measured m=172.3, s=134.8
            est = {"m": 172.3, "s": 134.8, "s_aug": 134.8}
            remaining = sum(est[k] * 80 for k, _ in [(x["key"], None) for x in MODELS
                                                      if not self.status["models"].get(x["key"], {}).get("best_sha256")])
        else:
            remaining = 0
            for x in MODELS:
                if not self.status["models"].get(x["key"], {}).get("best_sha256"):
                    remaining += last_ep * 80
        return remaining

    def _train_one(self, spec):
        import yaml
        hist = yaml.safe_load((FD / f"runs/detect/runs/train/{spec['hist']}/args.yaml")
                              .read_text(encoding="utf-8"))
        init = FD / spec["init"]
        if not init.is_file():
            self._do_abort(f"init weight missing: {init}")
            return
        name = f"reg1_{spec['hist']}"
        project = HERE / "formal_runs"
        run_dir = project / name
        self.cur = {"model": spec["key"], "run_dir": run_dir}
        resume = run_dir / "weights" / "last.pt"
        resume_count = self.status["resume_ledger"].get(spec["key"], 0)

        train_args = {k: v for k, v in hist.items() if k not in ("save_dir", "cls_remap")}
        overrides = {"data": str(HERE / "dataset/data_reg1.yaml"), "project": str(project),
                     "name": name, "model": str(init), "device": 0}
        if resume.exists() and not (run_dir / "weights" / "best.pt").exists():
            if resume_count >= MAX_RESUME:
                self._do_abort(f"{spec['key']} resume limit {MAX_RESUME} exceeded")
                return
            overrides["resume"] = True
            self.status["resume_ledger"][spec["key"]] = resume_count + 1
            self._save_status(f"resuming_{spec['key']}")
            print(f"RESUME {spec['key']} from last.pt (resume {resume_count + 1}/{MAX_RESUME})")
        train_args.update(overrides)
        train_args.pop("resume", None) if "resume" not in overrides else None

        cfg_json = HERE / f"_train_args_{spec['key']}_formal.json"
        cfg_json.write_text(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                                        for k, v in train_args.items()}), encoding="utf-8")
        child_script = HERE / "_train_child.py"
        child_script.write_text(
            "import sys,yaml,json\nfrom ultralytics import YOLO, settings\n"
            "args=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))\n"
            "try: settings.update({'sync':False,'hub':False})\nexcept Exception: pass\n"
            "m=YOLO(args.pop('model'))\n"
            "resume=args.pop('resume',False)\n"
            "m.train(resume=resume, **args)\nprint('TRAIN_CHILD_DONE')\n", encoding="utf-8")

        env = {**os.environ, "YOLO_CONFIG_DIR": str(CFG_DIR),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        logf = open(HERE / f"logs_train_{spec['key']}_formal.log", "w", encoding="utf-8")
        self._save_status(f"training_{spec['key']}")
        t0 = time.time()
        self.child = subprocess.Popen([sys.executable, str(child_script), str(cfg_json)],
                                      stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(FD))
        rc = self.child.wait()
        wall = time.time() - t0
        logf.close()
        self.child = None

        if self.abort.is_set():
            self._save_status(f"aborted_{spec['key']}:{self.abort_reason}")
            return
        best = run_dir / "weights" / "best.pt"
        if rc != 0 or not best.is_file():
            self._do_abort(f"{spec['key']} failed rc={rc} (no best.pt)")
            return
        self.status["models"][spec["key"]] = {
            "done": True, "wall_s": round(wall, 1),
            "best_sha256": sha256(best),
            "last_sha256": sha256(run_dir / "weights" / "last.pt"),
            "run_dir": str(run_dir), "args_freezed": str(cfg_json),
            "config_sha256": sha256(cfg_json),
            "env_python": sys.executable,
        }
        self.status["cumulative_train_s"] = self.status.get("cumulative_train_s", 0.0) + wall
        # measure per-epoch for projection of remaining models
        rcsv = run_dir / "results.csv"
        if rcsv.exists():
            try:
                import csv as _csv
                rows = list(_csv.DictReader(rcsv.read_text(encoding="utf-8").splitlines()))
                if rows:
                    self.status["last_measured_per_epoch_s"] = float(rows[-1]["time"]) / len(rows)
            except Exception:
                pass
        self._save_status(f"done_{spec['key']}")
        print(f"DONE {spec['key']} rc={rc} wall={wall:.0f}s sha={self.status['models'][spec['key']]['best_sha256'][:16]}")


if __name__ == "__main__":
    c = Controller()
    sys.exit(c.run_all())
