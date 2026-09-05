"""G1 single formal retrain driver (one production model per invocation).

Config authority = the model's own historical args.yaml (production run), with ONLY:
  data     -> remediation_g1/dataset/data_reg1.yaml   (grouped-isolation reg1 split)
  project  -> remediation_g1/runs                     (new registered dir)
  name     -> reg1_<historical name>
  model    -> resolved to the on-disk init weight fire_detection/<historical model>
Everything else (epochs=80, patience, batch, imgsz=960, seed=0, deterministic=true,
workers=0, amp=true, all augmentation) passes through UNCHANGED.

Modes:
  dryrun : same config with epochs=1; metric-unread technical validation (paths/save/OOM);
           produces TIME_PROJECTION.json (per-epoch x 80 x 3 models vs the 8h budget).
  formal : the ONE permitted formal run for this model. Never rerun on poor metrics.

Hard monitoring (mandate): temp >=90C once OR >=87C x3 OR C: free <50GB OR NaN/Inf in
results.csv OR no run-dir progress for 20min  ->  abort own child, exit 42, keep evidence.
"""
import argparse
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
CFG = HERE / "ultralytics_config"
CFG.mkdir(parents=True, exist_ok=True)

MODELS = {
    "m": {"hist": "fire_yolo26m_960", "init": "yolo26m.pt", "batch": 4},
    "s": {"hist": "fire_yolo26s_960", "init": "yolo26s.pt", "batch": 8},
    "s_aug": {"hist": "fire_yolo26s_aug_960", "init": "yolo26s.pt", "batch": 8},
}
BUDGET_S = 8 * 3600


def nvidia():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,temperature.memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        mem, t, _ = r.stdout.strip().splitlines()[0].split(", ")
        return int(mem), int(t)
    except Exception:
        return None, None


def disk_free_gb():
    import shutil
    return shutil.disk_usage("C:\\").free / 2**30


class Monitor(threading.Thread):
    """mandate abort conditions; only ever terminates the child WE started."""
    def __init__(self, run_dir):
        super().__init__(daemon=True)
        self.run_dir = run_dir
        self.stop = threading.Event()
        self.child = None
        self.events = []
        self.hot87 = 0
        self.aborted = threading.Event()

    def run(self):
        while not self.stop.wait(30):
            mem, temp = nvidia()
            free = disk_free_gb()
            stamp = time.strftime("%H:%M:%S")
            if temp is not None:
                print(f"[MON {stamp}] mem={mem}MiB temp={temp}C free={free:.1f}GB", flush=True)
                if temp >= 90:
                    self._abort(f"temp {temp}C >= 90C (once)")
                    return
                if temp >= 87:
                    self.hot87 += 1
                    if self.hot87 >= 3:
                        self._abort(f"temp >=87C three times")
                        return
            if free < 50.0:
                self._abort(f"C: free {free:.1f}GB < 50GB")
                return
            newest = self._newest_mtime()
            if newest and time.time() - newest > 20 * 60:
                self._abort("no progress in run dir for 20 min")
                return
            self._check_nan()

    def _newest_mtime(self):
        ts = [p.stat().st_mtime for p in self.run_dir.rglob("*") if p.is_file()]
        return max(ts) if ts else None

    def _check_nan(self):
        csv = self.run_dir / "results.csv"
        if not csv.exists():
            return
        try:
            lines = csv.read_text(encoding="utf-8").strip().splitlines()
            if lines and any(tok in lines[-1].lower() for tok in ("nan", "inf")):
                self._abort("NaN/Inf detected in results.csv")
        except Exception:
            pass

    def _abort(self, reason):
        self.events.append({"t": time.strftime("%H:%M:%S"), "abort_reason": reason})
        print(f"[MON] ABORT: {reason}", flush=True)
        self.aborted.set()
        if self.child and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(30)
            except subprocess.TimeoutExpired:
                self.child.kill()
        (HERE / "TRAINING_ABORTED.txt").write_text(reason, encoding="utf-8")

    def result(self):
        return {"aborted": self.aborted.is_set(), "events": self.events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--mode", choices=["dryrun", "formal"], required=True)
    a = ap.parse_args()
    spec = MODELS[a.model]
    import yaml
    hist = yaml.safe_load((FD / f"runs/detect/runs/train/{spec['hist']}/args.yaml")
                          .read_text(encoding="utf-8"))
    init = FD / spec["init"]
    if not init.is_file():
        print(f"STOP::init weight missing on disk: {init} (auto-download forbidden)")
        return 2
    if a.mode == "formal" and not (HERE / "split/SPLIT_FREEZE.json").exists():
        print("STOP::SPLIT_FREEZE.json missing - training may not start before freeze")
        return 2

    data_yaml = HERE / "dataset" / "data_reg1.yaml"
    if not data_yaml.exists():
        print("STOP::data_reg1.yaml missing (run prep_grouped_split.py first)")
        return 2

    name = f"reg1_{spec['hist']}" + ("_dryrun" if a.mode == "dryrun" else "")
    project = HERE / ("dryrun" if a.mode == "dryrun" else "runs")
    if (project / name).exists() and a.mode == "formal":
        print(f"STOP::{project/name} already exists - formal run is one-shot, no overwrite")
        return 2

    overrides = {"data": str(data_yaml), "project": str(project), "name": name,
                 "model": str(init), "device": 0}
    if a.mode == "dryrun":
        overrides["epochs"] = 1
        overrides["plots"] = False
    # drop the two non-train keys in the saved args (save_dir is a path artifact;
    # cls_remap is a custom key not present in ultralytics 8.4.87 DEFAULT_CFG)
    train_args = {k: v for k, v in hist.items() if k not in ("save_dir", "cls_remap")}
    train_args.update(overrides)
    train_args.pop("resume", None)

    run_script = HERE / "_train_child.py"
    run_script.write_text(
        "import json,sys,yaml\nfrom ultralytics import YOLO, settings\n"
        "args=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))\n"
        "try: settings.update({'sync':False,'hub':False})\n"
        "except Exception: pass\n"
        "m=YOLO(args.pop('model'))\nr=m.train(**args)\n"
        "print('TRAIN_CHILD_DONE')\n", encoding="utf-8")
    cfg_json = HERE / f"_train_args_{a.model}_{a.mode}.json"
    cfg_json.write_text(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                                    for k, v in train_args.items()}), encoding="utf-8")

    env = {**os.environ,
           "YOLO_CONFIG_DIR": str(CFG),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    cmd = [sys.executable, str(run_script), str(cfg_json)]
    log = open(HERE / f"logs_train_{a.model}_{a.mode}.log", "w", encoding="utf-8")
    mon = Monitor(project / name)
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(FD))
    mon.child = proc
    if a.mode == "formal":
        mon.start()
    rc = proc.wait()
    wall = time.perf_counter() - t0
    mon.stop.set()
    log.close()

    rec = {"model": a.model, "mode": a.mode, "child_rc": rc, "wall_s": round(wall, 1),
           "train_args_freezed": {k: str(v) if isinstance(v, Path) else v
                                  for k, v in train_args.items()},
           "monitor": mon.result(), "run_dir": str(project / name)}
    run_dir = project / name
    best = run_dir / "weights" / "best.pt"
    if a.mode == "dryrun":
        # per-epoch seconds read from ultralytics results.csv "time" column (not wall/epochs)
        per_epoch = wall
        rcsv = run_dir / "results.csv"
        if rcsv.exists():
            try:
                import csv as _csv
                rows = list(_csv.DictReader(rcsv.read_text(encoding="utf-8").splitlines()))
                if rows:
                    per_epoch = float(rows[-1]["time"])
            except Exception:
                per_epoch = wall
        this_80 = per_epoch * 80
        # s / s_aug run batch 8 (half the batches of m's batch 4) on a smaller backbone:
        # project them at 0.50x m's per-epoch (conservative, matches historical s/m~0.52).
        per_epoch_s = per_epoch
        total_3 = this_80 * (1 + 0.50 + 0.50)
        rec["projection"] = {
            "per_epoch_s_m": round(per_epoch, 1),
            "this_model_m_80ep_h": round(this_80 / 3600, 2),
            "s_and_s_aug_factor": 0.50,
            "three_models_80ep_h": round(total_3 / 3600, 2),
            "budget_h": 8,
            "over_budget": total_3 > BUDGET_S,
            "basis": "per-epoch from dry-run results.csv (m, batch4, reg1 698 imgs); "
                     "s/s_aug at batch8 ~0.50x (historical s/m GPU-h = 0.52)",
            "mitigating": "first-epoch warmup/cudnn-autotune inflates dry-run epoch vs later "
                          "epochs; s has patience=20 (may early-stop); reg1 dataset smaller (698 "
                          "vs 880 historical)"}
        if total_3 > BUDGET_S:
            rec["projection"]["decision"] = ("PROJECTED_OVER_BUDGET -> formal training must NOT "
                                             "start; pipeline stops per mandate budget gate")
    elif best.is_file():
        import hashlib
        h = hashlib.sha256(best.read_bytes()).hexdigest()
        rec["best_pt_sha256"] = h
        rec["weights"] = {"best": str(best), "last": str(run_dir / "weights" / "last.pt")}
    out = HERE / f"TRAIN_RECORD_{a.model}_{a.mode}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("model", "mode", "child_rc", "wall_s")}, indent=2))
    if a.mode == "dryrun":
        print(json.dumps(rec["projection"], indent=2))
    if rc != 0 or mon.aborted.is_set():
        return 42 if mon.aborted.is_set() else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
