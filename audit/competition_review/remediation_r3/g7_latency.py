"""G7 latency / VRAM profiling driver (measurement only; production code untouched).

Run with the NEW venv python. Method (recorded verbatim in the output JSON):
  A. cold_start        : production entry as a fresh subprocess over the P6 12 images,
                         wall time around the whole process (imports + weight loads + inference).
  B. warm_end_to_end   : two further identical subprocess runs (OS page cache warm);
                         median of the two reported.
  C. stage_decomp      : in-process, production stage functions called exactly as main() calls
                         them (same order, same argparse defaults obtained via parse_args),
                         one full warm-up pass discarded, then 3 timed passes
                         (torch.cuda.synchronize around each stage).
  D. single_image_chain: production stage functions on a ONE-image list, 3 different images,
                         weights reload per call -> upper-bound per-image chain latency.
  E. peak VRAM         : torch.cuda.max_memory_allocated for in-process passes + an
                         nvidia-smi sampler thread (0.5 s) around every subprocess run.

Batched VLM stages load weights once per CALL, so a true per-image marginal cost is not
attributable without modifying production code (forbidden); G7 therefore reports
amortized per-image means (stage_total/12, warm totals/12) and medians over runs,
with n stated. Numbers are recorded, not judged.
"""
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FD_SRC = ROOT / "fire_detection" / "src"
ENTRY = FD_SRC / "predict_best_fusion.py"
HUB = ROOT / "hf_cache" / "hub"
OUT = HERE / "g7"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FD_SRC))

ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
IMGS = sorted(p for p in (HERE / "p6" / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
assert len(IMGS) == 12, f"expected 12 P6 images, found {len(IMGS)}"


def nvidia_snapshot():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        u, util, temp, drv = r.stdout.strip().splitlines()[0].split(", ")
        return {"mem_MiB": int(u), "util_pct": int(util), "temp_C": int(temp), "driver": drv}
    except Exception as e:
        return {"error": str(e)}


class Sampler:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            s = nvidia_snapshot()
            s["t"] = time.perf_counter()
            self.samples.append(s)
            self.stop.wait(0.5)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join(timeout=3)

    def peak(self):
        mems = [s["mem_MiB"] for s in self.samples if "mem_MiB" in s]
        temps = [s["temp_C"] for s in self.samples if "temp_C" in s]
        return {"peak_mem_used_MiB": max(mems) if mems else None,
                "max_temp_C": max(temps) if temps else None, "n_samples": len(self.samples)}


def cli_for(source: Path, output: Path, py: str):
    # production ROOT is hardcoded to the dead C:\AI path (F-01/F-02), so EVERY
    # weight/checkpoint/cache path must be passed explicitly (zero production edits).
    import predict_best_fusion as pbf
    sys.argv = ["probe", "--source", str(source), "--output", str(output)]
    args = pbf.parse_args()
    args.siglip_checkpoint = ROOT / "fire_detection/runs_siglip/siglip2_linear_seed2026/best_head.pt"
    args.dino_checkpoint = ROOT / "fire_detection/runs_dinov3/dinov3_vitb16_linear_seed42/best_head.pt"
    args.crop_checkpoint = ROOT / "fire_detection/runs_siglip/siglip2_crop_v2_seed2026/best_head.pt"
    args.cache_dir = HUB
    args.yolo_m_weights = ROOT / "fire_detection/runs/detect/runs/train/fire_yolo26m_960/weights/best.pt"
    args.yolo_s_weights = ROOT / "fire_detection/runs/detect/runs/train/fire_yolo26s_960/weights/best.pt"
    args.yolo_s_aug_weights = ROOT / "fire_detection/runs/detect/runs/train/fire_yolo26s_aug_960/weights/best.pt"
    cmd = [py, str(ENTRY),
           "--source", str(source), "--output", str(output),
           "--siglip-checkpoint", str(args.siglip_checkpoint),
           "--dino-checkpoint", str(args.dino_checkpoint),
           "--crop-checkpoint", str(args.crop_checkpoint),
           "--cache-dir", str(args.cache_dir),
           "--yolo-m-weights", str(args.yolo_m_weights),
           "--yolo-s-weights", str(args.yolo_s_weights),
           "--yolo-s-aug-weights", str(args.yolo_s_aug_weights)]
    return args, cmd


def subprocess_run(cmd, tag):
    import os
    env = {**os.environ, **ENV}
    with Sampler() as s:
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=1800)
        wall = time.perf_counter() - t0
    (OUT / f"run_{tag}.log").write_text(r.stdout[-20000:] + "\n===STDERR===\n" + r.stderr[-20000:],
                                        encoding="utf-8")
    return {"tag": tag, "wall_s": round(wall, 3), "rc": r.returncode, **s.peak()}


def main():
    py = sys.executable
    args, cmd = cli_for(HERE / "p6" / "images", OUT / "p6_g7.json", py)
    import torch
    import predict_best_fusion as pbf

    prof = {"method": __doc__.strip(), "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "imgsz": args.imgsz,
            "device": args.device, "n_images": 12, "venv_python": py}

    # A. cold start
    runs = [subprocess_run(cmd, "cold")]
    # B. warm end-to-end (subprocess, warm caches)
    for i in range(2):
        runs.append(subprocess_run(cmd, f"warm{i + 1}"))
    prof["subprocess_runs"] = runs
    prof["cold_start_s"] = runs[0]["wall_s"]
    prof["warm_end_to_end_median_s"] = statistics.median(r["wall_s"] for r in runs[1:])

    # C. stage decomposition (in-process, production functions, production defaults)
    dev = torch.device("cpu" if args.device.lower() == "cpu" else "cuda")
    stages = ["siglip", "dino", "yolo_m", "yolo_s", "yolo_s_aug", "proposals", "crop"]

    def chain(paths, failures, timeit=False):
        out, tt = {}, {}
        for name in stages:
            torch.cuda.synchronize() if dev.type == "cuda" else None
            t0 = time.perf_counter()
            if name == "siglip":
                r = pbf.predict_siglip(paths, args.siglip_checkpoint, args.cache_dir,
                                       args.siglip_batch_size, dev, failures=failures)
            elif name == "dino":
                r = pbf.predict_dinov3(paths, args.dino_checkpoint, args.cache_dir,
                                       args.dino_batch_size, dev, failures=failures)
            elif name.startswith("yolo"):
                w, c = {"yolo_m": (args.yolo_m_weights, args.yolo_m_conf),
                        "yolo_s": (args.yolo_s_weights, args.yolo_s_conf),
                        "yolo_s_aug": (args.yolo_s_aug_weights, args.yolo_s_aug_conf)}[name]
                r = pbf.predict_yolo(paths, w, c, args.imgsz, args.iou, args.min_area,
                                     args.device, name, failures=failures)
            elif name == "proposals":
                r = pbf.collect_yolo_proposals(paths, [
                    ("yolo26m", args.yolo_m_weights), ("yolo26s", args.yolo_s_weights),
                    ("yolo26s_aug", args.yolo_s_aug_weights)],
                    args.crop_proposal_conf, args.imgsz, args.iou, args.device,
                    args.crop_max_proposals, failures=failures)
            else:
                r = pbf.predict_crop_siglip(paths, prev_proposals, args.crop_checkpoint,
                                            args.cache_dir, args.crop_batch_size,
                                            args.crop_context, args.crop_min_size, dev,
                                            failures=failures)
            (torch.cuda.synchronize() if dev.type == "cuda" else None)
            if name == "proposals":
                prev_proposals = r
            out[name] = r
            tt[name] = round(time.perf_counter() - t0, 4)
        return out, tt

    chain(IMGS, pbf.bs.new_failure_log())  # warm-up pass (kernels/cudnn warm), discarded
    passes = []
    peak_alloc = 0
    for i in range(3):
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        _, tt = chain(IMGS, pbf.bs.new_failure_log())
        passes.append(tt)
        if dev.type == "cuda":
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated() / 2**20)
    prof["stage_decomp_timed_passes_s"] = passes
    prof["stage_medians_s"] = {s: statistics.median(p[s] for p in passes) for s in stages}
    prof["warm_inprocess_total_s"] = round(sum(prof["stage_medians_s"].values()), 3)
    prof["amortized_warm_per_image_s"] = round(prof["warm_inprocess_total_s"] / 12, 4)
    prof["peak_vram_allocated_MiB"] = round(peak_alloc, 1)

    # D. single-image upper bound (weights reload per stage call)
    singles = []
    for p in (IMGS[0], IMGS[5], IMGS[11]):
        t0 = time.perf_counter()
        chain([p], pbf.bs.new_failure_log())
        singles.append(round(time.perf_counter() - t0, 3))
    prof["single_image_full_chain_s"] = {"runs": singles, "median_s": statistics.median(singles),
                                         "note": "per-stage weight reload included (upper bound)"}

    # sanity: production subprocess outputs from warm run must exist & have 12 keys
    sub = json.loads((OUT / "p6_g7.json").read_text(encoding="utf-8"))
    prof["subprocess_output_keys"] = len(sub)
    (OUT / "latency_profile.json").write_text(json.dumps(prof, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps({k: prof[k] for k in (
        "cold_start_s", "warm_end_to_end_median_s", "warm_inprocess_total_s",
        "amortized_warm_per_image_s", "peak_vram_allocated_MiB", "subprocess_output_keys")},
        indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
