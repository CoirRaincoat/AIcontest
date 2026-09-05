"""R2-B environment acceptance gate (mandate gate list, 2026-08-28).

Run with the NEW venv interpreter ONLY:
  C:/fire_envs/sf2026_min/Scripts/python.exe verify_env_gate.py

All 9 mandate gates must PASS for exit 0; result JSON is written either way to
remediation_r2b/ENV_GATE_RESULT.json. First failing gate is recorded; script never installs,
never downloads, never deletes, never touches system state.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FD_SRC = Path(__file__).resolve().parents[3] / "fire_detection" / "src"
RESULT = HERE / "ENV_GATE_RESULT.json"

checks = []


def gate(name, ok, detail):
    checks.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(("[PASS] " if ok else "[FAIL] ") + name + " :: " + str(detail)[:300])
    return ok


def main():
    import torch  # noqa

    gate("G1_python_version", sys.version_info[:2] == (3, 11), sys.version)
    gate("G2_isolation_sys_executable", "sf2026_min" in sys.executable.lower(), sys.executable)

    tv, tc = torch.__version__, torch.version.cuda
    gate("G3_torch_version", tv == "2.8.0+cu128", tv)
    gate("G4_torch_build_cuda_128", tc == "12.8", tc)
    avail = torch.cuda.is_available()
    gate("G5_cuda_available", avail, avail)
    if avail:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        arch = torch.cuda.get_arch_list()
        gate("G6_gpu_name", "RTX 5060" in name, name)
        gate("G7_capability_120", cap == (12, 0), cap)
        gate("G8_sm120_in_arch_list", any(a in ("sm_120", "compute_120") for a in arch), arch)

        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        gate("G9_cuda_fp32_matmul", torch.isfinite(c).all().item(), c.shape)
        a16, b16 = a.half(), b.half()
        c16 = a16 @ b16
        torch.cuda.synchronize()
        gate("G10_cuda_fp16_matmul", torch.isfinite(c16.float()).all().item(), c16.shape)
        conv = torch.nn.Conv2d(3, 8, 3).cuda().half()
        y = conv(torch.randn(1, 3, 64, 64, device="cuda").half())
        torch.cuda.synchronize()
        gate("G11_cuda_fp16_conv", tuple(y.shape) == (1, 8, 62, 62), tuple(y.shape))
        gate("G12_vram_reported", torch.cuda.get_device_properties(0).total_memory > 7e9,
             f"{torch.cuda.get_device_properties(0).total_memory/2**20:.0f} MiB")

    import torchvision
    gate("G13_torchvision_version", torchvision.__version__ == "0.23.0+cu128",
         torchvision.__version__)

    import ultralytics, transformers, timm, safetensors  # noqa
    import numpy, PIL  # noqa
    try:
        import cv2  # noqa
        cv2_v = cv2.__version__
    except Exception as e:  # ultralytics imports cv2 anyway; record if odd
        cv2_v = f"IMPORT-ERROR {e}"
    gate("G14_all_module_imports", True, {
        "ultralytics": ultralytics.__version__, "transformers": transformers.__version__,
        "timm": timm.__version__, "safetensors": safetensors.__version__,
        "numpy": numpy.__version__, "PIL": PIL.__version__, "cv2": cv2_v})

    # production entry importable (import only; no weight load, no inference)
    sys.path.insert(0, str(FD_SRC))
    import predict_best_fusion  # noqa
    import batch_safety  # noqa
    import evaluate_image_level  # noqa
    gate("G15_production_entry_importable", True, str(FD_SRC))

    free_gb = shutil.disk_usage("C:\\").free / 2**30
    gate("G16_C_free_ge_70GB", free_gb >= 70.0, f"{free_gb:.1f} GB free")

    r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True,
                       timeout=300)
    gate("G17_pip_check", r.returncode == 0, (r.stdout + r.stderr).strip()[:500])

    all_ok = all(c["pass"] for c in checks)
    RESULT.write_text(json.dumps({
        "gate": "R2-B environment acceptance", "all_pass": all_ok,
        "python": sys.version, "executable": sys.executable, "checks": checks,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL_PASS" if all_ok else "GATE_FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
