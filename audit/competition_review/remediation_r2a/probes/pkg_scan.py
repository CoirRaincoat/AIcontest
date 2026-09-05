"""R2-A probe: report interpreter identity + installed package versions.

Low side-effect by design: uses importlib.metadata only — never imports the
target packages, never loads models, never initialises CUDA.
"""
import importlib.metadata as md
import platform
import sys

PACKAGES = [
    "torch", "torchvision", "torchaudio", "ultralytics", "transformers",
    "timm", "gradio", "safetensors", "numpy", "pillow", "opencv-python",
    "huggingface-hub", "accelerate", "pandas", "matplotlib",
    "onnx", "onnxruntime", "nvidia-cudnn-cu12", "nvidia-cublas-cu12",
]

print("executable:", sys.executable)
print("version:", sys.version.replace("\n", " "))
print("platform:", platform.platform())
print("architecture:", platform.machine())
for name in PACKAGES:
    try:
        print(f"PKG {name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"PKG {name} MISSING")
    except Exception as exc:  # noqa: BLE001 — probe must not die on one entry
        print(f"PKG {name} ERROR {type(exc).__name__}: {exc}")
