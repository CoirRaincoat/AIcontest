$ErrorActionPreference = "Stop"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:POLARS_SKIP_CPU_CHECK = "1"
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) ".ultralytics"
$env:MPLCONFIGDIR = Join-Path (Get-Location) ".matplotlib"
$env:WINDIR = if ($env:WINDIR) { $env:WINDIR } else { "C:\Windows" }
$env:SystemRoot = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }

New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

$weights = if ($args.Count -ge 1) { $args[0] } else { "runs\detect\runs\train\fire_yolov8s_960\weights\last.pt" }

& "C:\ANACONDA2\envs\torch_gpu\python.exe" "src\tune_image_threshold.py" `
  --weights $weights `
  --source data\val\images `
  --gt outputs\val_gt.json `
  --imgsz 960 `
  --device 0 `
  --output outputs\threshold_search.json
exit $LASTEXITCODE
