$ErrorActionPreference = "Stop"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:POLARS_SKIP_CPU_CHECK = "1"
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) ".ultralytics"
$env:MPLCONFIGDIR = Join-Path (Get-Location) ".matplotlib"
$env:WINDIR = if ($env:WINDIR) { $env:WINDIR } else { "C:\Windows" }
$env:SystemRoot = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }

New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& "C:\ANACONDA2\envs\torch_gpu\python.exe" "src\train.py" `
  --model yolov8s.pt `
  --data data\data.yaml `
  --epochs 120 `
  --batch 8 `
  --imgsz 960 `
  --device 0 `
  --workers 0 `
  --patience 30 `
  --project runs\train `
  --name fire_yolov8s_960
exit $LASTEXITCODE
