$ErrorActionPreference = "Stop"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:POLARS_SKIP_CPU_CHECK = "1"
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) ".ultralytics"
$env:MPLCONFIGDIR = Join-Path (Get-Location) ".matplotlib"
$env:WINDIR = if ($env:WINDIR) { $env:WINDIR } else { "C:\Windows" }
$env:SystemRoot = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }

New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

if ($args.Count -lt 1) {
  throw "Usage: scripts\predict_submission.ps1 <image_dir> [weights] [output] [conf] [min_area]"
}

$source = $args[0]
$weights = if ($args.Count -ge 2) { $args[1] } else { "runs\detect\runs\train\fire_yolov8n_960\weights\best.pt" }
$output = if ($args.Count -ge 3) { $args[2] } else { "outputs\submission.json" }
$conf = if ($args.Count -ge 4) { $args[3] } else { "0.15" }
$minArea = if ($args.Count -ge 5) { $args[4] } else { "0.0" }

& "C:\ANACONDA2\envs\torch_gpu\python.exe" "src\predict_submit.py" `
  --weights $weights `
  --source $source `
  --output $output `
  --imgsz 960 `
  --conf $conf `
  --min_area $minArea `
  --device 0
exit $LASTEXITCODE
