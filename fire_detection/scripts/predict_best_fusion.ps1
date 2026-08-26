param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [string]$Output = "C:\AI\fire_detection\outputs\submission_best_fusion.json"
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:POLARS_SKIP_CPU_CHECK = "1"
$env:YOLO_CONFIG_DIR = "C:\AI\fire_detection\.ultralytics"
$env:MPLCONFIGDIR = "C:\AI\fire_detection\.matplotlib"

$project = "C:\AI\fire_detection"
$python = "C:\ANACONDA2\envs\siglip_learn\python.exe"

Set-Location $project
& $python "src\predict_best_fusion.py" --source $Source --output $Output
exit $LASTEXITCODE
