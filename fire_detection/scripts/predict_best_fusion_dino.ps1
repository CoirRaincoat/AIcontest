param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [string]$Output = "C:\AI\fire_detection\outputs\submission_best_fusion_dino.json"
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:POLARS_SKIP_CPU_CHECK = "1"
$env:HF_HOME = "C:\AI\hf_cache"
$env:HF_HUB_CACHE = "C:\AI\hf_cache\hub"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:YOLO_CONFIG_DIR = "C:\AI\fire_detection\.ultralytics"
$env:MPLCONFIGDIR = "C:\AI\fire_detection\.matplotlib"

$project = "C:\AI\fire_detection"
$python = "C:\ANACONDA2\envs\siglip_learn\python.exe"
$dinoCheckpoint = "C:\AI\fire_detection\runs_dinov3\dinov3_vitb16_linear_seed42\best_head.pt"

Set-Location $project
& $python "src\predict_best_fusion.py" `
    --source $Source `
    --output $Output `
    --dino-checkpoint $dinoCheckpoint `
    --dino-threshold 0.08
exit $LASTEXITCODE
