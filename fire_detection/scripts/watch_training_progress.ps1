$ErrorActionPreference = "Stop"

$run = if ($args.Count -ge 1) { $args[0] } else { "runs\detect\runs\train\fire_yolov8n_960" }
$seconds = if ($args.Count -ge 2) { [int]$args[1] } else { 30 }
$results = Join-Path $run "results.csv"
$weights = Join-Path $run "weights"

while ($true) {
  Clear-Host
  Write-Host ("Watching: {0}" -f $run)
  Write-Host ("Time: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  Write-Host ""

  Write-Host "=== GPU ==="
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
  Write-Host ""

  Write-Host "=== Latest Metrics ==="
  if (Test-Path $results) {
    $rows = Import-Csv $results
    if ($rows.Count -gt 0) {
      $last = $rows[-1]
      Write-Host ("Epoch: {0}" -f $last.epoch)
      Write-Host ("Precision: {0}" -f $last.'metrics/precision(B)')
      Write-Host ("Recall: {0}" -f $last.'metrics/recall(B)')
      Write-Host ("mAP50: {0}" -f $last.'metrics/mAP50(B)')
      Write-Host ("mAP50-95: {0}" -f $last.'metrics/mAP50-95(B)')
      Write-Host ("Train box/cls/dfl: {0} / {1} / {2}" -f $last.'train/box_loss', $last.'train/cls_loss', $last.'train/dfl_loss')
      Write-Host ("Val box/cls/dfl: {0} / {1} / {2}" -f $last.'val/box_loss', $last.'val/cls_loss', $last.'val/dfl_loss')
    } else {
      Write-Host "No epochs written yet."
    }
  } else {
    Write-Host "No results.csv found: $results"
  }
  Write-Host ""

  Write-Host "=== Weights ==="
  if (Test-Path $weights) {
    Get-ChildItem $weights | Sort-Object LastWriteTime -Descending | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
  } else {
    Write-Host "No weights directory found."
  }
  Write-Host ""
  Write-Host ("Refreshing every {0}s. Press Ctrl+C to stop watching; training will keep running." -f $seconds)
  Start-Sleep -Seconds $seconds
}
