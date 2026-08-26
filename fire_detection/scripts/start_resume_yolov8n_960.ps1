$ErrorActionPreference = "Stop"

$root = (Get-Location).Path
$resumeScript = Join-Path $root "scripts\resume_gpu_yolov8n_960.ps1"
$logDir = Join-Path $root "outputs\logs"
$outLog = Join-Path $logDir "resume_yolov8n_960.out.log"
$errLog = Join-Path $logDir "resume_yolov8n_960.err.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$command = "& '$resumeScript' 1>> '$outLog' 2>> '$errLog'"

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = "powershell.exe"
$psi.Arguments = '-NoProfile -ExecutionPolicy Bypass -Command "' + $command + '"'
$psi.WorkingDirectory = $root
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true

$pathValue = [System.Environment]::GetEnvironmentVariable("Path", "Process")
if (-not $pathValue) {
  $pathValue = [System.Environment]::GetEnvironmentVariable("PATH", "Process")
}
if ($pathValue) {
  $psi.Environment.Clear()
  $psi.Environment["Path"] = $pathValue
  $psi.Environment["SystemRoot"] = if ($env:SystemRoot) { $env:SystemRoot } else { "C:\Windows" }
  $psi.Environment["WINDIR"] = if ($env:WINDIR) { $env:WINDIR } else { "C:\Windows" }
  $psi.Environment["TEMP"] = if ($env:TEMP) { $env:TEMP } else { $root }
  $psi.Environment["TMP"] = if ($env:TMP) { $env:TMP } else { $root }
  $psi.Environment["USERNAME"] = $env:USERNAME
  $psi.Environment["USERPROFILE"] = $env:USERPROFILE
}

$process = [System.Diagnostics.Process]::Start($psi)
[PSCustomObject]@{
  Id = $process.Id
  ProcessName = $process.ProcessName
  Started = $process.StartTime
  OutLog = $outLog
  ErrLog = $errLog
}
