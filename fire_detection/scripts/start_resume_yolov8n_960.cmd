@echo off
cd /d "%~dp0\.."
if not exist outputs\logs mkdir outputs\logs
start "fire-yolov8n-train" /min cmd.exe /d /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\resume_gpu_yolov8n_960.ps1 > outputs\logs\resume_yolov8n_960.out.log 2> outputs\logs\resume_yolov8n_960.err.log"
