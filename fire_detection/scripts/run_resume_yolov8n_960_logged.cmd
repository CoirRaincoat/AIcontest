@echo off
cd /d "%~dp0\.."
if not exist outputs\logs mkdir outputs\logs
echo %date% %time% runner started>> outputs\logs\resume_yolov8n_960.launch.log
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\resume_gpu_yolov8n_960.ps1 > outputs\logs\resume_yolov8n_960.out.log 2> outputs\logs\resume_yolov8n_960.err.log
echo %date% %time% runner exited with %errorlevel%>> outputs\logs\resume_yolov8n_960.launch.log
