$ErrorActionPreference = "Stop"

& "C:\ANACONDA2\envs\torch_gpu\python.exe" "src\make_image_level_gt.py" `
  --labels data\images\train\train_image.json `
  --image_dir data\val\images `
  --output outputs\val_gt.json
