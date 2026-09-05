import sys,yaml,json
from ultralytics import YOLO, settings
args=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
try: settings.update({'sync':False,'hub':False})
except Exception: pass
m=YOLO(args.pop('model'))
resume=args.pop('resume',False)
m.train(resume=resume, **args)
print('TRAIN_CHILD_DONE')
