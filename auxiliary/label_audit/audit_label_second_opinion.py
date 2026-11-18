"""Independent pretrained text OCR cross-check for disputed nonempty labels."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
from PIL import Image
import numpy as np
from paddleocr import TextRecognition
from auxiliary.label_audit.audit_all_labels import norm
out=ROOT/'outputs/full-label-audit'
rows=[json.loads(s) for s in (out/'results.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(rows)==20125
rows=[r for r in rows if r['status']=='mismatch']
m=TextRecognition(model_name='PP-OCRv6_medium_rec',model_dir='C:/Users/saguraaa/.paddlex/official_models/PP-OCRv6_medium_rec',device='gpu:0')
with (out/'second-opinion.jsonl').open('w',encoding='utf-8') as f:
 for start in range(0,len(rows),32):
  batch=rows[start:start+32];imgs=[]
  for r in batch:
   with Image.open(ROOT/'datasets_oriented/yolo/images'/r['split']/r['image']) as im:imgs.append(np.array(im.convert('RGB').crop(r['bbox']))[:,:,::-1].copy())
  for r,p in zip(batch,m.predict(imgs,batch_size=32),strict=True):
   r['text_model']=p['rec_text'];r['text_score']=float(p['rec_score']);r['models_agree']=norm(r['text_model'])==norm(r['prediction']);f.write(json.dumps(r,ensure_ascii=False)+'\n')
  f.flush()
  print(f'{min(start+32,len(rows))}/{len(rows)}',flush=True)
