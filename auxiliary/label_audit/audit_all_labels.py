"""Read-only, resumable audit of all original annotation boxes."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os,sys,json,re,hashlib,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['PADDLE_PDX_CACHE_HOME']=str(ROOT/'weights/paddlex')
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
sys.path.insert(0,str(ROOT/'vendor/formula_deps'))
from evaluate_ocr import normalize
from prepare_data import box_to_yolo
from PIL import Image

def norm(s):
 for a,b in [(r'\times','*'),(r'\div','/'),(r'\cdot','*'),(r'\left',''),(r'\right',''),(r'\,',''),(r'\!','')]:s=s.replace(a,b)
 s=re.sub(r'\\(?:d?frac)\{([0-9.]+)\}\{([0-9.]+)\}',r'\1/\2',s)
 return normalize(s).replace('{','').replace('}','')

def main():
 from paddleocr import FormulaRecognition
 import numpy as np
 out=ROOT/'outputs/full-label-audit';out.mkdir(exist_ok=True)
 source=ROOT.parent/'实验3/train/annotations.json'
 data=source.read_bytes(); records=json.loads(data)
 manifest=json.loads((ROOT/'datasets_oriented/split_manifest.json').read_text(encoding='utf-8'))
 splits={name:s for s,names in manifest['splits'].items() for name in names}
 config={'source_sha256':hashlib.sha256(data).hexdigest(),'model':'PP-FormulaNet_plus-S','boxes':sum(len(p['boxes']) for p in records),'pages':len(records)}
 cfg=out/'config.json'
 if cfg.exists():assert json.loads(cfg.read_text())==config,'Source changed; use a new audit directory.'
 else:cfg.write_text(json.dumps(config,indent=2))
 path=out/'results.jsonl';done=set()
 if path.exists():
  for line in path.read_text(encoding='utf-8').splitlines():
   r=json.loads(line);done.add((r['image'],r['box_index']))
 model=FormulaRecognition(model_name=config['model'],device='gpu:0')
 start=time.time()
 with path.open('a',encoding='utf-8') as f:
  for page in records:
   if (out/'stop').exists(): return
   name=page['image'];split=splits[name]
   if all((name,i) in done for i in range(len(page['boxes']))):continue
   with Image.open(ROOT/'datasets_oriented/yolo/images'/split/name) as im:
    im=im.convert('RGB');w,h=im.size;batch=[];meta=[]
    for i,b in enumerate(page['boxes']):
     if (name,i) in done:continue
     x,y,bw,bh=box_to_yolo(b,w,h)
     bounds=[max(0,math.floor((x-bw/2)*w)),max(0,math.floor((y-bh/2)*h)),min(w,math.ceil((x+bw/2)*w)),min(h,math.ceil((y+bh/2)*h))]
     batch.append(np.array(im.crop(bounds))[:,:,::-1].copy())
     meta.append(dict(image=name,split=split,box_index=i,bbox=bounds,label=b.get('text') or ''))
    for result,r in zip(model.predict(batch,batch_size=8),meta,strict=True):
     pred=result['rec_formula'];r['prediction']=pred
     r['status']='empty_label' if not r['label'].strip() else ('match' if norm(pred)==norm(r['label']) else 'mismatch')
     f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();done.add((name,r['box_index']))
   print(f'{len(done)}/{config["boxes"]} elapsed={time.time()-start:.0f}s page={name}',flush=True)
 print('AUDIT COMPLETE',flush=True)
if __name__=='__main__':main()
