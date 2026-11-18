"""Evaluate formula checkpoints on the fixed pilot splits with identical preprocessing."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os,sys,json,copy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.modules['torch']=None
for folder in ['vendor/PaddleOCR','vendor/formula_deps','vendor/ocr_deps']:
 sys.path.insert(0,str(ROOT/folder))
import paddle,yaml,numpy as np
from ppocr.modeling.architectures import build_model
from ppocr.data.imaug import create_operators,transform
from ppocr.postprocess import build_post_process
from evaluate_ocr import edit_distance

cfg=yaml.safe_load((ROOT/'runs/F1-formula/train.yaml').read_text())
paddle.set_device('gpu:0')
post=build_post_process(cfg['PostProcess'],cfg['Global'])
transforms=copy.deepcopy(cfg['Eval']['dataset']['transforms'])
transforms=[t for t in transforms if 'UniMERNetLabelEncode' not in t]
transforms[-1]={'KeepKeys':{'keep_keys':['image']}}
ops=create_operators(transforms,cfg['Global'])
def norm(s):return ''.join(s.split()).replace(r'\left','').replace(r'\right','')
results=[]
for group,weights in [('F0',ROOT/'weights/PP-FormulaNet_plus-S_pretrained.pdparams'),('F1',ROOT/'runs/F1-formula/best_accuracy.pdparams'),('F2',ROOT/'runs/F2-formula/best_accuracy.pdparams'),('F1-head',ROOT/'runs/F1-formula-head/best_accuracy.pdparams'),('F2-head',ROOT/'runs/F2-formula-head/best_accuracy.pdparams')]:
 if len(sys.argv)>1 and group not in sys.argv[1:]:continue
 model=build_model(cfg['Architecture']);state=paddle.load(str(weights))
 assert set(model.state_dict())==set(state)
 assert all(tuple(model.state_dict()[k].shape)==tuple(v.shape) for k,v in state.items())
 model.set_state_dict(state);model.eval()
 for split in ['val','test']:
  records=[]
  for line in (ROOT/'datasets_formula'/f'{split}.txt').read_text(encoding='utf-8').splitlines():
   filename,label=line.split('\t',1)
   img=transform({'filename':str(ROOT/'datasets_formula'/filename),'image':(ROOT/'datasets_formula'/filename).read_bytes()},ops)[0]
   with paddle.no_grad():pred=post(model(paddle.to_tensor(np.expand_dims(img,0))))[0]
   records.append(dict(image=filename,label=label,prediction=pred,exact=norm(pred)==norm(label)))
  payload=dict(group=group,split=split,samples=len(records),correct=sum(r['exact'] for r in records),exact_rate=sum(r['exact'] for r in records)/len(records),cer=sum(edit_distance(norm(r['prediction']),norm(r['label'])) for r in records)/sum(len(norm(r['label'])) for r in records),weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),records=records)
  (ROOT/'outputs/formula-experiment'/f'{group}-{split}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');print(group,split,payload['correct'],len(records),flush=True)
  results.append({k:v for k,v in payload.items() if k!='records'})
 del model
all_results=[]
for path in sorted((ROOT/'outputs/formula-experiment').glob('F*-*.json')):
 payload=json.loads(path.read_text(encoding='utf-8'))
 if 'records' in payload:all_results.append({k:v for k,v in payload.items() if k!='records'})
(ROOT/'outputs/formula-experiment/comparison.json').write_text(json.dumps(all_results,indent=2))
