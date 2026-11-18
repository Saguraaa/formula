import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os,sys,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['PADDLE_PDX_CACHE_HOME']=str(ROOT/'weights/paddlex')
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
sys.path.insert(0,str(ROOT/'vendor/formula_deps'))
from PIL import Image
from paddleocr import FormulaRecognition
from evaluate_ocr import normalize

def norm(s):
    for a,b in [(r'\times','*'),(r'\div','/'),(r'\cdot','*'),(r'\left',''),(r'\right',''),(r'\,',''),(r'\!','')]: s=s.replace(a,b)
    s=re.sub(r'\\(?:d?frac)\{([0-9.]+)\}\{([0-9.]+)\}', r'\1/\2', s)
    return normalize(s).replace('{','').replace('}','')

out=ROOT/'outputs/formula-recheck'
cases=json.loads((out/'cases.json').read_text(encoding='utf-8'))
model=FormulaRecognition(model_name='PP-FormulaNet_plus-S',device='gpu:0')
with (out/'results.jsonl').open('w',encoding='utf-8') as f:
 for i,c in enumerate(cases):
    crop=out/f'{i:03d}.png'
    with Image.open(ROOT/'datasets_oriented/yolo/images/test'/c['image']) as im: im.convert('RGB').crop(c['bbox']).save(crop)
    r=list(model.predict(str(crop)))[0]
    c.update(crop=str(crop),new=r['rec_formula'])
    c['matches_label']=norm(c['new'])==norm(c['expected'])
    f.write(json.dumps(c,ensure_ascii=False)+'\n');f.flush()
    print(f"{i+1}/{len(cases)} match={c['matches_label']} new={c['new']}",flush=True)
summary={'model':'PP-FormulaNet_plus-S','cases':len(cases),'matches_label':sum(c['matches_label'] for c in cases),'source':'Y2-geometry-O2','scope':'previously detected but text mismatched; identical detector crops; no label edits'}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(summary)
