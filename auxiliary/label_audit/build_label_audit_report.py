"""Build a local, searchable audit report; never edits source labels."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json,re,csv,html,base64,io,collections
from pathlib import Path
from PIL import Image
from auxiliary.label_audit.audit_all_labels import norm
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'outputs/full-label-audit'
rows=[json.loads(s) for s in (out/'results.jsonl').read_text(encoding='utf-8').splitlines()]
def loose(s):
 return re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',norm(s))
for r in rows:
 if r['status']=='mismatch' and loose(r['label'])==loose(r['prediction']):r['status']='format_only'
reviews={(r['image'],r['box_index']):r for r in json.loads((out/'reviewed.json').read_text(encoding='utf-8'))}
retained_path=out/'confirmed-unchanged.json'
retained={(r['image'],r['box_index']) for r in json.loads(retained_path.read_text(encoding='utf-8'))} if retained_path.exists() else set()
second={(r['image'],r['box_index']):r for r in [json.loads(s) for s in (out/'second-opinion.jsonl').read_text(encoding='utf-8').splitlines()]}
for r in rows:
 key=(r['image'],r['box_index']);r['corrected_label']=reviews.get(key,{}).get('after','');r['text_model']=second.get(key,{}).get('text_model','')
 r['review_status']='confirmed_unchanged' if key in retained else 'confirmed_fixed' if key in reviews else ('models_agree_unreviewed' if r['status']=='mismatch' and second.get(key,{}).get('models_agree') else ('needs_review' if r['status'] in ['mismatch','empty_label'] else 'screened_only'))
counts={split:dict(collections.Counter(r['status'] for r in rows if r['split']==split)) for split in ['train','val','test']}
summary={'checked':len(rows),'expected':json.loads((out/'config.json').read_text())['boxes'],'unique':len({(r['image'],r['box_index']) for r in rows}),'counts':counts,'confirmed_fixed':len(reviews),'review_counts':dict(collections.Counter(r['review_status'] for r in rows)),'note':'Model screening, not ground truth. Mismatches require visual review. Empty labels may be intentionally blank. Mathematical correctness is not a label correction criterion.'}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
with (out/'all-results.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['image','split','box_index','label','prediction','status','text_model','review_status','corrected_label']);w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
suspects=[r for r in rows if r['status'] in ['mismatch','empty_label']]
suspects.sort(key=lambda r:({'confirmed_fixed':0,'confirmed_unchanged':1,'models_agree_unreviewed':2,'needs_review':3}.get(r['review_status'],3),r['image'],r['box_index']))
parts=['<!doctype html><meta charset="utf-8"><title>全数据集标注检查</title><style>body{font:16px system-ui;margin:30px;background:#f5f5f5}article{background:white;padding:16px;margin:12px 0;border-radius:8px}img{max-width:100%;max-height:180px}code{white-space:pre-wrap}input,select{padding:10px} .bar{position:sticky;top:0;background:#eee;padding:10px}</style><h1>全数据集标注检查</h1><p>模型输出不是标准答案。这里列出内容不一致及空标签；仅 confirmed_fixed 项已看图确认并修正，其余未改。原标签为检查开始时的快照。编号从 0 开始。</p>',f'<p>已检查 {len(rows)} / {summary["expected"]} 框；已修正 {len(reviews)} 条；待复核 {len(suspects)-len(reviews)-len(retained)} 条。</p>','<div class="bar"><input id="q" placeholder="搜索图片名、算式、数据划分" oninput="filter()"><select id="s" onchange="filter()"><option value="">全部候选</option><option>confirmed_fixed</option><option>confirmed_unchanged</option><option>models_agree_unreviewed</option><option>needs_review</option></select></div>']
cache_name=None
for r in suspects:
 key=(r['split'],r['image'])
 if key!=cache_name:
  if cache_name is not None:im.close()
  im=Image.open(ROOT/'datasets_oriented/yolo/images'/r['split']/r['image']).convert('RGB');cache_name=key
 crop=im.crop(r['bbox']);crop.thumbnail((1000,220));buf=io.BytesIO();crop.save(buf,format='JPEG',quality=85)
 txt=f"{r['split']} / {r['image']} / 框 {r['box_index']}"
 parts.append(f'<article data-status="{r["review_status"]}"><b>{html.escape(txt)}</b> [{r["status"]} / {r["review_status"]}]<p>原标签：<code>{html.escape(r["label"])}</code><br>模型：<code>{html.escape(r["prediction"])}</code><br>第二 OCR：<code>{html.escape(r["text_model"])}</code><br>已确认修正：<code>{html.escape(r["corrected_label"])}</code></p><img loading="lazy" src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"></article>')
parts.append('<script>function filter(){let q=document.querySelector("#q").value.toLowerCase(),s=document.querySelector("#s").value;document.querySelectorAll("article").forEach(a=>a.hidden=!(a.textContent.toLowerCase().includes(q)&&(!s||a.dataset.status===s)))}</script>')
(out/'review.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False));print('report candidates',len(suspects))
