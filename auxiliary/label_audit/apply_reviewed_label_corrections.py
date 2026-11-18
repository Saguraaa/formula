"""Apply only visually reviewed corrections with exact preconditions and backups."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json,hashlib,copy
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=root/'outputs/full-label-audit';changes=json.loads((out/'reviewed.json').read_text(encoding='utf-8'))
source=root.parent/'实验3/train/annotations.json';raw=source.read_bytes();a=json.loads(raw);original=copy.deepcopy(a)
manifest=json.loads((root/'datasets_oriented/split_manifest.json').read_text(encoding='utf-8'));splits={n:s for s,ns in manifest['splits'].items() for n in ns}
paths=[root/'datasets_oriented/ocr'/f'{s}.txt' for s in ['train','val','test']]+[root/'experiment_data/O2/ocr/train.txt']
contents={p:p.read_bytes() for p in paths};lines={p:contents[p].decode('utf-8').splitlines() for p in paths};counts={str(p):0 for p in paths}
assert len({(c['image'],c['box_index']) for c in changes})==len(changes)
for c in changes:
 page=next(p for p in a if p['image']==c['image']);b=page['boxes'][c['box_index']];assert b['text']==c['before'];b['text']=c['after']
 split=splits[c['image']];key=f"crops/{split}/{Path(c['image']).stem}_{c['box_index']:04d}.jpg"
 targets=[root/'datasets_oriented/ocr'/f'{split}.txt']
 if split=='train':targets.append(root/'experiment_data/O2/ocr/train.txt')
 for p in targets:
  idx=[i for i,line in enumerate(lines[p]) if line.split('\t',1)[0]==key];assert len(idx)==1,(p,key)
  i=idx[0];assert lines[p][i].split('\t',1)[1]==' '.join(c['before'].split()),(p,key)
  lines[p][i]=key+'\t'+c['after'];counts[str(p)]+=1
backup=root/'reports/label-corrections-20261113';backup.mkdir(exist_ok=False)
(backup/'annotations.json.before').write_bytes(raw)
for i,p in enumerate(paths):(backup/f'{i}-{p.name}.before').write_bytes(contents[p])
source.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
for p in paths:
 if counts[str(p)]:p.write_text('\n'.join(lines[p])+'\n',encoding='utf-8')
check=json.loads(source.read_text(encoding='utf-8'))
for c in changes:
 b=next(p for p in check if p['image']==c['image'])['boxes'][c['box_index']];assert b['text']==c['after'];b['text']=c['before']
assert check==original
for p in paths:assert p.read_text(encoding='utf-8').splitlines()==lines[p]
record={'changes':changes,'updated_label_counts':counts,'source_sha256_before':hashlib.sha256(raw).hexdigest(),'source_sha256_after':hashlib.sha256(source.read_bytes()).hexdigest(),'note':'Only visually confirmed text edits. Geometry, split assignments, existing trained weights and historical evaluation records unchanged.'}
(backup/'changes.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified corrections:',len(changes));print(counts)
