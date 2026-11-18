"""Re-score fixed test detections against current labels and list false accepts."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import base64
import csv
import hashlib
import html
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from arithmetic import judge

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'reports/results'
OUT = ROOT / 'outputs/wrong-accept-recheck'
SOURCE = ROOT.parent / '实验3/train/annotations.json'
PAGES = ROOT / 'datasets_oriented/yolo/images/test'
SPLIT = ROOT / 'datasets_oriented/split_manifest.json'


def crop_uri(image, bounds):
    crop = image.crop(tuple(round(v) for v in bounds))
    crop.thumbnail((620, 180))
    buffer = io.BytesIO()
    crop.convert('RGB').save(buffer, format='JPEG', quality=90)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output-name', help='Keep this scoring snapshot under a separate name')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    metadata = json.loads((RESULTS / f'{args.run}.json').read_text(encoding='utf-8'))
    pages = [json.loads(line) for line in (RESULTS / f'{args.run}.jsonl').read_text(
        encoding='utf-8').splitlines()]
    labels = {r['image']: r['boxes'] for r in json.loads(SOURCE.read_text(encoding='utf-8'))}
    assert [p['image'] for p in pages] == json.loads(SPLIT.read_text(encoding='utf-8'))['splits']['test']
    assert hashlib.sha256(SPLIT.read_bytes()).hexdigest() == metadata['config']['manifest_sha256']
    confusion, failures, notes = Counter(), [], []
    labeled = exact = changed = missed = matched = 0
    fields = ['image','box_index','label','ocr','label_left','label_right','ocr_left','ocr_right','target_bbox','ocr_bbox']
    for page in pages:
        name = page['image']
        assert len(page['targets']) == len(labels[name])
        pairs = page['matches']
        for i, (box, historical) in enumerate(zip(labels[name], page['targets'], strict=True)):
            target = judge(box.get('text') or '')
            idx = pairs.get(str(i))
            prediction = page['predictions'][idx] if idx is not None else {}
            matched += idx is not None
            changed += target['raw_text'] != historical['raw_text']
            if target['normalized_text']:
                labeled += 1
                exact += prediction.get('normalized_text') == target['normalized_text']
            truth = target['is_correct']
            if truth is None:
                continue
            decision = prediction.get('is_correct')
            state = 'correct' if truth else 'wrong'
            outcome = 'correct' if decision is True else 'wrong' if decision is False else 'unknown'
            confusion[state+'_'+outcome] += 1
            missed += idx is None
            if truth is False and decision is True:
                failures.append(dict(zip(fields, (name,i,target['raw_text'],prediction['raw_text'],
                    target['left_value'],target['right_value'],prediction['left_value'],prediction['right_value'],
                    historical['bbox'],prediction['bbox']), strict=True)))
    confusion = {a+'_'+b:confusion[a+'_'+b] for a in ('correct','wrong') for b in ('correct','wrong','unknown')}
    total = sum(confusion.values())
    wrong = sum(n for k,n in confusion.items() if k.startswith('wrong_'))
    false_accepts = confusion['wrong_correct']
    good = confusion['correct_correct'] + confusion['wrong_wrong']
    assert false_accepts == len(failures)
    report = dict(pages=len(pages),matched=matched,changed_vs_run_labels=changed,
                  labeled=labeled,exact=exact,grade_targets=total,grade_correct=good,
                  grading_success=good/total,wrong_targets=wrong,wrong_accepted=false_accepts,
                  wrong_accept_rate=false_accepts/wrong,missed_grade_targets=missed,confusion=confusion)
    stem = OUT / (args.output_name or args.run)
    stem.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with stem.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
    for r in failures:
        with Image.open(PAGES / r['image']) as image:
            gt, pred = (crop_uri(image,r[key]) for key in ('target_bbox','ocr_bbox'))
        e = lambda k: html.escape(str(r[k]))
        notes.append(f'<article><h2>{e("image")} · 框 {e("box_index")}</h2>'
                     f'<p>标注：<code>{e("label")}</code></p><img src="{gt}" alt="标注裁图">'
                     f'<p>OCR：<code>{e("ocr")}</code></p><img src="{pred}" alt="预测裁图"></article>')
    body = ''.join(notes)
    stem.with_suffix('.html').write_text(
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>错题误判明细</title>'
        '<style>body{max-width:850px;margin:2rem auto;padding:0 1rem;font:16px system-ui}article{border-top:1px solid #bbb;padding:1rem 0}img{max-width:100%}</style>'
        f'<h1>错题被判为正确的明细</h1><p>按当前标注统计，共 {false_accepts}/{wrong} 道。框编号从 0 开始。</p>{body}</html>',
        encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
