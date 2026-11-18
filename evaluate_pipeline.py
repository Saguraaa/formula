"""Evaluate a detector/recognizer pair on the fixed page split."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from arithmetic import judge
from predict import DEFAULT_OCR_MODE, FormulaRecognizer
from prepare_data import box_to_yolo

ROOT = Path(__file__).resolve().parent


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - w*h
    return w*h / union if union else 0


def match(predictions, targets, threshold):
    remaining = set(range(len(targets)))
    pairs = {}
    for index in sorted(range(len(predictions)), key=lambda i: -predictions[i]['detection_confidence']):
        if remaining:
            target = max(sorted(remaining), key=lambda j: iou(predictions[index]['bbox'], targets[j]['bbox']))
            if iou(predictions[index]['bbox'], targets[target]['bbox']) >= threshold:
                pairs[target] = index
                remaining.remove(target)
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--rec-dir', type=Path)
    p.add_argument('--ocr-mode', choices=['direct', 'pipeline'], default=DEFAULT_OCR_MODE,
                   help='OCR mode (default: %(default)s)')
    p.add_argument('--name', required=True)
    p.add_argument('--split', choices=['test', 'val'], default='test')
    p.add_argument('--dataset', type=Path, default=ROOT / 'datasets_oriented')
    p.add_argument('--annotations', type=Path, default=ROOT.parent / '实验3/train/annotations.json')
    p.add_argument('--device', default='0')
    p.add_argument('--ocr-device', default='gpu:0')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=.5)
    p.add_argument('--iou', type=float, default=.5)
    p.add_argument('--swanlab-mode', choices=['online', 'offline', 'disabled'], default='online')
    a = p.parse_args()
    import swanlab

    manifest = (a.dataset / 'split_manifest.json').read_bytes()
    splits = json.loads(manifest)['splits']
    names = splits[a.split]
    annotations = a.annotations.read_bytes()
    source = {r['image']: r['boxes'] for r in json.loads(annotations)}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config['manifest_sha256'] = hashlib.sha256(manifest).hexdigest()
    config['annotations_sha256'] = hashlib.sha256(annotations).hexdigest()
    config['detector_sha256'] = hashlib.sha256(a.weights.read_bytes()).hexdigest()
    if a.rec_dir:
        config['recognizer_files_sha256'] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(a.rec_dir.iterdir()) if path.is_file()
        }
    engine = FormulaRecognizer(a.weights, a.device, a.ocr_device, a.rec_dir, ocr_mode=a.ocr_mode)
    engine.predict(a.dataset / 'yolo/images/val' / splits['val'][0], conf=a.conf, imgsz=a.imgsz)
    directory = ROOT / 'reports' / 'results'
    directory.mkdir(parents=True, exist_ok=True)
    counts = dict(targets=0, matched=0, false_positives=0, text_targets=0, exact=0,
                  grade_targets=0, covered=0, grade_correct=0, wrong_targets=0, wrong_accepted=0)
    elapsed = 0
    with (directory / f'{a.name}.jsonl').open('w', encoding='utf-8') as records:
        for page_index, name in enumerate(names, 1):
            start = perf_counter()
            output = engine.predict(a.dataset / 'yolo/images' / a.split / name, conf=a.conf, imgsz=a.imgsz)
            elapsed += perf_counter() - start
            w, h = output['width'], output['height']
            targets = []
            for box in source[name]:
                x, y, bw, bh = box_to_yolo(box, w, h)
                text = box.get('text')
                targets.append(dict(judge(text if isinstance(text, str) else ''),
                                    bbox=[(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]))
            predictions = output['formulas']
            pairs = match(predictions, targets, a.iou)
            counts['targets'] += len(targets)
            counts['matched'] += len(pairs)
            counts['false_positives'] += len(predictions) - len(pairs)
            for index, target in enumerate(targets):
                pred = predictions[pairs[index]] if index in pairs else {}
                if target['normalized_text']:
                    counts['text_targets'] += 1
                    counts['exact'] += pred.get('normalized_text') == target['normalized_text']
                truth = target['is_correct']
                guessed = pred.get('is_correct')
                if truth is not None:
                    counts['grade_targets'] += 1
                    counts['covered'] += guessed is not None
                    counts['grade_correct'] += guessed is truth
                    counts['wrong_targets'] += truth is False
                    counts['wrong_accepted'] += truth is False and guessed is True
            records.write(json.dumps({'image': name, 'targets': targets, 'predictions': predictions,
                                      'matches': pairs}, ensure_ascii=False) + '\n')
            if page_index % 10 == 0 or page_index == len(names):
                print(f'{a.name}: {page_index}/{len(names)} pages', flush=True)
    def ratio(numerator, denominator):
        return counts[numerator] / counts[denominator] if counts[denominator] else None

    metrics = dict(detection_recall=ratio('matched', 'targets'),
                   exact_rate=ratio('exact', 'text_targets'),
                   grading_success=ratio('grade_correct', 'grade_targets'),
                   grading_coverage=ratio('covered', 'grade_targets'),
                   wrong_accept_rate=ratio('wrong_accepted', 'wrong_targets'),
                   ms_per_page=1000 * elapsed / len(names))
    payload = {'config': config, 'counts': counts, 'metrics': metrics}
    path = directory / f'{a.name}.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    run = swanlab.init(project='ModelAdjust', experiment_name=a.name, config=config, mode=a.swanlab_mode)
    try:
        run.log({f'{a.split}/{k}': v for k, v in (counts | metrics).items() if v is not None})
        payload['swanlab_url'] = run.url if a.swanlab_mode == 'online' else None
    finally:
        swanlab.finish()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
