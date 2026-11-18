import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=['Y0', 'Y1', 'Y2', 'Y2-geometry'])
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets_oriented/yolo/data.yaml')
    parser.add_argument('--device', default='0')
    parser.add_argument('--swanlab-mode', choices=['online', 'offline', 'disabled'], default='online')
    args = parser.parse_args()

    from ultralytics import YOLO
    import swanlab

    weights = args.weights.resolve(strict=True)
    data = args.data.resolve(strict=True)
    split_manifest = data.parent.parent / 'split_manifest.json'
    training_results = weights.parent.parent / 'results.csv'
    with training_results.open(encoding='utf-8-sig', newline='') as source:
        rows = list(csv.DictReader(source))
    best_validation = max(rows, key=lambda row: float(row['metrics/mAP50-95(B)']))
    training_log = ROOT / f'{args.group}-train.log'
    training_url = None
    if training_log.is_file():
        match = re.search(r'https://swanlab\.cn/[^\s]+/runs/[^\s]+', training_log.read_text(encoding='utf-8-sig'))
        if match:
            training_url = match.group()
    provenance = {
        'group': args.group, 'split': 'test', 'weights': str(weights),
        'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'data': str(data), 'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
        'split_manifest_sha256': hashlib.sha256(split_manifest.read_bytes()).hexdigest() if split_manifest.is_file() else None,
        'imgsz': 640, 'conf': 0.001, 'iou': 0.7, 'max_det': 300,
        'validation_map50_95': float(best_validation['metrics/mAP50-95(B)']),
        'training_swanlab_url': training_url,
    }
    run = swanlab.init(project='ModelAdjust', experiment_name=f'{args.group}-test',
                       mode=args.swanlab_mode, config=provenance)
    try:
        validation = YOLO(str(weights)).val(
            data=str(data), split='test', device=args.device, imgsz=640, workers=0,
            conf=0.001, iou=0.7, max_det=300, plots=False,
            project=str(ROOT / 'runs/eval'), name=args.group,
        )
        box = validation.box
        metrics = {'precision': float(box.mp), 'recall': float(box.mr),
                   'map50': float(box.map50), 'map50_95': float(box.map)}
        output = ROOT / 'reports/results' / f'{args.group}.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        result = {**metrics, 'group': args.group, 'split': 'test', 'metrics': metrics,
                  'config': provenance, 'swanlab_url': run.url if args.swanlab_mode == 'online' else None}
        output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        run.log({f'test/{key}': value for key, value in metrics.items()})
    finally:
        swanlab.finish()


if __name__ == '__main__':
    main()
