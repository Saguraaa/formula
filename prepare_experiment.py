"""Generate train-only augmented views of the frozen experiment split."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parent
SPLITS = ('train', 'val', 'test')
SEED = 42


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def ocr_rows(source, split):
    return [line.split('\t', 1) for line in
            (source / 'ocr' / f'{split}.txt').read_text(encoding='utf-8').splitlines()]


def audit(source):
    manifest = source / 'split_manifest.json'
    splits = json.loads(manifest.read_text(encoding='utf-8'))['splits']
    seen_images = {}
    seen_crops = set()
    crop_counts = Counter()
    for split in SPLITS:
        image_dir = source / 'yolo' / 'images' / split
        names = {path.name for path in image_dir.iterdir() if path.is_file()}
        if names != set(splits[split]) or len(names) != len(splits[split]):
            raise ValueError(f'{split}: images differ from frozen manifest')
        stems = {Path(name).stem for name in names}
        if len(stems) != len(names):
            raise ValueError(f'{split}: duplicated image stems')
        for name in sorted(names):
            path = image_dir / name
            image = read_rgb(path)
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            previous = seen_images.setdefault(digest, split)
            if previous != split:
                raise ValueError(f'{path}: image content also present in {previous}')
            if not (source / 'yolo' / 'labels' / split / f'{path.stem}.txt').is_file():
                raise FileNotFoundError(f'Missing YOLO label for {path}')
        for relative, label in ocr_rows(source, split):
            path = Path(relative)
            if (path.parts[:2] != ('crops', split) or len(path.parts) != 3
                    or path.stem.rsplit('_', 1)[0] not in stems
                    or not label.strip() or not (source / 'ocr' / path).is_file()
                    or path in seen_crops):
                raise ValueError(f'Invalid {split} OCR row: {relative}')
            seen_crops.add(path)
            crop_counts[split] += 1
    return {
        'split_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'images': {split: len(splits[split]) for split in SPLITS},
        'ocr_crops': dict(crop_counts),
        'cross_split_identical_images': 0,
    }


def save_rgb(path, image):
    Image.fromarray(image).save(path, quality=95)


def prepare_yolo(source, stage, geometric, rng):
    from optimizations.yolo_optimization import augment

    yolo = source / 'yolo'
    output_images = stage / 'images' / 'train'
    output_labels = stage / 'labels' / 'train'
    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)
    records = []
    for path in sorted((yolo / 'images' / 'train').iterdir()):
        label = yolo / 'labels' / 'train' / f'{path.stem}.txt'
        rows = np.asarray([line.split() for line in label.read_text().splitlines()
                           if line.strip()], dtype=np.float64).reshape(-1, 5)
        if np.any(rows[:, 0] != 0):
            raise ValueError(f'{label}: expected formula class 0')
        image_rng = np.random.default_rng(rng.integers(0, 2**63))
        image, boxes = augment(read_rgb(path), rows[:, 1:], image_rng, geometric=geometric)
        boxes = np.asarray(boxes).reshape(-1, 4)
        corners_min = boxes[:, :2] - boxes[:, 2:] / 2
        corners_max = boxes[:, :2] + boxes[:, 2:] / 2
        if (len(boxes) != len(rows) or not np.isfinite(boxes).all()
                or np.any(boxes[:, 2:] <= 0)
                or np.any(corners_min < -1e-5) or np.any(corners_max > 1 + 1e-5)):
            raise ValueError(f'{label}: augmentation dropped a box or produced invalid bounds')
        save_rgb(output_images / path.name, image)
        (output_labels / label.name).write_text(
            ''.join('0 ' + ' '.join(f'{coord:.8f}' for coord in box) + '\n'
                    for box in boxes), encoding='utf-8')
        records.append({'source': str(path), 'derived': f'images/train/{path.name}',
                        'boxes': len(boxes),
                        'geometry_changed': bool(np.any(boxes != rows[:, 1:]))})
    return records


def prepare_ocr(source, stage, rng):
    from optimizations.ocr_optimization import augment

    destination = stage / 'ocr'
    (destination / 'crops' / 'train').mkdir(parents=True)
    rows = ocr_rows(source, 'train')
    for relative, _ in rows:
        original = source / 'ocr' / relative
        save_rgb(destination / relative, augment(read_rgb(original), rng))
    (destination / 'train.txt').write_text(
        ''.join(f'{relative}\t{label}\n' for relative, label in rows), encoding='utf-8')
    return [{'source': str(source / 'ocr' / relative), 'derived': f'ocr/{relative}'}
            for relative, _ in rows]


def generate(source, output, group, audit_result):
    target = output / group
    if target.exists():
        raise FileExistsError(f'{target} already exists. Choose a new --output directory.')
    stage = Path(tempfile.mkdtemp(prefix=f'.{group}-', dir=output))
    rng = np.random.default_rng(SEED)
    if group == 'O2':
        records = prepare_ocr(source, stage, rng)
    else:
        records = prepare_yolo(source, stage, group == 'Y2', rng)
        config = {'path': target.as_posix(), 'train': 'images/train',
                  'val': (source / 'yolo/images/val').as_posix(),
                  'test': (source / 'yolo/images/test').as_posix(),
                  'names': {0: 'formula'}}
        (stage / 'yolo.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    metadata = {'group': group, 'seed': SEED, 'source': str(source),
                'data_audit': audit_result, 'records': records}
    (stage / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    stage.rename(target)
    print(f'{group}: {len(records)} train samples at {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'datasets_oriented')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiment_data')
    parser.add_argument('--groups', nargs='+', choices=('Y1', 'Y2', 'O2'), default=[])
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    result = audit(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    for group in args.groups:
        generate(source, output, group, result)


if __name__ == '__main__':
    main()
