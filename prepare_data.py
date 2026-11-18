import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT.parent / '实验3/train')
    p.add_argument('--output', type=Path, default=ROOT / 'datasets_oriented')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a new --output to preserve splits.')
    records = json.loads((args.source / 'annotations.json').read_text(encoding='utf-8-sig'))
    groups = {}
    size_mismatches = []
    for record in records:
        path = args.source / 'images' / record['image']
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            if image.size != (record['original_width'], record['original_height']):
                raise ValueError(f'Unresolved image size mismatch: {path}')
            digest = hashlib.sha256(image.convert('RGB').tobytes() + str(image.size).encode()).hexdigest()
        groups.setdefault(digest, []).append(record)
    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    n = len(keys)
    splits = {'train': keys[:int(n*.7)], 'val': keys[int(n*.7):int(n*.85)], 'test': keys[int(n*.85):]}
    manifest = {'seed': 42, 'grouping': 'identical decoded pixels', 'splits': {}, 'empty_text_count': 0, 'size_mismatches': size_mismatches}
    for split, hashes in splits.items():
        selected = [r for h in hashes for r in groups[h]]
        manifest['splits'][split] = [r['image'] for r in selected]
        image_dir = args.output / 'yolo/images' / split
        label_dir = args.output / 'yolo/labels' / split
        crop_dir = args.output / 'ocr/crops' / split
        for directory in (image_dir, label_dir, crop_dir): directory.mkdir(parents=True, exist_ok=True)
        ocr_lines = []
        for record in selected:
            path = args.source / 'images' / record['image']
            labels = []
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.save(image_dir / path.name)
                w, h = image.size
                for i, box in enumerate(record['boxes']):
                    cx, cy, bw, bh = box_to_yolo(box, w, h)
                    labels.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
                    text = box.get('text', '')
                    if not isinstance(text, str) or not text.strip():
                        manifest['empty_text_count'] += 1
                        continue
                    text = ' '.join(text.split())
                    filename = f'{path.stem}_{i:04d}.jpg'
                    bounds = (max(0, math.floor((cx-bw/2)*w)), max(0, math.floor((cy-bh/2)*h)),
                              min(w, math.ceil((cx+bw/2)*w)), min(h, math.ceil((cy+bh/2)*h)))
                    image.crop(bounds).save(crop_dir / filename)
                    ocr_lines.append(f'crops/{split}/{filename}\t{text}')
            (label_dir / (path.stem + '.txt')).write_text('\n'.join(labels), encoding='utf-8')
        (args.output / 'ocr' / f'{split}.txt').write_text('\n'.join(ocr_lines), encoding='utf-8')
    (args.output / 'yolo/data.yaml').write_text('path: ' + json.dumps((args.output / 'yolo').resolve().as_posix()) + '\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: formula\n', encoding='utf-8')
    (args.output / 'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print({k: len(v) for k, v in manifest['splits'].items()})
    print('Empty OCR texts:', manifest['empty_text_count'])

def box_to_yolo(box, image_width, image_height):
    """百分比旋转框 → 四个角 → 水平外接框 → YOLO 归一化坐标。"""
    values = [float(value) for value in (
        image_width, image_height, box["x"], box["y"],
        box["width"], box["height"], box.get("rotation", 0),
    )]
    image_width, image_height, x, y, w, h, rotation = values
    x, y = x / 100 * image_width, y / 100 * image_height
    w, h = w / 100 * image_width, h / 100 * image_height
    theta = math.radians(rotation)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = [
        (x, y),
        (x + w * cos_t, y + w * sin_t),
        (x + w * cos_t - h * sin_t, y + w * sin_t + h * cos_t),
        (x - h * sin_t, y + h * cos_t),
    ]
    x_min = max(0, min(point[0] for point in corners))
    y_min = max(0, min(point[1] for point in corners))
    x_max = min(image_width, max(point[0] for point in corners))
    y_max = min(image_height, max(point[1] for point in corners))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("框裁剪到图片范围后为空，请检查原始标注")
    return (
        (x_min + x_max) / 2 / image_width,
        (y_min + y_max) / 2 / image_height,
        (x_max - x_min) / image_width,
        (y_max - y_min) / image_height,
    )

if __name__ == '__main__': main()
