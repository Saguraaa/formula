"""Add mild, label-preserving geometry to the existing O2 training crops."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import yaml

from prepare_experiment import ROOT, read_rgb, save_rgb


def warp(image, rng, rotation_degrees, perspective_fraction):
    height, width = image.shape[:2]
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    angle = rng.uniform(-rotation_degrees, rotation_degrees)
    rotation = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1)
    moved = cv2.transform(corners[None], rotation)[0]
    moved += rng.uniform(-1, 1, (4, 2)).astype(np.float32) * perspective_fraction * (width - 1, height - 1)
    # Fit the transformed corners on a larger canvas so the formula stays intact.
    lower = np.floor(moved.min(axis=0))
    upper = np.ceil(moved.max(axis=0))
    matrix = cv2.getPerspectiveTransform(corners, (moved - lower).astype(np.float32))
    size = tuple(map(int, upper - lower + 1))
    return cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def main():
    config = yaml.safe_load((ROOT / 'configs/ocr/O3-geometry.yaml').read_text(encoding='utf-8'))
    source = ROOT / 'experiment_data/O2/ocr'
    destination = (ROOT / config['train_data']).parent
    if destination.exists():
        raise FileExistsError(destination)
    labels = (source / 'train.txt').read_bytes()
    rows = [line.split('\t', 1) for line in labels.decode('utf-8').splitlines()]
    reference = (ROOT / 'datasets_oriented/ocr/train.txt').read_text(encoding='utf-8').splitlines()
    if [f'{name}\t{label}' for name, label in rows] != reference:
        raise ValueError('O2 training labels differ from the current OCR split')

    geometry = config['geometry']
    rng = np.random.default_rng(geometry['seed'])
    with tempfile.TemporaryDirectory(prefix='.ocr-geometry-', dir=destination.parent) as temp:
        stage = Path(temp) / destination.name
        crops = stage / 'ocr/crops/train'
        crops.mkdir(parents=True)
        transformed = []
        for name, _ in rows:
            original = source / name
            output = stage / 'ocr' / name
            if rng.random() < geometry['probability']:
                image = read_rgb(original)
                if min(image.shape[:2]) >= 2:
                    save_rgb(output, warp(image, rng, geometry['rotation_degrees'], geometry['perspective_fraction']))
                    transformed.append(name)
                    continue
            shutil.copyfile(original, output)
        (stage / 'ocr/train.txt').write_bytes(labels)
        metadata = {'source': str(source), 'geometry': geometry, 'crops': len(rows),
                    'labels_sha256': hashlib.sha256(labels).hexdigest(), 'transformed': transformed}
        (stage / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        stage.rename(destination)
    print(f'{destination.name}: transformed {len(transformed)}/{len(rows)} training crops')


if __name__ == '__main__':
    main()
