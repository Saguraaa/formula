"""Evaluate the four detector and OCR combinations after both training queues finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'reports/results'


def result(name):
    return json.loads((RESULTS / f'{name}.json').read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swanlab-mode', choices=['online', 'offline', 'disabled'], default='online')
    args = parser.parse_args()

    # The two augmented candidates compete on validation only.
    y_best = max(('Y1', 'Y2'), key=lambda group: result(group)['config']['validation_map50_95'])
    o_best = 'O2'
    for group in ('Y0', y_best, 'O1', o_best):
        record = result(group)
        if args.swanlab_mode == 'online' and not record.get('swanlab_url'):
            raise ValueError(f'Missing online evaluation for {group}')
    combinations = [
        ('Y0-O1', 'Y0', 'O1'),
        ('Ybest-O1', y_best, 'O1'),
        ('Y0-Obest', 'Y0', o_best),
        ('Ybest-Obest', y_best, o_best),
    ]
    for name, detector, recognizer in combinations:
        subprocess.run([
            sys.executable, str(ROOT / 'evaluate_pipeline.py'), '--name', name,
            '--weights', result(detector)['config']['weights'],
            '--rec-dir', result(recognizer)['config']['model_dir'],
            '--swanlab-mode', args.swanlab_mode,
        ], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / 'auxiliary/reporting/build_report.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
