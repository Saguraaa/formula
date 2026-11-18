"""Verify the packaged inference weights before serving the application."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'models/manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise SystemExit(f'Missing or damaged model: {path}')
    print('Model checksums verified: Y2-geometry-2 + controlled O3', flush=True)


if __name__ == '__main__':
    main()
