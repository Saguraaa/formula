"""Local worksheet review UI using the best controlled full-page combination."""
import argparse
import io
import json
import os
import threading
import time
import uuid
import warnings
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get('FORMULA_WEIGHTS', ROOT / 'models/yolo/best.pt'))
OCR_DIR = Path(os.environ.get('FORMULA_OCR_DIR', ROOT / 'models/ocr'))
MODEL_INFO = {'detector': 'Y2-geometry-2', 'ocr': 'O3 · 统一重训', 'conf': 0.5, 'imgsz': 640}
COLORS = {'correct': '#168365', 'incorrect': '#dc4545', 'uncertain': '#b7791f'}


def create_app(engine=None, output_dir=None, device='auto'):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=20 * 1024 * 1024, ENGINE=engine)
    output_dir = Path(output_dir or os.environ.get('FORMULA_OUTPUT_DIR', ROOT / 'outputs/web'))
    lock = threading.Lock()

    def get_engine():
        if app.config['ENGINE'] is None:
            for path in (WEIGHTS, *(OCR_DIR / name for name in ('inference.json', 'inference.pdiparams', 'inference.yml'))):
                if not path.is_file():
                    raise FileNotFoundError(f'缺少模型文件：{path}')
            os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
            # Torch must load first on Windows to avoid conflicting Paddle DLLs.
            import torch
            import paddle
            from predict import FormulaRecognizer
            use_torch_gpu = device != 'cpu' and torch.cuda.is_available()
            use_paddle_gpu = device != 'cpu' and paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            app.config['ENGINE'] = FormulaRecognizer(
                weights=WEIGHTS, rec_dir=OCR_DIR, device='0' if use_torch_gpu else 'cpu',
                ocr_device='gpu:0' if use_paddle_gpu else 'cpu', ocr_mode='direct')
        return app.config['ENGINE']

    @app.get('/')
    def home():
        return render_template('index.html', cloud=os.environ.get('FORMULA_DEPLOYMENT') == 'cloud')

    @app.get('/api/health')
    def health():
        return jsonify(model=MODEL_INFO, loaded=app.config['ENGINE'] is not None)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error='图片过大，请上传小于 20 MB 的图片。'), 413

    @app.post('/api/predict')
    def predict_image():
        upload = request.files.get('image')
        if upload is None or not upload.filename:
            return jsonify(error='请选择一张算式图片。'), 400
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(upload.read())) as source:
                    if source.width * source.height > 25_000_000:
                        return jsonify(error='图片超过 2500 万像素，请先缩小图片。'), 400
                    image = ImageOps.exif_transpose(source).convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            return jsonify(error='无法读取图片，请使用有效的 JPG、PNG 或 WEBP 图片。'), 400
        if not lock.acquire(blocking=False):
            return jsonify(error='正在识别另一张图片，请稍后重试。'), 409
        started = time.perf_counter()
        try:
            recognizer = get_engine()
            job_id = uuid.uuid4().hex
            folder = output_dir / job_id
            folder.mkdir(parents=True)
            source_path = folder / 'original.png'
            image.save(source_path)
            result = recognizer.predict(source_path, conf=0.5, imgsz=640)
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            counts = dict(total=len(result['formulas']), correct=0, incorrect=0, uncertain=0)
            line_width = max(2, round(image.width / 550))
            def url(name):
                return f'/results/{job_id}/{name}'
            for row in result['formulas']:
                counts[row['status']] += 1
                name = f'crop_{row["id"]:03d}.png'
                image.crop(row['bbox']).save(folder / name)
                row['crop_url'] = url(name)
                color = COLORS[row['status']]
                x1, y1, x2, y2 = row['bbox']
                draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
                label = str(row['id'])
                font_size = max(14, round(image.width / 65))
                label_width = int(len(label) * font_size * .7 + 8)
                top = max(0, y1 - font_size - 6)
                draw.rectangle((x1, top, x1 + label_width, top + font_size + 6), fill=color)
                draw.text((x1 + 4, top + 2), label, fill='white', font_size=font_size)
            annotated.save(folder / 'annotated.png')
            result.pop('image', None)
            result.update(filename=upload.filename, counts=counts, model=MODEL_INFO,
                          elapsed_seconds=round(time.perf_counter() - started, 2),
                          original_url=url('original.png'), annotated_url=url('annotated.png'),
                          download_url=url('results.zip'))
            (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            with zipfile.ZipFile(folder / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(folder.glob('*.png')):
                    archive.write(path, path.name)
                archive.write(folder / 'result.json', 'result.json')
            return jsonify(result)
        except Exception:
            app.logger.exception('Image recognition failed')
            return jsonify(error='识别失败，请检查启动窗口中的错误信息。可尝试用 --device cpu 启动。'), 500
        finally:
            lock.release()

    @app.get('/results/<job_id>/<name>')
    def result_file(job_id, name):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            return jsonify(error='结果不存在。'), 404
        return send_from_directory(output_dir / job_id, name, as_attachment=name.endswith('.zip'))

    return app


def main():
    parser = argparse.ArgumentParser(description='本地算式识别与批改页面')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    args = parser.parse_args()
    app = create_app(device=args.device)
    print(f'\n打开 http://127.0.0.1:{args.port} · 首次识别将加载模型\n', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
