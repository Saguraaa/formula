import argparse
import json
import math
from pathlib import Path

from arithmetic import judge

ROOT = Path(__file__).resolve().parent
DEFAULT_OCR_MODE = 'direct'


def read_bgr(path):
    import cv2
    import numpy as np
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def detect_crops(detector, image, device, conf, imgsz):
    height, width = image.shape[:2]
    boxes = detector.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0].boxes
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        bounds = [max(0, math.floor(x1)), max(0, math.floor(y1)),
                  min(width, math.ceil(x2)), min(height, math.ceil(y2))]
        left, top, right, bottom = bounds
        if right > left and bottom > top:
            detections.append({'bbox': bounds, 'detection_confidence': float(box.conf[0])})
    detections.sort(key=lambda row: (row['bbox'][1], row['bbox'][0]))
    return [dict(row, id=index) for index, row in enumerate(detections, 1)]


class FormulaRecognizer:
    #权重选择
    def __init__(self, weights=ROOT / 'weights/formula_detector.pt', device='cpu', ocr_device='cpu',
                 rec_dir=None, ocr_mode=DEFAULT_OCR_MODE):
        from ultralytics import YOLO
        self.detector = YOLO(str(weights))
        self.ocr_mode = ocr_mode
        if ocr_mode == 'direct':
            from paddleocr import TextRecognition
            self.ocr = TextRecognition(model_name='PP-OCRv6_medium_rec', device=ocr_device,
                                       model_dir=str(rec_dir) if rec_dir else None)
        elif ocr_mode == 'pipeline':
            from paddleocr import PaddleOCR
            options = dict(lang='en', ocr_version='PP-OCRv6', device=ocr_device,
                           use_doc_orientation_classify=False, use_doc_unwarping=False,
                           use_textline_orientation=False)
            if rec_dir:
                options['text_recognition_model_dir'] = str(rec_dir)
            self.ocr = PaddleOCR(**options)
        else:
            raise ValueError(f'Unsupported OCR mode: {ocr_mode}')
        self.device = device

    def _recognize(self, crop):
        outputs = list(self.ocr.predict(crop))
        segments = []
        if outputs:
            item = outputs[0]
            if self.ocr_mode == 'direct':
                height, width = crop.shape[:2]
                if item['rec_text']:
                    segments.append({'text': item['rec_text'], 'confidence': float(item['rec_score']),
                                     'box': [0.0, 0.0, float(width), float(height)]})
            else:
                for text, score, rect in zip(item['rec_texts'], item['rec_scores'], item['rec_boxes']):
                    segments.append({'text': text, 'confidence': float(score), 'box': [float(v) for v in rect]})
        segments.sort(key=lambda s: s['box'][0])
        decision = judge(''.join(s['text'] for s in segments))
        # Fragments on different rows do not have a reliable left-to-right reading order.
        if len(segments) > 1 and max(s['box'][1] for s in segments) >= min(s['box'][3] for s in segments):
            decision = dict(decision, status='uncertain', is_correct=None, reason='multiple_text_lines')
        return dict(decision, ocr_confidence=min((s['confidence'] for s in segments), default=0.0),
                    segments=segments)

    def predict(self, path, conf=0.5, imgsz=640):
        image = read_bgr(path)
        h, w = image.shape[:2]
        formulas = []
        for detection in detect_crops(self.detector, image, self.device, conf, imgsz):
            left, top, right, bottom = detection['bbox']
            decision = self._recognize(image[top:bottom, left:right].copy())
            formulas.append(dict(decision, **detection))
        return {'image': str(Path(path).resolve()), 'width': w, 'height': h, 'formulas': formulas}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('image', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/result.json')
    p.add_argument('--weights', type=Path, default=ROOT / 'weights/formula_detector.pt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--ocr-device', default='cpu')
    p.add_argument('--rec-dir', type=Path)
    p.add_argument('--ocr-mode', choices=['direct', 'pipeline'], default=DEFAULT_OCR_MODE,
                   help='OCR mode (default: %(default)s)')
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--imgsz', type=int, default=640)
    args = p.parse_args()
    engine = FormulaRecognizer(args.weights, args.device, args.ocr_device, args.rec_dir, ocr_mode=args.ocr_mode)
    result = engine.predict(args.image, args.conf, args.imgsz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {len(result["formulas"])} formulas to {args.output}')


if __name__ == '__main__': main()
