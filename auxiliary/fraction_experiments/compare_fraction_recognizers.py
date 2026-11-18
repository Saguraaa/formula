"""Compare text OCR and formula OCR on the fixed fraction pilot."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import csv
import json
import os
import re

import numpy as np
from PIL import Image

from evaluate_ocr import edit_distance, normalize
from train_ocr import MODEL, ROOT
from auxiliary.fraction_experiments.train_ocr_fraction_pilot import FRACTION

OUT = ROOT / "outputs/formula-experiment"
DATA = ROOT / "datasets_formula"


def transcription(text):
    text = FRACTION.sub(r"\1/\2", text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac"))
    for token, replacement in ((r"\div", "/"), (r"\times", "*"), (r"\cdot", "*")):
        text = text.replace(token, replacement)
    return normalize(text)


def main():
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition

    rows = []
    comparisons = []
    for group in ("OCR-F0", "OCR-F1", "OCR-F2", "F0", "F1-head", "F2-head"):
        text_ocr = group.startswith("OCR")
        if text_ocr:
            model = TextRecognition(model_name=MODEL, model_dir=ROOT / "runs" / group / "inference",
                                    device="gpu:0")
        for split, count in (("val", 24), ("test", 23)):
            samples = [line.split("\t", 1) for line in (DATA / f"{split}.txt").read_text(
                encoding="utf-8").splitlines()]
            if text_ocr:
                images = []
                for name, _ in samples:
                    with Image.open(DATA / name) as image:
                        images.append(np.array(image.convert("RGB"))[:, :, ::-1].copy())
                predictions = (result["rec_text"] for result in model.predict(images, batch_size=4))
            else:
                records = json.loads((OUT / f"{group}-{split}.json").read_text(encoding="utf-8"))["records"]
                assert [[r["image"], r["label"]] for r in records] == samples
                predictions = (r["prediction"] for r in records)
            correct = edits = characters = 0
            for (name, label), prediction in zip(samples, predictions, strict=True):
                expected, actual = transcription(label), transcription(prediction)
                exact = expected == actual
                correct += exact
                edits += edit_distance(expected, actual)
                characters += len(expected)
                rows.append(dict(model=group, split=split, image=name, expected=expected,
                                 raw_prediction=prediction, prediction=actual, correct=exact))
            assert len(samples) == count
            comparison = dict(model=group, split=split, count=count, correct=correct,
                              exact_rate=correct/count, cer=edits/characters)
            comparisons.append(comparison)
            print(comparison, flush=True)
        if text_ocr:
            del model

    (OUT / "ocr-vs-formula.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    with (OUT / "ocr-vs-formula.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
