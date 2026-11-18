"""Score text and formula OCR using the same full validation and test splits."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets_oriented/ocr"
OUT = ROOT / "outputs/ocr-full-comparison"

sys.modules["torch"] = None
for folder in ("vendor/PaddleOCR", "vendor/formula_deps", "vendor/ocr_deps"):
    sys.path.insert(0, str(ROOT / folder))

import numpy as np
import paddle
import yaml

from ppocr.data.imaug import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from evaluate_ocr import edit_distance
from auxiliary.fraction_experiments.compare_fraction_recognizers import transcription

CASES = {
    "text-base": ("text", ROOT / "weights/PP-OCRv6_medium_rec_pretrained.pdparams"),
    "text-finetuned": ("text", ROOT / "runs/full-text/iter_epoch_1.pdparams"),
    "formula-base": ("formula", ROOT / "weights/PP-FormulaNet_plus-S_pretrained.pdparams"),
    "formula-finetuned": ("formula", ROOT / "runs/full-formula/iter_epoch_1.pdparams"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=CASES)
    args = parser.parse_args()
    group = args.group
    mode, weights = CASES[group]
    config = yaml.safe_load((ROOT / "runs" / f"full-{mode}" / "train.yaml").read_text())
    options = config["Global"]
    paddle.set_device("gpu:0")
    if mode == "text":
        options["character_dict_path"] = str(ROOT / "vendor/PaddleOCR" / options["character_dict_path"])
    decode = build_post_process(config["PostProcess"], options)
    if mode == "text":
        alphabet = len(decode.character)
        config["Architecture"]["Head"]["out_channels_list"] = {
            "CTCLabelDecode": alphabet, "NRTRLabelDecode": alphabet + 3,
        }
    model = build_model(config["Architecture"])
    state = paddle.load(str(weights))
    params = model.state_dict()
    assert params.keys() == state.keys() and all(params[k].shape == state[k].shape for k in params)
    model.set_state_dict(state)
    model.eval()
    if mode == "formula":
        # Cap generation only, leaving all pretrained parameter shapes intact.
        model.head.max_seq_len = 128

    transforms = [op for op in config["Eval"]["dataset"]["transforms"]
                  if not any(k.endswith("Encode") or k == "KeepKeys" for k in op)]
    transforms.append({"KeepKeys": {"keep_keys": ["image"]}})
    operators = create_operators(transforms, options)
    manifest = json.loads((OUT / "manifest.json").read_text())
    for split in ("val", "test"):
        labels = DATA / f"{split}.txt"
        assert hashlib.sha256(labels.read_bytes()).hexdigest() == manifest[split]["source_sha256"]
        samples = [line.split("\t", 1) for line in labels.read_text(encoding="utf-8").splitlines()]
        rows = []
        for start in range(0, len(samples), 16):
            batch = samples[start:start + 16]
            images = []
            for name, _ in batch:
                path = DATA / name
                images.append(transform({"filename": str(path), "image": path.read_bytes()}, operators)[0])
            with paddle.no_grad():
                outputs = decode(model(paddle.to_tensor(np.stack(images))))
            if mode == "text":
                outputs = [r[0] for r in outputs]
            for (name, label), predicted in zip(batch, outputs, strict=True):
                expected, actual = transcription(label), transcription(predicted)
                rows.append(dict(image=name, label=expected, raw_prediction=predicted, prediction=actual,
                                 correct=actual == expected, edits=edit_distance(expected, actual)))
            if start % 320 == 0:
                print(group, split, len(rows), "/", len(samples), flush=True)
        assert len(rows) == manifest[split]["count"]
        summary = dict(group=group, split=split, count=len(rows), correct=sum(r["correct"] for r in rows),
                       cer=sum(r["edits"] for r in rows) / sum(len(r["label"]) for r in rows),
                       weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                       labels_sha256=manifest[split]["source_sha256"])
        print(summary, flush=True)
        with (OUT / f"{group}-{split}.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (OUT / f"{group}-{split}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
