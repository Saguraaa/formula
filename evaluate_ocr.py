import argparse
import json
import unicodedata
from pathlib import Path

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets_oriented/ocr"
MODEL = "PP-OCRv6_medium_rec"


def normalize(text):
    return "".join(unicodedata.normalize("NFKC", text).split()).translate(
        str.maketrans({"×": "*", "∗": "*", "÷": "/", "−": "-", "–": "-"})
    )


def edit_distance(left, right):
    row = list(range(len(right) + 1))
    for index, a in enumerate(left, 1):
        next_row = [index]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


def evaluate(model_dir, device, split="test", limit=None):
    from paddleocr import TextRecognition

    recognizer = TextRecognition(model_name=MODEL, model_dir=model_dir, device=device)
    records = [line.split("\t", 1) for line in (DATA / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("No labeled OCR crops.")
    correct = edits = characters = 0
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        images = []
        for name, _ in batch:
            with Image.open(DATA / name) as image:
                images.append(np.array(image.convert("RGB"))[:, :, ::-1].copy())
        results = recognizer.predict(images, batch_size=16)
        for (_, label), result in zip(batch, results, strict=True):
            expected = normalize(label)
            predicted = normalize(result["rec_text"])
            correct += expected == predicted
            edits += edit_distance(expected, predicted)
            characters += len(expected)
    return {
        "metrics": {"exact_rate": correct / len(records), "cer": edits / characters},
        "counts": {"samples": len(records), "correct": correct, "edits": edits, "characters": characters},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["O0", "O1", "O2", "O2-geometry-control", "O3-geometry"], required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--swanlab-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    model_dir = args.model_dir
    if args.group != "O0" and model_dir is None:
        model_dir = ROOT / "runs" / args.group / "inference"
    if model_dir is not None and not model_dir.is_dir():
        parser.error(f"Missing exported model: {model_dir}")
    config = {"group": args.group, "model": MODEL, "model_dir": str(model_dir) if model_dir else None,
              "split": args.split, "device": args.device, "limit": args.limit}
    import hashlib
    config["labels_sha256"] = hashlib.sha256((DATA / f"{args.split}.txt").read_bytes()).hexdigest()
    config["split_manifest_sha256"] = hashlib.sha256((ROOT / "datasets_oriented/split_manifest.json").read_bytes()).hexdigest()
    if model_dir is not None:
        config["model_files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sorted(model_dir.iterdir()) if p.is_file()}
        training_record = model_dir.parent / "run.json"
        if training_record.is_file():
            config["training"] = json.loads(training_record.read_text(encoding="utf-8"))
    result = evaluate(str(model_dir) if model_dir else None, args.device, args.split, args.limit)
    result["config"] = config
    import swanlab
    with swanlab.init(project="ModelAdjust", experiment_name=f"{args.group}-{args.split}",
                      config=config, mode=args.swanlab_mode) as run:
        run.log(result["metrics"])
        result["swanlab_url"] = run.url if args.swanlab_mode == "online" else None
    suffix = "" if args.split == "test" else "-val"
    if args.limit is not None:
        suffix += "-smoke"
    output = ROOT / "reports/results" / f"{args.group}{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

