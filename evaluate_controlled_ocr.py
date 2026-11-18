"""Evaluate full crops and shared YOLO detections for the controlled OCR runs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from arithmetic import judge
from evaluate_ocr import MODEL, normalize
from evaluate_pipeline import iou, match
from predict import detect_crops, read_bgr
from prepare_data import box_to_yolo
from run_controlled_ocr import ANNOTATIONS, DATA, GROUPS, OCR, OUT, ROOT, load_json, records, save_json, sha

RESULTS = OUT / "evaluation"
DETECTOR = ROOT / "runs/Y2-geometry-2/weights/best.pt"
DETECTION_CONFIG = {"conf": 0.50, "imgsz": 640, "match_iou": 0.50}
FRACTION_PAGES = {"10", "225", "419"}
SCOPES = ("val", "test", "pages", "fraction-crop", "fraction-page", "ordinary-crop", "ordinary-page")


def check_inputs():
    manifest = load_json(OUT / "manifest.json")
    if sha(ANNOTATIONS) != manifest["source_annotations_sha256"] or sha(DATA / "split_manifest.json") != manifest["split_sha256"]:
        raise ValueError("Annotations or page split changed after the training audit")
    for split in ("val", "test"):
        if sha(OUT / f"{split}.txt") != manifest["splits"][split]["normalized_manifest_sha256"]:
            raise ValueError(f"{split} labels changed after the training audit")
    return manifest


def detect(device):
    from ultralytics import YOLO

    check_inputs()
    manifest = DATA / "split_manifest.json"
    pages = load_json(manifest)["splits"]["test"]
    if len(pages) != 88 or "10.jpg" not in pages:
        raise ValueError("Unexpected test pages")
    detector = YOLO(str(DETECTOR))
    output = []
    for index, name in enumerate(pages, 1):
        path = DATA / "yolo/images/test" / name
        image = read_bgr(path)
        height, width = image.shape[:2]
        detections = detect_crops(detector, image, device, DETECTION_CONFIG["conf"], DETECTION_CONFIG["imgsz"])
        output.append(dict(image=name, sha256=sha(path), width=width, height=height, detections=detections))
        if index % 20 == 0 or index == len(pages):
            print(f"YOLO: {index}/{len(pages)} pages", flush=True)
    save_json(RESULTS / "shared-detections.json", {
        "detector_path": str(DETECTOR), "detector_sha256": sha(DETECTOR), "split_sha256": sha(manifest),
        "settings": dict(DETECTION_CONFIG, device=device), "pages": output,
    })


def recognize(recognizer, images):
    if not images:
        return []
    predictions = list(recognizer.predict(images, batch_size=16))
    if len(predictions) != len(images):
        raise RuntimeError(f"TextRecognition returned {len(predictions)} results for {len(images)} crops")
    return [dict(ocr_text=item["rec_text"], ocr_score=float(item["rec_score"])) for item in predictions]


def score(expected_raw, prediction):
    expected = normalize(expected_raw)
    actual = prediction["ocr_text"] if prediction else None
    normalized_actual = normalize(actual) if actual is not None else None
    truth = judge(expected)
    guess = judge(normalized_actual or "")
    return dict(expected_raw=expected_raw, expected=expected, ocr_text=actual,
                ocr_normalized=normalized_actual, ocr_score=prediction["ocr_score"] if prediction else None,
                exact=expected == normalized_actual if expected else None,
                edits=Levenshtein.distance(expected, normalized_actual or "") if expected else None,
                truth_grade=truth["is_correct"], ocr_grade=guess["is_correct"],
                truth_grade_reason=truth.get("reason"),
                ocr_grade_reason=guess.get("reason") if prediction else "no_iou_match")


def evaluate_crops(recognizer, split, annotations, pages_by_stem):
    rows = records(OUT / f"{split}.txt")
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        predictions = recognize(recognizer, [read_bgr(OCR / path) for path, _ in batch])
        for (path, expected), prediction in zip(batch, predictions, strict=True):
            stem, index = Path(path).stem.rsplit("_", 1)
            page = pages_by_stem[stem]
            index = int(index)
            raw = annotations[page][index]["text"]
            if normalize(raw) != expected:
                raise ValueError(f"Crop source-label mismatch: {path}")
            yield dict(crop=path, page=page, box_index=index, **score(raw, prediction))


def candidate_boxes(target, predictions, assigned):
    def rank(item):
        index, candidate = item
        x1, y1, x2, y2 = candidate["bbox"]
        a1, b1, a2, b2 = target
        distance = (x1 + x2 - a1 - a2) ** 2 + (y1 + y2 - b1 - b2) ** 2
        return -iou(target, candidate["bbox"]), distance, -candidate["detection_confidence"], index

    best = sorted(enumerate(predictions), key=rank)[:3]
    return [dict(candidate, iou=iou(target, candidate["bbox"]), assigned_box_index=assigned.get(index))
            for index, candidate in best]


def evaluate_pages(recognizer, cached, annotations):
    for page_index, page in enumerate(cached["pages"], 1):
        name, detections = page["image"], page["detections"]
        path = DATA / "yolo/images/test" / name
        if sha(path) != page["sha256"]:
            raise ValueError(f"Page image changed: {name}")
        image = read_bgr(path)
        height, width = image.shape[:2]
        crops = [image[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in (d["bbox"] for d in detections)]
        predictions = [dict(box, **ocr) for box, ocr in zip(detections, recognize(recognizer, crops), strict=True)]
        targets = []
        for box in annotations[name]:
            x, y, w, h = box_to_yolo(box, width, height)
            raw_text = box.get("text")
            targets.append(dict(bbox=[(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height],
                                text=raw_text if isinstance(raw_text, str) else ""))
        pairs = match(predictions, targets, DETECTION_CONFIG["match_iou"])
        assigned = {pred: target for target, pred in pairs.items()}
        rows = []
        for index, target in enumerate(targets):
            pred = predictions[pairs[index]] if index in pairs else None
            overlap = iou(target["bbox"], pred["bbox"]) if pred else None
            row = dict(page=name, box_index=index, target_bbox=target["bbox"],
                       matched=pred is not None, prediction_id=pred["id"] if pred else None,
                       prediction_bbox=pred["bbox"] if pred else None, match_iou=overlap,
                       **score(target["text"], pred))
            if pred is None:
                candidates = candidate_boxes(target["bbox"], predictions, assigned)
                row["candidates"] = candidates
                row["match_failure"] = ("no_candidates" if not candidates else "below_iou_threshold"
                                        if candidates[0]["iou"] < DETECTION_CONFIG["match_iou"] else "candidate_assigned_elsewhere")
            rows.append(row)
        for j, prediction in enumerate(predictions):
            prediction["assigned_box_index"] = assigned.get(j)
        yield dict(page=name, targets=rows, predictions=predictions)
        if page_index % 20 == 0 or page_index == len(cached["pages"]):
            print(f"OCR pages: {page_index}/{len(cached['pages'])}", flush=True)


def summarize(rows):
    counts = dict(boxes=len(rows), labeled=0, exact=0, edits=0, characters=0, gradeable=0,
                  grade_correct=0, wrong_targets=0, wrong_accepted=0, undetermined=0, truth_ungradable=0)
    if rows and "matched" in rows[0]:
        counts.update(matched=sum(row["matched"] for row in rows),
                      labeled_unmatched=sum(bool(row["expected"]) and not row["matched"] for row in rows))
        counts.update(Counter(row["match_failure"] for row in rows if row["expected"] and not row["matched"]))
    for row in rows:
        if not row["expected"]:
            continue
        counts["labeled"] += 1
        counts["exact"] += row["exact"]
        counts["edits"] += row["edits"]
        counts["characters"] += len(row["expected"])
        truth, guess = row["truth_grade"], row["ocr_grade"]
        if truth is None:
            counts["truth_ungradable"] += 1
        else:
            counts["gradeable"] += 1
            counts["grade_correct"] += guess is truth
            counts["wrong_targets"] += truth is False
            counts["wrong_accepted"] += truth is False and guess is True
            counts["undetermined"] += guess is None
    def rate(a, b):
        return counts[a] / counts[b] if counts[b] else None

    return {"counts": counts, "metrics": dict(exact_rate=rate("exact", "labeled"), cer=rate("edits", "characters"),
                                               grading_accuracy=rate("grade_correct", "gradeable"),
                                               wrong_accept_rate=rate("wrong_accepted", "wrong_targets"))}


def character_errors(rows):
    details = []
    categories, edit_counts = Counter(), Counter()
    for row in rows:
        expected, actual = row["expected"], row["ocr_normalized"]
        if not expected or actual is None or expected == actual:
            continue
        edits = []
        affected = set()
        for op in Levenshtein.editops(expected, actual):
            before = expected[op.src_pos] if op.tag != "insert" else ""
            after = actual[op.dest_pos] if op.tag != "delete" else ""
            tags = {"digit" if char in "0123456789" else "decimal_point" if char == "." else
                    "operator_or_equals" if char in "+-*/=()" else "other" for char in before + after}
            affected.update(tags)
            edit_counts[(op.tag, before, after)] += 1
            edits.append(dict(operation=op.tag, expected_pos=op.src_pos, ocr_pos=op.dest_pos,
                              expected_character=before, ocr_character=after, categories=sorted(tags)))
        categories.update(affected)
        details.append(dict(page=row["page"], box_index=row["box_index"], expected=expected,
                            ocr_text=row["ocr_text"], ocr_normalized=actual, categories=sorted(affected), edits=edits))
    return {"incorrect_with_ocr": len(details), "questions_by_category": dict(categories),
            "common_edits": [dict(operation=k[0], expected=k[1], ocr=k[2], count=n)
                             for k, n in edit_counts.most_common(25)]}, details


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def evaluate(group, device):
    # Paddle and Torch share DLL names on this Windows installation. Load Torch first.
    if os.name == "nt":
        import torch  # noqa: F401
    from paddleocr import TextRecognition

    manifest = check_inputs()
    official = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex")) / "official_models" / MODEL
    model_dir = {"O0": official, "full-text": ROOT / "runs/full-text/inference"}.get(group, OUT / group / "inference")
    model_hashes = {name: sha(model_dir / name) for name in ("inference.json", "inference.pdiparams", "inference.yml")}
    if sha(official / "inference.yml") != model_hashes["inference.yml"]:
        import yaml
        config = yaml.safe_load((model_dir / "inference.yml").read_text(encoding="utf-8"))
        baseline = yaml.safe_load((official / "inference.yml").read_text(encoding="utf-8"))
        for key in ("PreProcess", "PostProcess"):
            if config[key] != baseline[key]:
                raise ValueError(f"{group}: TextRecognition {key} differs from the official model")
    recognizer = TextRecognition(model_name=MODEL, model_dir=str(model_dir), device=device)
    annotations = {page["image"]: page["boxes"] for page in load_json(ANNOTATIONS)}
    splits = load_json(DATA / "split_manifest.json")["splits"]
    cached = load_json(RESULTS / "shared-detections.json")
    if cached["detector_sha256"] != sha(DETECTOR) or cached["split_sha256"] != manifest["split_sha256"] or any(
        cached["settings"][key] != value for key, value in DETECTION_CONFIG.items()
    ) or [page["image"] for page in cached["pages"]] != splits["test"]:
        raise ValueError("Shared detection cache does not match the fixed experiment")
    destination = RESULTS / group
    destination.mkdir(exist_ok=True)
    results = {}
    crop_rows = {}
    for split in ("val", "test"):
        stems = {Path(name).stem: name for name in splits[split]}
        rows = list(evaluate_crops(recognizer, split, annotations, stems))
        if len(rows) != manifest["splits"][split]["count"]:
            raise RuntimeError(f"{split}: incomplete crop evaluation")
        print(f"{group}: {split} {len(rows)} crops", flush=True)
        crop_rows[split] = rows
        results[split] = summarize(rows)
        save_jsonl(destination / f"{split}.jsonl", rows)
    pages = list(evaluate_pages(recognizer, cached, annotations))
    save_jsonl(destination / "pages-predictions.jsonl", pages)
    page_rows = [row for page in pages for row in page["targets"]]
    save_jsonl(destination / "pages.jsonl", page_rows)
    results["pages"] = summarize(page_rows)
    if results["pages"]["counts"]["labeled"] != results["test"]["counts"]["labeled"]:
        raise RuntimeError("Page and crop test annotations disagree")
    results["pages"]["counts"]["false_positive_boxes"] = sum(p["assigned_box_index"] is None for page in pages for p in page["predictions"])
    for mode, rows in (("crop", crop_rows["test"]), ("page", page_rows)):
        fraction = [row for row in rows if row["page"] == "10.jpg" and row["expected"]]
        if len(fraction) != 23:
            raise RuntimeError(f"Expected 23 labeled fractions, got {len(fraction)}")
        results[f"fraction-{mode}"] = summarize(fraction)
        save_jsonl(destination / f"fraction-{mode}.jsonl", fraction)
        ordinary = [row for row in rows if Path(row["page"]).stem not in FRACTION_PAGES]
        results[f"ordinary-{mode}"] = summarize(ordinary)
        breakdown, details = character_errors(ordinary)
        results[f"ordinary-{mode}"]["character_errors"] = breakdown
        save_jsonl(destination / f"ordinary-{mode}-character-errors.jsonl", details)
    for section, rows in (("val", crop_rows["val"]), ("test", crop_rows["test"]), ("pages", page_rows)):
        errors = [row for row in rows if row.get("matched") is False or row["exact"] is False or
                  (row["truth_grade"] is not None and row["ocr_grade"] is not row["truth_grade"])]
        save_jsonl(destination / f"{section}-errors.jsonl", errors)
    save_json(destination / "summary.json", {"group": group, "model_dir": str(model_dir), "model_sha256": model_hashes,
              "ocr_device": device, "detection_cache_sha256": sha(RESULTS / "shared-detections.json"),
              "source_annotations_sha256": manifest["source_annotations_sha256"], "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("detect", "O0", *GROUPS, "full-text"))
    parser.add_argument("--device")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.target == "detect":
        detect(args.device or "0")
    else:
        evaluate(args.target, args.device or "gpu:0")


if __name__ == "__main__":
    main()
