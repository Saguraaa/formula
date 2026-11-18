"""Data preparation and formal TextRecognition evaluation for the OCR weight edits."""
import argparse
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

import evaluate_controlled_ocr as controlled
from evaluate_controlled_ocr import (
    ANNOTATIONS, DATA, DETECTOR, DETECTION_CONFIG, MODEL, OCR, ROOT,
    character_errors, evaluate_crops, evaluate_pages, load_json, normalize,
    read_bgr, recognize, records, save_json, save_jsonl, sha, summarize,
)
from run_controlled_ocr import COUNTS, SOURCE, check_crops

OUT = ROOT / "runs/ocr-weight-edit-v1"
O3 = controlled.OUT / "O3"
BASE = O3 / "inference"
BEST = O3 / "best_accuracy.pdparams"
SEED = 42
STEPS = (0, 50, 100, 200)
GROUPS = {"E1": (0.0, 0.0), "E2-lambda-0.1": (0.1, 0.01), "E2-lambda-1.0": (1.0, 0.01)}
MODEL_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
CONFIG = dict(seed=SEED, optimizer="Adam", learning_rate=1e-4, batch_size=16,
              edit_per_step=8, retain_per_step=8, rank=8, maximum_update_steps=200,
              validation_steps=list(STEPS), temperature=1, epsilon=1e-12,
              candidates=GROUPS, device="gpu:0", swanlab_mode="offline")


def read_jsonl(path):
    import json
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def annotations_and_split():
    annotations = {page["image"]: page["boxes"] for page in load_json(ANNOTATIONS)}
    return annotations, load_json(DATA / "split_manifest.json")["splits"]


def model_hashes(directory):
    return {name: sha(directory / name) for name in MODEL_FILES}


def recognizer(directory):
    if os.name == "nt":
        import torch  # noqa: F401
    from paddleocr import TextRecognition

    return TextRecognition(model_name=MODEL, model_dir=str(directory), device=CONFIG["device"])


def evaluate_split(rec, split, annotations, splits):
    stems = {Path(page).stem: page for page in splits[split]}
    rows = list(evaluate_crops(rec, split, annotations, stems))
    if len(rows) != COUNTS[split] or sum(bool(row["expected"]) for row in rows) != COUNTS[split]:
        raise ValueError(f"Incomplete {split} evaluation")
    return rows


def verify_labels(split, annotations, splits, baseline):
    path = controlled.OUT / f"{split}.txt"
    exported = dict(records(path))
    original = dict(records(OCR / f"{split}.txt"))
    source = {}
    for page in splits[split]:
        for index, box in enumerate(annotations[page]):
            text = box.get("text")
            if isinstance(text, str) and text.strip():
                crop = f"crops/{split}/{Path(page).stem}_{index:04d}.jpg"
                if crop in source:
                    raise ValueError(f"Duplicate source crop {crop}")
                source[crop] = normalize(text)
    differences = [
        dict(crop=name, source=source.get(name), baseline=exported.get(name), original=original.get(name))
        for name in sorted(source.keys() | exported.keys() | original.keys())
        if name not in original or source.get(name) != exported.get(name) or
        source.get(name) != normalize(original[name])
    ]
    if differences:
        save_json(OUT / f"{split}-label-disagreements.json", differences)
        raise ValueError(f"{split}: {len(differences)} source/export label disagreements")
    if len(source) != COUNTS[split] or sha(path) != baseline["splits"][split]["normalized_manifest_sha256"]:
        raise ValueError(f"Invalid controlled {split} manifest")
    check_crops(OCR, split, source.keys())


def fraction_types(label):
    # A simple slash in an ordinary division expression is not by itself a fraction.
    if "/" in label:
        if "(" in label or label.count("/") > 1 or "/" in label.partition("=")[2]:
            return "fraction-like"
        return "division"
    return "no-slash"


def strata(label):
    length = len(label)
    return "|".join([("<=8" if length <= 8 else "<=12" if length <= 12 else ">12"),
                     fraction_types(label), "decimal" if "." in label else "no-decimal",
                     "digits:" + "".join(sorted(set(label) & set("0123456789"))),
                     "operators:" + "".join(sorted(set(label) & set("+-*/=()")))])


def choose_retain(rows):
    groups = defaultdict(list)
    for row in rows:
        row["stratum"] = strata(row["expected"])
        groups[row["stratum"]].append(row)
    rng = random.Random(SEED)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    selected = []
    while keys and len(selected) < 2048:
        remaining = []
        for key in keys:
            selected.append(groups[key].pop())
            if len(selected) == 2048:
                break
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return selected


def write_schedule(edit, retain):
    def draws(rows, seed):
        rng = random.Random(seed)
        while True:
            names = sorted(row["crop"] for row in rows)
            rng.shuffle(names)
            yield from names

    edit_draws, retain_draws = draws(edit, SEED), draws(retain, SEED + 1)
    schedule = [dict(step=step, edit=[next(edit_draws) for _ in range(8)],
                     retain=[next(retain_draws) for _ in range(8)]) for step in range(1, 201)]
    save_jsonl(OUT / "sampling-sequence.jsonl", schedule)


def probe_inference(rec, crops):
    predictor = rec.paddlex_predictor
    images = [read_bgr(OCR / crop) for crop in crops]
    expected = recognize(rec, images)
    raw = predictor.pre_tfs["Read"](imgs=images)
    resized = predictor.pre_tfs["ReisizeNorm"](imgs=raw)
    x = predictor.pre_tfs["ToBatch"](imgs=resized)[0]
    (probabilities,) = predictor.runner(x=[x])
    widths = [min(3200, math.ceil(48 * (im.shape[1] / im.shape[0])), item.shape[-1])
              for im, item in zip(raw, resized, strict=True)]
    np.savez(OUT / "parity-probe.npz", input=x, probabilities=probabilities, widths=widths)
    save_json(OUT / "parity-probe.json", dict(crops=crops, predictions=expected,
              batch_width=x.shape[-1], actual_widths=widths))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "config.json", CONFIG)
    baseline = load_json(controlled.OUT / "manifest.json")
    previous = load_json(controlled.RESULTS / "O3/summary.json")
    actual_hashes = model_hashes(BASE)
    if actual_hashes != previous["model_sha256"]:
        raise ValueError("O3 inference files differ from the audited evaluation")
    export = yaml.safe_load((O3 / "export.yaml").read_text(encoding="utf-8"))
    if Path(export["Global"]["pretrained_model"]).resolve() != BEST.resolve():
        raise ValueError("O3 export does not reference the selected training checkpoint")
    dictionary = SOURCE / export["Global"]["character_dict_path"]
    sources = {"annotations": ANNOTATIONS, "split": DATA / "split_manifest.json",
               "dictionary": dictionary, "best_training_weight": BEST, "best_training_states": O3 / "best_accuracy.states",
               "epoch_3_training_weight": O3 / "iter_epoch_3.pdparams",
               "actual_train_config": O3 / "config.yml", "train_config": O3 / "train.yaml",
               "export_config": O3 / "export.yaml", "train_labels": controlled.OUT / "train.txt",
               "validation_labels": controlled.OUT / "val.txt"}
    hashes = {key: dict(path=str(path), sha256=sha(path)) for key, path in sources.items()}
    if (hashes["annotations"]["sha256"] != baseline["source_annotations_sha256"] or
            hashes["split"]["sha256"] != baseline["split_sha256"] or
            hashes["dictionary"]["sha256"] != baseline["dictionary_sha256"] or
            hashes["best_training_weight"]["sha256"] != hashes["epoch_3_training_weight"]["sha256"]):
        raise ValueError("Source hashes disagree with prior O3 metadata")
    annotations, splits = annotations_and_split()
    sets = [set(pages) for pages in splits.values()]
    if sum(map(len, sets)) != len(set.union(*sets)):
        raise ValueError("Page split overlap")
    for split in ("train", "val"):
        verify_labels(split, annotations, splits, baseline)
    save_json(OUT / "provenance.json", dict(sources=hashes, inference=actual_hashes, o3_best_epoch=3))

    rec = recognizer(BASE)
    rows = evaluate_split(rec, "val", annotations, splits)
    summary = summarize(rows)
    old = read_jsonl(controlled.RESULTS / "O3/val.jsonl")
    if summary != previous["results"]["val"] or [(r["crop"], r["ocr_text"]) for r in rows] != [
            (r["crop"], r["ocr_text"]) for r in old]:
        save_jsonl(OUT / "E0-validation-reproduction-mismatch.jsonl", rows)
        raise ValueError("O3 validation does not reproduce historical predictions")
    directory = OUT / "E0"
    directory.mkdir(exist_ok=True)
    save_json(directory / "val-summary.json", summary)
    save_jsonl(directory / "val.jsonl", rows)
    print(f"O3 validation reproduced: {summary['counts']}", flush=True)

    stems = {Path(page).stem: page for page in splits["train"]}
    inventory = []
    audited_hashes = dict(records(controlled.OUT / "O1/train-image-sha256.tsv"))
    records_train = records(controlled.OUT / "train.txt")
    for start in range(0, len(records_train), 16):
        batch = records_train[start:start + 16]
        images = [read_bgr(OCR / name) for name, _ in batch]
        predictions = recognize(rec, images)
        for (name, label), prediction, image in zip(batch, predictions, images, strict=True):
            stem, box = Path(name).stem.rsplit("_", 1)
            digest = sha(OCR / name)
            if digest != audited_hashes[name] or not name.startswith("crops/train/"):
                raise ValueError(f"Invalid original train crop: {name}")
            height, width = image.shape[:2]
            inventory.append(dict(crop=name, page=stems[stem], box_index=int(box), expected=label, sha256=digest,
                                  ocr_normalized=normalize(prediction["ocr_text"]), width=width, height=height,
                                  **prediction))
        if (start // 16 + 1) % 100 == 0:
            print(f"Original train baseline: {len(inventory)}/{len(records_train)}", flush=True)
    edit = [r for r in inventory if r["ocr_normalized"] != r["expected"]]
    correct = [r for r in inventory if r["ocr_normalized"] == r["expected"]]
    retain = choose_retain(correct)
    if not edit or not retain or len(inventory) != COUNTS["train"] or (
        {r["crop"] for r in edit} & {r["crop"] for r in retain}
    ):
        raise ValueError("Invalid edit/retain eligibility")
    for name, values in (("train-baseline", inventory), ("edit", edit), ("retain", retain)):
        save_jsonl(OUT / f"{name}.jsonl", values)
    write_schedule(edit, retain)
    wide = sorted(inventory, key=lambda r: r["width"] / r["height"], reverse=True)
    probe_inference(rec, [r["crop"] for r in edit[:8] + retain[:6] + wide[:2]])
    provenance = load_json(OUT / "provenance.json")
    provenance["data_preparation"] = dict(edit=len(edit), retain=len(retain), correct_pool=len(correct),
        retain_strata=dict(Counter(r["stratum"] for r in retain)),
        retain_rule="One disjoint composite stratum per label: length bucket, slash form, decimal flag, "
                    "digit set, operator set. Shuffle strata and members with seed 42, then round-robin without replacement. "
                    "Fraction-like: slash with a parenthesis, multiple slashes, or slash after equals. "
                    "A remaining single slash is classified as division.",
        sequence_rule="Each pool gets its own deterministic shuffled cycles (edit seed 42, retain seed 43, "
                      "derived by adding 1). Each cycle includes every pool item once.",
        sha256={name: sha(OUT / name) for name in ("train-baseline.jsonl", "edit.jsonl",
                                                   "retain.jsonl", "sampling-sequence.jsonl", "parity-probe.npz")})
    save_json(OUT / "provenance.json", provenance)
    audit_prepared_data()
    print(f"Prepared {len(edit)} edit and {len(retain)} retain crops", flush=True)



def audit_prepared_data():
    import re
    from arithmetic import judge

    validation = [dict(crop=name, sha256=sha(OCR / name))
                  for name, _ in records(controlled.OUT / "val.txt")]
    save_jsonl(OUT / "validation-image-sha256.jsonl", validation)
    candidates = []
    grammar_limits = []
    for role in ("edit", "retain"):
        for row in read_jsonl(OUT / f"{role}.jsonl"):
            label = row["expected"]
            reason = judge(label).get("reason")
            if not reason:
                continue
            data = {key: row[key] for key in ("crop", "page", "box_index", "expected", "ocr_text", "sha256")}
            item = dict(data, dataset=role, judge_reason=reason)
            if re.search(r"[0-9],[0-9]{3}(?![0-9])", label) or label.endswith("="):
                grammar_limits.append(item)
            else:
                candidates.append(item)
    save_json(OUT / "train-label-review.json", dict(
        criteria="Review leads only: syntactically unusual or incomplete transcriptions in selected subsets. "
                 "Unanswered text and conventional comma-grouped numbers are listed separately. "
                 "The parser and OCR output do not establish what the student wrote. All labels are kept verbatim after normalization.",
        candidates=candidates, parser_or_unanswered=grammar_limits))
    provenance = load_json(OUT / "provenance.json")
    paths = {"config": OUT / "config.json", "validation_images": OUT / "validation-image-sha256.jsonl",
             "label_review": OUT / "train-label-review.json"}
    provenance["additional_audit"] = {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}
    save_json(OUT / "provenance.json", provenance)


def export_checkpoint(candidate, step):
    import copy
    from train_ocr import launch

    directory = OUT / candidate / f"step-{step:03}"
    weights = directory / "merged.pdparams"
    meta = load_json(directory / "checkpoint.json")
    if sha(weights) != meta["weights_sha256"]:
        raise ValueError(f"Merged checkpoint changed: {weights}")
    cfg = copy.deepcopy(yaml.safe_load((O3 / "export.yaml").read_text(encoding="utf-8")))
    cfg["Global"].update(pretrained_model=weights.as_posix(),
                         save_model_dir=directory.as_posix(),
                         save_inference_dir=(directory / "inference").as_posix())
    path = directory / "export.yaml"
    if (directory / "inference").exists():
        raise FileExistsError(directory / "inference")
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    launch(SOURCE, "export_model.py", path, directory / "export.log")


def verify_export(candidate, step, rec):
    directory = OUT / candidate / f"step-{step:03}"
    cfg = yaml.safe_load((directory / "inference/inference.yml").read_text(encoding="utf-8"))
    original = yaml.safe_load((BASE / "inference.yml").read_text(encoding="utf-8"))
    if any(cfg[key] != original[key] for key in ("PreProcess", "PostProcess")):
        raise ValueError("Edited export changed preprocessing or decoding")
    meta = load_json(directory / "checkpoint.json")
    probe = load_json(OUT / "parity-probe.json")
    exported = recognize(rec, [read_bgr(OCR / crop) for crop in probe["crops"]])
    if [row["ocr_text"] for row in exported] != meta["probe_decodes"]:
        raise ValueError(f"TextRecognition decode differs from merged dynamic model: {candidate} {step}")
    if step == 0 and [row["ocr_text"] for row in exported] != [row["ocr_text"] for row in probe["predictions"]]:
        raise ValueError("Zero update export differs from E0")
    inputs = np.load(OUT / "parity-probe.npz")["input"]
    (actual,) = rec.paddlex_predictor.runner(x=[inputs])
    expected = np.load(directory / "parity-probabilities.npy")
    if actual.shape != expected.shape:
        raise ValueError(f"Export probability shape mismatch: {actual.shape} vs {expected.shape}")
    error = np.abs(actual - expected)
    if error.max() > 0.075 or error.mean() > 1e-5:
        raise ValueError(f"Export probability drift exceeds tolerance: max={error.max()}, mean={error.mean()}")
    return dict(decode_exact=True, checked_crops=len(exported),
                probability_max_abs=float(error.max()), probability_mean_abs=float(error.mean()),
                probability_argmax_disagreements=int((actual.argmax(-1) != expected.argmax(-1)).sum()),
                sha256=model_hashes(directory / "inference"))


def validate_checkpoint(candidate, step, smoke=False):
    directory = OUT / candidate / f"step-{step:03}"
    rec = recognizer(directory / "inference")
    parity = verify_export(candidate, step, rec)
    save_json(directory / "export-verification.json", parity)
    if smoke:
        print(f"Smoke export verified: {candidate} {step} {parity}", flush=True)
        return
    annotations, splits = annotations_and_split()
    rows = evaluate_split(rec, "val", annotations, splits)
    save_jsonl(directory / "val.jsonl", rows)
    summary = summarize(rows)
    if step == 0:
        baseline = read_jsonl(OUT / "E0/val.jsonl")
        if [(r["crop"], r["ocr_text"]) for r in rows] != [(r["crop"], r["ocr_text"]) for r in baseline]:
            raise ValueError("Step 0 does not reproduce E0 on the complete validation set")
    save_json(directory / "val-summary.json", summary)
    print(f"{candidate} step {step}: {summary['counts']}", flush=True)


def rank_candidate(point):
    counts = point["val"]["counts"]
    # The character denominator is fixed across all validation runs.
    return (-counts["exact"], counts["wrong_accepted"], counts["edits"], point["step"], point["lambda_kl"])


def lock_selection():
    target = OUT / "selection-lock.json"
    if target.exists():
        raise FileExistsError(target)
    baseline = load_json(OUT / "E0/val-summary.json")["counts"]
    rows = []
    for candidate, (lam, mu) in GROUPS.items():
        directory = OUT / candidate
        if load_json(directory / "completed.json")["optimizer_updates"] != 200:
            raise ValueError(f"Incomplete training: {directory}")
        for step in STEPS:
            checkpoint = directory / f"step-{step:03}"
            summary = load_json(checkpoint / "val-summary.json")
            counts = summary["counts"]
            if (counts["characters"], counts["labeled"]) != (baseline["characters"], baseline["labeled"]):
                raise ValueError("Candidate evaluated on a different validation set")
            checks = dict(exact_increased=counts["exact"] > baseline["exact"],
                          wrong_accepts_not_increased=counts["wrong_accepted"] <= baseline["wrong_accepted"],
                          cer_not_increased=counts["edits"] <= baseline["edits"])
            hashes = load_json(checkpoint / "export-verification.json")["sha256"]
            if model_hashes(checkpoint / "inference") != hashes:
                raise ValueError(f"Model changed after validation: {checkpoint}")
            rows.append(dict(candidate=candidate, step=step, val=summary, criteria=checks,
                             accepted=all(checks.values()), lambda_kl=lam, mu=mu,
                             relative_weight_delta=load_json(checkpoint / "checkpoint.json")["relative_weight_delta"],
                             inference_sha256=hashes,
                             val_sha256=sha(checkpoint / "val.jsonl")))
    initializations = [load_json(OUT / name / "step-000/checkpoint.json")["factor_hashes"]
                       for name in (*GROUPS, "smoke")]
    usage = [load_json(OUT / name / "usage.json") for name in GROUPS]
    if any(item != initializations[0] for item in initializations) or any(item != usage[0] for item in usage):
        raise ValueError("Groups used differing initializations or samples")

    selected, exploratory = {}, {}
    for family in ("E1", "E2"):
        options = [point for point in rows if point["candidate"].startswith(family) and point["step"] > 0]
        eligible = [point for point in options if point["accepted"]]
        selected[family] = min(eligible, key=rank_candidate) if eligible else None
        # Test one locked edited representative even if the family has no accepted edit.
        exploratory[family] = selected[family] or min(options, key=rank_candidate)
    accepted = [point for point in selected.values() if point]
    lock = dict(validation_points=rows, selected=selected, final_test_models=exploratory,
                recommended=min(accepted, key=rank_candidate) if accepted else "E0",
                baseline_hashes=model_hashes(BASE), fallback="If a family has no accepted edit, test its highest "
                "ranked nonzero checkpoint for diagnostic comparison only. E0 remains eligible for recommendation.",
                recorded_before_test=True)
    save_json(target, lock)
    print(f"Locked models: {[(k, v['candidate'], v['step'], v['accepted']) for k, v in exploratory.items()]}", flush=True)


def test_family(family):
    lock_path = OUT / "selection-lock.json"
    lock = load_json(lock_path)
    provenance = load_json(OUT / "provenance.json")
    if sha(ANNOTATIONS) != provenance["sources"]["annotations"]["sha256"] or (
        sha(DATA / "split_manifest.json") != provenance["sources"]["split"]["sha256"]
    ):
        raise ValueError("Source annotations or fixed page split changed")
    point = lock["final_test_models"].get(family)
    model_dir = OUT / point["candidate"] / f"step-{point['step']:03}" / "inference" if point else BASE
    hashes = point["inference_sha256"] if point else lock["baseline_hashes"]
    if model_hashes(model_dir) != hashes:
        raise ValueError("Model changed after the selection lock")
    baseline = load_json(controlled.OUT / "manifest.json")
    annotations, splits = annotations_and_split()
    verify_labels("test", annotations, splits, baseline)
    cache = controlled.RESULTS / "shared-detections.json"
    cached = load_json(cache)
    historical = load_json(controlled.RESULTS / "O3/summary.json")
    if (cached["detector_sha256"] != sha(DETECTOR) or cached["split_sha256"] != baseline["split_sha256"] or
            sha(cache) != historical["detection_cache_sha256"] or
            any(cached["settings"][key] != value for key, value in DETECTION_CONFIG.items()) or
            [page["image"] for page in cached["pages"]] != splits["test"] or len(splits["test"]) != 88):
        raise ValueError("Detection cache differs from the fixed YOLO experiment")
    directory = OUT / "final" / family
    directory.mkdir(parents=True, exist_ok=True)
    rec = recognizer(model_dir)
    val_rows = evaluate_split(rec, "val", annotations, splits)
    results = {"val": summarize(val_rows)}
    expected_val = load_json(OUT / "E0/val-summary.json") if point is None else point["val"]
    val_file = OUT / "E0/val.jsonl" if point is None else model_dir.parent / "val.jsonl"
    if results["val"] != expected_val or [(r["crop"], r["ocr_text"]) for r in val_rows] != [
        (r["crop"], r["ocr_text"]) for r in read_jsonl(val_file)
    ]:
        raise ValueError("Final validation re-evaluation differs from selection")
    crop_rows = evaluate_split(rec, "test", annotations, splits)
    pages = list(evaluate_pages(rec, cached, annotations))
    if [page["page"] for page in pages] != splits["test"]:
        raise ValueError("Missing or out-of-order test pages")
    page_rows = [row for page in pages for row in page["targets"]]
    save_jsonl(directory / "pages-predictions.jsonl", pages)
    save_jsonl(directory / "val.jsonl", val_rows)
    save_jsonl(directory / "test.jsonl", crop_rows)
    save_jsonl(directory / "pages.jsonl", page_rows)
    results.update(test=summarize(crop_rows), pages=summarize(page_rows))
    results["pages"]["counts"]["false_positive_boxes"] = sum(
        p["assigned_box_index"] is None for page in pages for p in page["predictions"])
    if results["pages"]["counts"]["labeled"] != COUNTS["test"]:
        raise ValueError("Page annotations and test crop labels disagree")
    for mode, rows in (("crop", crop_rows), ("page", page_rows)):
        fractions = [row for row in rows if row["page"] == "10.jpg" and row["expected"]]
        if len(fractions) != 23:
            raise ValueError(f"Expected 23 labeled fractions, got {len(fractions)}")
        save_jsonl(directory / f"fraction-{mode}.jsonl", fractions)
        results[f"fraction-{mode}"] = summarize(fractions)
        ordinary = [row for row in rows if Path(row["page"]).stem not in controlled.FRACTION_PAGES]
        results[f"ordinary-{mode}"] = summarize(ordinary)
        breakdown, details = character_errors(ordinary)
        results[f"ordinary-{mode}"]["character_errors"] = breakdown
        save_jsonl(directory / f"ordinary-{mode}-character-errors.jsonl", details)
    if family == "E0" and results != historical["results"]:
        raise ValueError("E0 did not reproduce the historical OCR metrics after locking model selection")
    save_json(directory / "summary.json", dict(family=family, selected=point, model_dir=str(model_dir),
        selection_lock_sha256=sha(lock_path), model_sha256=hashes, detector_sha256=sha(DETECTOR),
        detection_cache_sha256=sha(cache), test_labels_sha256=sha(controlled.OUT / "test.txt"),
        test_crop_sha256_manifest=save_test_hashes(crop_rows), results=results))
    print(f"Final {family}: {results['test']['counts']}", flush=True)


def save_test_hashes(rows):
    path = OUT / "final/test-image-sha256.jsonl"
    hashes = [dict(crop=row["crop"], sha256=sha(OCR / row["crop"])) for row in rows]
    if path.exists():
        if read_jsonl(path) != hashes:
            raise ValueError("Test crop bytes changed during evaluation")
    else:
        save_jsonl(path, hashes)
    return sha(path)


def track_swanlab(summaries):
    import socket
    import swanlab

    run_dir = OUT / "swanlab"
    run_dir.mkdir(exist_ok=True)
    status = []
    lock = load_json(OUT / "selection-lock.json")
    try:
        socket.getaddrinfo("api.swanlab.cn", 443)
        offline_reason = "local-first tracking; synchronization not attempted"
    except OSError as exc:
        offline_reason = f"api.swanlab.cn DNS unavailable: {exc}"

    def scalars(prefix, summary):
        return {f"{prefix}/{k}": v for k, v in (summary["counts"] | summary["metrics"]).items()
                if v is not None}

    for name in ("E0", *GROUPS):
        status_path = run_dir / f"{name}.json"
        if status_path.exists():
            status.append(load_json(status_path))
            continue
        with swanlab.init(project="ModelAdjust", name=f"ocr-weight-edit-v1-{name}", group="ocr-weight-edit-v1",
                          mode="offline", log_dir=str(run_dir), config=CONFIG | {"candidate": name},
                          settings=swanlab.Settings(interactive=False)) as run:
            validations = {point["step"]: point["val"] for point in lock["validation_points"]
                           if point["candidate"] == name}
            if name == "E0":
                run.log(scalars("validation", summaries[name]["results"]["val"]), step=0)
                last_step = 1
            else:
                run.log(scalars("validation", validations.pop(0)), step=0)
                for row in read_jsonl(OUT / name / "training.jsonl"):
                    step = row.pop("step")
                    metrics = {"train/" + k: v for k, v in row.items()}
                    if step in validations:
                        metrics |= scalars("validation", validations[step])
                    run.log(metrics, step=step)
                last_step = 201
            final = [summary for summary in summaries.values()
                     if (summary["selected"]["candidate"] if summary["selected"] else "E0") == name]
            if final:
                run.log({key: value for scope, summary in final[0]["results"].items()
                         for key, value in scalars(f"final/{scope}", summary).items()}, step=last_step)
            item = dict(run=name, status="offline-saved", synchronized=False, run_id=run.id,
                        path=str(run.dir), url=None, offline_reason=offline_reason)
        save_json(status_path, item)
        status.append(item)
    save_json(OUT / "swanlab-status.json", status)



def paired_results():
    scopes = ("val", "test", "pages", "fraction-crop", "fraction-page", "ordinary-crop", "ordinary-page")
    totals = {}
    for scope in scopes:
        totals[scope] = {}
        source_scope = "test" if scope == "ordinary-crop" else "pages" if scope == "ordinary-page" else scope
        all_rows = {family: read_jsonl(OUT / "final" / family / f"{source_scope}.jsonl")
                    for family in ("E0", "E1", "E2")}
        if scope.startswith("ordinary"):
            all_rows = {family: [row for row in rows if Path(row["page"]).stem not in controlled.FRACTION_PAGES]
                        for family, rows in all_rows.items()}
        for family in ("E1", "E2"):
            counts = Counter()
            paired = []
            for before, after in zip(all_rows["E0"], all_rows[family], strict=True):
                if any(before.get(key) != after.get(key) for key in (
                    "crop", "page", "box_index", "expected", "truth_grade",
                    "matched", "prediction_id", "prediction_bbox", "target_bbox", "match_iou",
                )):
                    raise ValueError(f"Inconsistent paired evidence in {scope}")
                exact_change = {
                    (False, True): "repair", (True, False): "regression",
                    (False, False): "both_wrong", (True, True): "both_correct",
                    (None, None): "unlabeled",
                }[(before["exact"], after["exact"])]
                grade_changed = before["ocr_grade"] is not after["ocr_grade"]
                new_wrong_accept = before["truth_grade"] is False and after["ocr_grade"] is True and (
                    before["ocr_grade"] is not True)
                removed_wrong_accept = before["truth_grade"] is False and before["ocr_grade"] is True and (
                    after["ocr_grade"] is not True)
                new_false_reject = before["truth_grade"] is True and before["ocr_grade"] is True and (
                    after["ocr_grade"] is False)
                new_undetermined_on_correct = before["truth_grade"] is True and before["ocr_grade"] is True and (
                    after["ocr_grade"] is None)
                counts.update([exact_change])
                counts["grade_changed"] += grade_changed
                counts["new_wrong_accept"] += new_wrong_accept
                counts["removed_wrong_accept"] += removed_wrong_accept
                counts["new_false_reject"] += new_false_reject
                counts["new_undetermined_on_correct"] += new_undetermined_on_correct
                if exact_change in ("repair", "regression", "both_wrong") or grade_changed or (
                    before.get("matched") is False or scope.startswith("fraction")
                ):
                    paired.append(dict(status=exact_change, grade_changed=grade_changed,
                        new_wrong_accept=new_wrong_accept, removed_wrong_accept=removed_wrong_accept,
                        new_false_reject=new_false_reject, new_undetermined_on_correct=new_undetermined_on_correct,
                        E0=before, edited=after))
            counts["total"] = len(all_rows["E0"])
            for key in ("repair", "regression", "both_wrong"):
                counts.setdefault(key, 0)
            totals[scope][family] = dict(counts)
            destination = OUT / "final" / f"{scope}-{family}-paired.jsonl"
            save_jsonl(destination, paired)
    return totals


def build_details():
    import html

    def esc(value):
        return html.escape(str(value))

    def evidence(row):
        if row.get("matched") is False:
            candidates = "".join(
                f"<li>#{c['id']} {esc(c['bbox'])} · IoU {c['iou']:.6f} · "
                f"<code>{esc(c['ocr_text'])}</code> · assigned={esc(c['assigned_box_index'])}</li>"
                for c in row["candidates"])
            return f"匹配失败 · {esc(row['match_failure'])}<ol>{candidates or '<li>无候选框</li>'}</ol>"
        geometry = ""
        if row.get("matched"):
            geometry = f"<br>#{row['prediction_id']} {esc(row['prediction_bbox'])} · IoU {row['match_iou']:.6f}"
        return f"<code>{esc(row['ocr_text'])}</code><br>判题 {esc(row['ocr_grade'])}{geometry}"

    parts = ['<!doctype html><meta charset="utf-8"><title>OCR 权重编辑逐题对照</title>',
             '<style>body{font:16px system-ui;margin:2rem;color:#203040}table{border-collapse:collapse;width:100%}'
             'td,th{padding:.45rem;border:1px solid #ccd3da;text-align:left;vertical-align:top}'
             'th{background:#edf2f8}code{white-space:pre-wrap}</style>',
             '<h1>OCR 权重编辑逐题对照</h1><p>题号是原标注的零基下标。'
             '前后均错误的题保留两次实际识别，未匹配的整页题展示检测候选。</p>']
    labels = {"repair": "修复", "regression": "退步", "both_wrong": "前后均错",
              "both_correct": "前后均对", "unlabeled": "无文本标注"}
    for scope in ("val", "test", "pages", "fraction-crop", "fraction-page"):
        for family in ("E1", "E2"):
            rows = read_jsonl(OUT / "final" / f"{scope}-{family}-paired.jsonl")
            parts.append(f"<h2>{scope} · {family}</h2><table><tr><th>页与题号</th><th>标注文本</th>"
                         f"<th>E0</th><th>{family}</th><th>变化</th></tr>")
            for row in rows:
                before, after = row["E0"], row["edited"]
                flags = [labels[row["status"]]]
                for key, label in (("new_wrong_accept", "新增错题判对"), ("removed_wrong_accept", "消除错题判对"),
                                   ("new_false_reject", "新增正确题判错"),
                                   ("new_undetermined_on_correct", "原判对的正确题变成无法判定"),
                                   ("grade_changed", "判题变化")):
                    if row[key]:
                        flags.append(label)
                parts.append(f"<tr><td>{esc(before['page'])} #{before['box_index']}</td>"
                             f"<td><code>{esc(before['expected_raw'])}</code><br>判题 {esc(before['truth_grade'])}</td>"
                             f"<td>{evidence(before)}</td><td>{evidence(after)}</td>"
                             f"<td>{' · '.join(flags)}</td></tr>")
            parts.append("</table>")
    (OUT / "paired-details.html").write_text("\n".join(parts), encoding="utf-8")


def render_report(summaries, paired):
    lock = load_json(OUT / "selection-lock.json")
    data = load_json(OUT / "provenance.json")["data_preparation"]
    smoke = load_json(OUT / "smoke/parity.json")
    architecture = load_json(OUT / "E1/architecture.json")
    sample_uses = load_json(OUT / "E1/usage.json")["counts"]
    edit_used = [sample_uses[row["crop"]] for row in read_jsonl(OUT / "edit.jsonl")]
    retain_used = [sample_uses[row["crop"]] for row in read_jsonl(OUT / "retain.jsonl")]
    report = [
        "# OCR 权重编辑实验",
        "",
        "本次以已选出的 O3 第 3 轮训练权重为 E0，比较局部低秩适配 E1 与带保持约束的 E2。"
        "这是单次固定种子实验，当前测试数字属于已有测试集上的后续探索。",
        "",
        f"编辑矩阵为 {architecture['parameter']}，形状 {architecture['shape']}，"
        f"秩 {architecture['rank']}，可训练因子参数 {architecture['trainable_parameters']:,} 个。"
        "前级是 LightSVTR 特征编码器，主支路的 conv_reduce 与旁路 skip_conv 将骨干通道投影到 192 维，"
        "主支路经过注意力和归一化后与旁路相加，再送入 CTC 分类器。"
        f"骨干、特征投影、CTC 偏置、NRTR 以及运行统计均冻结。"
        f"同一批 {smoke['crops']} 张原图上，零增量概率最大差 {smoke['zero_delta_max_abs']:.3g}，"
        f"训练权重动态图与现有导出概率最大差 {smoke['export_vs_dynamic_max_abs']:.3g}，"
        f"全概率张量平均绝对差 {smoke['export_vs_dynamic']['mean_abs']:.3g}，时间步 argmax 与解码逐条一致。"
        "编辑时复用了导出过程的卷积重参数化，单矩阵合并回原始训练检查点后使用原有入口导出。"
        "Windows 沿用 CPU WarpCTC，通过实测等价的一阶梯度传递避免框架的重复反传故障。",
        "",
        f"原训练固定宽 320，训练态池化将时间维压到 40。导出使用动态宽和推理态池化。"
        f"探针实际批宽 {smoke['exported_width']}，在推理态固定宽度输入有 "
        f"{smoke['fixed_320_eval_decodes_different']} 条解码与动态输入不同。"
        "本次编辑使用与正式推理相同的宽度逻辑。CTC 沿用完整输出长度，KL 只计算有效时步。"
        "同一冻结特征上的原投影生成教师分布，KL 温度为 1。",
        "",
        f"训练原图全量识别得到 {data['edit']} 张错误图和 {data['correct_pool']:,} 张正确候选。"
        f"保持集抽取 {data['retain']:,} 张；三个正式候选各使用 1,600 次编辑样本与 1,600 次保持样本。"
        f"每张编辑图实际用 {min(edit_used)}–{max(edit_used)} 次；"
        f"{sum(n > 0 for n in retain_used):,} 张保持图各用 1 次，{sum(n == 0 for n in retain_used):,} 张未被抽到。",
        "",
        "保持集按长度档、分式形态、是否含小数点、数字字符集、运算符字符集的组合做互斥分层。"
        "含斜线且带括号、有多个斜线或等号右侧有斜线归为疑似分式，其余单斜线归为普通除法。"
        "分层和批次细节见 provenance.json 与 sampling-sequence.jsonl。"
        "疑似标注或语法异常见 train-label-review.json，原始标签保留。",
        "",
        "## 验证集筛选",
        "",
        "| 候选 | 步数 | 全文正确 / 2738 | CER | 错题判对 | 无法判定 | 相对权重改变量 | 通过 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for point in lock["validation_points"]:
        counts, metrics = point["val"]["counts"], point["val"]["metrics"]
        report.append(f"| {point['candidate']} | {point['step']} | {counts['exact']} | {metrics['cer']:.4%} |"
                      f" {counts['wrong_accepted']} | {counts['undetermined']} | "
                      f"{point['relative_weight_delta']:.6g} | {'是' if point['accepted'] else '否'} |")
    recommended = lock["recommended"]
    description = recommended if isinstance(recommended, str) else f"{recommended['candidate']} · 步 {recommended['step']}"
    report.extend(["", f"按验证集规则推荐 **{description}**。", ""])
    for family, point in lock["final_test_models"].items():
        remark = "通过接受条件" if point["accepted"] else "仅作为未达接受条件的诊断对照"
        report.append(f"- {family} 锁定 {point['candidate']} 的第 {point['step']} 步，{remark}。")
    report.extend(["", "按验证集全文正确数降序、错题判对数升序、CER 升序及较早步数排序。"
                   "前四项完全相同时采用较小的 λ。未达接受条件的组仍可展示锁定的诊断模型。"])
    report.extend(["", "## 锁定后的评估", "",
        "88 页使用既有 YOLO 检测框，权重为 runs/Y2-geometry-2/weights/best.pt，IoU 阈值为 0.50。"
        "未匹配题在逐题明细中列出实际候选框、IoU 和该框的 OCR。",
        "判题准确率与无法判定数沿用原评估口径，仅统计真值可判定的题。真值无法判定数保存在 comparison.json。",
        "", "| 范围 | 模型 | 全文正确 | 全文准确率 | CER | 判题准确率 | 错题判对 | 无法判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"])
    names = {"val": "验证裁图", "test": "测试裁图", "pages": "整页",
             "fraction-crop": "10.jpg 标注裁图", "fraction-page": "10.jpg 检测裁图",
             "ordinary-crop": "普通算式标注裁图", "ordinary-page": "普通算式检测裁图"}
    for scope, label in names.items():
        for family, result in summaries.items():
            item = result["results"][scope]
            counts, metrics = item["counts"], item["metrics"]
            grade = f"{metrics['grading_accuracy']:.4%}" if metrics["grading_accuracy"] is not None else "—"
            report.append(f"| {label} | {family} | {counts['exact']}/{counts['labeled']} | "
                          f"{metrics['exact_rate']:.4%} | {metrics['cer']:.4%} | {grade} | "
                          f"{counts['wrong_accepted']} | {counts['undetermined']} |")
    report.extend(["", "## 与 E0 逐题比较", "",
        "| 范围 | 模型 | 修复 | 退步 | 前后均错 | 判题变化 | 新增错题判对 | 消除错题判对 | 原判对的正确题变成判错 | 原判对的正确题变成无法判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for scope, by_family in paired.items():
        for family, counts in by_family.items():
            report.append(f"| {names[scope]} | {family} | {counts['repair']} | {counts['regression']} | "
                          f"{counts['both_wrong']} | {counts['grade_changed']} | {counts['new_wrong_accept']} | "
                          f"{counts['removed_wrong_accept']} | {counts['new_false_reject']} | "
                          f"{counts['new_undetermined_on_correct']} |")
    report.extend(["", "## 普通算式字符错误", "",
        "同一题可命中多个字符类别。整页的检测未匹配题单独计数，字符分类只对有实际 OCR 结果的题使用。",
        "", "| 范围 | 模型 | 数字 | 小数点 | 运算符、等号或括号 | 参与字符错误统计的题 |",
        "|---|---|---:|---:|---:|---:|"])
    for scope in ("ordinary-crop", "ordinary-page"):
        for family, result in summaries.items():
            breakdown = result["results"][scope]["character_errors"]
            counts = breakdown["questions_by_category"]
            report.append(f"| {names[scope]} | {family} | {counts.get('digit', 0)} | "
                          f"{counts.get('decimal_point', 0)} | {counts.get('operator_or_equals', 0)} | "
                          f"{breakdown['incorrect_with_ocr']} |")
    report.extend(["", "## 损失与梯度尺度", "",
        "表中梯度为同一子批次上的两个因子梯度的联合 L2 范数。KL 和相对惩罚列已乘对应系数。"
        "初始化时两者为零，来源是初始增量为零。",
        "", "| 候选 | 步数 | 编辑 CTC 梯度 | 保持 CTC 梯度 | λ × KL 梯度 | μ × 相对惩罚梯度 |",
        "|---|---:|---:|---:|---:|---:|"])
    for candidate in GROUPS:
        points = read_jsonl(OUT / candidate / "training.jsonl")
        for point in points:
            if "grad_ctc_edit" not in point:
                continue
            report.append(f"| {candidate} | {point['step']} | {point['grad_ctc_edit']:.3g} | "
                          f"{point['grad_ctc_retain']:.3g} | {point['lambda_kl']*point['grad_kl_retain']:.3g} | "
                          f"{point['mu']*point['grad_relative_delta_squared']:.3g} |")
    last = read_jsonl(OUT / "E2-lambda-1.0/training.jsonl")[-1]
    ratio = last["lambda_kl"] * last["grad_kl_retain"] / last["grad_ctc_retain"]
    report.extend(["", f"λ=1 候选到第 200 步时，加权 KL 梯度约为保持 CTC 的 {ratio:.1%}，"
                   "仍低于编辑 CTC。相对改变量惩罚保持为弱正则，正式组使用预设系数。"])
    report.extend(["", "## E2 与 E1", ""])
    for scope in ("val", "test", "pages"):
        e1 = summaries["E1"]["results"][scope]["counts"]
        e2 = summaries["E2"]["results"][scope]["counts"]
        report.append(f"- {names[scope]}中，E2 相对 E1 全文正确数 {e2['exact']-e1['exact']:+d}、"
                      f"字符编辑数 {e2['edits']-e1['edits']:+d}、"
                      f"错题判对数 {e2['wrong_accepted']-e1['wrong_accepted']:+d}。")
    statuses = load_json(OUT / "swanlab-status.json")
    synced = sum(row["synchronized"] for row in statuses)
    report.extend(["", f"SwanLab 共记录 {len(statuses)} 组，已同步 {synced} 组。同步明细见 swanlab-status.json。", ""])
    (OUT / "weight-edit-results.md").write_text("\n".join(report), encoding="utf-8")


def report():
    from importlib.metadata import version
    import platform

    summaries = {family: load_json(OUT / "final" / family / "summary.json")
                 for family in ("E0", "E1", "E2")}
    if any(row["selection_lock_sha256"] != sha(OUT / "selection-lock.json") for row in summaries.values()):
        raise ValueError("Selection lock changed after testing")
    paired = paired_results()
    save_json(OUT / "comparison.json", dict(models=summaries, paired=paired))
    build_details()
    track_swanlab(summaries)
    render_report(summaries, paired)
    code = ("weight_editing/weight_edit.py", "weight_editing/train_weight_edit.py", "weight_editing/run_weight_edit.ps1",
            "evaluate_controlled_ocr.py", "evaluate_ocr.py", "optimizations/ocr_runtime.py")
    save_json(OUT / "reproduction.json", dict(
        platform=platform.platform(), python=platform.python_version(),
        packages={name: version(name) for name in ("paddlepaddle-gpu", "paddleocr", "paddlex", "numpy", "swanlab")},
        code_sha256={name: sha(ROOT / name) for name in code}, config_sha256=sha(OUT / "config.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "export", "validate", "lock", "test", "report"))
    parser.add_argument("--candidate", choices=(*GROUPS, "smoke"))
    parser.add_argument("--step", type=int)
    parser.add_argument("--family", choices=("E0", "E1", "E2"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action in ("export", "validate"):
        if not args.candidate or args.step not in ((0, 3) if args.candidate == "smoke" else STEPS):
            parser.error("Specify a valid --candidate and --step")
        if args.action == "export":
            export_checkpoint(args.candidate, args.step)
        else:
            validate_checkpoint(args.candidate, args.step, smoke=args.candidate == "smoke")
    elif args.action == "lock":
        lock_selection()
    elif args.action == "test":
        if not args.family:
            parser.error("Specify --family")
        test_family(args.family)
    else:
        report()

