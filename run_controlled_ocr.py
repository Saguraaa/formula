"""Audit and train the three fixed-split OCR controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from evaluate_ocr import MODEL, normalize
from train_ocr import ROOT, launch

NAME = "ocr-controlled-v1"
OUT = ROOT / "runs" / NAME
SOURCE = ROOT / "vendor/PaddleOCR"
DATA = ROOT / "datasets_oriented"
OCR = DATA / "ocr"
ANNOTATIONS = ROOT.parent / "实验3/train/annotations.json"
INITIAL = ROOT / "weights" / f"{MODEL}_pretrained.pdparams"
GROUPS = {
    "O1": OCR,
    "O2": ROOT / "experiment_data/O2/ocr",
    "O3": ROOT / "experiment_data/O3-geometry/ocr",
}
COUNTS = {"train": 13206, "val": 2738, "test": 2767}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def records(path):
    rows = [tuple(line.split("\t", 1)) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if len(dict(rows)) != len(rows):
        raise ValueError(f"Duplicate crop identifiers in {path}")
    return rows


def save_rows(path, rows):
    path.write_bytes("".join(f"{name}\t{label}\n" for name, label in rows).encode("utf-8"))


def check_crops(base, split, names):
    actual = {p.relative_to(base).as_posix() for p in (base / "crops" / split).glob("*") if p.is_file()}
    if actual != names:
        raise ValueError(f"Crop list mismatch at {base}: missing={sorted(names - actual)[:12]}, "
                         f"extra={sorted(actual - names)[:12]}")


def verify_geometry(out):
    from io import BytesIO

    import numpy as np
    from PIL import Image

    from prepare_experiment import read_rgb
    from prepare_ocr_geometry import warp

    o1, o2, o3 = (GROUPS[group] for group in GROUPS)
    manifest = load_json(o3.parent / "manifest.json")
    quality = load_json(o2.parent / "manifest.json")["records"]
    quality_pairs = {item["derived"]: Path(item["source"]) for item in quality}
    geometry = manifest["geometry"]
    rng = np.random.default_rng(geometry["seed"])
    transformed = []
    rows = records(o3 / "train.txt")
    if len(quality_pairs) != len(quality) or len(quality) != len(rows):
        raise ValueError("Invalid O2 image provenance count")
    for name, _ in rows:
        if quality_pairs[f"ocr/{name}"].relative_to(o1).as_posix() != name:
            raise ValueError(f"O2: crop provenance differs for {name}")
        selected = rng.random() < geometry["probability"]
        with Image.open(o1 / name) as original, Image.open(o2 / name) as quality_crop:
            if original.size != quality_crop.size:
                raise ValueError(f"O2: crop dimensions changed for {name}")
            if not selected or min(quality_crop.size) < 2:
                continue
        result = warp(read_rgb(o2 / name), rng, geometry["rotation_degrees"], geometry["perspective_fraction"])
        with BytesIO() as encoded:
            Image.fromarray(result).save(encoded, format="JPEG", quality=95)
            if encoded.getvalue() != (o3 / name).read_bytes():
                raise ValueError(f"O3: recomputed geometry image differs for {name}")
        transformed.append(name)
    if transformed != manifest["transformed"]:
        raise ValueError("O3: replayed geometry selection differs from the manifest")
    save_json(out / "geometry-verification.json", {"settings": geometry, "O2_name_and_shape_matches": len(rows),
              "O3_exact_reproduction": len(transformed), "O3_unchanged_sha256_matches": len(rows) - len(transformed)})
    print(f"Verified all {len(rows)} O2 crop mappings and {len(transformed)} byte-identical O3 geometry replays", flush=True)


def audit(out, config):
    manifest_path = DATA / "split_manifest.json"
    splits = load_json(manifest_path)["splits"]
    annotations = {page["image"]: page["boxes"] for page in load_json(ANNOTATIONS)}
    dictionary = SOURCE / config["Global"]["character_dict_path"]
    alphabet = set(dictionary.read_text(encoding="utf-8").splitlines())
    max_length = config["Global"]["max_text_length"]
    audit_data = {"source_annotations_sha256": sha(ANNOTATIONS), "split_sha256": sha(manifest_path),
                  "dictionary_sha256": sha(dictionary), "initial_weights_sha256": sha(INITIAL),
                  "splits": {}, "groups": {}}
    latest = {}
    differences = []
    incompatible = []
    for split, expected in COUNTS.items():
        original = OCR / f"{split}.txt"
        exported = records(original)
        source = {}
        for image in splits[split]:
            for index, box in enumerate(annotations[image]):
                label = box.get("text")
                if isinstance(label, str) and label.strip():
                    name = f"crops/{split}/{Path(image).stem}_{index:04d}.jpg"
                    if name in source:
                        raise ValueError(f"Duplicate source crop {name}")
                    source[name] = label
        actual = dict(exported)
        split_differences = []
        for name in sorted(source.keys() | actual.keys()):
            a, b = source.get(name), actual.get(name)
            if a is None or b is None or normalize(a) != normalize(b):
                split_differences.append(dict(split=split, crop=name, source=a, exported=b))
        differences.extend(split_differences)
        if split_differences:
            continue
        if len(source) != expected or len(exported) != expected:
            raise ValueError(f"{split}: expected {expected}, source={len(source)}, exported={len(exported)}")
        check_crops(OCR, split, actual.keys())
        normalized = [(name, normalize(source[name])) for name, _ in exported]
        incompatible.extend(dict(split=split, crop=name, label=label, unsupported=sorted(set(label) - alphabet), length=len(label))
                            for name, label in normalized if not label or len(label) > max_length or set(label) - alphabet)
        latest[split] = normalized
        audit_data["splits"][split] = {
            "pages": len(splits[split]), "count": expected, "original_export_sha256": sha(original),
            "normalized_label_sha256": hashlib.sha256("\n".join(label for _, label in normalized).encode()).hexdigest(),
            "recovered_asterisk": [name for name, label in exported if "∗" in label and "∗" not in alphabet],
            "max_length": max((len(label) for _, label in normalized), default=0),
        }
    if differences:
        save_json(out / "source-label-disagreements.json", differences)
        raise ValueError(f"{len(differences)} source/export disagreements; see source-label-disagreements.json")
    if incompatible:
        save_json(out / "incompatible.json", incompatible)
        raise ValueError(f"{len(incompatible)} incompatible samples; see incompatible.json")
    for split, rows in latest.items():
        path = out / f"{split}.txt"
        save_rows(path, rows)
        audit_data["splits"][split]["normalized_manifest_sha256"] = sha(path)

    reference = dict(latest["train"])
    geometry = load_json(GROUPS["O3"].parent / "manifest.json")
    transformed = set(geometry["transformed"])
    if not transformed <= reference.keys() or len(transformed) != len(geometry["transformed"]):
        raise ValueError("Invalid geometry manifest")
    for group, base in GROUPS.items():
        existing = dict(records(base / "train.txt"))
        if existing.keys() != reference.keys():
            raise ValueError(f"{group}: existing training crop identifiers differ")
        stale = [dict(crop=name, previous=label, latest=reference[name]) for name, label in existing.items()
                 if normalize(label) != reference[name]]
        check_crops(base, "train", reference.keys())
        if stale:
            save_json(out / f"{group}-label-disagreements.json", stale)
            raise ValueError(f"{group}: {len(stale)} unexplained label differences")
        if group == "O3" and sha(base / "train.txt") != geometry["labels_sha256"]:
            raise ValueError("O3 source label file is inconsistent with its geometry manifest")
        directory = out / group
        directory.mkdir(exist_ok=True)
        for split, rows in latest.items():
            save_rows(directory / f"{split}.txt", rows)
        image_hashes = {name: sha(base / name) for name in reference}
        save_rows(directory / "train-image-sha256.tsv", image_hashes.items())
        if group == "O2":
            quality_hashes = image_hashes
        if group == "O3" and {name for name in image_hashes if image_hashes[name] != quality_hashes[name]} != transformed:
            raise ValueError("O2/O3 changed images differ from the geometry manifest")
        audit_data["groups"][group] = {
            "image_dir": str(base), "train_image_manifest_sha256": sha(directory / "train-image-sha256.tsv"),
            "labels": {split: {"count": len(rows), "sha256": sha(directory / f"{split}.txt")}
                       for split, rows in latest.items()}, "stale_source_labels": stale,
        }
    audit_data["geometry"] = dict(geometry["geometry"], transformed=len(transformed), total=len(reference))
    save_json(out / "manifest.json", audit_data)
    print({split: (r["count"], len(r["recovered_asterisk"])) for split, r in audit_data["splits"].items()}, flush=True)
    print(f"O3: {len(transformed)} transformed; all training labels agree", flush=True)


def train(out, group, template):
    directory = out / group
    config = copy.deepcopy(template)
    batch = 16
    steps = (COUNTS["train"] + batch - 1) // batch
    config["Global"].update(use_gpu=True, distributed=False, seed=42, epoch_num=3,
                            save_model_dir=directory.as_posix(), save_epoch_step=1,
                            print_batch_step=20, pretrained_model=INITIAL.as_posix(),
                            complete_ocr_batches=True, eval_batch_step=[0, steps],
                            cal_metric_during_train=False)
    config["Optimizer"]["lr"].update(learning_rate=1e-4, warmup_epoch=0)
    config["Metric"].update(normalize_equivalent=True, ignore_space=False)
    transforms = copy.deepcopy(template["Eval"]["dataset"])
    for section, split, base in (("Train", "train", GROUPS[group]), ("Eval", "val", OCR)):
        config[section].pop("sampler", None)
        config[section]["dataset"] = copy.deepcopy(transforms)
        config[section]["dataset"].update(data_dir=base.as_posix(),
                                         label_file_list=[(directory / f"{split}.txt").as_posix()])
        config[section]["loader"].update(shuffle=section == "Train", batch_size_per_card=batch,
                                        num_workers=0, drop_last=False)
    config_path = directory / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if not (directory / "iter_epoch_3.pdparams").is_file():
        launch(SOURCE, "train.py", config_path, directory / "train.log")
    best = directory / "best_accuracy.pdparams"
    config["Global"].update(pretrained_model=best.as_posix(),
                            save_inference_dir=(directory / "inference").as_posix())
    export_path = directory / "export.yaml"
    export_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if not best.is_file():
        raise FileNotFoundError(best)
    if not (directory / "inference/inference.pdiparams").is_file():
        launch(SOURCE, "export_model.py", export_path, directory / "export.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-geometry", action="store_true")
    parser.add_argument("--group", choices=GROUPS, action="append")
    args = parser.parse_args()
    template = yaml.safe_load((SOURCE / f"configs/rec/PP-OCRv6/{MODEL}.yml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    if args.audit_only or not (OUT / "manifest.json").is_file():
        audit(OUT, template)
    if args.verify_geometry:
        verify_geometry(OUT)
    if not args.audit_only and not args.verify_geometry:
        for group in args.group or GROUPS:
            train(OUT, group, template)


if __name__ == "__main__":
    main()
