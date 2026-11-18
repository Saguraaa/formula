"""Prepare a matched full-data experiment without editing source labels."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import hashlib
import json
import math
import re
from pathlib import Path

import yaml

from evaluate_ocr import normalize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets_oriented/ocr"
OUT = ROOT / "outputs/ocr-full-comparison"
FRACTION = re.compile(r"(\d+)/(\d+)")
MODES = {"text": "runs/O1/train.yaml", "formula": "runs/F1-formula-head/train.yaml"}


def latex_label(name, normalized):
    # These are the only worksheet pages with stacked fractions in the original split.
    if Path(name).stem.split("_")[0] in {"10", "225", "419"}:
        normalized = FRACTION.sub(lambda match: r"\frac{" + match[1] + "}{" + match[2] + "}", normalized)
    return normalized.replace("/", r"\div ").replace("*", r"\times ")


def main():
    OUT.mkdir(exist_ok=True)
    alphabet = set((ROOT / "vendor/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt")
                   .read_text(encoding="utf-8").splitlines())
    manifest = {}
    for split, count in (("train", 13206), ("val", 2738), ("test", 2767)):
        original = DATA / f"{split}.txt"
        samples = [line.split("\t", 1) for line in original.read_text(encoding="utf-8").splitlines()]
        assert len({name for name, _ in samples}) == len(samples) == count
        assert all((DATA / name).is_file() for name, _ in samples)
        manifest[split] = {"count": count, "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest()}
        for mode in MODES:
            records = []
            for name, target in samples:
                target = normalize(target)
                if mode == "text":
                    assert len(target) <= 25 and set(target) <= alphabet
                else:
                    target = latex_label(name, target)
                records.append(f"{name}\t{target}")
            (OUT / f"{mode}-{split}.txt").write_text("\n".join(records) + "\n", encoding="utf-8")

    for mode, template in MODES.items():
        run = ROOT / "runs" / f"full-{mode}"
        run.mkdir(exist_ok=True)
        config = yaml.safe_load((ROOT / template).read_text(encoding="utf-8"))
        config["Global"].update(epoch_num=1, save_epoch_step=1, print_batch_step=25,
                                save_model_dir=run.as_posix(), eval_batch_step=[0, 10000000],
                                cal_metric_during_train=False)
        for section, split in (("Train", "train"), ("Eval", "val")):
            config[section]["dataset"].update(data_dir=DATA.as_posix(),
                  label_file_list=[(OUT / f"{mode}-{split}.txt").as_posix()])
            config[section]["loader"].update(batch_size_per_card=16, drop_last=False)
        (run / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
