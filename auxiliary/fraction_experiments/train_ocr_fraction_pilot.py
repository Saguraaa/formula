"""Train and export PP-OCRv6 on the same fraction crops as the formula pilot."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import copy
import re

import yaml

from train_ocr import MODEL, ROOT, launch

FRACTION = re.compile(r"\\frac\{(\d+)\}\{(\d+)\}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="+", choices=["OCR-F0", "OCR-F1", "OCR-F2"])
    args = parser.parse_args()
    source = ROOT / "vendor/PaddleOCR"
    data = ROOT / "datasets_formula"
    template = yaml.safe_load((ROOT / "runs/O1/train.yaml").read_text(encoding="utf-8"))
    dictionary = source / template["Global"]["character_dict_path"]
    alphabet = set(dictionary.read_text(encoding="utf-8").splitlines())

    for group in args.groups:
        run = ROOT / "runs" / group
        run.mkdir(exist_ok=False)
        config = copy.deepcopy(template)
        config["Global"].update(epoch_num=3, save_model_dir=run.as_posix(),
                                print_batch_step=1, save_epoch_step=3, eval_batch_step=[0, 5])
        config["Optimizer"]["lr"]["learning_rate"] = 0.0001
        train_labels = "train_aug.txt" if group == "OCR-F2" else "train.txt"
        for section, label_file in (("Train", train_labels), ("Eval", "val.txt")):
            records = []
            for line in (data / label_file).read_text(encoding="utf-8").splitlines():
                image, latex = line.split("\t", 1)
                target = FRACTION.sub(r"\1/\2", latex)
                assert (data / image).is_file()
                assert set(target) <= alphabet and len(target) <= config["Global"]["max_text_length"]
                records.append(f"{image}\t{target}")
            assert len(records) == 24
            labels = run / label_file
            labels.write_text("\n".join(records) + "\n", encoding="utf-8")
            config[section]["dataset"].update(data_dir=data.as_posix(), label_file_list=[labels.as_posix()])
            config[section]["loader"]["batch_size_per_card"] = 4

        if group != "OCR-F0":
            train_config = run / "train.yaml"
            train_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            launch(source, "train.py", train_config, run / "train.log")
            config["Global"]["pretrained_model"] = (run / "iter_epoch_3.pdparams").as_posix()
        config["Global"]["save_inference_dir"] = (run / "inference").as_posix()
        export_config = run / "export.yaml"
        export_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        launch(source, "export_model.py", export_config, run / "export.log")


if __name__ == "__main__":
    main()
