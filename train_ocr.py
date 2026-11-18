import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parent
MODEL = "PP-OCRv6_medium_rec"


def launch(source, tool, config, log, run=None):
    command = [sys.executable, "-u", str(ROOT / "optimizations/ocr_runtime.py"),
               str(source / "tools" / tool), "-c", str(config)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", NO_ALBUMENTATIONS_UPDATE="1")
    if os.name == "nt":
        cudnn = Path(sys.prefix) / "Lib/site-packages/torch/lib"
        if (cudnn / "cudnn64_9.dll").exists():
            env["PATH"] = os.pathsep.join((str(cudnn), env["PATH"]))
    deps = str(ROOT / "vendor/ocr_deps")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (deps, env.get("PYTHONPATH"))))
    step = 0
    with log.open("w", encoding="utf-8") as output, subprocess.Popen(
        command, cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace"
    ) as child:
        for line in child.stdout:
            print(line, end="", flush=True)
            output.write(line)
            if run is None:
                continue
            match = re.search(r"global_step:\s*(\d+)", line)
            if match:
                step = int(match.group(1))
                fields = {"loss", "loss_ctc", "loss_gtc", "CTCLoss", "NRTRLoss", "lr"}
                prefix = "train"
            elif "cur metric," in line:
                fields = {"acc", "norm_edit_dis", "exact", "samples"}
                prefix = "val"
            else:
                continue
            values = {f"{prefix}/{key}": float(value) for key, value in re.findall(
                r"(\w+):\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line
            ) if key in fields}
            if values:
                run.log(values, step=step)
        if child.wait():
            raise subprocess.CalledProcessError(child.returncode, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "vendor/PaddleOCR")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "weights" / f"{MODEL}_pretrained.pdparams")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--swanlab-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    source = args.source.resolve()
    train_data = ROOT / spec["train_data"]
    val_data = ROOT / "datasets_oriented/ocr"
    original = source / "configs/rec/PP-OCRv6" / f"{MODEL}.yml"
    for path in (original, args.pretrained, train_data / "train.txt", val_data / "val.txt"):
        if not path.is_file():
            parser.error(f"Required input missing: {path}")
    group = spec["group"] + ("-smoke" if args.limit else "")
    output = ROOT / "runs" / group
    if args.limit:
        import tempfile
        output = Path(tempfile.mkdtemp(prefix=group + "-", dir=ROOT / "runs"))
        output.rmdir()
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(original.read_text(encoding="utf-8"))
    cfg["Global"].update(use_gpu=not args.cpu, distributed=False, epoch_num=args.epochs,
                         save_model_dir=output.as_posix(), save_epoch_step=1, print_batch_step=20,
                         pretrained_model=args.pretrained.resolve().as_posix(), seed=42)
    cfg["Optimizer"]["lr"].update(learning_rate=0.0001, warmup_epoch=0)
    dictionary = source / cfg["Global"]["character_dict_path"]
    alphabet = set(dictionary.read_text(encoding="utf-8").splitlines()) | {" "}
    size = {}
    for key, folder, split in (("Train", train_data, "train"), ("Eval", val_data, "val")):
        section = cfg[key]
        section.pop("sampler", None)
        section["dataset"] = copy.deepcopy(cfg["Eval"]["dataset"])
        records = (folder / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        accepted = [line for line in records if line and len(line.split("\t", 1)[1]) <= cfg["Global"]["max_text_length"]
                    and set(line.split("\t", 1)[1]) <= alphabet]
        if args.limit:
            accepted = accepted[:args.limit]
        if not accepted:
            raise ValueError(f"No usable {split} samples")
        labels = output / f"{split}.txt"
        labels.write_text("\n".join(accepted) + "\n", encoding="utf-8")
        size[split] = {"source": len(records), "used": len(accepted)}
        section["dataset"].update(data_dir=folder.as_posix(), label_file_list=[labels.as_posix()])
        section["loader"].update(batch_size_per_card=args.batch, num_workers=0, drop_last=False)
    steps = (size["train"]["used"] + args.batch - 1) // args.batch
    # PaddleOCR's Windows loop stops before the last loader batch.
    cfg["Global"]["eval_batch_step"] = [0, max(1, steps - (os.name == "nt"))]
    config = output / "train.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    import swanlab
    with swanlab.init(project="ModelAdjust", experiment_name=group, mode=args.swanlab_mode,
                      config={"group": group, "model": MODEL, "epochs": args.epochs,
                              "batch": args.batch, "counts": size, "train_data": str(train_data),
                              "train_config": cfg}) as run:
        (output / "run.json").write_text(json.dumps({"group": group,
            "swanlab_url": run.url if args.swanlab_mode == "online" else None,
            "counts": size}, indent=2), encoding="utf-8")
        launch(source, "train.py", config, output / "train.log", run)
        best = output / "best_accuracy.pdparams"
        if not best.is_file():
            raise FileNotFoundError(best)
        cfg["Global"].update(pretrained_model=best.as_posix(), save_inference_dir=(output / "inference").as_posix())
        export = output / "export.yaml"
        export.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        launch(source, "export_model.py", export, output / "export.log")


if __name__ == "__main__":
    main()



