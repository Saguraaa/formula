"""Verify OCR results, record them in SwanLab, and package comparison data."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from evaluate_controlled_ocr import RESULTS, SCOPES
from run_controlled_ocr import GROUPS, INITIAL, NAME, OUT, ROOT, load_json, save_json, sha

MAIN = ("O0", *GROUPS)
HISTORICAL = "full-text"
GROUP_LABELS = {"O0": "O0 官方预训练", "O1": "O1 原图微调", "O2": "O2 画质增强", "O3": "O3 画质 + 几何",
                HISTORICAL: "full-text（历史，1 轮）"}
SCOPE_LABELS = dict(zip(SCOPES, ("验证裁图", "测试裁图", "整页检测裁图", "10.jpg 标注裁图", "10.jpg 检测裁图",
                                "普通算式标注裁图", "普通算式检测裁图"), strict=True))
NUMBER = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
FIELDS = re.compile(rf"([\w]+):\s*{NUMBER}")
COVERAGE = re.compile(r"OCR epoch coverage: epoch=(\d+), samples=(\d+), steps=(\d+)")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def train_curve(group):
    directory = OUT / group
    log = (directory / "train.log").read_text(encoding="utf-8")
    values = defaultdict(dict)
    validation = []
    step = 0
    for line in log.splitlines():
        data = dict(FIELDS.findall(line))
        if "epoch: [" in line and "global_step" in data:
            step = int(data["global_step"])
            values[step].update({f"train/{name}": float(data[name]) for name in ("loss", "CTCLoss", "NRTRLoss", "lr")})
        elif "cur metric," in line:
            samples, correct = int(data["samples"]), int(data["exact"])
            expected_step = 826 * (len(validation) + 1)
            if samples != 2738 or step != expected_step or float(data["acc"]) != correct / samples:
                raise ValueError(f"{group}: unexpected validation coverage at step {step}: {data}")
            point = dict(epoch=step // 826, step=step, samples=samples, exact=correct, acc=float(data["acc"]),
                         norm_edit_dis=float(data["norm_edit_dis"]))
            validation.append(point)
            values[step].update({f"validation/{key}": v for key, v in point.items() if key != "step"})
    expected = [(str(i), "13206", "826") for i in range(1, 4)]
    if COVERAGE.findall(log) != expected or len(validation) != 3 or "load pretrain successful" not in log:
        raise ValueError(f"{group}: the training log has incomplete coverage")
    selected = max(validation, key=lambda point: point["exact"])["epoch"]
    best = sha(directory / "best_accuracy.pdparams")
    if best != sha(directory / f"iter_epoch_{selected}.pdparams"):
        raise ValueError(f"{group}: selected checkpoint differs from the correct epoch {selected}")
    manifest = load_json(OUT / "manifest.json")
    for split, item in manifest["groups"][group]["labels"].items():
        if sha(directory / f"{split}.txt") != item["sha256"]:
            raise ValueError(f"{group}: {split} sample manifest changed")
    if sha(directory / "train-image-sha256.tsv") != manifest["groups"][group]["train_image_manifest_sha256"]:
        raise ValueError(f"{group}: training image manifest changed")
    report = {"selected_epoch": selected, "best_checkpoint_sha256": best, "validation": validation,
              "coverage": [dict(epoch=int(i), samples=int(n), steps=int(s)) for i, n, s in expected],
              "train_config_sha256": sha(directory / "train.yaml"), "train_log_sha256": sha(directory / "train.log"),
              "export_config_sha256": sha(directory / "export.yaml"), "export_log_sha256": sha(directory / "export.log"),
              "initial_weights_sha256": manifest["initial_weights_sha256"],
              "curve": [dict(step=k, metrics=v) for k, v in sorted(values.items())]}
    save_json(directory / "training-metrics.json", report)
    return report


def sync(group, mode):
    import swanlab

    directory = RESULTS / group
    outcome = directory / "swanlab.json"
    if outcome.is_file():
        previous = load_json(outcome)
        if previous["status"] == "online-completed":
            return
    status = {"requested_mode": mode, "status": "pending", "url": None}
    summary = load_json(directory / "summary.json")
    training = train_curve(group) if group in GROUPS else None
    manifest = load_json(OUT / "manifest.json")
    run_dir = OUT / "swanlab"
    run_dir.mkdir(exist_ok=True)

    def record(actual_mode):
        config = {"group": group, "annotation_sha256": manifest["source_annotations_sha256"],
                  "split_sha256": manifest["split_sha256"], "model_sha256": summary["model_sha256"],
                  "detection_cache_sha256": summary["detection_cache_sha256"],
                  "dataset_counts": {split: row["count"] for split, row in manifest["splits"].items()}}
        if training:
            config.update(initial_weights_sha256=manifest["initial_weights_sha256"], selected_epoch=training["selected_epoch"],
                          train_config_sha256=training["train_config_sha256"], epochs=3, batch=16, seed=42,
                          learning_rate=1e-4, lr_schedule="Cosine", optimizer="Adam")
        elif group == HISTORICAL:
            config.update(epochs=1, historical=True)
        with swanlab.init(project="ModelAdjust", name=f"{NAME}-{group}", group=NAME, mode=actual_mode,
                          log_dir=str(run_dir), config=config, settings=swanlab.Settings(interactive=False)) as run:
            if training:
                for point in training["curve"]:
                    run.log(point["metrics"], step=point["step"])
            metrics = {f"final/{scope}/{key}": value for scope, data in summary["results"].items()
                       for key, value in (data["counts"] | data["metrics"]).items() if value is not None}
            run.log(metrics, step=2478 if training else 0)
            url = run.url if actual_mode == "online" else None
            metadata = {"url": url, "run_id": run.id, "local_log_dir": str(run.dir)}
            print(f"SwanLab {group}: {actual_mode} {url or run.dir}", flush=True)
        if actual_mode == "online" and not url:
            raise RuntimeError("SwanLab did not provide an online run URL")
        return metadata

    if mode == "online":
        try:
            status.update(status="online-completed", **record("online"))
        except Exception as exc:
            status["online_error"] = f"{type(exc).__name__}: {exc}"
            print(f"SwanLab {group} online failed: {type(exc).__name__}: {exc}", flush=True)
            status.update(status="offline-saved", **record("offline"))
    else:
        status.update(status="offline-saved", **record("offline"))
    save_json(outcome, status)


def paired_differences(groups):
    deltas = {}
    for scope, filenames in {"val": "val", "test": "test", "pages": "pages", "fraction-crop": "fraction-crop",
                             "fraction-page": "fraction-page"}.items():
        by_group = {group: read_jsonl(RESULTS / group / f"{filenames}.jsonl") for group in groups}
        deltas[scope] = {}
        for a, b in zip(MAIN, MAIN[1:]):
            changes = dict(gained=0, lost=0, wrong_accept_added=0, wrong_accept_removed=0, grade_delta=0)
            for before, after in zip(by_group[a], by_group[b], strict=True):
                evidence = ("page", "box_index", "expected", "truth_grade", "matched", "prediction_bbox", "match_iou")
                if any(before.get(key) != after.get(key) for key in evidence):
                    raise ValueError(f"{scope}: {a} and {b} do not share the same question and detection")
                changes["gained"] += before["exact"] is False and after["exact"] is True
                changes["lost"] += before["exact"] is True and after["exact"] is False
                truth = before["truth_grade"]
                if truth is not None:
                    changes["grade_delta"] += (after["ocr_grade"] is truth) - (before["ocr_grade"] is truth)
                if truth is False:
                    changes["wrong_accept_added"] += before["ocr_grade"] is not True and after["ocr_grade"] is True
                    changes["wrong_accept_removed"] += before["ocr_grade"] is True and after["ocr_grade"] is not True
            changes["exact_delta"] = changes["gained"] - changes["lost"]
            changes["wrong_accept_delta"] = changes["wrong_accept_added"] - changes["wrong_accept_removed"]
            deltas[scope][f"{a}→{b}"] = changes
    return deltas


def percent(value):
    return f"{value:.2%}" if value is not None else "—"


def group_table(results, scope):
    lines = ["| 组别 | 全文正确 | 字符错误率 | 判题正确 | 错题判对 | 无法判定 |",
             "|---|---:|---:|---:|---:|---:|"]
    for group, data in results.items():
        counts, rates = (data["results"][scope][key] for key in ("counts", "metrics"))
        lines.append(f"| {GROUP_LABELS[group]} | {counts['exact']}/{counts['labeled']} ({percent(rates['exact_rate'])}) | "
                     f"{percent(rates['cer'])} | {counts['grade_correct']}/{counts['gradeable']} ({percent(rates['grading_accuracy'])}) | "
                     f"{counts['wrong_accepted']}/{counts['wrong_targets']} | {counts['undetermined']} |")
    return lines


def render_report(results, training, deltas, sync_status):
    manifest = load_json(OUT / "manifest.json")
    pages = results["O0"]["results"]["pages"]["counts"]
    g = manifest["geometry"]
    lines = ["# OCR 对照实验结果", "", f"实验名称 `{NAME}`。O1–O3 使用同一份官方训练权重，"
             "全参数微调 3 轮，batch 16，seed 42，Adam，初始学习率 1e-4，余弦衰减，"
             "输入 3×48×320，最大文本长度 25。",
             "", "按整页固定划分，训练 13,206 张、验证 2,738 张、测试 2,767 张。源标注与已有 OCR 导出在折叠空白后逐条一致。"
             "训练、验证和测试使用相同的算符规范化。"
             "训练集原先因 `∗` 被过滤的 122 条全部保留，验证集对应 57 条也全部保留。每轮 826 个训练批次、172 个验证批次。",
             "", f"O3 沿用已有图片，{g['transformed']:,}/{g['total']:,} 张经过 ±{g['rotation_degrees']:g}° 旋转和"
             f"最多 {g['perspective_fraction']:.0%} 的角点偏移。扩大画布容纳四个变换后的角点。"
             "按原种子复算后，所有变换图均与已有 JPEG 逐字节一致。",
             "", "## 每轮验证与导出选择", "", "| 组别 | 第 1 轮 | 第 2 轮 | 第 3 轮 | 导出轮次 |",
             "|---|---:|---:|---:|---:|"]
    for group, record in training.items():
        values = " | ".join(f"{v['exact']}/{v['samples']} ({v['acc']:.2%})" for v in record["validation"])
        lines.append(f"| {group} | {values} | {record['selected_epoch']} |")
    if (OUT / "interrupted/O2-cuda-999/train.log").is_file():
        lines += ["", "O2 曾因 CUDA 999 和执行环境断开中断。表中 O2 从同一官方权重和种子重新完成了完整的三轮训练。"
                  "中断日志保存在 `interrupted/`。"]
    lines += ["", "先按全文准确率选最佳权重，同分保留较早轮次。训练内验证使用固定 320 像素宽的输入。"
              "下方四组统一用 TextRecognition 的默认动态宽度处理完整裁图，评分沿用相同的规范化规则。",
              "", "## 四组统一评估", "", "全文分母为有文字标注的题目。判题分母为原标注可判定的题目，"
              "无法判定列统计其中 OCR 结果无法判题的数量。错题判对的分母为原标注判错的题目。CER 以规范化标注字符数为分母。",
              "", f"88 页统一使用 YOLO `runs/Y2-geometry-2/weights/best.pt`，置信度 0.50、输入尺寸 640、匹配 IoU ≥ 0.50。"
              f"共匹配 {pages['matched']:,}/{pages['boxes']:,} 个标注框，有文字标注的未匹配题 {pages['labeled_unmatched']} 道，"
              f"未匹配预测框 {pages['false_positive_boxes']} 个。未匹配题计全文错误，CER 计参考字符删除，判题计无法判定，"
              "明细另列实际候选框、IoU 与原始 OCR。"]
    for scope in ("val", "test", "pages", "fraction-crop", "fraction-page"):
        lines += ["", f"### {SCOPE_LABELS[scope]}", "", *group_table(results, scope)]
    lines += ["", "## 同题对照", "", "| 范围 | 对照 | 新读全对 | 由对变错 | 净增全文正确 | 净增判题正确 | 新增错判对 | 消除错判对 | 净增错判对 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scope, pairs in deltas.items():
        for pair, d in pairs.items():
            lines.append(f"| {SCOPE_LABELS[scope]} | {pair} | {d['gained']} | {d['lost']} | {d['exact_delta']:+d} | "
                         f"{d['grade_delta']:+d} | {d['wrong_accept_added']} | {d['wrong_accept_removed']} | {d['wrong_accept_delta']:+d} |")
    lines += ["", "## 普通算式字符错误", "", "按最小编辑距离对齐规范化后的文字。每题可计入多个类别。整页分项统计实际获得 OCR 的题目，"
              "检测未匹配题在整页总表与逐题明细中另外计数。", "",
              "| 输入 | 组别 | OCR 文字有误 | 涉及数字 | 涉及小数点 | 涉及算符、等号或括号 | 涉及其它字符 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for scope in ("ordinary-crop", "ordinary-page"):
        for group, data in results.items():
            chars = data["results"][scope]["character_errors"]
            categories = chars["questions_by_category"]
            values = " | ".join(str(categories.get(key, 0)) for key in ("digit", "decimal_point", "operator_or_equals", "other"))
            lines.append(f"| {SCOPE_LABELS[scope]} | {group} | {chars['incorrect_with_ocr']} | {values} |")
    lines += ["", "测试集普通算式标注裁图中，各字符类别的主要混淆如下。`∅` 表示插入或删除位置的空字符。", "",
              "| 组别 | 数字 | 小数点 | 算符、等号或括号 |", "|---|---|---|---|"]
    for group in MAIN:
        edits = results[group]["results"]["ordinary-crop"]["character_errors"]["common_edits"]
        cells = []
        for alphabet in (set("0123456789"), {"."}, set("+-*/=()")):
            relevant = [item for item in edits if set(item["expected"] + item["ocr"]) & alphabet][:2]
            cells.append("、".join(f"`{item['expected'] or '∅'}` → `{item['ocr'] or '∅'}` ({item['count']})"
                                  for item in relevant) or "—")
        lines.append(f"| {group} | " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |")
    lines += ["", "每组的字符替换、插入和删除频次见 `comparison.json`，对应题目见 `evaluation/` 下的 `ordinary-*-character-errors.jsonl`。",
              "", "`full-text` 使用最新标签重新评分。该历史权重训练 1 轮，参考时需考虑训练设置差异。O0→O1、O1→O2、O2→O3 的差值描述本次固定种子运行。",
            "", "## SwanLab", "", "| 组别 | 同步状态 | 实验链接 / 离线运行编号 |", "|---|---|---|"]
    for group, status in sync_status.items():
        url = status.get("url")
        sync_label = "在线记录完成" if status["status"] == "online-completed" else "离线记录待同步"
        lines.append(f"| {group} | {sync_label} | {f'[打开实验]({url})' if url else status['run_id']} |")
    if any(status["status"] == "offline-saved" for status in sync_status.values()):
        lines += ["", "实验环境无法解析 `api.swanlab.cn`。压缩包中 `swanlab/` 包含可同步的离线运行文件。"
                  "恢复网络后可对各运行目录使用 `swanlab sync`。已有线上项目入口见下方。",
                  "", "[SwanLab ModelAdjust](https://swanlab.cn/@Saguraaa/ModelAdjust)"]
    lines += ["", "## 追溯文件", "", "- `manifest.json` 和各组 `train.txt`、`val.txt`、`test.txt` 记录文件清单、标签与权重哈希。",
              "- 各组 `train-image-sha256.tsv` 可核对原图和两套增强图，`train.yaml` 和 `export.yaml` 保存运行配置。",
              "- `training-curves.png` 为训练损失、完整验证集准确率及共用学习率曲线，空心标记所在轮次用于导出。",
              "- `error-details.html` 可按组和范围展开原始 OCR 与错误详情。`evaluation/` 包含全部逐题结果、单独的错误明细和检测预测框。",
              "- 新模型权重保存在项目 `runs/ocr-controlled-v1/O1`、`O2`、`O3` 各组 `inference/` 中。"]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_details(groups):
    def cell(value):
        return escape(str(value)) if value is not None else "—"

    def grade(value):
        return {True: "对", False: "错", None: "无法判定"}[value]

    html = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>OCR 逐题错误明细</title>",
            "<style>body{font-family:system-ui,sans-serif;padding:1rem;color:#202530}table{border-collapse:collapse;width:100%}"
            "th,td{padding:.4rem;border:1px solid #d6dbe2;text-align:left;vertical-align:top}th{background:#f3f5f8}"
            "code{white-space:pre-wrap}summary{cursor:pointer;margin:.5rem 0}li{margin-bottom:.3rem}</style>",
            "<h1>OCR 逐题错误明细</h1><p>框序号从 0 开始。10.jpg 显示全部 23 道。OCR 列保留 TextRecognition 原始输出。</p>"]
    for group in groups:
        html.append(f"<h2>{cell(GROUP_LABELS[group])}</h2>")
        for name in ("val", "test", "pages", "fraction-crop", "fraction-page"):
            filename = name if name.startswith("fraction") else f"{name}-errors"
            rows = read_jsonl(RESULTS / group / f"{filename}.jsonl")
            html.append(f"<details><summary>{cell(SCOPE_LABELS[name])}（{len(rows)} 条）</summary>"
                        "<table><thead><tr><th>页 / 框</th><th>源标注</th><th>实际 OCR</th><th>标注 / OCR 判题</th>"
                        "<th>框、IoU、候选框 OCR</th></tr></thead><tbody>")
            for row in rows:
                actual = "未通过框匹配" if row.get("matched") is False else cell(row["ocr_text"]) or "（OCR 返回空字符串）"
                context = "—"
                if row.get("matched") is False:
                    candidates = [f"<li>框 {c['id']} {cell(c['bbox'])}，IoU {c['iou']:.6f}，OCR <code>"
                                  f"{cell(c['ocr_text']) or '（空字符串）'}</code>，匹配到标注框 "
                                  f"{cell(c['assigned_box_index'])}</li>" for c in row["candidates"]]
                    description = {"no_candidates": "无检测框", "below_iou_threshold": "候选 IoU 低于 0.50",
                                   "candidate_assigned_elsewhere": "候选已匹配到其它标注框"}[row["match_failure"]]
                    context = f"标注框 {cell(row['target_bbox'])}。{description}。"
                    if candidates:
                        context += "<ul>" + "".join(candidates) + "</ul>"
                elif row.get("matched") is True:
                    context = f"框 {row['prediction_id']} {cell(row['prediction_bbox'])}，IoU {row['match_iou']:.6f}"
                html.append(f"<tr><td>{cell(row['page'])} / {row['box_index']}</td><td><code>{cell(row['expected_raw'])}</code></td>"
                            f"<td><code>{actual}</code></td><td>{grade(row['truth_grade'])} / {grade(row['ocr_grade'])}</td>"
                            f"<td>{context}</td></tr>")
            html.append("</tbody></table></details>")
    html.append("</html>")
    (OUT / "error-details.html").write_text("\n".join(html), encoding="utf-8")


def render_curves(training):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, (loss_ax, accuracy_ax, lr_ax) = plt.subplots(1, 3, figsize=(12.5, 3.5), layout="constrained")
    for group, color in zip(GROUPS, ("#245e9f", "#a35416", "#25806b"), strict=True):
        record = training[group]
        steps = [point for point in record["curve"] if "train/loss" in point["metrics"]]
        loss_ax.plot([p["step"] for p in steps], [p["metrics"]["train/loss"] for p in steps],
                     label=group, color=color, linewidth=1.2)
        validation = record["validation"]
        accuracy_ax.plot([p["epoch"] for p in validation], [p["acc"] for p in validation],
                         marker="o", color=color, label=group)
        winner = validation[record["selected_epoch"] - 1]
        accuracy_ax.scatter(winner["epoch"], winner["acc"], s=105, facecolors="none", edgecolors=color, zorder=4)
        if group == "O1":
            lr_ax.plot([p["step"] for p in steps], [p["metrics"]["train/lr"] for p in steps], color="#444b56")
    loss_ax.set(xlabel="Optimizer step", ylabel="Loss", title="Training loss")
    accuracy_ax.set(xlabel="Epoch", ylabel="Exact accuracy", title="Full validation set", xticks=[1, 2, 3])
    accuracy_ax.yaxis.set_major_formatter(PercentFormatter(1))
    lr_ax.set(xlabel="Optimizer step", ylabel="Learning rate", title="Shared cosine schedule")
    lr_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for axis in (loss_ax, accuracy_ax, lr_ax):
        axis.grid(alpha=0.23, linewidth=0.7)
    loss_ax.legend(frameon=False)
    figure.savefig(OUT / "training-curves.png", dpi=160)
    plt.close(figure)


def verify_configs():
    manifest = load_json(OUT / "manifest.json")
    if sha(INITIAL) != manifest["initial_weights_sha256"]:
        raise ValueError("Official initialization changed after the audit")
    shared = None
    for group, data_dir in GROUPS.items():
        directory = OUT / group
        config = yaml.safe_load((directory / "train.yaml").read_text(encoding="utf-8"))
        if Path(config["Global"]["pretrained_model"]) != INITIAL:
            raise ValueError(f"{group}: unexpected initialization path")
        config["Global"].pop("save_model_dir")
        if Path(config["Train"]["dataset"].pop("data_dir")) != data_dir:
            raise ValueError(f"{group}: unexpected training image directory")
        for section, split in (("Train", "train"), ("Eval", "val")):
            dataset = config[section]["dataset"]
            label_file = [Path(path) for path in dataset.pop("label_file_list")]
            if label_file != [directory / f"{split}.txt"] or sha(label_file[0]) != manifest["splits"][split]["normalized_manifest_sha256"]:
                raise ValueError(f"{group}: unexpected {split} label manifest")
        if shared is None:
            shared = config
        elif config != shared:
            raise ValueError(f"{group}: shared training or validation settings differ")


def package():
    verify_configs()
    all_groups = (*MAIN, HISTORICAL)
    results = {group: load_json(RESULTS / group / "summary.json") for group in all_groups}
    if len({(r["detection_cache_sha256"], r["source_annotations_sha256"], r["ocr_device"]) for r in results.values()}) != 1:
        raise ValueError("Evaluation inputs differ")
    training = {group: train_curve(group) for group in GROUPS}
    swanlab = {group: load_json(RESULTS / group / "swanlab.json") for group in all_groups}
    deltas = paired_differences(all_groups)
    save_json(OUT / "comparison.json", {"results": results, "training": training, "paired_differences": deltas, "swanlab": swanlab})
    render_report(results, training, deltas, swanlab)
    render_details(all_groups)
    render_curves(training)
    paths = [OUT / name for name in ("comparison.md", "comparison.json", "error-details.html", "training-curves.png",
                                     "manifest.json", "train.txt", "val.txt", "test.txt")]
    if (OUT / "environment.json").is_file():
        paths.append(OUT / "environment.json")
    for verification in ("geometry-verification.json", "initial-weight-verification.json"):
        if (OUT / verification).is_file():
            paths.append(OUT / verification)
    paths.extend((OUT / "interrupted").glob("*/train.log"))
    paths.extend(p for p in RESULTS.rglob("*") if p.is_file())
    for group in GROUPS:
        paths.extend(p for p in (OUT / group).iterdir() if p.suffix in {".yaml", ".yml", ".log", ".tsv", ".txt", ".json"})
    for status in swanlab.values():
        if status["status"] == "offline-saved":
            run_dir = Path(status["local_log_dir"])
            paths.append(run_dir / f"run-{status['run_id']}.swanlab")
    source_paths = ("run_controlled_ocr.py", "evaluate_controlled_ocr.py", "auxiliary/reporting/publish_controlled_ocr.py", "evaluate_ocr.py",
                    "predict.py", "arithmetic.py", "evaluate_pipeline.py", "prepare_ocr_geometry.py", "optimizations/ocr_runtime.py",
                    "train_ocr.py", "vendor/PaddleOCR/tools/program.py", "vendor/PaddleOCR/ppocr/data/simple_dataset.py",
                    "vendor/PaddleOCR/ppocr/metrics/rec_metric.py", "vendor/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt",
                    "vendor/PaddleOCR/configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml")
    with ZipFile(OUT / f"{NAME}-results.zip", "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, path.relative_to(OUT))
        for relative in source_paths:
            archive.write(ROOT / relative, f"source/{relative}")
    print(OUT / f"{NAME}-results.zip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("package", *MAIN, HISTORICAL))
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    if args.target == "package":
        package()
    else:
        sync(args.target, args.mode)


if __name__ == "__main__":
    main()
