"""Calibrate each detector on validation IoU matches, then score the locked operating points."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weight_editing import official_weight_edit as experiment
from weight_editing.official_weight_edit import DATA, COMBINATIONS, MODES, OUT, detector_path, box_to_yolo, match, read_bgr
from weight_editing.official_weight_edit import detect_crops, grade_summary, load_json, save_json, save_jsonl, sha

ROOT = OUT / "calibrated"
CONFIDENCES = (0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
PAIRS = {name: modes for name, modes in COMBINATIONS.items() if "initial" not in modes}
LOCK = ROOT / "threshold-lock.json"


def scan(split):
    from ultralytics import YOLO

    annotations, splits = experiment.old.annotations_and_split()
    locked = load_json(LOCK) if split == "test" else None
    if locked:
        if (locked["official_test_lock_sha256"] != sha(OUT/"test-lock.json") or
                any(sha(Path(path)) != h for path,h in locked["inputs_sha256"].items())):
            raise ValueError("Threshold calibration inputs changed")
        experiment.check_lock()
    by_mode = {}
    for mode in MODES:
        path = detector_path(mode)
        folder = ROOT / split / f"detector-{mode}"
        folder.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(path))
        if locked:
            thresholds = (locked["selected"][mode]["confidence"],)
        else:
            thresholds = CONFIDENCES
        totals = {conf:dict(matched=0, predicted=0, targets=0) for conf in thresholds}
        pages = []
        for name in splits[split]:
            source=DATA/"yolo/images"/split/name
            image=read_bgr(source)
            height,width=image.shape[:2]
            boxes=detect_crops(model,image,0,min(thresholds),640)
            pages.append(dict(page=name,sha256=sha(source),detections=boxes))
            if not locked:
                targets=[]
                for box in annotations[name]:
                    x,y,w,h=box_to_yolo(box,width,height)
                    targets.append(dict(bbox=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]))
                for confidence, counts in totals.items():
                    filtered=[b for b in boxes if b["detection_confidence"]>=confidence]
                    counts["matched"] += len(match(filtered,targets,.5))
                    counts["targets"] += len(targets)
                    counts["predicted"] += len(filtered)
        if locked:
            threshold=thresholds[0]
        else:
            trials=[]
            for confidence,c in totals.items():
                denominator=c["targets"]+c["predicted"]
                f1=2*c["matched"]/denominator if denominator else 0.0
                trials.append(dict(confidence=confidence,f1=f1,**c))
            selected=max(trials,key=lambda r:(r["f1"],r["matched"],r["confidence"]))
            threshold=selected["confidence"]
            by_mode[mode]=selected
            save_json(folder/"trials.json",trials)
        for page in pages:
            page["detections"]=[b for b in page["detections"] if b["detection_confidence"]>=threshold]
        save_json(folder/"boxes.json",dict(weights=sha(path),confidence=threshold,pages=pages))
        print(split,mode,"confidence",threshold,"boxes",sum(len(p["detections"]) for p in pages),flush=True)
    if not locked:
        inputs=[detector_path(mode) for mode in MODES]
        inputs += [ROOT/"val"/f"detector-{mode}"/"trials.json" for mode in MODES]
        inputs += [experiment.ANNOTATIONS, DATA/"split_manifest.json"]
        save_json(LOCK,dict(selected=by_mode,grid=CONFIDENCES,
            rule="Highest validation detection F1 at IoU 0.50, then most matches, then highest confidence",
            official_test_lock_sha256=sha(OUT/"test-lock.json"),
            inputs_sha256={str(p):sha(p) for p in inputs}))


def score_pages():
    experiment.check_lock()
    calibration=load_json(LOCK)
    annotations,splits=experiment.old.annotations_and_split()
    results={}
    for mode in MODES:
        rec=experiment.old.recognizer(experiment.ocr_path(mode))
        for split in ("val","test"):
            results.setdefault(split,{})
            for pair,(yolo,ocr) in PAIRS.items():
                if ocr!=mode:
                    continue
                cache=ROOT/split/f"detector-{yolo}"/"boxes.json"
                if load_json(cache)["confidence"]!=calibration["selected"][yolo]["confidence"]:
                    raise ValueError("Detection cache threshold disagrees with lock")
                rows,fp=experiment.page_rows(rec,split,yolo,annotations,splits,detection_cache=cache)
                summary=grade_summary(rows)
                if summary["counts"]["labeled"]!=experiment.old.COUNTS[split]:
                    raise ValueError(f"Missing {split} labeled pages for {pair}")
                summary["counts"]["false_positive_boxes"]=fp
                folder=ROOT/split/pair
                folder.mkdir(parents=True,exist_ok=True)
                save_jsonl(folder/"rows.jsonl",rows)
                save_json(folder/"metrics.json",summary)
                results[split][pair]=summary
                print(split,pair,"matched",summary["counts"]["matched"],"grade correct",summary["counts"]["grade_correct"],flush=True)
        del rec
    save_json(ROOT/"comparison.json",results)
    fixed=load_json(OUT/"comparison.json")
    lines=["# 验证集置信度校准后的判题对照","",
           "使用同一批 200 步权重。两个检测器各自在 87 页验证集的十个置信度点中选择检测 F1 最高的点。"
           "匹配阈值 IoU 固定为 0.50。选定后锁定，再用 88 页已有测试集评估。","",
           "| 检测器 | 锁定置信度 | 验证匹配框 | 验证输出框 | 验证检测 F1 |",
           "|---|---:|---:|---:|---:|"]
    for mode, row in calibration["selected"].items():
        lines.append(f"| {mode} | {row['confidence']:.3g} | {row['matched']} | {row['predicted']} | {row['f1']:.2%} |")
    lines.extend(["","| 阈值 | 划分 | 组合 | 匹配框 | 多余框 | 判题正确数 | 判题准确率 | 平衡准确率 | 错题判对 | 无法判定 |",
                   "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for label,data in (("0.50",fixed),("已校准",results)):
        for split in ("val","test"):
            for pair in PAIRS:
                c=data[split][pair]["counts"]
                m=data[split][pair]["metrics"]
                lines.append(f"| {label} | {split} | {pair} | {c['matched']} | {c['false_positive_boxes']} | "
                    f"{c['grade_correct']}/{c['gradeable']} | {m['grading_accuracy']:.2%} | "
                    f"{m['balanced_grading_accuracy']:.2%} | {c['wrong_accepted']} | {c['undetermined']} |")
    (ROOT/"results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("val","test","report"))
    stage=parser.parse_args().stage
    if stage=="report":
        score_pages()
    else:
        scan(stage)

