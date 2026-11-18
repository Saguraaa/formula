"""Verify full-data coverage and summarize the paired OCR evaluation."""
import sys as _sys
from pathlib import Path as _Path

if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets_oriented/ocr"
OUT = ROOT / "outputs/ocr-full-comparison"
GROUPS = {
    "text-base": "PP-OCRv6 官方预训练",
    "text-finetuned": "PP-OCRv6 全量微调",
    "formula-base": "PP-FormulaNet_plus-S 官方预训练",
    "formula-finetuned": "PP-FormulaNet_plus-S 全量微调",
}


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    lines = ["# 普通 OCR 与算式模型全量对照", "",
             "两组微调使用原实验划分中完全相同的 13,206 张训练裁图。"
             "使用当前标注，在同一批 2,738 张验证裁图与 2,767 张测试裁图上评估。"
             "训练各一轮，批大小 16，随机种子 42，并处理 Windows 上的最后一批。", "",
             "普通 OCR 全参数微调，初始学习率 1e-4。算式模型固定视觉编码器，仅微调解码器，初始学习率 1e-6。"
             "两组均测固定一轮权重，不通过测试集挑选权重。", "",
             "普通 OCR 使用一行文字标签。算式模型使用 LaTeX 标签，识别时最多生成 128 个标记。"
             "评分统一分式和常见算符写法，保留数字、分子分母、符号及学生原答。"
             "测试集包含线性算式和上下分式。上下分式子集单独统计原测试页 10.jpg 的 23 张有文字标签的裁图。", "",
             "| 模型 | 验证全文匹配 | 测试全文匹配 | 测试字符错误率 | 测试上下分式全文匹配 |",
             "|---|---:|---:|---:|---:|"]
    for group, description in GROUPS.items():
        results = {}
        for split in ("val", "test"):
            summary = json.loads((OUT / f"{group}-{split}.json").read_text())
            with (OUT / f"{group}-{split}.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            labels = DATA / f"{split}.txt"
            expected_names = [line.split("\t", 1)[0] for line in labels.read_text(
                encoding="utf-8").splitlines()]
            assert summary["labels_sha256"] == manifest[split]["source_sha256"] == hashlib.sha256(
                labels.read_bytes()).hexdigest()
            assert [r["image"] for r in rows] == expected_names
            assert summary["count"] == len(rows) == manifest[split]["count"]
            assert summary["correct"] == sum(r["correct"] == "True" for r in rows)
            results[split] = summary
        fraction_rows = [r for r in rows if Path(r["image"]).stem.startswith("10_")]
        assert len(fraction_rows) == 23
        fraction_correct = sum(r["correct"] == "True" for r in fraction_rows)
        val, test = results["val"], results["test"]
        lines.append(f"| {description} | {val['correct']}/{val['count']} ({val['correct']/val['count']:.2%}) | "
                     f"{test['correct']}/{test['count']} ({test['correct']/test['count']:.2%}) | {test['cer']:.2%} | "
                     f"{fraction_correct}/{len(fraction_rows)} |")
    lines += ["", "每组的逐图识别结果、训练日志和标签版本哈希保存在 outputs/ocr-full-comparison。"]
    report = "\n".join(lines) + "\n"
    (ROOT / "reports/ocr-full-comparison.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
