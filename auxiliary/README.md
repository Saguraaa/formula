# 辅助脚本

常用训练、测试、预测和数据集准备入口保留在 `ModelAdjust/`。这里集中放标签审计、结果汇总、专项排查和历史分式对照脚本，权重编辑代码单独放在 `weight_editing/`。

## 目录和用途

| 目录 | 文件 | 用途 |
|---|---|---|
| `label_audit/` | `audit_all_labels.py` | 全量标注初筛 |
| `label_audit/` | `audit_label_second_opinion.py` | 使用另一识别模型交叉核查 |
| `label_audit/` | `apply_reviewed_label_corrections.py` | 应用已人工确认的标签修正 |
| `label_audit/` | `build_label_audit_report.py` | 生成标签审计页面 |
| `reporting/` | `build_report.py` | 汇总常规检测、OCR 和整页结果 |
| `reporting/` | `publish_controlled_ocr.py` | 统一 OCR 结果打包及 SwanLab 同步 |
| `reporting/` | `report_full_ocr_comparison.py` | 汇总历史全文 OCR 对照 |
| `diagnostics/` | `recheck_formula_errors.py` | 分式错误专项复核 |
| `diagnostics/` | `recheck_wrong_accepts.py` | 错题判对复查与详情输出 |
| `fraction_experiments/` | `compare_fraction_recognizers.py` | 普通 OCR 与公式识别器对照 |
| `fraction_experiments/` | `train_ocr_fraction_pilot.py` | 分式子集 OCR 微调 |
| `fraction_experiments/` | `evaluate_formula_pilot.py` | 公式识别试验评估 |
| `fraction_experiments/` | `prepare_full_ocr_comparison.py` | 历史全文对照数据与配置准备 |
| `fraction_experiments/` | `evaluate_full_ocr_comparison.py` | 历史全文对照评估 |

## 运行方式

沿用 `D:/conda/envs/labelstudioml/python.exe` 和原来的依赖设置，从 `ModelAdjust/` 执行，只调整脚本路径，原参数继续使用。例如：

```powershell
$Python = 'D:/conda/envs/labelstudioml/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = (Resolve-Path vendor/ocr_deps).Path

& $Python auxiliary/reporting/build_report.py
& $Python auxiliary/reporting/publish_controlled_ocr.py --help
& $Python auxiliary/diagnostics/recheck_wrong_accepts.py --help
& $Python auxiliary/fraction_experiments/train_ocr_fraction_pilot.py --help
```

也支持从 `ModelAdjust/` 使用包入口，例如 `python -m auxiliary.reporting.build_report`。数据、权重和结果路径仍相对于 `ModelAdjust/`，继续使用已有的 `datasets_oriented/`、`datasets_formula/`、`runs/`、`outputs/` 和 `reports/`。

部分历史脚本直接执行时就会运行模型、生成报告或应用标签修正：`audit_label_second_opinion.py`、`build_label_audit_report.py`、`apply_reviewed_label_corrections.py`、`recheck_formula_errors.py` 和 `evaluate_formula_pilot.py`。这些脚本沿用原来的执行方式，没有通用的 `--help` 参数；应按对应实验记录使用。

根目录原有的七份 `.log` 文件已移入 `logs/`，文件名和内容保持原样。历史实验记录中的旧命令保留作为当时的运行记录，新执行命令使用上表路径。
