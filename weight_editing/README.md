# 权重空间编辑

权重编辑代码统一放在这个目录。常规 YOLO、OCR 微调入口保留在上一级 `ModelAdjust/`，评估与数据工具继续共享。

## 文件用途

| 文件 | 用途 |
|---|---|
| `weight_edit.py` | O3 低秩编辑的数据准备、导出、评估、候选锁定与报告 |
| `train_weight_edit.py` | O3 字符分类矩阵低秩更新及保持约束训练 |
| `run_weight_edit.ps1` | 上述实验的分阶段启动入口 |
| `official_weight_edit.py` | 官方初始化下 YOLO、OCR 全量适配与低秩编辑对照 |
| `calibrate_official_weight_edit.py` | 官方起点对照的检测阈值校准与评估 |
| `run_official_weight_edit.ps1` | 官方起点对照启动入口 |
| `direct_ocr_edit.py` | 无梯度字符列定向编辑与随机对照 |
| `diagnose_ocr_mechanism.py` | 错误分布、模块交换和骨干机制诊断 |
| [官方OCR无梯度骨干编辑执行方案.md](官方OCR无梯度骨干编辑执行方案.md) | 后续骨干直接编辑方案 |

## 运行方式

在 `ModelAdjust/` 下使用原有 `labelstudioml` 环境。两个 PowerShell 启动脚本会自动切换到 `ModelAdjust/` 并设置依赖路径。

```powershell
# O3 低秩编辑，仍支持 prepare / smoke / formal / final / all
.\weight_editing\run_weight_edit.ps1 -Stage smoke

# 官方起点对照；-Calibrate 追加阈值校准流程
.\weight_editing\run_official_weight_edit.ps1 -Calibrate
```

单独运行 Python 入口前，沿用启动脚本中的环境设置：

```powershell
$Python = 'D:/conda/envs/labelstudioml/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = (Resolve-Path vendor/ocr_deps).Path
$torchLib = Join-Path (Split-Path -Parent $Python) 'Lib/site-packages/torch/lib'
$env:Path = "$torchLib;$env:Path"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = 'True'

& $Python weight_editing/weight_edit.py --help
& $Python weight_editing/official_weight_edit.py --help
# 也支持 python -m weight_editing.weight_edit --help
```

直接编辑入口改为 `weight_editing/direct_ocr_edit.py`，原子命令继续使用：`baseline train`、`baseline val`、`discover`、`extract`、`design`、`pack`、`evaluate direct val`、`evaluate random val`、`lock`、`evaluate direct test`、`evaluate random test`、`report`。`design` 需要已有的训练核图记录。机制诊断入口改为 `weight_editing/diagnose_ocr_mechanism.py`，保留原有子命令和参数。

这些命令会执行对应实验步骤；整理目录后的检查仅验证入口、导入和路径。

## 数据和实验记录

所有路径仍以 `ModelAdjust/` 为根目录，已有数据、模型和运行记录继续放在原位置：

- O3 低秩编辑：`runs/ocr-weight-edit-v1/`
- 官方起点对照及校准：`runs/official-weight-edit-v1/`
- 无梯度字符编辑：`runs/direct-ocr-edit-v1/`
- 机制诊断：`runs/ocr-mechanism-diagnosis-v1/`

历史记录里的旧命令和源码哈希保留作为当时的运行证据；再次执行入口时使用本目录的新路径。新生成的源码校验记录会指向 `weight_editing/` 中的文件。实验算法、参数、选择条件和输出目录沿用原设置。
