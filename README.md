# ModelAdjust：独立算式识别与优化实验

## 网页应用与云端部署

Windows：双击 `start_local.bat`，打开 `http://127.0.0.1:7860`。

CloudStudio：按 [部署说明](deploy/CLOUDSTUDIO.md) 导入仓库，启动命令为 `bash deploy/start-cloud.sh`，主预览端口为 `7860`。`.vscode/preview.yml` 已配置。CloudStudio 实际部署和识别验收由用户完成。

仓库内 `models/` 已包含最佳检测器 Y2-geometry-2 和统一重训 OCR O3，无需下载历史训练记录。`python deploy/check_models.py` 校验四个模型文件的 SHA-256。网页默认读取这些打包文件；训练代码和历史说明仍保留原实验路径，训练数据、虚拟环境、日志和输出不纳入版本库。云端使用独立 CPU 依赖，不改动本地 GPU 环境。

输入一张作业图片，YOLO 检测算式框，PaddleOCR 识别文字，受限算术解析器计算等号两边，输出 JSON。无需启动 Label Studio。已有 backend 未修改。

## 代码目录

- 根目录只保留常用入口：YOLO/OCR 训练、评估、整页对照、预测、判题，以及数据集准备与增强。
- `optimizations/`：数据增强与训练运行工具。
- [weight_editing/](weight_editing/README.md)：权重空间编辑、官方起点对照、阈值校准和机制诊断，含专用启动脚本及骨干编辑方案。
- [auxiliary/](auxiliary/README.md)：标签审计、报告汇总、错误排查及分式专项实验。
- `logs/`：原先散落在根目录的训练与测试日志。
- `runs/`、`weights/`、`datasets_oriented/`、`experiment_data/`：已有实验记录、权重和数据，路径沿用原设置。

## 运行环境

继续使用 D:/conda/envs/labelstudioml/python.exe。依赖见 requirements.txt；GPU 版 torch、PaddlePaddle 沿用当前环境，不要为接入日志重装它们。SwanLab 云端使用前在该环境执行 swanlab login。正式训练和评估默认 online，使用当前已登录账号上传云端；无网络时可显式 --swanlab-mode offline。

## 预测

### 本地网页：拖图识别与批改

双击项目上一级目录的 `start_formula_ui.bat`，然后打开 `http://127.0.0.1:7860`。保持启动窗口运行，关闭窗口或按 Ctrl+C 停止服务。

也可从 ModelAdjust 目录启动：

```powershell
& 'D:/conda/envs/labelstudioml/python.exe' web_app.py
# 修改端口，或强制使用 CPU
& 'D:/conda/envs/labelstudioml/python.exe' web_app.py --port 7861 --device cpu
```

网页默认组合是 `runs/Y2-geometry-2/weights/best.pt` + `runs/ocr-controlled-v1/O3/inference`，使用 TextRecognition 读取完整检测裁图，检测阈值 0.50、尺寸 640。根据 `reports/实验3报告正文汇总-20261118.md` 的统一整页对照，此组合全文正确 2730/2767（98.66%），判题正确 2733/2757（99.13%），为本次候选中整页表现最佳的组合。此处不使用历史 `weights/formula_detector.pt`。

支持拖入或选择 JPG、PNG、WEBP 图片（最多 20 MB、2500 万像素），显示全部检测裁图、OCR 原文、置信度、等号两侧数值和正确/错误/待确认状态。可按状态筛选，点击图片查看原尺寸，并下载原图、标注图、全部裁图及 result.json 的 ZIP。判题依据 OCR 文字，未检测到的题目不会显示；无法解析或未填写答案时保留待确认。

默认自动使用可用 GPU，模型首次识别时加载，后续复用。所有结果保存在 `outputs/web/<任务编号>/`，不会自动清理。页面资源与推理均在本机，不需要前端构建或外部 CDN。启动脚本使用当前已有的 labelstudioml 环境；迁移机器时需修改脚本中的 Python 路径，并保留上述权重目录。网页额外依赖 Flask，已列入 requirements.txt。

### 命令行预测

从 ModelAdjust 目录执行：

```powershell
& 'D:/conda/envs/labelstudioml/python.exe' predict.py '../实验3/train/images/1.jpg' --device 0 --ocr-device gpu:0 --output outputs/result.json
```

默认使用 weights/formula_detector.pt（从原 backend 复制的历史模型）。整页识别和评估都默认用 PP-OCRv6 的 `TextRecognition` 直接读取完整的 YOLO 裁图。复现旧流程时可指定 `--ocr-mode pipeline`。初次运行可能下载官方模型。--weights 可指定新 YOLO 权重，--rec-dir 可指定配套导出的 OCR 识别模型目录。CPU 使用 --device cpu --ocr-device cpu。

每条结果包含 bbox、检测/OCR 置信度、原始文字、规范化文字、等号两侧数值、is_correct 和 status。correct/incorrect 分别代表计算正确/错误；空答案、无法解析、除零或多行歧义返回 uncertain，is_correct=null。支持四则运算、括号、负数、小数；分数用精确有理数计算。暂不支持竖式、余数、方程、连等式，不会把这些强制判错。没有识别出的漏检题目无法出现在结果中，仍需要单独评估检测召回率。判题仅依据 OCR 文字，不保证 OCR 本身正确。

## 数据准备

```powershell
& 'D:/conda/envs/labelstudioml/python.exe' prepare_data.py
```

读取 ../实验3/train/annotations.json，按整图固定种子划分约 70/15/15，像素完全相同的图片归入同一组，避免跨集合。输出 datasets_oriented/yolo 和 datasets_oriented/ocr，后者包含裁剪图片与路径 TAB 文字的标签列表。保留原始错误答案；空文字只跳过 OCR 导出，检测框保留。旋转框转换为水平外接框。输出目录已存在时拒绝覆盖；新实验可指定 --output datasets_v2。生成的配置需要传给训练入口的 --data。

split_manifest.json 记录划分、空文字数量、尺寸差异。坐标为原始百分比，按实际像素尺寸换算；图片先按 EXIF 转正再保存，转正后尺寸仍与标注不符则报错。近似重复、同一学生/试卷来源目前无法自动识别，应在正式实验前核查。pairing_report.json 中已跳过的缺图、空框样本不会自动作为负样本加入。人工修正后的标注应单独保留版本。

## YOLO 与 SwanLab

```powershell
& 'D:/conda/envs/labelstudioml/python.exe' train_yolo.py --epochs 1 --name smoke --swanlab-mode offline
& 'D:/conda/envs/labelstudioml/python.exe' train_yolo.py --epochs 100 --name baseline --swanlab-mode online
```

训练起点为 weights/yolo11n.pt，输出 runs/。官方回调记录配置、损失、学习率、验证指标和生成的图表。基线显式关闭主要随机增强；后续增强组应记录新配置并使用相同划分。训练不会覆盖推理权重，训练完使用 --weights 指向 runs 下的 best.pt。历史 formula_detector.pt 曾训练过实验2中的图片，不适合作为新划分的公平测试基线，应从原始预训练权重重训。

已提供 YOLO 与 PaddleOCR 微调、训练前增强、统一测试和整页判题评估。最终实测对比见 [reports/comparison.md](reports/comparison.md)，OCR 环境与训练细节见 [configs/ocr/README.md](configs/ocr/README.md)。测试集不用于选参。

## 验证

```powershell
& 'D:/conda/envs/labelstudioml/python.exe' -m unittest discover -s tests -v
```

正式默认数据为 datasets_oriented/，integration-smoke 是早期调试记录，不用于实验结论。Windows 日志如出现 GBK 解码异常，运行前设置 $env:PYTHONUTF8='1'。



## 对照实验

### 新增几何增强候选 Y2-geometry

`configs/yolo/Y2-geometry.yaml` 使用 Y1 已生成的画质增强训练图，再通过 YOLO 原生参数动态施加旋转（±3°）、平移（宽高 ±3%）、缩放（约 0.9～1.1 倍）、错切（x/y 轴各 ±3°）和透视（强度 0.0002）。这些变换随机组合，不按互斥图片比例分配。验证集与测试集不做上述增强。

```powershell
python train_yolo.py --config configs/yolo/Y2-geometry.yaml --swanlab-mode online
python evaluate_yolo.py --group Y2-geometry --weights runs/Y2-geometry/weights/best.pt --swanlab-mode online
```

运行前需要已有 `experiment_data/Y1/yolo.yaml`（缺少时运行 `python prepare_experiment.py --groups Y1`）。不要使用旧 Y2 图片作为输入，以免重复施加旋转。原生增强会同步变换、裁剪和过滤检测框，但不保证边缘算式完整保留；训练后应检查生成的训练批次预览。

旧 `configs/yolo/Y2.yaml`、`runs/Y2` 和报告中的 Y2 仍代表离线旋转实验。新组使用独立训练目录、SwanLab 名称及 `reports/results/Y2-geometry.json`；已有对比报告和 `run_pipeline_comparison.py` 尚不纳入这个新候选。先比较验证成绩，再决定是否更新最终测试与整页组合。新增配置本身不代表已完成训练或获得新成绩。

从本目录运行，沿用当前已登录的 SwanLab 账号。默认实验项目为 ModelAdjust。

```powershell
$env:PYTHONUTF8 = '1'
python train_yolo.py --config configs/yolo/Y0.yaml --swanlab-mode online
python train_yolo.py --config configs/yolo/Y1.yaml --swanlab-mode online
python train_yolo.py --config configs/yolo/Y2.yaml --swanlab-mode online
python evaluate_yolo.py --group Y0 --weights runs/Y0/weights/best.pt
python evaluate_yolo.py --group Y1 --weights runs/Y1/weights/best.pt
python evaluate_yolo.py --group Y2 --weights runs/Y2/weights/best.pt
python train_ocr.py --config configs/ocr/O1.yaml --swanlab-mode online
python train_ocr.py --config configs/ocr/O2.yaml --swanlab-mode online
python evaluate_ocr.py --group O0 --swanlab-mode online
python evaluate_ocr.py --group O1 --swanlab-mode online
python evaluate_ocr.py --group O2 --swanlab-mode online
python run_pipeline_comparison.py --swanlab-mode online
python auxiliary/reporting/build_report.py
```

Y0 使用方向校正后的原训练图。Y1 使用画质增强图。Y2 在同样的画质增强上增加轻微旋转，并同步转换检测框。O2 对 OCR 裁剪图做画质增强，文字标签保持一致。增强数据及来源映射位于 experiment_data，三个训练组使用同一份验证与测试数据。

训练前先运行 `python prepare_experiment.py --groups Y1 Y2 O2` 生成固定的增强版本。各组保持相同训练样本数与训练预算，不使用水平/垂直翻转文字。数据审计位于 experiment_data/data_audit.json。已经存在的训练输出目录拒绝覆盖；重新实验应先在新目录保留旧结果。

整页评估示例（权重目录应使用实际完成的训练组）：

```powershell
python evaluate_pipeline.py --name Y0-O1 --weights runs/Y0/weights/best.pt --rec-dir runs/O1/inference --swanlab-mode online
python auxiliary/reporting/build_report.py
```

整页评估按检测置信度顺序，以 IoU ≥ 0.5 对预测框和真值框进行一对一匹配。全文匹配率以有文本真值的题目为分母。判题成功率以能够从标注文字计算对错的题目为分母，其中漏检和无法判定计为未成功。逐页结果和计数在 reports/results，对比表由实际评估 JSON 生成。
