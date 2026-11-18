# YOLO 几何增强对比（2026-11-11）

固定测试集：88 页，3014 个标注框。四组均为 YOLO11n、30 轮、batch 8、640 像素、seed 42，按验证成绩选择最佳权重。

| 组别 | precision | recall | mAP50 | mAP50–95 | 测试记录 |
| --- | ---: | ---: | ---: | ---: | --- |
| Y0 | 98.32% | 99.00% | 98.88% | 58.89% | [SwanLab](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/7vzxdcac) |
| Y1 | 98.58% | 99.12% | 98.58% | 59.65% | [SwanLab](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/xo2yr0nl) |
| Y2 | 98.62% | 99.25% | 98.54% | 58.77% | [SwanLab](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/nlm9sied) |
| Y2-geometry | 98.86% | 99.40% | 98.53% | 59.22% | [SwanLab](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/iqh3cs3g) |

Y0：基础微调；Y1：画质增强；Y2：历史画质增强＋离线旋转；Y2-geometry：Y1 图片＋原生在线几何增强。

新组参数：degrees=3，shear=3，translate=0.03，scale=0.1，perspective=0.0002。
新组最佳验证轮次：28，验证 mAP50–95：60.12%。

实际权重：runs/Y2-geometry-2/weights/best.pt。首次网络受限启动未训练，重试自动使用该目录。
训练记录：https://swanlab.cn/@Saguraaa/ModelAdjust/runs/3fbka5id

新组精确率和召回率最高，但测试 mAP50–95 仍低于 Y1。单次固定种子实验不能证明统计显著性，也不能单独归因于透视。整页 OCR/判题复评见下表。

## 整页 OCR 与判题复评（2026-11-11）

固定 O2 识别权重、相同 88 页测试图片、标注与阈值，仅替换 YOLO。数据与 OCR 权重哈希已核对一致。

| 指标 | Y1＋O2（历史结果） | Y2-geometry＋O2 | 差值（百分点） |
| --- | ---: | ---: | ---: |
| 检测召回率 | 98.94% | 99.27% | +0.33 |
| 算式全文识别成功率 | 94.69% | 95.19% | +0.51 |
| 判题成功率 | 95.39% | 95.86% | +0.47 |
| 判题覆盖率 | 98.66% | 98.87% | +0.22 |
| 错题误判正确率（越低越好） | 35.06% | 34.42% | -0.65 |

全文识别正确：2620 → 2634 / 2767，多识别正确 14 道；判题正确：2626 → 2639 / 2753，多判对 13 道。
检测漏框：32 → 22；未匹配预测框：41 → 50。错题误判为正确：54 → 53 / 154。
单页耗时：历史 859.85 ms，本次 980.96 ms；两次测量不同时段，不能据此单独归因于模型速度变化。
全文识别成功率按有文字标注的算式统计，包含漏检失败；不表示整张作业全部识别正确的比例。
[新组 SwanLab 记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/ehfu7ui8)

结论：此次新几何组的整页识别与判题指标小幅提高，但错题误判仍较多；单次实验不能证明统计显著性。
