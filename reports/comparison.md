# 模型对比实验

正式测试结果取自下列各组的评估文件。数值为比例，耗时单位为毫秒。
“未测”表示缺少评估文件。流程验证和验证集成绩不进入此表。

Y0 为无额外增强的检测微调组，Y1 加入画质增强，Y2 再加入小角度旋转。
O0 为官方预训练识别模型，O1 使用原始裁剪图微调，O2 使用增强裁剪图微调。
整页组合中的 Ybest 是 Y1、Y2 中验证 mAP50–95 较高的一组，Obest 表示 OCR 增强候选 O2。

## YOLO 检测

| 组别 | precision | recall | map50 | map50_95 | SwanLab |
| --- | --- | --- | --- | --- | --- |
| [Y0](results/Y0.json) | 0.9832 | 0.9900 | 0.9888 | 0.5889 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/7vzxdcac) |
| [Y1](results/Y1.json) | 0.9858 | 0.9912 | 0.9858 | 0.5965 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/xo2yr0nl) |
| [Y2](results/Y2.json) | 0.9862 | 0.9925 | 0.9854 | 0.5877 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/nlm9sied) |

## OCR 识别

| 组别 | exact_rate | cer | SwanLab |
| --- | --- | --- | --- |
| [O0](results/O0.json) | 0.9017 | 0.0196 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/u6r3jiti) |
| [O1](results/O1.json) | 0.9617 | 0.0050 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/19pb2xf2) |
| [O2](results/O2.json) | 0.9631 | 0.0049 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/koxbuen2) |

## 整页判题

| 组别 | detection_recall | exact_rate | grading_success | grading_coverage | wrong_accept_rate | ms_per_page | SwanLab |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Y0-O1](results/Y0-O1.json) | 0.9721 | 0.9317 | 0.9393 | 0.9702 | 0.3312 | 880.1830 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/fvp5vlds) |
| [Ybest-O1](results/Ybest-O1.json) | 0.9894 | 0.9458 | 0.9528 | 0.9858 | 0.3571 | 860.1948 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/w6a58gul) |
| [Y0-Obest](results/Y0-Obest.json) | 0.9721 | 0.9339 | 0.9408 | 0.9724 | 0.3312 | 894.5990 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/ri49i88s) |
| [Ybest-Obest](results/Ybest-Obest.json) | 0.9894 | 0.9469 | 0.9539 | 0.9866 | 0.3506 | 859.8471 | [线上记录](https://swanlab.cn/@Saguraaa/ModelAdjust/runs/l10yhbc3) |

## 实验口径

- 固定整页划分：训练 406、验证 87、测试 88；OCR 原始有效裁剪为 13206 / 2738 / 2767。增强仅作用于训练集，测试图片不增强。
- Y0/Y1/Y2 均从 YOLO11n 官方预训练参数开始，30 轮、batch 8、640 像素、seed 42，按验证 mAP50–95 保存最佳权重。
- O1/O2 均从 PP-OCRv6_medium_rec 官方训练参数开始，3 轮、batch 16、学习率 0.0001、seed 42。按原生字典与长度上限过滤后，训练 13084、验证 2681；最终评估保留全部 2767 个测试裁剪。
- 画质增强包含亮度、对比度、高斯噪声及概率性轻微模糊；Y2 额外尝试 ±3° 旋转，检测框同步变换。文字不翻转，错误算式标签不改写。
- OCR exact_rate 为规范化后全文匹配率；cer 为总编辑距离除以真值总字符数，越低越好。
- 整页检测以 IoU ≥ 0.5 按置信度进行一对一匹配；exact_rate 包含漏检失败。grading_success 以真值文字能计算对错的题目为分母，漏检与无法判断计为失败；grading_coverage 是其中给出明确判定的比例。
- wrong_accept_rate 为真值错误题目被判为正确的比例，越低越好。仅比较真假判定不能证明全文识别正确，需结合 exact_rate。空文本/不支持的算式不计入判题分母，原始计数保存在 JSON。
- 测试集共 3014 个标注框，2767 个有文字；规则可判题 2753 个，其中正确 2599、错误 154。其余 261 个不进入判题分母。类别明显不均衡，因此单看判题准确率不足以评价系统。
- 整页耗时包含单页读取、YOLO、PaddleOCR 及算术规则，排除模型加载与一次预热；同机串行测试。各组合使用相同 OCR 文本检测器，仅替换识别权重。
- Windows GPU CTC 兼容问题采用 CPU CTC 前后向与 GPU 梯度桥接，其余 OCR 网络在 GPU 训练。详见 configs/ocr/README.md。
- 每组为一次固定种子实验，差异不是统计显著性结论。完全重复页面未跨划分；近似重复或来源相关性仍需人工核查。

Y1/Y2 的画质扰动使用同一分布；几何采样会改变随机数消耗，因此两组并非逐像素相同的画质扰动。Y2 的结果只能说明该组合在本次运行中的表现，不能单独证明旋转的因果效果。
