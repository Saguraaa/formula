# OCR 对照实验

在 ModelAdjust 目录使用已有的 GPU Paddle 环境。O0 是官方 PP-OCRv6_medium_rec 推理权重；O1 使用原始训练裁图微调；O2 使用同样标签的增强训练裁图微调。评估三组使用相同的整页划分与识别接口。

首次准备依赖时使用下列命令。训练脚本在独立子进程中加载 vendor/ocr_deps，当前环境的 Python 包无需变更。

```powershell
git clone https://github.com/PaddlePaddle/PaddleOCR.git vendor/PaddleOCR
git -C vendor/PaddleOCR checkout dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf
python -m pip install --target vendor/ocr_deps --no-deps -r configs/ocr/training-requirements.txt
curl.exe -fL -o weights/PP-OCRv6_medium_rec_pretrained.pdparams https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams
```

先准备 O2 数据，要求 experiment_data/O2/ocr/train.txt 与原始训练裁图清单拥有相同标签和相同顺序。所有正式训练和评估默认使用已登录的 SwanLab 在线项目 ModelAdjust。

```powershell
python -X utf8 evaluate_ocr.py --group O0
python -X utf8 train_ocr.py --config configs/ocr/O1.yaml --epochs 3 --batch 16
python -X utf8 train_ocr.py --config configs/ocr/O2.yaml --epochs 3 --batch 16
python -X utf8 evaluate_ocr.py --group O1 --split val
python -X utf8 evaluate_ocr.py --group O2 --split val
python -X utf8 evaluate_ocr.py --group O1
python -X utf8 evaluate_ocr.py --group O2
```

训练参数保存于 runs/O1/train.yaml 和 runs/O2/train.yaml。两个微调组共享 Adam、初始学习率 0.0001、种子 42、预训练参数、识别尺寸 48×320、25 字长度限制。只在训练和官方验证过滤字符字典之外或超长的标签，完整的 OCR 对照评估保持相同测试清单。Windows 版官方训练程序略过每轮末批，按实际步数触发验证；使用其验证全文准确率选 best_accuracy，再用同一个评估脚本测完整验证集和测试集。运行目录拒绝覆盖已有实验。

当前 Windows Paddle 3.0.0 开发版的 GPU warp-CTC 在 RTX 5080 上报 workspace 错误。项目 runtime 使用官方 CPU CTC 算子与自定义 autograd 桥接，前后向梯度传回 GPU，NRTR 和模型仍在 GPU 运行。微型用例已验证非零 GPU 梯度。导出临时文件存入本次运行目录，官方源码未修改。导出的 runs/O1/inference 与 runs/O2/inference 可传给 predict.py 的 --rec-dir。评估 JSON 保存到 reports/results。

几何增强对照从同一份 O2 画质增强裁图出发。`O2-geometry-control` 用当前标注重训画质组，`O3-geometry` 在同一批训练裁图中随机挑选约一半，增加 ±3° 旋转和最大 3% 的透视角点偏移。变换后的画布容纳整张裁图，避免裁掉文字。两组使用同样的初始权重、训练参数和验证集。用验证集选择每组最佳权重，再测固定测试集。

```powershell
python -X utf8 prepare_ocr_geometry.py
python -X utf8 train_ocr.py --config configs/ocr/O2-geometry-control.yaml --epochs 3 --batch 16 --swanlab-mode offline
python -X utf8 train_ocr.py --config configs/ocr/O3-geometry.yaml --epochs 3 --batch 16 --swanlab-mode offline
foreach ($group in @('O2-geometry-control', 'O3-geometry')) {
    python -X utf8 evaluate_ocr.py --group $group --split val --swanlab-mode offline
    python -X utf8 evaluate_ocr.py --group $group --split test --swanlab-mode offline
    python -X utf8 evaluate_pipeline.py --name "Y2-geometry-$group-direct" --weights runs/Y2-geometry-2/weights/best.pt --rec-dir "runs/$group/inference" --ocr-mode direct --swanlab-mode offline
}
```
