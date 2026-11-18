# 分式图片识别验收

日期：2026-11-12
模型：PaddleOCR PP-FormulaNet_plus-S 官方预训练模型，gpu:0。
输入：outputs/fraction-check/input.png（此前用户提供原图，未修改）。
模型原始输出：`\frac{2}{4}-\frac{1}{4}=\frac{1}{4}`。
人工逐项核对：三个分数的分子、分母、减号和等号均正确。
结论：通过“正确识别一个分式图片”的验收。仅为单样本验证，不代表批量准确率。
未训练、未修改任何数据标注。此次结果保存在本地，未上传 SwanLab。
依赖：vendor/formula_deps 中补充 ftfy；运行时将该目录加入 PYTHONPATH。
模型缓存：weights/paddlex/official_models/PP-FormulaNet_plus-S。
原始输出见 outputs/fraction-check/input_res.json。
