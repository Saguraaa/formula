# CloudStudio 部署

本仓库已包含网页、判题代码与最佳 YOLO/OCR 推理权重。仓库根目录就是原 ModelAdjust 目录，无需整个实验数据集。

1. 在 CloudStudio 应用管理中新建应用，从本 GitHub 私有仓库导入；使用能访问该仓库的 GitHub 账号授权。
2. 选择 Linux x86_64、Python 3.11 环境。建议至少 4 核、8 GB 内存，预留 10 GB 磁盘用于依赖。默认 CPU 推理，无需 CUDA。
3. 启动配置已放在 `.vscode/preview.yml`：工作目录 `./`、命令 `bash deploy/start-cloud.sh`、端口 `7860`、`mainPort: true`。
4. 首次启动会创建独立环境并安装依赖，需要访问 PyPI 和 PyTorch 官方 CPU 下载源；以后配置未变化会复用环境。如果提示缺少 venv，请在环境中安装 python3-venv。
5. 等待启动日志出现 `Listening at: http://0.0.0.0:7860`，回到应用管理打开应用，即进入算式工作台。首次识别需加载权重。

手动启动与配置中的启动方式一致：

```bash
bash deploy/start-cloud.sh
```

若默认 python3 不是 3.11，可使用 `PYTHON_BIN=python3.11 bash deploy/start-cloud.sh`。

验收：打开主页，上传一张作业图，确认检测图、所有裁图、OCR 原文、正确/错误/待确认统计均有返回；再下载 ZIP，检查包含裁图和 result.json。`/api/health` 返回服务状态，但只有实际上传并完成识别才能证明模型可用。

结果保存在服务端 `outputs/web/`；云端页面会明确提示上传至应用处理。结果通过随机任务 URL 访问，本应用没有账号权限管理，适合作为个人实验演示。不要将敏感作业上传到公开共享实例。结果目录可通过 FORMULA_OUTPUT_DIR 指向持久化存储。

若失败，查看启动日志：缺少模型先运行 `python deploy/check_models.py`；安装下载失败检查外网；内存不足提高实例规格；页面无法打开确认服务绑定 0.0.0.0、端口和 preview.yml 一致。

本地 Windows 验证不等于云端验收。CloudStudio 最终部署及实际上传验收由使用者完成。

官方配置参考：https://cloudstudio.net/docs/guide/code_editing/basic_operations/how-to-run/
