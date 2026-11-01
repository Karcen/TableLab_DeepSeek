查找模型请先打开 index.csv。每行是一项训练配置，directory 一列给出对应目录。

benchmarks 保存论文的 12 项基准配置。classroom 保存网页的 9 个例题，其中 5 个训练了模型。teaching 是讲解用的聚合计算，api_checks 是原工程的手工参数测试。每次新训练写入 runs，不改交付的基准结果。

训练目录中的 plan.json 记录目标、特征、算法和划分。split.csv 标明每条记录的用途。validation_predictions.csv 保存逐条预测。metrics.csv、explanation.csv、transformation_audit.json 分别记录指标、系数或置换重要性、预处理与参数。

这里没有可直接反序列化的预训练模型。网页按选定的方案现场拟合；关闭服务后，先重训，再做单条预测。只想复现某项基准，在 codes 目录运行：
.venv/bin/python refit.py house_primary_random_forest

Qwen 由 Mac 上的 Ollama 管理。权重不随本包分发。prepare_model.command 将实际模型的版本指纹写到 models/runtime/model.lock.json。
