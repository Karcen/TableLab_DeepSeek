# TableLab：中文表格问答与预测

[English](README.md) · 中文

TableLab 是一个机器学习课程项目。用户用中文描述问题，程序检索本地字段说明，再通过 DeepSeek API 生成 JSON 分析方案。确认字段和条件后，本机程序读取表格，完成统计或模型训练。

项目包含房价、Adult 收入分类、NBA 球员得分和学生数学成绩四个任务，也保留了多模型比较、网格与 TPE 超参数搜索、拟合诊断、SHAP 和特征组消融的实验结果。

## 启动

在 macOS 上解压整个文件夹，双击 `TableLab-DeepSeek.command`。首次启动需要联网，并已安装 Python 3.11、3.12 或 3.13。启动器会创建 `codes/.venv`，安装依赖后打开网页。

按终端提示输入自己的 DeepSeek API Key。输入时不显示字符，密钥默认保存在当前用户的 Mac 钥匙串中，后续启动会自动读取。也可在本机设置启动器中的 `PRIVATE_API_KEY`；发布或分享项目时应保留该项为空。

网页打开后，直接在网页中提问，并保持终端窗口打开。如果浏览器没有自动打开，看到“网页已就绪”后访问 <http://127.0.0.1:8765>。结束时在终端按 `Control-C`。

如果双击提示没有执行权限，在终端输入 `bash` 和一个空格，将 `TableLab-DeepSeek.command` 拖入终端后按回车。

## 论文与展示

- [课程论文](docs/paper.pdf)
- [Beamer 展示稿](docs/presentation.pdf)
- [实验结果说明](docs/research_summary.md)

LaTeX 和 BibTeX 源码在 `docs/source`。`docs/demo_on_mac.txt` 是现场操作说明；`docs/classroom_replay.html` 可在不调用 API 的情况下回放固定例题，回答新问题需启动网页。

## 扩展模型实验

四个原任务各比较 21 个模型条目，包括线性方法、树集成、近邻、SVM、XGBoost、LightGBM、CatBoost、MLP、残差与门控 MLP、Deep & Cross、字段 Transformer、ELM 和 RBF 网络。NBA 另设三季历史预测下一季的任务，共 15 项对照，加入 RNN、GRU、BiGRU、LSTM、BiLSTM、因果 TCN 和序列 Transformer。扩展模型通过批量入口运行，网页保留原有交互算法。

每项可训练模型分别做网格与 TPE。大部分候选域为八点，无惩罚线性项只比较两种截距配置。外层三折评价、内层三折选参数；Adult 按全部原始预测字段指纹分组分层，NBA 按完整赛季向前验证。预处理和回归目标缩放均只在训练数据拟合。每项特征消融重新网格调参，五项表格网络另做结构控制；端到端网络固定训练四十轮。

| 任务 | 本轮网格流程中的较优条目 | 外层均值 ± 标准差 |
| --- | --- | --- |
| 房价 | RBF 网络 | RMSE 31,202.70 ± 4,473.78 美元 |
| 收入分类 | CatBoost | AUC 0.8926 ± 0.0023 |
| NBA 得分 | 随机森林 | RMSE 5.687 ± 0.393 分 |
| 学生数学 | 线性 SVM | RMSE 13.540 ± 0.387 分 |
| NBA 下一赛季 | BiGRU | RMSE 2.888 ± 0.098 分 |

Adult 对 21 项网格或固定条目另做后续扩展测试。按 CV 事先预选的 CatBoost 测试 AUC 为 0.8924，95% 分组 bootstrap 区间为 [0.8871, 0.8977]。原测试已用于前期十三流程，因此不称新的盲测；测试成绩没有用于回调参数或阈值。其余任务仍缺独立新样本。NBA 历史任务改变了输入和样本，不能用它与原 NBA 任务的误差差值直接说明网络收益。

SHAP 固定解释提升树，覆盖全部外层验证记录；消融删除输入或关闭部件后重新训练。字段贡献与删减不直接识别现实中的因果效应。检索仍只检查十六道固定题，其中十五道含英文字段名，真实 DeepSeek API 方案质量尚未评价。完整数字、解释和复现办法见 [实验说明](docs/research_summary.md)。

## 复现与核对

先启动一次网页准备 Python 环境，停止服务后运行 `Run-Expanded.command`。它会安装固定版本依赖，完成嵌套搜索、消融、双精度预测导出和结果核对，再绘图并编译论文与 Madrid 展示稿。批量实验不调用 DeepSeek API，每个模型单线程，已完成组合按冻结协议复用。

扩展结果在 `artifacts/expanded`。前期十三流程保留在 `artifacts/research`，其 Adult 测试保留在 `artifacts/holdout/adult`；新增扩展测试另存于 `artifacts/expanded/holdout_extension`，不覆盖前期记录。修改冻结训练代码前，应先归档不兼容结果。

绘图使用指定 Nature Skills，默认路径为 `~/Downloads/SKILLS/nature-skills-main/skills/nature-figure`，可用 `NATURE_FIGURE_SKILL` 指定其他位置。编译需要 XeLaTeX、BibTeX 与 Font Awesome 5。

```bash
# 复算扩展指标，检查分折、搜索、扩展测试与 SHAP 加和
codes/.venv/bin/python codes/tools/verify_expanded.py
codes/.venv/bin/python codes/tools/verify_expanded_holdout.py
codes/.venv/bin/python codes/tools/verify_expanded_shap.py

# 重建论文与展示稿
bash docs/build_docs.sh

# 修改文件前核对交付清单
shasum -a 256 -c codes/verification/research/files.sha256
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `data` | 四份数据、来源清单和处理后的 CSV |
| `codes` | 网页、字段检索、DeepSeek 接口、建模与复核代码 |
| `artifacts/expanded` | 扩展搜索、消融、模型、预测与后续测试 |
| `artifacts/research` | 四任务嵌套验证的模型和逐条结果 |
| `figures`、`tables` | 原单次划分复核的图与结果表 |
| `models` | 原实验的训练方案、划分、预测和演示记录 |
| `docs` | 论文、展示稿、源码、操作说明和历史版本 |

数据来源及其核实情况见论文附录 C、D 和 `data/manifest.json`。
