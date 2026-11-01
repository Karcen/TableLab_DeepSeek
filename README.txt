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

## 四任务实验

九种预设方法包括常数基线、线性与正则化模型、决策树、随机森林、Extra Trees、直方图梯度提升和近邻。森林与提升树分别增加网格和 TPE 搜索，每个任务共比较十三条训练流程。

实验使用外层三折评价、内层三折选参数。房价与学生任务随机分折；Adult 按原始字段指纹分组并按标签分层；NBA 按完整赛季向前验证。内层沿用同一分折规则，插补、编码和标准化只在当前训练折拟合。网格与 TPE 在相同的十二点参数域内各评估十二次，TPE 允许重复组合。

| 任务 | 本次较优流程 | 外层三折平均结果 |
| --- | --- | --- |
| 房价 | 提升树＋网格或 TPE | RMSE 32,338.23 美元 |
| Adult 收入分类 | 提升树＋网格或 TPE | AUC 0.8913 |
| NBA 得分 | 预设随机森林 | RMSE 5.698 分 |
| 学生数学成绩 | Ridge | RMSE 13.545 分 |

这些排名描述本次三折结果。论文同时报告折间波动；原单次划分与嵌套验证使用的样本和配置不同，两组数值不能直接解释为调参收益。训练与验证曲线用于检查拟合差距，SHAP 描述既定模型的预测贡献，消融则删除字段后重新调参和训练。两种解释都需要考虑字段之间的替代关系。

问答部分检查了 16 道固定题的字段检索，其中 15 道带英文字段名，MRR 为 0.96875。DeepSeek API 的方案生成质量尚未评价。

## 复现与核对

先启动一次网页以准备 Python 环境，再双击 `Run-Research.command`。它会安装实验依赖，运行四任务实验，复算指标，生成图形并重建两份文档。批量实验不调用 DeepSeek API，使用两个任务进程并行，每个模型单线程。编译文档需要 XeLaTeX 和 BibTeX。

结果保存在 `artifacts/research`，包含分折行号、搜索记录、拟合模型、逐行预测、SHAP 值和消融结果。完整折会复用；修改实验代码后，应先归档旧结果再重跑。

绘图使用 Nature Skills。默认路径为 `~/Downloads/SKILLS/nature-skills-main/skills/nature-figure`，也可用环境变量 `NATURE_FIGURE_SKILL` 指定该技能目录。

在工程根目录运行以下命令，可单独核对已有结果或重建文档：

```bash
# 复核原单次划分结果
codes/.venv/bin/python codes/check_results.py

# 从保存的预测复算扩展实验指标，并检查分折与 SHAP 加和
codes/.venv/bin/python codes/tools/verify_research.py

# 编译论文与 Beamer
bash docs/build_docs.sh

# 核对当前研究交付文件（修改文件前）
shasum -a 256 -c codes/verification/research/files.sha256
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `data` | 四份数据、来源清单和处理后的 CSV |
| `codes` | 网页、字段检索、DeepSeek 接口、建模与复核代码 |
| `artifacts/research` | 四任务嵌套验证的模型和逐条结果 |
| `figures`、`tables` | 原单次划分复核的图与结果表 |
| `models` | 原实验的训练方案、划分、预测和演示记录 |
| `docs` | 论文、展示稿、源码、操作说明和历史版本 |

数据来源及其核实情况见论文附录 C、D 和 `data/manifest.json`。
