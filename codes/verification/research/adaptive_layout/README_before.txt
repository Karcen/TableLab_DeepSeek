# TableLab｜DeepSeek 联网演示

## 启动

解压整个文件夹。双击同目录的 `TableLab-DeepSeek.command`。这是正式网页的唯一启动入口，不用选菜单，不用下载 Qwen 或安装 Ollama。

首次运行需要联网，电脑上还要装有 Python 3.11、3.12 或 3.13。启动器会在 `codes/.venv` 建立项目环境，安装 `requirements.txt` 中的依赖。安装完成后会接着启动网页，不用退出再开。

看到密钥提示时，在终端粘贴自己的新 DeepSeek API Key，按回车。粘贴时不会显示字符。密钥默认存入当前用户的 Mac 钥匙串，下一次自动读取。已经贴到聊天里的旧密钥请撤销。喜欢手动配置的用户也可在自己电脑上编辑启动器开头的 `PRIVATE_API_KEY`；提交作业请交原始 ZIP，不要把填过密钥的副本发给别人。

浏览器打开后，在网页里提问，不用继续在终端输入。整个演示期间保持终端窗口打开。浏览器未自动打开时，看到「网页已就绪」后访问 `http://127.0.0.1:8765`。结束时在终端按 `Control-C`。

双击时如果提示没有权限，在终端里输入 `bash` 和一个空格，把本文件夹中的 `TableLab-DeepSeek.command` 拖进终端，按回车。不要运行旧工程的「`运行入口.command`」。

## 交作业

要交两份文档：`docs/paper.pdf`（课程论文），`docs/presentation.pdf`（Beamer 展示稿）。两份文档的 LaTeX 和 BibTeX 源码都在 `docs/source`。`docs/demo_on_mac.txt` 是演示操作卡。`docs/classroom_replay.html` 是不用 API 的例题回放，不能回答新问题。

## 目录

- `data`：四份数据、下载清单和处理后的 CSV。数据已经齐全。
- `codes`：网页、字段检索、DeepSeek 接口、建模、测试和复算代码；`download_tools` 是原数据下载器，不需要重跑。
- `figures`：报告和展示中使用的 PDF 图。
- `tables`：数据核查与模型结果表。
- `models`：训练方案、样本划分、逐行预测、模型参数及演示记录；不含大语言模型权重。
- `docs`：论文、展示稿、源码与使用说明。`archive` 仅保留先前版本说明和旧菜单文本。原离线实验的源文件与记录供复核，不影响 DeepSeek 启动。

## 复核

论文数值：首次安装依赖后，在工程根目录运行 `codes/.venv/bin/python codes/check_results.py`。重新编译文档用 `bash docs/build_docs.sh`，现场演示不用编译。

核对整个解压包（修改文件之前）：在工程根目录运行 `shasum -a 256 -c codes/verification/files.sha256`。

## 四任务批量模型实验

双击 `Run-Research.command` 可依次运行多模型、嵌套调参、拟合诊断、SHAP 与特征组消融，并重建论文和 Beamer。批量实验不调用 DeepSeek API；两个任务进程并行，模型内部单线程。结果写入 `artifacts/research`，独立复算记录写入 `codes/verification/research`。

本轮协议为外层 3 折、内层 3 折，森林和提升树的网格与 TPE 各评估 12 组参数；四个任务使用各自的随机、分组或时间约束。图形依赖指定 Nature Skills 的本地目录，可用环境变量 `NATURE_FIGURE_SKILL` 指向该技能根目录。已有完整折会复用；实验代码变化后应先归档旧结果，再按新协议重跑。
