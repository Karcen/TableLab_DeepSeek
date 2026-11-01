Karcen Zheng，这个工具在你的 Mac 上联网下载四份课程数据，生成可上传给助手的 tablelab_datasets_upload.zip。它只准备数据，不下载语言模型，也不运行回归或预测。

使用前需要 Python 3.11 或更新版本。MacBook Air M4 可以使用面向 Apple Silicon 的 Python；当前代码没有在你的电脑上实测。没有 Python 时请从 https://www.python.org/downloads/macos/ 安装，不要从陌生镜像复制安装命令。

先在浏览器登录 Kaggle。打开 https://www.kaggle.com/settings/api ，用 Generate New Token 创建 API Token。再打开 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules ，阅读规则；你接受后，在页面完成加入或授权步骤。脚本不会代替你同意规则，也不会绕过账号权限。

解压本工具包，保持 download_data.command、download_tablelab_data.py 和 requirements_download.txt 在同一文件夹。打开 Mac 的“终端”，输入 bash 和一个空格，再把 download_data.command 拖入终端，按回车。这样不需要先修改文件的可执行权限。

启动文件会在工具目录建立独立的 .venv-download 环境，从 PyPI 安装 requirements_download.txt 中固定的 kagglehub 版本，然后调用 Python 下载器。它不会修改原 TableLab 工程的依赖。下载器会显示正在处理的数据集；Kaggle 下载进度由官方库输出。

终端提示输入 Token 时，粘贴刚创建的 API Token，然后按回车。输入内容不会显示，这是隐藏输入的正常行为。不要输入 Kaggle 登录密码，也不要把整个 kaggle.json 文本粘进去。脚本把 Token 放在本次进程的环境变量里，不写入项目、下载清单或上传包。不要把 Token、密码、Cookie 或 kaggle.json 发给助手。

直接使用 Python 的方式、仅下载一个数据集的方式以及参数说明，写在 download_tablelab_data.py 顶部的说明和 --help 输出里。只下载 Adult 不需要 kagglehub 或 Kaggle Token。直接运行时，不加 --login 会让 Kaggle 官方库使用已经配置好的本机认证信息；支持的认证来源以官方文档为准。

输出位置默认是本工具目录内的 data/incoming，不取决于终端当前目录。四个文件分别为 adult.zip、nba.zip、students.zip、house-prices.zip。download_manifest.json 保存来源页面、下载时间、文件大小、实际读取的行数、字段和 SHA-256。清单里的行数是结构检查结果，不是已经完成数据清洗后的分析样本数。哈希用于识别本地副本和检查改动，不能单独证明来源真实性。

UCI 的 adult.zip 保留官方 ZIP 字节。Kaggle 官方库返回下载后的文件，脚本把指定原文件重新封装成 ZIP，文件内容字节不作改动。因此，三个 Kaggle ZIP 是本工具重新封装的容器，不宣称与网页 Download All 的 ZIP 哈希相同；清单明确记录这一差别。

Adult 包保留 adult.data、adult.test、adult.names；NBA 包包含 all_seasons.csv；学生包包含 StudentsPerformance.csv；房价包包含 train.csv、test.csv、data_description.txt、sample_submission.csv。房价 test.csv 没有真实成交价标签，sample_submission.csv 是提交格式样例，不能拿二者计算预测准确性。

四份数据全部下载并通过校验后，脚本生成 data/incoming/tablelab_datasets_upload.zip。这个上传包只含四个数据 ZIP 和一份下载清单，不包含 Python 环境、模型、其他目录或凭据。把这一个 ZIP 上传到当前对话即可；不需要你手工解压、合并、清洗或改名。

脚本会拒绝不是 ZIP 的登录页、缺少必需文件或字段的下载结果、损坏的压缩包和不一致的哈希。一次失败会停止任务并返回非零退出码，已经完成并登记的文件保留。修正网络或授权问题后，用同一方式重跑；脚本会重新校验这些文件并跳过下载。这是按完整文件复用，不是承诺对中断文件进行字节级断点续传。

需要重新取得某个来源的版本时，使用 --force 明确允许重新下载。没有 --force 时，本地已登记且通过校验的副本不会自动更新。使用未固定版本的 Kaggle 数据集句柄时，清单只表明下载时间与文件指纹，不保证以后重新下载仍是同一版本。

出现 HTTP 401 或 403 时，先核对 Token 是否有效，以及浏览器中接受规则的账号是否与 Token 所属账号一致；不要仅因浏览器已登录就认为脚本也已认证。遇到限流时停止连续重试。遇到 DNS、代理、网络或证书错误时先修复环境，不要关闭 HTTPS 证书校验。

输出文件名与 TableLab v0.1 的本地 ZIP 导入参数对应。下载器不会自动执行 TableLab 清洗或修改其 data/manifest.json；本工具的 download_manifest.json 是独立下载记录。原工程导入时，应显式提供四个本地 ZIP，避免再次走网络下载路径。

本工具已运行 32 项离线单元测试，使用极小合成夹具和模拟下载函数。测试覆盖文件结构、缓存校验、错误停止、原文件字节保留和凭据排除；合成夹具不会写入正式下载输出。真实 UCI 下载在当前执行环境失败；Kaggle 下载及 Mac 联调未实测。完整记录在 test_results.txt、live_download_check.txt 和 VERIFICATION.txt。

查阅的官方资料：
https://github.com/Kaggle/kagglehub
https://github.com/Kaggle/kagglehub/blob/v1.0.2/src/kagglehub/datasets.py
https://pypi.org/project/kagglehub/1.0.2/
https://github.com/Kaggle/kaggle-cli/blob/main/skills/SKILL.md
https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md
https://www.kaggle.com/docs/api
https://archive.ics.uci.edu/dataset/2/adult
https://www.kaggle.com/datasets/justinas/nba-players-data
https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
