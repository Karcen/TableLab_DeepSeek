"""Write source-backed Chinese results into the course paper and Beamer deck."""
from pathlib import Path
import json,re
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'artifacts/research';SRC=ROOT/'docs/source';BACKUP=ROOT/'docs/archive/before_multimodel/source'
DS=['house','adult','nba','students'];CN={'house':'房价','adult':'收入分类','nba':'NBA 得分','students':'学生成绩'}
NAMES={'Mean':'常数基线','Linear':'逻辑回归 / OLS','Ridge':'强 L2 / Ridge','Sparse':'L1 / Elastic Net','Tree':'单棵决策树','RF':'随机森林','ExtraTrees':'Extra Trees','HGB':'直方图梯度提升','KNN':'K 近邻','RF_grid':'森林＋网格','RF_tpe':'森林＋TPE','HGB_grid':'提升树＋网格','HGB_tpe':'提升树＋TPE'}
ABL_CN={'No neighborhood':'社区','No size':'面积组','No quality':'质量','No year':'建成年份','Area + quality':'仅保留面积与质量','No education':'教育','No work':'工作组','No marital':'婚姻状况','No age':'年龄','No physique':'身体属性组','No draft':'选秀年份','No gender':'性别','No parental education':'父母教育','No lunch':'午餐','No preparation':'备考课程'}
def display(name,ds):
 return {'Linear':'逻辑回归' if ds=='adult' else 'OLS','Ridge':'强 L2 逻辑回归' if ds=='adult' else 'Ridge','Sparse':'L1 逻辑回归' if ds=='adult' else 'Elastic Net'}.get(name,NAMES[name])
def row(ds,name):return SUMMARY.query('dataset==@ds and model==@name').iloc[0]
def value(ds,name):
 r=row(ds,name);scale=1000 if ds=='house' else 1;dec=4 if ds=='adult' else 2 if ds=='house' else 3
 return f'{r["mean"]/scale:.{dec}f} $\\pm$ {r["sd"]/scale:.{dec}f}'
def fig(name,caption,label):return '\\begin{figure}[!htbp]\\centering\n\\includegraphics[width=\\linewidth]{figures/research/'+name+'.pdf}\n\\caption{'+caption+'}\\label{'+label+'}\n\\end{figure}\n'
def frame(title,body):return '\\begin{frame}{'+title+'}\n'+body+'\n\\end{frame}\n\n'
def image(name):
 if (SRC/'figures/research'/f'{name}_slide.pdf').exists():name+='_slide'
 height='.65' if name.startswith('benchmark_') else '.72'
 return '\\begin{center}\\includegraphics[width=.92\\textwidth,height='+height+'\\textheight,keepaspectratio]{figures/research/'+name+'.pdf}\\end{center}\n'
def main():
 global SUMMARY
 SUMMARY=pd.read_csv(DATA/'summary.csv');tuning=pd.read_csv(DATA/'tuning_summary.csv');abl=pd.read_csv(DATA/'ablation_deltas.csv');shap=pd.read_csv(DATA/'shap_importance.csv');metrics=pd.read_csv(DATA/'metrics.csv')
 wins={ds:SUMMARY.query('dataset==@ds').sort_values('mean',ascending=ds!='adult').iloc[0] for ds in DS}
 rhouse=wins['house'];radult=wins['adult'];rnba=wins['nba'];rstudent=wins['students']
 detail=(f'房价的较小平均 RMSE 为 {rhouse["mean"]:,.2f} 美元，来自{display(rhouse.model,"house")}；收入分类中，{display(radult.model,"adult")}的 AUC 为 {radult["mean"]:.4f}。'
         f'NBA 得分由{display(rnba.model,"nba")}取得较小平均误差（{rnba["mean"]:.3f} 分），学生成绩则是 {display(rstudent.model,"students")}（{rstudent["mean"]:.3f} 分）。')
 results='\\subsection{多模型与嵌套验证}\n本轮比较九种预设方法，并为随机森林、梯度提升各增加网格和 TPE 搜索，共形成每个任务十三条比较流程。'+detail+'这些是三折验证中的描述性排序，各折结果见图~\\ref{fig:new-benchmark}。\n\n'
 results+='\\inlinetab{\\captionof{table}{三折外层验证的均值与折间标准差}\\label{tab:new-models}\n\\begin{tabularx}{\\linewidth}{@{}Yrrrr@{}}\\toprule\n方法 & 房价 RMSE & 收入 AUC & NBA RMSE & 成绩 RMSE\\\\\\midrule\n'
 for name in NAMES:results+=NAMES[name]+' & '+' & '.join(value(ds,name) for ds in DS)+r'\\'+'\n'
 results+='\\bottomrule\\end{tabularx}\n\\tabnote{房价单位为千美元，NBA 与成绩单位为分；AUC 越高越好，其余指标越低越好。误差项为三折间样本标准差。各折训练集有重合，标准差用于描述波动。NBA 和学生的 Elastic Net 在各自三折中斜率均为零，因此结果与训练均值基线一致；惩罚参数见方法部分。}}\n'
 results+=fig('benchmark','九种预设方法及四条调参流程的外层表现。空心菱形与误差线为三折均值和标准差，圆点、方块与三角分别表示三个外层折；颜色区分方法组。浅色带标出本次较优均值，重合者均保留。各任务按自身指标排序，房价以千美元显示。','fig:new-benchmark')
 from adult_holdout_docs import results as holdout_results, methods as holdout_methods, brief as holdout_brief, interpretation, write_report
 results+=holdout_results()
 results+='\\subsection{网格与 TPE 贝叶斯搜索的实际收益}\n'
 gains={}
 for ds in DS:
  q=tuning.query('dataset==@ds and model=="HGB"').set_index('method');gains[ds]=(q.loc['grid','improvement'],q.loc['tpe','improvement'])
 results+=(f'房价中，提升树的网格与 TPE 搜索都使平均 RMSE 比预设参数低 {gains["house"][0]:.2f} 美元；NBA 的降幅为 {gains["nba"][0]:.3f} 分。学生任务两条流程分别降低 {gains["students"][0]:.3f} 和 {gains["students"][1]:.3f} 分，收入分类的平均 AUC 增量约为 {gains["adult"][0]:.4f}。\n\n')
 results+='森林在收入分类与 NBA 任务中的调参结果反而略差。这些参数由内层选择，外层退化说明内层最优组合的收益未必能在另一批记录上保持。\n\n'
 results+='两种搜索使用相同的十二点离散域；网格遍历全部组合，TPE 允许重复访问。每个任务中提升树的 TPE 平均访问约七至八种不同组合，其最终外层表现有时与网格相同。比较范围是本次候选域和预算；搜索轨迹、重复次数及耗时一并保留。\n'
 results+=fig('search_validation','内层搜索与外层收益并排比较，每行依次为房价、收入、NBA 与学生。左列为提升树的内层最优分数轨迹，阴影为三个外层训练子集间的标准差；右列比较调参流程与同折预设模型，正值为改善，负值为退化，点为同折差值，菱形与误差线为均值和标准差。收入收益以 0.001 AUC 为单位。','fig:new-search')
 results+='\\subsection{训练误差与验证误差}\n'
 def tree_error(ds,part):
  m='log_loss' if ds=='adult' else 'rmse'
  return metrics.query('dataset==@ds and model=="Tree" and partition==@part')[m].mean()
 results+=(f'完整决策树在收入分类中的平均训练对数损失为 {tree_error("adult","train"):.3f}，外层验证为 {tree_error("adult","validation"):.3f}；房价训练 RMSE 为 {tree_error("house","train"):,.2f} 美元，验证为 {tree_error("house","validation"):,.2f} 美元。模型在训练记录上的低误差没有保持到验证记录，过拟合迹象较为清楚。复杂度曲线再观察树深增加后，训练和验证误差如何分离。\n\n')
 diagnostics=pd.read_csv(DATA/'diagnostics.csv')
 def depth_error(ds,depth):
  m='log_loss' if ds=='adult' else 'rmse'
  return diagnostics.query('dataset==@ds and kind=="complexity" and depth==@depth and partition=="validation"')[m].mean()
 results+=(f'浅树也会遗漏预测信息。房价的树深从 1 增至 4 后，平均验证 RMSE 从 {depth_error("house",1):,.2f} 降至 {depth_error("house",4):,.2f} 美元；学生任务从 {depth_error("students",1):.3f} 降至 {depth_error("students",4):.3f} 分，训练误差也同时下降。收入分类的验证对数损失由 {depth_error("adult",1):.3f} 降至 {depth_error("adult",4):.3f}，这些变化与浅树拟合不足的解释一致。NBA 从 {depth_error("nba",1):.3f} 降至 {depth_error("nba",4):.3f} 分，变化相对折间波动很小，需要结合其他模型判断。\n\n')
 results+='学习曲线固定各外层训练折选出的提升树参数，再增加训练数据量。训练和验证误差都高时，还需检查输入信息是否充分；仅凭误差水平，难以将特征不足与模型欠拟合分开。\n'
 results+=fig('fit_diagnostics','学习曲线与复杂度诊断。每行一个任务，左列增加训练数据，右列改变决策树深度；Full 表示不限制深度。曲线为三折均值，阴影为折间标准差，空心圆为训练、实心方块为验证。每个子样本重新拟合预处理，分类用对数损失，回归用 RMSE。两列提供不同的拟合诊断。','fig:new-learning').replace('\\end{figure}','\\label{fig:new-complexity}\\end{figure}')
 results+='\\subsection{SHAP 解释与特征组消融}\nSHAP 与消融统一采用事先指定的网格调参提升树，避免为了得到某种解释而挑选模型。\n\n'
 def loss(ds,variant):return abl.query('dataset==@ds and variant==@variant').delta.mean()
 results+=(f'房价的平均绝对 SHAP 以整体质量最高，但删去面积组后的退化最大，RMSE 平均增加 {loss("house","No size"):,.2f} 美元。SHAP 分配的是既定模型中的预测贡献；重新训练的消融允许其余字段替代被删信息，两种排序因而可以不同。\n\n')
 results+=(f'其余任务中，收入分类以婚姻状况、NBA 以选秀年份、学生成绩以午餐类别的平均绝对贡献较大。删去对应字段后，收入 AUC 平均下降 {loss("adult","No marital"):.4f}，NBA 与学生的 RMSE 分别增加 {loss("nba","No draft"):.3f} 和 {loss("students","No lunch"):.3f} 分。\n')
 results+=fig('explanation_ablation','既定模型贡献与重新训练的消融。每行一个任务，左列为原始字段的平均绝对 SHAP，右列为删去字段后重新网格调参的同折性能变化。柱端误差线为三折标准差，右列还保留各折差值。右列正值为损失、负值为改善；Only area + quality 仅保留地上居住面积与整体质量。两列分别检查预测贡献与其余字段的替代信息。','fig:new-shap').replace('\\end{figure}','\\label{fig:new-ablation}\\end{figure}')
 results+=fig('shap_structure','全部外层验证记录的贡献分布与取值依赖。左列保留各记录的 SHAP 值与分布轮廓，纵向抖动仅显示重叠；蓝、橙表示贡献的负、正方向，深浅表示各字段内相对大小，并非原始类别的顺序。右列的数值字段按六边形汇总记录数，颜色按对数计数加深；午餐类别保留全部记录，白点与粗线分别为中位数与四分位距。房价单位为千美元。','fig:new-beeswarm').replace('\\end{figure}','\\label{fig:new-dependence}\\end{figure}')
 results+=fig('local','代表性单条预测的 SHAP 瀑布图。各任务均按全部外层验证记录的绝对预测误差排序，取中间记录。灰色为基准，橙、蓝分别提高和降低原始预测，青色为贡献加和后的最终输出，连线连接相邻步骤。分类在对数几率尺度分解，回归在目标单位分解；房价以千美元显示。具体行号、真实值、预测值与加和核对见计算记录。','fig:new-local')
 (SRC/'research_results.tex').write_text(results)
 methods=r'''\subsection{扩展实验的验证与搜索协议}\label{sec:new-protocol}
九种预设方法包括常数基线、线性模型、较强 L2 正则模型、稀疏线性模型、完整决策树、随机森林、Extra Trees、直方图梯度提升和 K 近邻。回归的三种线性模型为 OLS、Ridge 和 Elastic Net；分类对应 L2 逻辑回归、较强 L2 逻辑回归与 L1 逻辑回归。Ridge 的 $\alpha=10$；Elastic Net 的 $\alpha=10$、$\mathrm{l1\_ratio}=0.5$。普通与较强 L2 逻辑回归的 $C$ 分别为 1 和 0.1，L1 逻辑回归取 $C=0.1$，使用 \code{liblinear} 求解。以上均为预设参数，未参与本轮搜索。NBA 和学生的 Elastic Net 在各自三折中斜率均为零，截距等于训练均值；在此惩罚强度下，预测退化为常数。森林和 Extra Trees 使用 64 棵树、深度 12、叶节点最少 5 条；文中的“提升树”均指表~\ref{tab:new-models} 的直方图梯度提升，回归与分类分别使用 \code{HistGradientBoostingRegressor} 和 \code{HistGradientBoostingClassifier}；预设配置使用 100 次迭代并关闭内部随机早停。近邻数为 15。完整参数随源代码与拟合对象保存。

每个任务使用三折外层验证，内层仅在相应外层训练子集中，沿用同一分折规则再分三折搜索参数。房价和学生随机分折，种子为 42；Adult 在原训练文件内按全部非目标原始字段指纹分组并按标签分层；NBA 按完整赛季向前滚动，验证赛季晚于训练赛季。NBA 三个外层训练/验证规模分别为 3,972/2,706、6,678/2,883、9,561/3,283；最早的训练赛季不作为验证记录。其余任务每条合格记录恰好进入一次外层验证。嵌套验证将选参数与评价参数选择流程分开\citep{nestedcv}。

数值插补与标准化、分类缺失值处理和独热编码都在每次拟合的训练折完成。各方法使用相同字段、相同样本行号及相同内外层划分；房价保持美元原尺度，学生不输入同期其他考试成绩。原始记录与异常值的保留规则沿用数据核查。

网格搜索与基于 TPE 的贝叶斯超参数搜索，对同一个离散参数域各评估十二次。森林搜索最大深度 4、8 或不限，叶节点最少 3 或 10 条，候选特征比例 0.7 或 1；提升树搜索叶节点上限 7、15 或 31，L2 惩罚 0 或 1，学习率 0.05 或 0.1。两者均形成十二种组合。TPE（树结构 Parzen 估计）前四次为启动采样，后续根据已有参数及分数更新采样概率并选择参数，种子为 42 加外层折编号\citep{optunadocs}。TPE 可以重复组合，全部评估保留。分类按内层平均 AUC 选参数，回归按平均 RMSE 选参数。阈值固定为 0.5，未作阈值搜索。总计四个任务、两类模型、两种搜索、三折外层与三折内层，主搜索完成 1,728 次内层拟合；消融在此之外重复网格搜索。

\subsection{解释、消融与结果核对}
解释模型预先固定为网格调参提升树。Tree SHAP 使用训练树路径计数作为背景分布，解释全部外层验证样本，分类输出为原始对数几率\citep{shapdocs}。每条记录检查基准值加全部字段贡献能否重建原始预测。类别字段先将对应编码的有符号贡献相加，再汇总绝对重要性。

房价的面积组包含地上居住面积 \code{GrLivArea} 与地下室面积 \code{TotalBsmtSF}；NBA 的身体属性组包含身高 \code{player_height} 与体重 \code{player_weight}；Adult 的工作组包含周工时 \code{hours-per-week}、用工类别 \code{workclass} 和职业 \code{occupation}。房价分别删去社区、面积组、质量和建成年份，另设仅保留地上居住面积与整体质量 \code{OverallQual} 的两字段配置；这一配置不包含地下室面积。Adult 分别删去工作组、教育、婚姻和年龄；NBA 分别删去身体属性组、年龄和选秀年份；学生逐一删去四个背景字段。每个变体在同一外层训练记录中重新完成十二点网格搜索。消融以同折差值评价；缺少某个字段后误差变化小，也可能是其他字段提供了替代信息。SHAP 与消融用于描述预测模型，解释因果关系还需要额外的识别设计。

批量实验由独立入口运行，不调用语言模型。每折保存样本行号、内层划分、搜索轨迹、参数、拟合对象、逐条预测与 SHAP 值。另用核对脚本从逐条预测复算分类或回归指标，检查时间和分组边界、解释加和以及候选评估次数。搜索使用两个数据集进程并行，每个模型仅用一个工作线程。旧数据已参与前期分析，结果按重新定义协议的交叉验证估计报告。
'''
 methods+=holdout_methods()
 (SRC/'research_methods.tex').write_text(methods)
 old=(BACKUP/'paper.tex').read_text()
 intro=old[old.index('\\section{引言}')+len('\\section{引言}'):old.index('\\section{数据与样本划分}')]
 data=old[old.index('\\section{数据与样本划分}'):old.index('\\section{字段检索与统计计算}')].replace('\\section{数据与样本划分}','\\subsection{数据与原单次划分复核}').replace('主要比较的样本','原单次划分复核的样本').replace('四项预测任务的样本和输入变量','原单次划分复核的样本和输入变量').replace('本文只用五折分组器的第一个划分。','原单次划分只用五折分组器的第一个划分。')
 retrieval=old[old.index('\\section{字段检索与统计计算}'):old.index('\\section{预测模型与评价方法}')].replace('\\section{字段检索与统计计算}','\\subsection{字段检索与统计计算}')
 start=retrieval.index('用户先选择数据集。');end=retrieval.index('\n',start)
 retrieval=retrieval[:start]+'用户先选择数据集。程序通过 DeepSeek API 发送问题、字段名与类型、相关字段和方法说明，以及必要的上一轮方案，接收 JSON 分析参数。程序校验格式和列名，用户核对条件后，由本机读取原始表格计算或训练模型。生成新方案需要联网；原始 CSV 不随请求整表上传。本例采用人工指定方案，检索排名来自对原问句的实际查询。'+retrieval[end:]
 models=old[old.index('\\section{预测模型与评价方法}'):old.index('\\section{实验结果}')].replace('\\section{预测模型与评价方法}','\\subsection{预测模型与评价方法}').replace('各项实验均未搜索超参数。','上述单次划分复核固定超参数；扩展搜索协议见本节前文。')
 models+='\\subsection{代码公开与复现}\\label{sec:reproducibility}\n\\faGithub\\enspace 项目仓库：\\href{https://github.com/Karcen/TableLab_DeepSeek}{\\nolinkurl{github.com/Karcen/TableLab_DeepSeek}}。中英文 \\code{README.md} 与 \\code{README.zh-CN.md} 说明网页启动、实验入口和文件位置。复现时记录所用提交编号，并核对 \\code{artifacts/research/protocol.json} 中的依赖版本、随机种子与实验代码哈希。\n\n\\faTerminal\\enspace 在 macOS 上下载完整项目后，进入项目根目录，依次运行：\n\\begin{lstlisting}\nbash TableLab-DeepSeek.command\nbash Run-Research.command\n\\end{lstlisting}\n第一个入口用 Python 3.11、3.12 或 3.13 准备环境并打开网页；网页就绪后可按 Control-C 停止服务，再运行第二个入口。第二个入口安装实验依赖，完成四任务训练、指标复算、绘图和文档重建。完整折会复用；修改实验代码后，先归档旧结果再重跑。批量实验不调用 DeepSeek API，网页生成方案则需使用者自行提供密钥，默认保存在本机 Mac 钥匙串中。\n\n绘图使用 Nature Skills 的 \\code{nature-figure} 技能，默认目录为 \\code{~/Downloads/SKILLS/nature-skills-main/skills/nature-figure}，可通过 \\code{NATURE_FIGURE_SKILL} 指定其他位置。文档编译需要 XeLaTeX、BibTeX 和 Font Awesome 5 字体包。在项目根目录运行 \\code{bash}\\ \\code{docs/build_docs.sh} 可仅重建论文与展示稿。\n\n\\faCode\\enspace 已保存结果位于 \\code{artifacts/research}。运行 \\code{codes/.venv/bin/python}\\ \\code{codes/tools/verify_research.py} 可从逐条预测复算指标，检查内外层分折与 SHAP 加和。\\code{codes/verification/research/files.sha256} 用于核对交付文件；数据来源及其核实情况见附录 C、D。重跑所得模型应按相同协议比较，API 生成的新方案另受当时服务配置与响应影响。\n\n'
 models+=r'''Adult 补测另存于 \code{artifacts/holdout/adult}，冻结协议为该目录下的 \code{protocol.json}。环境准备完成后，在根目录运行以下入口：
\begin{lstlisting}
codes/.venv/bin/python codes/tools/run_adult_holdout.py
codes/.venv/bin/python codes/tools/verify_adult_holdout.py
\end{lstlisting}
已完成测试会直接复用，不根据测试成绩重新搜索。补测图由 \code{codes/tools/plot_adult_holdout.py} 生成，使用同一 Nature Skills；运行 \code{codes/tools/integrate_research_docs.py} 可将核对后的数值写入论文、展示稿和汇总文档，再编译 PDF。

'''
 previous=old[old.index('\\section{实验结果}')+len('\\section{实验结果}'):old.index('\\section{讨论与结论}')].replace('\\subsection{','\\subsubsection{')
 previous=previous.replace('自由中文提问的测试安排在讨论部分说明。',r'自由中文提问与 API 方案质量的后续评价办法见附录~\ref{app:qa-evaluation}。')
 previous=re.sub(r'\\begin\{figure\}\[H\].*?\\end\{figure\}\s*','',previous,flags=re.S)
 previous=previous.replace('图~\\ref{fig:roc} 比较','附录图~\\ref{fig:roc}b 比较')

 discussion=r'''四个任务使用不同的观测单位与预测信息。增加模型与搜索参数后，仍需按任务判断收益。完整树的训练与验证差距说明，仅降低训练误差不足以说明模型有效。SHAP 和重新训练的消融提供了两种检查字段依赖的办法；相关字段之间可以互相替代，因此解释结果应结合输入信息阅读。

嵌套验证允许比较包括调参在内的训练流程。三折结果保留各折样本构成的差异，尤其 NBA 各验证期来自不同赛季；折间标准差用于描述这种变化。原固定划分与新增验证协议评价的是不同样本，不把两组数值直接当成调参前后的变化。Adult 已另用预留记录评价，提升树＋网格与 TPE 的测试 AUC 接近，配对差值区间包含零，未分清两者优劣；与预设提升树和森林相比，选定流程的差值区间高于零，但这些为未经多重比较校正的描述性区间。房价、NBA 与学生任务仍缺少新样本独立测试。

TableLab 将问题、字段说明与分析参数连接到实际计算。DeepSeek API 的方案生成质量尚未评价，后续办法见附录~\ref{app:qa-evaluation}；本轮多模型实验以人工固定任务和特征检验数值训练部分。对社区或备考课程的解释限于模型利用的信息，学生数据的真实抽样过程仍需核实。
'''
 header=old[:old.index('\\begin{abstract}')]
 header=header.replace('\\usepackage{float}',r'\usepackage{float,fontawesome5}')
 header=header.replace('\\raggedbottom','% Allow figures and tables to share pages with text.\n\\renewcommand{\\topfraction}{.9}\n\\renewcommand{\\bottomfraction}{.85}\n\\renewcommand{\\textfraction}{.08}\n\\renewcommand{\\floatpagefraction}{.85}\n\\setcounter{topnumber}{3}\n\\setcounter{bottomnumber}{2}\n\\setcounter{totalnumber}{4}\n'+'\\raggedbottom')
 header=header.replace('\\newcommand{\\inlinetab}[1]{\\par\\smallskip\\noindent\\begin{minipage}{\\linewidth}\\centering\\small #1\\end{minipage}\\par\\smallskip}','\\newcommand{\\inlinetab}[1]{\\begin{table}[!htbp]\\centering\\small #1\\end{table}}')
 abstract='TableLab 在本地检索字段说明，通过 DeepSeek API 生成分析方案，再由本机程序完成统计与模型训练。字段检索检查限于 16 道固定题，其中 15 道含英文字段名；DeepSeek API 的方案生成质量尚未评价。本文用房价、收入分类、NBA 得分和学生成绩四个任务比较九种预设方法，并对森林与梯度提升分别进行网格和 TPE 搜索。外层三折评价训练流程，内层三折选择参数；训练与验证曲线检查拟合差距，SHAP 和重新调参的特征组消融检查模型依赖。'+detail+'各任务的较优方法与搜索收益不同，解释和消融结果也需结合字段之间的替代关系判断。'
 abstract+=holdout_brief().replace('%',r'\%')+'网格与 TPE 的测试差值区间包含零，其余三个任务尚无独立测试。'
 tail=old[old.index('\\clearpage\n\\appendices'):]
 paper=header+'\\begin{abstract}\n'+abstract+'\n\\end{abstract}\n\\renewcommand{\\IEEEkeywordsname}{关键词}\n\\begin{IEEEkeywords}表格问答；嵌套交叉验证；超参数搜索；SHAP；消融实验\\end{IEEEkeywords}\n\\clearpage\\tableofcontents\\clearpage\n\\section{研究问题}\n'+intro+'\n本文在原固定参数复核之外，比较多类模型及两种超参数搜索，检查训练误差与验证误差的分离，并用字段贡献和重新训练的消融分析预测依据。\n\\section{结果}\n\\input{research_results}\n\\subsection{原单次划分的计算复核}\n'+previous+'\n\\section{意义}\n'+discussion+'\n\\section{方法}\n'+data+'\n\\input{research_methods}\n'+retrieval+'\n'+models+tail
 paper=paper.replace('[H]','[!htbp]')
 paper_main,paper_appendix=paper.split('\\appendices',1)
 paper=paper_main.replace('\\FloatBarrier','')+'\\appendices'+paper_appendix
 from expanded_docs import ready as expanded_ready,apply as apply_expanded,report as expanded_report
 if expanded_ready():paper=apply_expanded(paper)
 (SRC/'paper.tex').write_text(paper)
 # Rebuild the concise oral presentation using the preserved author metadata.
 from compact_presentation import compact_deck
 (SRC/'presentation.tex').write_text(compact_deck(ROOT,SUMMARY))
 # Readable consolidated summary, with the requested four-part order.
 report='# TableLab 四任务多模型实验\n\n## 研究问题\n\n在固定可用字段和验证样本后，多模型与参数搜索能否改善预测？训练和验证差距来自哪里，模型依赖哪些字段？\n\n## 结果\n\n'+detail+'\n\n以上为外层三折均值，每个任务比较九种预设方法及四条调参流程。提升树调参使房价平均 RMSE 降低约 967 美元，收入分类 AUC 增量约 0.0001；森林调参在收入与 NBA 任务中反而略差。论文保留完整图表与推导；16 页展示稿集中呈现模型比较、调参收益、拟合诊断、SHAP 和特征组消融。\n\n## 意义\n\n模型选择应结合具体任务。训练误差和验证误差一起读，才能识别过拟合迹象；两者都高也可能源于输入信息不足。SHAP 说明模型如何使用字段，删除字段后重新训练检验其余信息能否替代它，两者均不直接估计因果效应。\n\n## 方法\n\n四个任务均采用外层三折、内层三折；两种搜索在同一十二点离散域内各评估十二次。Adult 保持指纹分组，NBA 按完整赛季向前验证；插补和编码在训练折拟合。SHAP 解释全部外层验证记录，逐条检查贡献加和；各消融重新进行内层调参。三折标准差描述折间波动，不作为独立重复试验的置信区间。旧数据已用于前期分析，结果属于交叉验证估计。6 项协议测试通过；546 项指标已从逐条预测独立复算。SHAP 覆盖 43,893 条外层验证记录，贡献加和最大误差约为 4.7×10⁻¹⁰。\n\n'
 report=report.replace('16 页展示稿','17 页 Madrid 展示稿')
 report=report.replace('## 意义',holdout_brief().replace('--','–')+'网格与 TPE 的配对差值区间包含零；房价、NBA 和学生仍缺少独立新样本。完整补测见 [Adult 预留测试说明](adult_holdout_results.md)。\n\n## 意义')
 report+='[课程论文]('+str(ROOT/'docs/paper.pdf')+') · [Beamer 展示稿]('+str(ROOT/'docs/presentation.pdf')+')\n'
 (ROOT/'docs/research_summary.md').write_text(report)
 if expanded_ready():
  (ROOT/'docs/research_summary.md').write_text(expanded_report())
  from expanded_docs import update_readmes
  update_readmes()
 write_report()
 # Primary, previously verified official documentation; no invented literature.
 bib=SRC/'merged_refs.bib';text=bib.read_text()
 entries={'nestedcv':('scikit-learn developers','Nested versus Non-nested Cross-validation','https://scikit-learn.org/1.8/auto_examples/model_selection/plot_nested_cross_validation_iris.html'),'optunadocs':('Optuna developers','Samplers','https://optuna.readthedocs.io/en/stable/reference/samplers/index.html'),'shapdocs':('SHAP developers','TreeExplainer','https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html')}
 for key,(author,title,url) in entries.items():
  if '@misc{'+key+',' not in text:text+='\n@misc{'+key+',author={{'+author+'}},title={'+title+'},year={n.d.},url={'+url+'},note={官方文档；访问日期：2026-11-01}}\n'
 bib.write_text(text)
 print('Updated paper, Beamer and consolidated summary from saved results.')
if __name__=='__main__':main()
