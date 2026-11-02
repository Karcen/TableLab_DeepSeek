"""Generate the concise academic presentation with the standard Madrid theme."""
from pathlib import Path
import re

REPO='https://github.com/Karcen/TableLab_DeepSeek'


def metadata_command(source,name,default):
 match=re.search(r'\\'+name+r'(?:\[[^\]]*\])?\{',source)
 if not match:return default
 depth=1;end=match.end()
 while end<len(source) and depth:
  if source[end]=='{' and (end==0 or source[end-1]!='\\'):depth+=1
  elif source[end]=='}' and source[end-1]!='\\':depth-=1
  end+=1
 return source[match.start():end] if depth==0 else default


def frame(title,body):
 return '\\begin{frame}{'+title+'}\n\\relax\n'+body+'\n\\end{frame}\n\n'


def plot(name,height='.66'):
 return '{\\centering\\includegraphics[width=\\linewidth,height='+height+'\\textheight,keepaspectratio]{figures/research/'+name+'.pdf}\\par}\n'


def compact_deck(root,summary):
 current=(root/'docs/source/presentation.tex').read_text()
 author=metadata_command(current,'author',r'\author[郑嘉诚]{\href{https://karcen.github.io/}{郑嘉诚}}')
 institute=metadata_command(current,'institute',r'\institute[马寅初经济学院]{马寅初经济学院\quad 应用经济学\\[.5em]学号：2026209004}')
 subtitle=metadata_command(current,'subtitle',r'\subtitle{天津大学机器学习与人工智能I课程作业}')
 date=metadata_command(current,'date',r'\date[2026.11.01]{2026年11月1日}')
 header=r'''% !TEX program = xelatex
\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[UTF8,fontset=fandol]{ctex}
\usepackage{amsmath,amssymb,booktabs,tabularx,array,graphicx,xurl,fontawesome5}
\usetheme{Madrid}
\usecolortheme{default}
\setbeamertemplate{navigation symbols}{}
\hypersetup{colorlinks=true,linkcolor=.,urlcolor=blue,pdftitle={TableLab：基于本地字段检索的中文表格问答与预测},pdfauthor={郑嘉诚}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\code}[1]{\nolinkurl{#1}}
\newcommand{\captionline}[1]{\par\vspace{1.5mm}{\fontsize{8}{10}\selectfont #1\par}}
\title[TableLab]{TableLab：基于本地字段检索的\\中文表格问答与预测}
'''
 header+='\n'.join([subtitle,author,institute,date])+'\n'
 header+=r'\titlegraphic{\scriptsize\href{https://github.com/Karcen/TableLab_DeepSeek}{\faGithub\enspace\nolinkurl{github.com/Karcen/TableLab_DeepSeek}}}'+'\n'
 slides=[r'''\begin{frame}[plain]
\titlepage
\end{frame}

''']
 slides.append(frame('研究问题与系统分工',r'''
\begin{itemize}\setlength{\itemsep}{.8em}
\item 本地检索字段与方法说明，将中文问题对应到表中的列。
\item DeepSeek API 根据问题与检索结果生成 JSON 分析方案。
\item 用户确认字段和条件后，本机读取原表，完成统计或模型训练。
\end{itemize}
\vspace{.25cm}
本文分别检查字段检索排名和模型预测误差。
\captionline{API 请求不整表上传原始 CSV。方案生成与本机数值计算分别记录。}
'''))
 slides.append(frame('数据与预测任务',r'''
\small
\begin{tabularx}{\textwidth}{@{}lr l Y@{}}\toprule
数据 & 记录数 & 预测目标 & 输入信息\\\midrule
房价 & 1,460 & 成交价 & 面积、质量、年份、地下室、车位、社区。\\[.6em]
Adult & 32,561 & 收入类别 & 年龄、教育、工时、用工类别、职业、婚姻。\\[.6em]
NBA & 12,844 & 场均得分 & 年龄、身高、体重、选秀年份。\\[.6em]
学生 & 1,000 & 数学成绩 & 性别、父母教育、午餐、备考课程。\\\bottomrule
\end{tabularx}
\captionline{本表为嵌套验证的样本数。Adult 另有 16,255 条预留测试记录，已排除跨文件重合的 26 条。学生任务排除同期阅读、写作成绩。}
'''))
 slides.append(frame('验证方法',r'''
\begin{itemize}\setlength{\itemsep}{.65em}
\item 外层三折评价训练流程；内层三折选择超参数。
\item 内层仅使用相应外层训练子集，并沿用同一分折规则。
\item 插补、缩放与独热编码均在当前训练折拟合。
\end{itemize}
\vspace{.2cm}
\small
\begin{tabularx}{\textwidth}{@{}lY@{}}\toprule
任务 & 分折规则\\\midrule
房价、学生 & 固定随机三折。\\
Adult & 按全部非目标原始字段指纹分组，并按标签分层。\\
NBA & 按完整赛季向前滚动，验证赛季晚于训练赛季。\\\bottomrule
\end{tabularx}
'''))
 slides.append(frame('模型与超参数搜索',r'''
\begin{itemize}\setlength{\itemsep}{.7em}
\item 九种预设方法：常数基线、三种线性或正则化模型、决策树、随机森林、Extra Trees、直方图梯度提升与近邻。
\item 森林与提升树分别增加网格和 TPE 贝叶斯搜索，每个任务共十三条流程。
\item 两种搜索在相同的十二点参数域内各评估十二次；TPE 允许重复组合。
\end{itemize}
\captionline{主搜索共 1,728 次内层拟合，消融另计。预设森林使用 64 棵树；原单次划分采用不同样本和配置，结果单独见论文。}
'''))
 body=r'\small\begin{tabularx}{\textwidth}{@{}lYl@{}}\toprule'+'\n'+r'任务 & 较优流程 & 外层均值 $\pm$ 标准差\\\midrule'+'\n'
 for ds,label,method in [('house','房价','提升树＋网格 / TPE'),('adult','收入分类','提升树＋网格 / TPE'),('nba','NBA 得分','预设随机森林'),('students','学生数学','Ridge')]:
  row=summary.query('dataset==@ds').sort_values('mean',ascending=ds!='adult').iloc[0]
  digits=2 if ds=='house' else 4 if ds=='adult' else 3
  body+=label+' & '+method+' & '+f'{row["mean"]:,.{digits}f} $\\pm$ {row["sd"]:,.{digits}f}'+r'\\[.7em]'+'\n'
 body+=r'''\bottomrule\end{tabularx}
\par\medskip
房价以美元 RMSE 评价，NBA 与学生以分数 RMSE 评价；收入以 AUC 评价。
\captionline{三折外层均值。标准差描述折间波动，训练集有重合；Adult 的预留测试另列下一页，其余任务尚无独立测试。}
'''
 slides.append(frame('四任务模型比较结果',body))
 from adult_holdout_docs import data as holdout_data
 heldout,paired=holdout_data()
 chosen=heldout.loc['HGB_grid'];tpe=heldout.loc['HGB_tpe'];difference=paired.loc['HGB_tpe']
 slides.append(frame('Adult 预留测试',plot('deck_adult_holdout','.65')+
  f'\\small 事先选定的网格提升树 AUC 为 {chosen.auc:.4f}，TPE 为 {tpe.auc:.4f}；差值区间包含零。\n'+
  r'\captionline{32,561 条训练、16,255 条测试；排除跨文件重合的 26 条。误差线为 2,000 次分组 bootstrap 的 95\% 区间；配对区间未作多重比较校正。测试后未回调参数。}'+'\n'))
 slides.append(frame('房价模型比较',plot('deck_benchmark_house','.66')+r'''
提升树＋网格与 TPE 的平均 RMSE 均为 32,338.23 美元。
\captionline{点形区分三个外层折，菱形与误差线为三折均值和标准差。各模型使用相同特征与分折。}
'''))
 slides.append(frame('超参数搜索结果',plot('deck_search','.56')+r'''
\small 提升树的外层变化：房价 RMSE 降 966.57 美元，NBA 降 0.014 分；学生分别降 0.115、0.082 分，收入 AUC 增约 0.0001。
\captionline{上图为内层搜索轨迹。学生降幅依次对应网格与 TPE；森林调参在收入和 NBA 上略差。外层收益均按同折预设模型与调参模型计算。}
'''))
 slides.append(frame('过拟合与欠拟合诊断',plot('deck_complexity','.66')+r'''
完整树的训练误差较低，但收入与 NBA 的验证误差上升。
\captionline{Full 表示不限深度。曲线为三折均值，阴影为标准差；浅树增深后的改善用于检查拟合不足。}
'''))
 slides.append(frame('学习曲线',plot('deck_learning','.66')+r'''
房价的验证误差随训练样本增加而下降，其余任务变化较小。
\captionline{各子样本重新拟合预处理。曲线为三折均值，阴影为标准差；训练与验证误差都高时，也需检查输入信息是否充分。}
'''))
 slides.append(frame('SHAP 全局解释',plot('deck_shap','.66')+r'''
四任务中平均绝对贡献较大的字段分别为质量、婚姻状况、选秀年份和午餐类别。
\captionline{解释模型预先固定为网格调参提升树，使用全部外层验证记录。误差线为三折标准差；分类贡献在对数几率尺度上分解。}
'''))
 slides.append(frame('特征组消融',plot('deck_ablation','.62')+r'''
\small 房价的 SHAP 以整体质量最高；删去面积组后，RMSE 平均增加 7,000.18 美元，损失最大。重新训练时，其余字段可提供替代信息。
\captionline{各变体重新进行网格调参。正值表示性能损失；Only area + quality 仅保留地上居住面积与整体质量。}
'''))
 slides.append(frame('单条预测的 SHAP 分解',plot('deck_local','.66')+r'''
基准值与字段贡献之和重建模型原始输出。
\captionline{房价和收入各选一条绝对误差居中的记录。橙、蓝分别表示正、负贡献；全部 43,893 条外层记录的贡献加和均已核对。}
'''))
 slides.append(frame('表格问答的计算与检索检查',r'''
\begin{columns}[T,onlytextwidth]
\begin{column}{.47\textwidth}
\textbf{社区均价计算}
\par\medskip
筛选 CollgCr 社区中成交价有记录的 150 条房屋交易。
\[
\bar y=\frac{29{,}694{,}866}{150}
\]
均价为 197,965.77 美元。
\end{column}
\begin{column}{.49\textwidth}
\textbf{字段检索检查}
\begin{itemize}\setlength{\itemsep}{.6em}
\item 四份表各 4 题，共 16 题。
\item 15 题排第一，1 题排第二；MRR 为 0.96875。
\item 15 题直接含英文字段名。
\end{itemize}
\end{column}
\end{columns}
\captionline{均价计算使用人工固定方案；真实 DeepSeek API 方案生成质量尚未评价。}
'''))
 slides.append(frame('代码与复现',r'''
\faGithub\enspace\href{https://github.com/Karcen/TableLab_DeepSeek}{\nolinkurl{github.com/Karcen/TableLab_DeepSeek}}
\par\medskip
\begin{enumerate}\setlength{\itemsep}{.7em}
\item 运行 \code{TableLab-DeepSeek.command}，准备 Python 环境。
\item 停止网页服务后，运行 \code{Run-Research.command}，完成实验、核对、绘图与文档重建。
\item 对照 \code{artifacts/research} 中的分折、逐条预测和 SHAP 值，记录提交编号与依赖版本。
\end{enumerate}
\captionline{Adult 补测：\code{codes/tools/run_adult_holdout.py}；复算：\code{codes/tools/verify_adult_holdout.py}。批量实验不调用 API；绘图需要 Nature Skills，编译需要 XeLaTeX、BibTeX 与 Font Awesome 5。}
'''))
 slides.append(frame('优点与不足',r'''
\begin{columns}[T,onlytextwidth]
\begin{column}{.46\textwidth}
\textbf{优点（Pros）}
\begin{itemize}\setlength{\itemsep}{.75em}
\item 四任务在相同外层分折上比较。
\item 调参与预处理限定在训练折内。
\item 保存逐条预测，支持指标与贡献加和的复算。
\end{itemize}
\end{column}
\begin{column}{.49\textwidth}
\textbf{不足（Cons）}
\begin{itemize}\setlength{\itemsep}{.55em}
\item 检索只检查 16 道固定题，其中 15 道含英文字段名。
\item 真实 API 方案质量尚未评价。
\item 房价、NBA 与学生尚无独立测试。
\item Adult 测试未分清网格与 TPE 优劣。
\item SHAP 与消融不直接估计因果效应。
\end{itemize}
\end{column}
\end{columns}
'''))
 assert len(slides)==17
 from expanded_docs import ready as expanded_ready
 if expanded_ready():
  from expanded_deck import render
  return render(header,frame)
 deck=header+'\\begin{document}\n'+''.join(slides)+'\\end{document}\n'
 assert '边界' not in deck
 return deck
