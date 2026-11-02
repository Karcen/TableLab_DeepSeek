"""Compose a concise oral Beamer deck with a custom academic visual system."""
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


def frame(kicker,title,body):
 return '\\begin{frame}[t]{'+title+'}\n\\framesubtitle{'+kicker+'}\n'+body+'\n\\end{frame}\n\n'


def plot(name,height='.61'):
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
\usepackage{amsmath,amssymb,booktabs,tabularx,array,graphicx,xurl,fontawesome5,listings}
\definecolor{Ink}{HTML}{18334A}
\definecolor{Accent}{HTML}{287F86}
\definecolor{Muted}{HTML}{627383}
\definecolor{Mist}{HTML}{F0F4F7}
\definecolor{Warm}{HTML}{B97557}
\setbeamercolor{normal text}{fg=Ink,bg=white}
\setbeamercolor{frametitle}{fg=Ink,bg=white}
\setbeamercolor{framesubtitle}{fg=Accent,bg=white}
\setbeamercolor{structure}{fg=Accent}
\setbeamercolor{codebox}{fg=Ink,bg=Mist}
\setbeamerfont{normal text}{size=\normalsize}
\setbeamerfont{frametitle}{size=\fontsize{19}{23}\selectfont,series=\bfseries}
\setbeamerfont{framesubtitle}{size=\scriptsize,series=\bfseries}
\setbeamersize{text margin left=.65cm,text margin right=.65cm}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{itemize item}{\raisebox{.2ex}{\scriptsize\textbullet}}
\setlength{\parskip}{4pt}
\setbeamertemplate{frametitle}{%
 \vspace{.22cm}
 {\usebeamerfont{framesubtitle}\color{Accent}\insertframesubtitle\par}
 \vspace{.06cm}
 {\usebeamerfont{frametitle}\insertframetitle\par}
 \vspace{.1cm}
}
\setbeamertemplate{footline}{%
 \hspace*{.65cm}{\color{Muted}\fontsize{6.5}{8}\selectfont TABLELAB\quad /\quad 郑嘉诚}
 \hfill{\color{Muted}\fontsize{7}{8}\selectfont\insertframenumber\,/\,\inserttotalframenumber}\hspace*{.65cm}\vspace{.18cm}
}
\hypersetup{colorlinks=true,linkcolor=Accent,urlcolor=Accent,pdftitle={TableLab：中文表格问答与预测},pdfauthor={郑嘉诚}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\code}[1]{\nolinkurl{#1}}
\newcommand{\captionline}[1]{{\color{Muted}\fontsize{8}{10}\selectfont #1\par}}
\newcommand{\takeaway}[1]{\par\vspace{.03cm}{\color{Accent}\bfseries\fontsize{11}{14}\selectfont #1\par}}
\newcommand{\infocard}[3][2.0cm]{%
 \begingroup\setlength{\fboxsep}{8pt}%
 \colorbox{Mist}{\begin{minipage}[t][#1][t]{\dimexpr\linewidth-16pt\relax}
 \raggedright{\color{Accent}\bfseries #2\par}\vspace{.09cm}
 {\small #3\par}
 \end{minipage}}\endgroup
}
\newcommand{\resultcard}[5]{%
 \begingroup\setlength{\fboxsep}{8pt}%
 \colorbox{Mist}{\begin{minipage}[t][1.9cm][t]{\dimexpr\linewidth-16pt\relax}
 \raggedright{\color{Muted}\small #1\par}
 {\color{Accent}\fontsize{22}{25}\selectfont\bfseries #2}\ {\scriptsize #3}\par
 {\small #4\par}{\color{Muted}\scriptsize #5\par}
 \end{minipage}}\endgroup
}
\title[TableLab]{中文表格问答与预测}
'''
 header+='\n'.join([subtitle,author,institute,date])+'\n'
 slides=[]
 slides.append(r'''\begin{frame}[plain]
\vspace{.25cm}
\begin{columns}[T,onlytextwidth]
\column{.8\textwidth}{\color{Accent}\bfseries\fontsize{18}{22}\selectfont TableLab}
\column{.12\textwidth}\raggedleft\includegraphics[height=1.15cm]{logo/TDLogo.pdf}
\end{columns}
\vspace{.38cm}
{\fontsize{27}{33}\selectfont\bfseries\inserttitle\par}
\vspace{.22cm}
{\color{Muted}\fontsize{12}{16}\selectfont 本地字段检索\quad /\quad DeepSeek API\quad /\quad 多模型验证\par}
\vspace{.45cm}
{\color{Accent}\rule{1.6cm}{1.5pt}\par}
\vspace{.3cm}
{\fontsize{14}{18}\selectfont\insertauthor\par}
\vspace{.14cm}
{\fontsize{10}{14}\selectfont\insertinstitute\par}
\vfill
{\color{Muted}\scriptsize\insertsubtitle\hfill\insertdate\par}
\end{frame}

''')
 slides.append(frame('01 / RESEARCH QUESTION','中文提问，怎样落到具体计算？',r'''
{\large “CollgCr 社区的平均成交价是多少？”\par}
\vspace{.22cm}
\begin{columns}[T,onlytextwidth]
\column{.315\textwidth}\infocard[2.15cm]{1\quad 查字段}{本地检索列名、中文释义和方法说明。}
\column{.315\textwidth}\infocard[2.15cm]{2\quad 定方案}{DeepSeek API 返回 JSON；用户确认条件。}
\column{.315\textwidth}\infocard[2.15cm]{3\quad 做计算}{本机读取表格，完成统计或模型训练。}
\end{columns}
\vspace{.22cm}
\takeaway{分别检查：字段是否找对，预测误差是否降低。}
\captionline{请求包含问题与字段、方法说明；原始 CSV 不整表上传。}
'''))
 slides.append(frame('02 / DATA','四个任务，四种预测目标',r'''
\begin{columns}[T,onlytextwidth]
\column{.485\textwidth}\infocard[1.72cm]{房价\quad 1,460 条}{目标：成交价。\par 面积、质量、年份、地下室、车位、社区。}
\column{.485\textwidth}\infocard[1.72cm]{Adult\quad 32,561 条}{目标：年收入是否超过 5 万美元。\par 年龄、教育、工时与就业、婚姻信息。}
\end{columns}
\vspace{.23cm}
\begin{columns}[T,onlytextwidth]
\column{.485\textwidth}\infocard[1.72cm]{NBA\quad 12,844 条}{目标：球员场均得分。\par 年龄、身高、体重、选秀年份。}
\column{.485\textwidth}\infocard[1.72cm]{学生\quad 1,000 条}{目标：数学成绩。\par 四个背景字段，不使用阅读、写作成绩。}
\end{columns}
\vspace{.12cm}
\captionline{Adult 仅使用原训练文件；独立测试文件的 16,281 条记录未参与。数据来源见论文附录 C、D。}
'''))
 slides.append(frame('03 / EVALUATION','外层评价，内层选参数',r'''
\begin{columns}[T,onlytextwidth]
\column{.54\textwidth}
\infocard[3.25cm]{三折外层验证}{留出本折验证记录\par\vspace{.1cm}
\hspace{.15cm}$\downarrow$\par
训练子集内再分三折，搜索参数\par\vspace{.1cm}
\hspace{.15cm}$\downarrow$\par
在外层留出记录上计算指标}
\column{.42\textwidth}
{\color{Accent}\bfseries 分折随任务而定\par}
\vspace{.12cm}
房价、学生：随机分折。\par
Adult：指纹分组并按标签分层。\par
NBA：按完整赛季向前滚动。
\end{columns}
\vspace{.22cm}
\takeaway{内层沿用同一分折规则；插补、缩放和编码只在训练折拟合。}
'''))
 slides.append(frame('04 / MODELS','每个任务比较十三条训练流程',r'''
\begin{columns}[T,onlytextwidth]
\column{.315\textwidth}\infocard[1.9cm]{\fontsize{25}{28}\selectfont 9}{预设方法}
\column{.315\textwidth}\infocard[1.9cm]{\fontsize{25}{28}\selectfont +4}{森林、提升树各用网格与 TPE}
\column{.315\textwidth}\infocard[1.9cm]{\fontsize{25}{28}\selectfont 1,728}{主搜索的内层拟合次数}
\end{columns}
\vspace{.22cm}
\small 常数基线、线性与正则化模型、决策树、随机森林、Extra Trees、直方图梯度提升、近邻。
\par\vspace{.12cm}
\takeaway{网格与 TPE 贝叶斯搜索在同一十二点参数域内，各评估十二次。}
\captionline{TPE 可重复组合。预设森林使用 64 棵树；原单次划分的样本和配置不同，详细结果留在论文。}
'''))
 def winner(ds):return summary.query('dataset==@ds').sort_values('mean',ascending=ds!='adult').iloc[0]
 cards=[]
 for ds,label,unit,method in [('house','房价','美元 RMSE','提升树＋网格 / TPE'),('adult','收入分类','AUC','提升树＋网格 / TPE'),('nba','NBA 得分','分 RMSE','预设随机森林'),('students','学生数学','分 RMSE','Ridge')]:
  w=winner(ds);digits=2 if ds=='house' else 4 if ds=='adult' else 3
  cards.append(r'\resultcard{'+label+'}{'+f'{w["mean"]:,.{digits}f}'+'}{'+unit+'}{'+method+'}{折间标准差 '+f'{w["sd"]:,.{digits}f}'+'}')
 body=''
 for j in [0,2]:
  body+='\\begin{columns}[T,onlytextwidth]\n\\column{.485\\textwidth}'+cards[j]+'\n\\column{.485\\textwidth}'+cards[j+1]+'\n\\end{columns}\n\\vspace{.22cm}\n'
 body+=r'\captionline{三折外层均值；标准差描述折间波动。房价与收入的两条搜索流程同值，排名仍需新样本确认。}'
 slides.append(frame('05 / MAIN RESULTS','较优模型随任务变化',body))
 slides.append(frame('06 / HOUSE PRICE','房价：提升树的平均误差较小',plot('deck_benchmark_house','.61')+r'''
\takeaway{提升树＋网格 / TPE：RMSE 为 32,338.23 美元。}
\captionline{点形区分三个外层折；菱形与误差线为均值和标准差。各模型使用相同字段与分折。}
'''))
 slides.append(frame('07 / HYPERPARAMETERS','调参收益随任务变化',plot('deck_search','.51')+r'''
\begin{columns}[T,onlytextwidth]
\column{.24\textwidth}{\color{Accent}\bfseries 966.57 美元}\par\scriptsize 房价 RMSE 降幅
\column{.24\textwidth}{\color{Accent}\bfseries 0.0001}\par\scriptsize 收入 AUC 增量
\column{.24\textwidth}{\color{Accent}\bfseries 0.014 分}\par\scriptsize NBA RMSE 降幅
\column{.24\textwidth}{\color{Accent}\bfseries 0.115 / 0.082 分}\par\scriptsize 学生：网格 / TPE 降幅
\end{columns}
\vspace{.16cm}
\captionline{上图是提升树内层搜索轨迹，数字是同折外层收益。森林调参在收入和 NBA 上反而略差。}
'''))
 slides.append(frame('08 / OVERFITTING','树更深，验证误差未必更低',plot('deck_complexity','.62')+r'''
\takeaway{完整树的训练误差下降，收入与 NBA 的验证误差却上升。}
\captionline{曲线为三折均值，阴影为标准差；Full 为不限制深度。浅树的改善支持拟合不足，训练与验证的分离支持过拟合诊断。}
'''))
 slides.append(frame('09 / LEARNING CURVES','增加数据后，再看训练与验证的差距',plot('deck_learning','.62')+r'''
\takeaway{房价验证误差随样本增加而下降；其余任务的变化更小。}
\captionline{预处理在各子样本重新拟合。两种误差都高时，也需检查输入信息是否充分。}
'''))
 slides.append(frame('10 / SHAP','四个任务分别依赖哪些字段？',plot('deck_shap','.62')+r'''
\takeaway{质量\quad /\quad 婚姻状况\quad /\quad 选秀年份\quad /\quad 午餐类别}
\captionline{预先指定网格调参提升树，解释全部外层验证记录。条形为平均绝对 SHAP；误差线为三折标准差。分类使用对数几率。}
'''))
 slides.append(frame('11 / ABLATION','贡献排名与删字段的损失可以不同',plot('deck_ablation','.575')+r'''
\takeaway{房价 SHAP 以质量最高，删去面积组却使 RMSE 增加 7,000.18 美元。}
\captionline{消融会重新调参和训练，其余字段可提供替代信息。Only area + quality 仅保留地上居住面积与整体质量；正值表示损失。}
'''))
 slides.append(frame('12 / LOCAL EXPLANATION','一条预测可以逐项还原',plot('deck_local','.62')+r'''
\takeaway{基准值＋各字段贡献＝模型原始输出。}
\captionline{房价与收入各选一条绝对误差居中的记录。橙、蓝表示正、负贡献；全部 43,893 条外层记录的贡献加和已核对。}
'''))
 slides.append(frame('13 / TABLE QUESTIONS','问答完成了哪些检查？',r'''
\begin{columns}[T,onlytextwidth]
\column{.485\textwidth}\infocard[3.4cm]{社区均价可以复算}{
CollgCr 的 150 条成交记录。\par\vspace{.15cm}
{\color{Accent}\fontsize{24}{28}\selectfont 197,965.77}\par 美元\par\vspace{.15cm}
由成交价合计 29,694,866 除以 150 得到。
}
\column{.485\textwidth}\infocard[3.4cm]{字段检索的固定题检查}{
四份表各 4 题，共 16 题。\par\vspace{.15cm}
{\color{Accent}\fontsize{24}{28}\selectfont 0.96875}\par MRR\par\vspace{.15cm}
15 题排第一，1 题排第二；15 题直接含英文字段名。
}
\end{columns}
\vspace{.22cm}
\captionline{社区计算使用人工固定方案；真实 DeepSeek API 方案生成质量尚未评价。}
'''))
 slides.append(frame('14 / CODE AND REPRODUCTION','代码与复现',r'''
{\large\faGithub\enspace\href{https://github.com/Karcen/TableLab_DeepSeek}{\nolinkurl{github.com/Karcen/TableLab_DeepSeek}}\par}
\vspace{.28cm}
\begin{beamercolorbox}[sep=9pt,wd=\textwidth]{codebox}
{\ttfamily\small bash TableLab-DeepSeek.command\par}
{\ttfamily\small bash Run-Research.command\par}
\end{beamercolorbox}
\vspace{.18cm}
先准备网页环境，再停止服务并运行批量实验。\par
\small 方案、分折、模型和逐条结果保存在 \code{artifacts/research}；复现时记录提交编号与依赖版本。
\vspace{.15cm}
\captionline{批量实验不调用 API；网页需自行配置密钥。绘图需 Nature Skills；编译需 XeLaTeX、BibTeX 与 Font Awesome 5。完整推导与参考文献见论文。}
'''))
 slides.append(frame('15 / PROS AND CONS','优点与不足',r'''
\begin{columns}[T,onlytextwidth]
\column{.46\textwidth}
{\color{Accent}\fontsize{16}{20}\selectfont\bfseries Pros\par}
\vspace{.17cm}
四任务在同折记录上比较。\par\vspace{.14cm}
调参和预处理限定在训练折内。\par\vspace{.14cm}
保存逐条预测，可复算指标与贡献加和。
\column{.49\textwidth}
{\color{Warm}\fontsize{16}{20}\selectfont\bfseries Cons\par}
\vspace{.17cm}
16 道检索题中，15 道含英文字段名。\par\vspace{.1cm}
真实 API 方案质量尚未评价。\par\vspace{.1cm}
三折排名仍需独立新样本确认。\par\vspace{.1cm}
SHAP 与消融均不直接估计因果效应。
\end{columns}
\vspace{.28cm}
\takeaway{下一步：评价自由中文提问的方案质量，并完成独立测试。}
'''))
 assert len(slides)==16,len(slides)
 deck=header+'\\begin{document}\n'+''.join(slides)+'\\end{document}\n'
 assert '边界' not in deck
 return deck
