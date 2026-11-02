"""Build the 20-slide oral presentation, including the cover and final slide."""
from pathlib import Path
import re

REPO='https://github.com/Karcen/TableLab_DeepSeek'

def frame(title,body):
 return '\\begin{frame}{'+title+'}\n'+body+'\n\\end{frame}\n\n'

def compact_deck(root,summary):
 full=(root/'docs/archive/presentation_70_pages/presentation.tex').read_text()
 header=full[:full.index('\\begin{document}')]
 header=header.replace(r'\date[2026.11.01]{\today}',r'\date[2026.11.01]{2026年11月1日}')
 frames=re.findall(r'\\begin\{frame\}.*?\\end\{frame\}',full,flags=re.S)
 assert len(frames)==70,len(frames)
 slides=[frames[0]+'\n\n']
 slides.append(frame('中文问题怎样变成可复算的结果',r'''
\begin{tabularx}{\textwidth}{@{}lY@{}}\toprule
环节 & TableLab 实际执行的步骤\\\midrule
本地检索 & 用问题查找字段中文释义、取值和方法说明。\\
DeepSeek API & 根据问题与检索说明，生成 JSON 分析方案。\\
本机计算 & 用户确认字段和条件后，读取原表并计算或训练模型。\\\bottomrule
\end{tabularx}
\par\medskip
问答检查选列与筛选，预测检查相同外层样本上的模型误差。
\smallnote{API 请求包含问题、字段与方法说明，原始 CSV 不整表上传；统计与模型训练在本机完成。}
'''))
 # Keep the four datasets visible, without the old fixed-split result sequence.
 data=frames[2]
 data=re.sub(r'数据来源见\\citep\{[^}]+\}。','数据来源见论文附录 C、D。',data)
 slides.append(data+'\n\n')
 slides.append(frames[5]+'\n\n')
 slides.append(frame('检索检查的结果与范围',r'''
\begin{tabularx}{\textwidth}{@{}lY@{}}\toprule
检查项 & 已有结果\\\midrule
固定问题 & 四份数据各 4 道，共 16 道。\\
目标说明排名 & 15 道排第一，1 道排第二；MRR 为 0.96875。\\
题目表述 & 15 道直接含英文字段名。\\
评价范围 & 检查字段检索排名；DeepSeek API 方案质量尚未评价。\\\bottomrule
\end{tabularx}
\par\medskip
后面的多模型实验使用人工固定的目标与特征，评价本机数值训练部分。
'''))
 slides.append(frames[20]+'\n\n')
 slides.append(frame('比较九种预设方法与四条调参流程',r'''
\small
\begin{tabularx}{\textwidth}{@{}lY@{}}\toprule
方法组 & 本轮配置\\\midrule
基线与线性模型 & 常数、OLS / 逻辑回归、Ridge / 强 L2、Elastic Net / L1。\\
树与近邻 & 决策树、随机森林、Extra Trees、直方图梯度提升、K 近邻。\\
搜索流程 & 森林、提升树分别使用网格与 TPE，共增加四条流程。\\\bottomrule
\end{tabularx}
\par\medskip
网格与 TPE 在同一十二点参数域内各评估十二次。TPE 允许重复组合，预算相同也未必覆盖相同候选。
\smallnote{每任务共十三条流程；主搜索完成 1,728 次内层拟合，消融另计。预设森林为 64 棵树。原单次划分使用不同样本和配置，详细结果留在论文。}
'''))
 # Preserve the measured winner means, SDs, and tied search results.
 body=r'\small\begin{tabularx}{\textwidth}{@{}lYl@{}}\toprule'+'\n'+r'任务 & 本次较优流程 & 外层均值 $\pm$ 标准差\\\midrule'+'\n'
 labels={'house':'房价','adult':'收入分类','nba':'NBA 得分','students':'学生数学'}
 methods={'house':'提升树＋网格 / TPE','adult':'提升树＋网格 / TPE','nba':'预设随机森林','students':'Ridge'}
 for ds in labels:
  row=summary.query('dataset==@ds').sort_values('mean',ascending=ds!='adult').iloc[0]
  digits=4 if ds=='adult' else 2 if ds=='house' else 3
  val=f'{row["mean"]:,.{digits}f} $\\pm$ {row["sd"]:,.{digits}f}'
  body+=labels[ds]+' & '+methods[ds]+' & '+val+r'\\'+'\n'
 body+=r'''\bottomrule\end{tabularx}
\par\medskip
房价使用美元 RMSE，NBA 与学生使用分数 RMSE；收入使用 AUC。各任务的较优方法不同。
\smallnote{结果为三折均值；标准差描述折间波动，训练集有重合。十三条流程的排名仍需新样本确认。}
'''
 slides.append(frame('四个任务的较优模型不同',body))
 # One readable comparison per task, followed by tuning and fitting diagnostics.
 for i in [21,22,23,24,25,26,27,28,31,32]:slides.append(frames[i]+'\n\n')
 # Keep the source link and reproduction entry in the main presentation.
 repo=frames[34].replace('\\end{frame}',r'\smallnote{推导、逐项结果与参考文献见课程论文。}'+'\n'+r'\end{frame}')
 slides.append(repo+'\n\n')
 slides.append(frame('结果与应用边界',r'''
模型选择随任务变化：房价与收入较优的是提升树，NBA 是随机森林，学生成绩是 Ridge。调参收益按同折外层结果判断。
\par\medskip
训练与验证曲线检查拟合差距；SHAP 描述既定模型贡献，消融检查删去字段并重新训练后的变化。相关字段可以提供替代信息。
\par\medskip
\small 检索证据限于 16 道固定题，其中 15 道带英文字段名；DeepSeek API 方案质量尚未评价。预测结果为三折交叉验证估计，最终盲测尚未开展；SHAP 与消融均不直接估计因果效应。
'''))
 assert len(slides)==20,len(slides)
 deck=header+'\\begin{document}\n'+''.join(slides)+'\\end{document}\n'
 assert '\\appendix' not in deck
 assert '\\cite' not in deck
 return deck
