"""Twenty-page Madrid talk centred on the completed expanded experiment."""
import json
from expanded_docs import load,best,score,NAMES,CN,search_counts,ROOT,DATA,holdout_data


def render(header,frame):
    summary,_,abl=load()
    def image(name,height='.64'):
        return r'{\centering\includegraphics[width=\linewidth,height='+height+r'\textheight,keepaspectratio]{figures/expanded/'+name+r'.pdf}\par}'+'\n'
    slides=[r'\begin{frame}[plain]\titlepage\end{frame}'+'\n']
    slides.append(frame('研究问题与系统分工',r'''
\begin{itemize}\setlength{\itemsep}{.7em}
\item 中文问题先检索本地字段与方法说明，DeepSeek API 再生成 JSON 分析方案。
\item 用户核对字段和条件后，本机读取原表完成计算与训练。
\item 实验比较模型、参数搜索与信息依赖，另检查字段检索排名。
\end{itemize}
\captionline{批量模型实验不调用语言模型。生成新问答方案需要联网；真实 API 方案质量尚未评价。}
'''))
    slides.append(frame('数据与预测任务',r'''
\small\begin{tabularx}{\linewidth}{@{}lrlY@{}}\toprule
任务 & 记录数 & 目标 & 输入\\\midrule
房价 & 1,460 & 成交价 & 面积、质量、年份、地下室、车位、社区。\\[.5em]
Adult & 32,561 & 收入类别 & 年龄、教育、工时、用工类别、职业、婚姻。\\[.5em]
NBA & 12,844 & 场均得分 & 年龄、身高、体重、选秀年份。\\[.5em]
学生 & 1,000 & 数学成绩 & 四项背景字段，不输入阅读或写作成绩。\\[.5em]
NBA 历史 & 6,228 & 下一季得分 & 三个历史赛季，每季七项信息。\\\bottomrule
\end{tabularx}
\captionline{最后一行为另设任务，样本和信息与原 NBA 任务不同。原始文件不改写，历史窗口的筛选原因逐项保存。}
'''))
    slides.append(frame('内外层验证与信息使用',r'''
\begin{itemize}\setlength{\itemsep}{.65em}
\item 外层三折评价；内层三折选择参数，沿用同一分折规则。
\item Adult 按十四个原始预测字段指纹分组分层；NBA 按完整赛季向前验证。
\item 插补、编码、缩放及回归目标标准化只在当前训练数据拟合。
\item 所有调参只用原训练文件，扩展测试在模型锁定后统一进行。
\end{itemize}
\captionline{房价、学生随机分折。原单次划分、前期十三流程与本轮扩展使用不同配置，数值分开报告。}
'''))
    slides.append(frame('模型与网络结构',r'''
\small
\begin{itemize}\setlength{\itemsep}{.75em}
\item 参照与线性：常数、无惩罚线性、L2、弹性网、线性 SVM。
\item 树模型：决策树、森林、Extra Trees、AdaBoost、直方图提升、XGBoost、LightGBM、CatBoost。
\item 邻域与基函数：K 近邻、ELM、RBF 网络。
\item 端到端网络：MLP、残差 MLP、Deep \& Cross、门控 MLP、字段 Transformer。
\end{itemize}
\captionline{共二十一个表格模型条目。分类采用相应分类器；网络为明确的小型本地实现，不声称复现同名论文的完整训练方案。}
'''))
    slides.append(frame('超参数搜索与消融规则',r'''
\begin{itemize}\setlength{\itemsep}{.65em}
\item 每项可训练模型分别做网格和 TPE，使用同一有限参数域和评价次数。
\item 大部分域为八点；无惩罚线性模型只比较两种截距配置。
\item 端到端网络固定四十轮，搜索宽度、学习率和权重衰减；验证曲线只记录，不选轮数。
\item 所有特征消融重新拟合预处理并网格调参；五项表格网络另做结构控制。
\end{itemize}
\captionline{常数与沿用上季得分是固定参照。TPE 允许重复配置，重复评分复用并保留轨迹；外层成绩不用于反复修改候选域。}
'''))
    body=r'\small\begin{tabularx}{\linewidth}{@{}lYl@{}}\toprule 任务 & 网格流程中较优条目 & 三折均值 $\pm$ 标准差\\\midrule'+'\n'
    for ds in ['house','adult','nba','students']:
        r=best(summary,ds)
        body+=CN[ds]+' & '+NAMES[r.model]+' & '+score(r,True)+r'\\[.7em]'+'\n'
    body+=r'\bottomrule\end{tabularx}\captionline{房价以千美元 RMSE、收入以 AUC、其余以分数 RMSE 评价。标准差描述三折波动；这些是本轮候选域和预算下的排序。}'
    slides.append(frame('四任务模型比较结果',body))
    slides.append(frame('树模型、线性方法与网络',image('deck_benchmark','.68')+r'\captionline{图中取每任务网格前七项及常数参照，论文保留全部二十一项。两种搜索同时显示，误差线为三折标准差，未按外层结果替模型选择搜索方法。}'))
    body=r'\small\begin{tabularx}{\linewidth}{@{}lrrr@{}}\toprule 任务 & TPE 改善 & 相同 & 退化\\\midrule'+'\n'
    for r in search_counts():body+=CN[r['dataset']]+f' & {r["better"]} & {r["same"]} & {r["worse"]}'+r'\\[.7em]'+'\n'
    body+=r'''\bottomrule\end{tabularx}
\par\medskip
内层最优组合的优势未必保持到外层；搜索方法按外层差值评价。
\captionline{改善指 AUC 增加或 RMSE 降低。相同使用数值容差 $10^{-9}$。TPE 重复次数、实际访问组合及每折差值完整保存。}
'''
    slides.append(frame('网格与 TPE 的外层差异',body))
    slides.append(frame('神经网络的拟合诊断',image('deck_neural_training')+r'\captionline{曲线固定展示残差 MLP 与字段 Transformer，不按外层成绩挑选。虚线为训练、实线为验证，阴影为三折标准差；固定训练四十轮，不以外层曲线选择轮数。}'))
    slides.append(frame('特征组消融',image('deck_ablation','.60')+r'''
\small 固定提升树删去面积组后，RMSE 平均增加 7,000.29 美元；SHAP 最高则为整体质量。重新训练检验其余字段能否替代。
\captionline{展示房价、收入的六项预定对照，完整四任务与全部模型见论文。每格三个色条依次为三个外层折；正值表示退化。Only area + quality 仅保留居住面积与整体质量。}
'''))
    slides.append(frame('网络部件的结构控制',image('deck_architecture','.60')+r'''
\small 关闭一个部件后，也重新进行内层网格搜索。
\captionline{依次关闭激活、残差连接、交叉输出、门控和自注意力。每个任务的三个色条为配对外层折，正值表示损失；不同部件的比较限于本地实现。}
'''))
    slides.append(frame('NBA：从历史赛季预测下一赛季',r'''
\[
\hat y_t=f\bigl(X_{t-3},X_{t-2},X_{t-1}\bigr)
\]
\begin{itemize}\setlength{\itemsep}{.65em}
\item 每季输入年龄、身高、体重、得分、篮板、助攻和出场数。
\item RNN、GRU、BiGRU、LSTM、BiLSTM、因果 TCN 和序列 Transformer 保留时间轴。
\item 线性、森林、提升树与 MLP 接收同样的历史信息；沿用上季得分作固定参照。
\end{itemize}
\captionline{按目标赛季向前验证。双向网络只编码预测时已知的历史；姓名等组合键并非已核实的球员编号。}
'''))
    r=best(summary,'nba_sequence')
    slides.append(frame('NBA 历史窗口的比较与消融',image('deck_sequence','.61')+
        r'\small 网格流程中较小平均 RMSE 来自 '+NAMES[r.model]+f'（{r["mean"]:.3f} 分）。\n'+
        r'\captionline{十五项条目共用 6,228 个目标窗口，误差线为三折标准差。十三项拟合模型分别删除字段组或改为两季输入，目标记录保持一致；不能与原四字段 NBA 误差直接比较。}'))
    slides.append(frame('固定提升树的 SHAP 全局解释',image('deck_shap')+r'\captionline{解释模型在实验前固定为网格提升树。覆盖四任务全部外层验证记录；贡献先合并到原字段再取绝对值，误差线为三折标准差。SHAP 与重新训练的消融回答不同问题。}'))
    slides.append(frame('单条预测的贡献分解',image('deck_local')+r'\captionline{房价、收入各选一条绝对误差居中的记录。基准加字段贡献重建原始输出，分类在对数几率尺度分解；行号、实际值、预测值及贡献加和核对随图保存。}'))
    slides.append(frame('表格问答的计算与检索检查',r'''
\begin{columns}[T,onlytextwidth]
\begin{column}{.47\textwidth}
\textbf{社区均价计算}
\par\medskip
CollgCr 社区中，成交价有记录的房屋共 150 条。
\[
\bar y=\frac{29{,}694{,}866}{150}
\]
均价为 197,965.77 美元。
\end{column}
\begin{column}{.49\textwidth}
\textbf{字段检索检查}
\begin{itemize}\setlength{\itemsep}{.6em}
\item 四份表各四题，共十六题。
\item 十五题排第一，一题排第二；MRR 为 0.96875。
\item 十五题直接含英文字段名。
\end{itemize}
\end{column}
\end{columns}
\captionline{均价使用人工固定方案；检索结果不能替代真实 DeepSeek API 的方案生成质量评价。}
'''))
    primary,heldout,_=holdout_data();h=heldout.loc[primary]
    slides.append(frame('Adult 后续扩展测试',image('deck_holdout','.60')+
        r'\small 按 CV 预选的 '+NAMES[primary]+f' 测试 AUC 为 {h.auc:.4f}，95\\% 区间 [{h.auc_ci_low:.4f}, {h.auc_ci_high:.4f}]。\n'+
        r'\captionline{原测试已用于前期十三流程，本轮按后续扩展报告，不称新的盲测。全部二十一项见论文；此处展示预选模型、固定参照和常数。16,255 条测试，排除跨文件重复 26 条；测试后未回调参数。}'))
    checks=json.loads((ROOT/'codes/verification/research/expanded/checks.json').read_text())
    slides.append(frame('开源与复现',r'''
\faGithub\enspace\href{https://github.com/Karcen/TableLab_DeepSeek}{\nolinkurl{github.com/Karcen/TableLab_DeepSeek}}
\begin{enumerate}\setlength{\itemsep}{.7em}
\item 启动一次网页准备环境，再运行 \code{Run-Expanded.command}。
\item 对照 \code{artifacts/expanded} 的协议、分折、参数、模型与逐条预测。
\item 运行核对入口，复算外层指标、SHAP 加和，并记录提交与依赖版本。
\end{enumerate}
'''+f'\\captionline{{本轮 {checks["jobs"]:,} 项流程、外层折与消融组合；{checks["validation_metrics_reconstructed"]:,} 项外层指标复算通过。批量实验不调用 API；绘图用指定 Nature Skills，文档用 XeLaTeX。}}'))
    slides.append(frame('优点与不足',r'''
\begin{columns}[T,onlytextwidth]
\begin{column}{.46\textwidth}
\textbf{优点（Pros）}
\begin{itemize}\setlength{\itemsep}{.65em}
\item 多类模型共用外层样本与输入。
\item 调参与预处理限定在训练折内。
\item 所有可训练模型重新调参做消融。
\item 保存参数、模型和逐条预测供复算。
\end{itemize}
\end{column}
\begin{column}{.49\textwidth}
\textbf{不足（Cons）}
\begin{itemize}\setlength{\itemsep}{.45em}
\item Adult 为后续扩展测试，其余任务缺新独立样本。
\item 网络仅用小型结构、四十轮和有限搜索域。
\item NBA 缺少已核实的球员编号。
\item 检索仅十六题；真实 API 方案质量未评价。
\item SHAP 与消融不直接估计因果效应。
\end{itemize}
\end{column}
\end{columns}
'''))
    assert len(slides)==20
    deck=header+'\\begin{document}\n'+''.join(slides)+'\\end{document}\n'
    assert '边界' not in deck
    return deck
