"""Verified expanded results for the paper, Madrid deck and one Chinese report."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'
SRC=ROOT/'docs/source'
DS=['house','adult','nba','students']
CN={'house':'房价','adult':'收入分类','nba':'NBA 得分','students':'学生数学','nba_sequence':'NBA 下一赛季'}
NAMES={'Mean':'常数基线','Persistence':'沿用上季得分','Linear':'无惩罚线性 / Logistic','Ridge':'L2 线性 / Logistic','Sparse':'Elastic Net / Logistic','Tree':'决策树','RF':'随机森林','ExtraTrees':'Extra Trees','HGB':'直方图梯度提升','KNN':'K 近邻','AdaBoost':'AdaBoost','SVM':'线性 SVM','XGB':'XGBoost','LightGBM':'LightGBM','CatBoost':'CatBoost','MLP':'MLP','ResMLP':'残差 MLP','DCN':r'Deep \& Cross','GatedMLP':'门控 MLP','FeatureTransformer':'字段 Transformer','ELM':'ELM','RBFNet':'RBF 网络','RNN':'RNN','GRU':'GRU','BiGRU':'BiGRU','LSTM':'LSTM','BiLSTM':'BiLSTM','TCN':'因果 TCN','SeqTransformer':'序列 Transformer'}
NN=['MLP','ResMLP','DCN','GatedMLP','FeatureTransformer','ELM','RBFNet']
REC=['RNN','GRU','BiGRU','LSTM','BiLSTM','TCN','SeqTransformer']
ARCH={'MLP':'关闭激活','ResMLP':'关闭残差连接','DCN':'交叉分支输出置零','GatedMLP':'门控置为 1','FeatureTransformer':'跳过自注意力'}


def ready():
    p=ROOT/'codes/verification/research/expanded/checks.json'
    files=[p]+[ROOT/f'codes/verification/research/expanded/{name}_checks.json' for name in ['shap','holdout_extension','diagnostic','optimization']]
    return all(f.exists() and json.loads(f.read_text()).get('status')=='passed' for f in files)


def load():
    assert ready()
    return pd.read_csv(DATA/'summary.csv'),pd.read_csv(DATA/'metrics.csv'),pd.read_csv(DATA/'ablation_deltas.csv')


def best(summary,ds,models=None):
    q=summary.query('dataset == @ds and variant == "Full" and method in ["grid","fixed"]')
    if models is not None:q=q.loc[q.model.isin(models)]
    return q.sort_values('mean',ascending=ds!='adult',kind='stable').iloc[0]


def score(row,latex=False):
    ds=row.dataset;digits=4 if ds=='adult' else 2 if ds=='house' else 3
    scale=1000 if latex and ds=='house' else 1
    plus=r' $\pm$ ' if latex else ' ± '
    return f'{row["mean"]/scale:,.{digits}f}'+plus+f'{row.sd/scale:,.{digits}f}'


def findings(summary):
    sentences=[]
    for ds in DS:
        row=best(summary,ds)
        unit='' if ds=='adult' else ' 美元' if ds=='house' else ' 分'
        metric='AUC' if ds=='adult' else 'RMSE'
        digits=4 if ds=='adult' else 2 if ds=='house' else 3
        sentences.append(f'{CN[ds]}中，{NAMES[row.model]}的网格流程取得较优三折均值，{metric} 为 {row["mean"]:,.{digits}f}{unit}')
    return '；'.join(sentences)+'。'


def fig(name,caption,label):
    return '\\begin{figure}[!htbp]\\centering\n\\includegraphics[width=\\linewidth]{figures/expanded/'+name+'.pdf}\n\\caption{'+caption+'}\\label{'+label+'}\n\\end{figure}\n'


def search_counts():
    q=pd.read_csv(DATA/'search_differences.csv')
    output=[]
    for ds in DS+['nba_sequence']:
        v=q.query('dataset == @ds').tpe_improvement
        output.append({'dataset':ds,'better':int((v>1e-9).sum()),'same':int((abs(v)<=1e-9).sum()),'worse':int((v < -1e-9).sum())})
    return output


def results():
    summary,metrics,abl=load()
    body=r'\subsection{扩展模型的共同验证}\label{sec:expanded-results}'+'\n'
    body+='本轮在四个原任务中比较二十一个模型条目，其中二十项分别完成网格与 TPE 搜索，常数基线不调参。七项使用神经网络或基函数网络。以下以网格流程比较模型，TPE 单独作为搜索方法比较；两者均保留，不按外层成绩为每个模型挑选更好的搜索方法。'+findings(summary)+'这些排序描述当前数据、字段、候选域与训练预算。\n\n'
    body+=r'\inlinetab{\captionof{table}{扩展实验：网格流程的三折外层均值及标准差}\label{tab:expanded-models}'+'\n'
    body+=r'\begin{tabularx}{\linewidth}{@{}Yrrrr@{}}\toprule'+'\n'+r'模型条目 & 房价 RMSE & 收入 AUC & NBA RMSE & 成绩 RMSE\\\midrule'+'\n'
    order=summary.query('dataset == "house" and variant == "Full" and method in ["grid","fixed"]').model.tolist()
    for model in order:
        values=[]
        for ds in DS:
            row=summary.query('dataset == @ds and model == @model and variant == "Full" and method in ["grid","fixed"]').iloc[0]
            values.append(score(row,True))
        body+=NAMES[model]+' & '+' & '.join(values)+r'\\'+'\n'
    body+=r'\bottomrule\end{tabularx}\tabnote{房价单位为千美元，其余回归单位为分。收入为 AUC，越高越好；RMSE 越低越好。标准差描述三个外层折，训练样本重合，不是独立重复试验的置信区间。Linear 在分类中为无惩罚 Logistic，Ridge 对应 L2 Logistic，稀疏项对应弹性网 Logistic。其余分类、回归实现及参数域见方法部分。}}'+'\n'
    body+=fig('benchmark','扩展模型在四个任务中的外层表现。圆点为网格，方块为 TPE，灰色菱形为不调参的常数参照；横线为三折标准差。每项同时展示两种搜索，以网格均值排序。全部记录及模型均保留，未按测试成绩筛选模型。','fig:expanded-benchmark')
    body+=r'\subsection{网格与 TPE 的外层比较}'+'\n'
    counts=search_counts()
    for r in counts[:4]:
        body+=f'{CN[r["dataset"]]}的二十项调参比较中，TPE 的外层平均指标相对网格改善 {r["better"]} 项、相同 {r["same"]} 项、退化 {r["worse"]} 项。'
    body+='每项搜索最多八次评价，无惩罚线性模型只有两种截距配置。网格遍历本轮候选域，TPE 可重复访问已有配置；重复评分复用，但搜索轨迹保留。内层最优值与外层效果需分别阅读，外层某一次改善不说明搜索方法普遍更好。逐折差值、实际访问配置数与耗时随结果保存。\n\n'
    body+=r'\subsection{神经网络拟合与消融}'+'\n'
    diagnostic=pd.read_csv(DATA/'diagnostics.csv')
    def error(ds,depth,part):
        key='log_loss' if ds=='adult' else 'rmse'
        return diagnostic.query('dataset == @ds and kind == "complexity" and value == @depth and partition == @part')[key].mean()
    body+=f'房价复杂度控制中，深度 1 的训练、验证 RMSE 分别为 {error("house",1,"train"):,.2f}、{error("house",1,"validation"):,.2f} 美元；深度 4 为 {error("house",4,"train"):,.2f}、{error("house",4,"validation"):,.2f} 美元。'
    if error('house',4,'train')<error('house',1,'train') and error('house',4,'validation')<error('house',1,'validation'):
        body+='训练和验证误差同时下降，与浅树拟合不足的解释一致。'
    body+=f'收入分类中，不限深度树的训练对数损失为 {error("adult",-1,"train"):.3f}，验证为 {error("adult",-1,"validation"):.3f}。训练误差低而验证误差高，提示泛化差距；若两者都高，还需分辨输入信息不足、容量限制与训练未充分。\n'
    body+=fig('fit_diagnostics','样本量与复杂度控制。每行一个任务，左列使用网格提升树，固定完整外层训练集内选出的参数，增加训练样本比例；每个子样本重新拟合预处理。Adult 以指纹组抽取，NBA 取最早完整赛季，因此其比例是赛季近似值。右列改变决策树深度，叶节点最少两条，Full 表示不限制深度。线与阴影为三折均值及标准差，验证记录保持不变。这些控制用于诊断，不增加排行榜条目。','fig:expanded-fit')
    grouped=abl.loc[abl.variant.isin(['No activations','No residuals','No cross branch','No gates','No attention'])].groupby(['dataset','model','variant']).loss.mean()
    same_direction=abl.loc[abl.variant.isin(['No activations','No residuals','No cross branch','No gates','No attention'])].groupby(['dataset','model','variant']).loss.apply(lambda x:bool((x>0).all()))
    body+=f'端到端网络固定训练四十轮。曲线展示残差 MLP 与字段 Transformer 的训练和验证误差，展示对象不按外层成绩挑选；其余端到端网络曲线完整保存。二十项任务与结构控制中，关闭部件后的平均性能损失有 {int((grouped>0).sum())} 项为正，其中 {int(same_direction.sum())} 项在三个外层折均为正。部件收益随任务与配置变化，这些计数只作描述。\n'
    body+=fig('neural_training','固定四十轮预算下的拟合诊断。颜色区分残差 MLP 与字段 Transformer，虚线为训练、实线为外层验证，阴影为三折标准差。展示模型固定为残差 MLP 与字段 Transformer，不按外层成绩挑选；曲线仅记录，未用于选择轮数或早停。回归损失转换回原目标尺度，分类为对数损失。','fig:expanded-training')
    body+=fig('ablation','全部二十项可训练模型的特征组消融。每个模型与字段组对应三个并排色条，依次为三个外层折，显示删减后相对同折完整模型的指标损失百分比；正值为退化，负值为改善。各面板使用各自的线性色标。删减后重新拟合预处理并网格调参；Only area + quality 为仅保留地上居住面积与整体质量的配置。所有逐折差值保留，未只展示有利结果。','fig:expanded-ablation')
    body+=fig('architecture','五项神经网络结构控制在四任务中的配对变化。每个任务含三个并排色条，对应三个外层折；指标损失定义同图~\\ref{fig:expanded-ablation}。控制分别关闭激活、残差相加、自注意力，或将交叉分支输出置零、门控置为一；其余训练规则一致，每项控制重新网格调参。该比较检验本地实现中的部件，不能解释为对同名论文完整模型的复现。','fig:expanded-architecture')
    body+=r'\subsection{NBA 历史赛季预测}\label{sec:nba-sequence}'+'\n'
    audit=json.loads((DATA/'nba_sequence/sequence_audit.json').read_text())
    winner=best(summary,'nba_sequence');recurrent=best(summary,'nba_sequence',REC)
    persistence=summary.query('dataset == "nba_sequence" and model == "Persistence" and variant == "Full"').iloc[0]
    body+=f'另将 NBA 定义为根据连续三个历史赛季预测下一赛季得分，得到 {audit["eligible_windows"]:,} 个窗口。十五项模型条目使用相同历史信息，其中十三项调参；均值与沿用上季得分作为固定参照。网格流程中，{NAMES[winner.model]}的平均 RMSE 为 {winner["mean"]:.3f} 分；七项循环、卷积及序列注意力网络中，{NAMES[recurrent.model]}为 {recurrent["mean"]:.3f} 分。沿用上季得分的平均 RMSE 为 {persistence["mean"]:.3f} 分。该任务与原 NBA 四字段任务的样本和信息不同，不能把误差之差归因于网络结构。\n\n'
    body+=r'\inlinetab{\captionof{table}{相同三赛季历史窗口上的 NBA 下一赛季预测}\label{tab:expanded-sequence}\begin{tabularx}{\linewidth}{@{}Yrr@{}}\toprule 模型 & 网格或固定参照 RMSE & TPE RMSE\\\midrule'+'\n'
    for model in summary.query('dataset == "nba_sequence" and variant == "Full"').model.unique():
        q=summary.query('dataset == "nba_sequence" and model == @model and variant == "Full"')
        g=q.query('method in ["grid","fixed"]').iloc[0]
        t=q.query('method == "tpe"')
        body+=NAMES[model]+' & '+score(g,True)+' & '+(score(t.iloc[0],True) if len(t) else '---')+r'\\'+'\n'
    body+=r'\bottomrule\end{tabularx}\tabnote{单位为分，误差项为三个向前验证折的标准差。验证块内按一年一步的滚动预测解释，各时点可使用当时已经观测的上一赛季；模型在整个验证块内不重新拟合。}}'+'\n'
    body+=fig('sequence','NBA 历史窗口任务的性能与信息依赖。a，十五项条目的网格、TPE 或固定参照结果，误差线为三折标准差。b，十三项可训练条目分别删去年龄、身体属性、历史产出、出场次数，或只保留最近两个赛季；每格三个色条保留同折损失。两赛季控制使用相同目标记录，未另换较容易的样本。','fig:expanded-sequence')
    body+=r'\subsection{固定提升树的 SHAP 解释}'+'\n'
    body+='解释模型在实验前固定为网格调参的直方图梯度提升，不随排行榜改变。对四任务全部外层验证记录解释，并将回归贡献换回原目标单位；类别编码的贡献先按原字段有符号相加。SHAP 描述既定模型的贡献，删去字段后重新调参检验其余信息是否能替代，二者不必给出相同排序。\n'
    importance=pd.read_csv(DATA/'shap_importance.csv')
    top=importance.query('dataset == "house"').groupby('field').mean_absolute_shap.mean().idxmax()
    area=abl.query('dataset == "house" and model == "HGB" and variant == "No size"').loss
    assert len(area)==3 and np.isfinite(area).all()
    body+=f'房价中，固定提升树的平均绝对 SHAP 以 {top} 最高；删去面积组并重新调参后，RMSE 平均增加 {area.mean():,.2f} 美元，三折标准差为 {area.std(ddof=1):,.2f} 美元。单个字段在既定模型中的贡献，与一组字段删除后其余输入的替代能力，是两种不同的比较。\n'
    body+=fig('shap','固定提升树对四任务全部外层验证记录的平均绝对 SHAP。柱与误差线分别为三折均值和标准差；分类在原始对数几率尺度，回归在目标单位，房价以千美元表示。没有只解释预测较准的记录，贡献加和逐条核对。','fig:expanded-shap')
    body+=fig('local','房价与收入的单条 SHAP 分解。各任务在全部外层验证记录中按绝对预测误差排序，取中间记录；字段按该记录贡献的绝对值排序。灰色为基准，蓝、橙为负、正贡献，青色为最终原始输出。分类输出为对数几率，须经 sigmoid 才得到概率。行号、真实值与预测值随图保存。','fig:expanded-local')
    body+=holdout_results()
    return body


def holdout_data():
    folder=DATA/'holdout_extension'
    primary=json.loads((folder/'selection.json').read_text())['chosen_before_test_prediction']
    return primary,pd.read_csv(folder/'metrics.csv').set_index('model'),pd.read_csv(folder/'paired_differences.csv').set_index('comparator')


def holdout_results():
    primary,scores,pairs=holdout_data();r=scores.loc[primary]
    anchor='HGB' if primary!='HGB' else 'XGB';d=pairs.loc[anchor]
    body=r'\subsection{Adult 后续扩展测试}\label{sec:adult-holdout}'+'\n'
    body+='前期十三流程已使用原测试文件评价，网格提升树 AUC 为 0.8922；其结果和冻结协议原样保留。本轮另对二十项网格模型及常数参照统一评价，模型与参数均先在原训练文件内确定。这是已经使用过该数据之后的后续扩展，不称新的盲测。\n\n'
    body+=f'按本轮嵌套验证预选的模型为{NAMES[primary]}，在 16,255 条记录上的 AUC 为 {r.auc:.4f}，95\\% 区间为 [{r.auc_ci_low:.4f}, {r.auc_ci_high:.4f}]。与{NAMES[anchor]}相比，AUC 差为 {d.auc_difference:+.5f}，配对区间为 [{d.ci_low:+.5f}, {d.ci_high:+.5f}]。'
    body+='该区间包含零，现有记录未分清这两项的优劣。' if d.ci_low<=0<=d.ci_high else '该区间在本次重抽样中未跨零；它是未作多重比较校正、给定最终模型的描述性区间。'
    body+='测试成绩没有用于回选参数、改阈值或更换预选模型。CV 与补充测试的训练规模、评价记录不同，两列之差不作为调参收益。\n\n'
    body+=r'\inlinetab{\captionof{table}{Adult 后续扩展测试：按训练数据中的 CV 均值排序}\label{tab:adult-holdout}\begin{tabularx}{\linewidth}{@{}Yrrrr@{}}\toprule 模型 & CV AUC & 测试 AUC & 测试 AUC 95\% 区间 & 测试 F1\\\midrule'+'\n'
    for model,row in scores.sort_values('cv_auc',ascending=False,kind='stable').iterrows():
        body+=f'{NAMES[model]} & {row.cv_auc:.4f} & {row.auc:.4f} & [{row.auc_ci_low:.4f}, {row.auc_ci_high:.4f}] & {row.f1:.4f}'+r'\\'+'\n'
    body+=r'\bottomrule\end{tabularx}\tabnote{CV 列为本轮网格流程的三折外层均值；最终模型在全部 32,561 条原训练记录中再做三折内层网格选择，然后重拟合。测试排除 26 条跨文件重复，保留内部重复；区间来自 16,249 个指纹组的 2,000 次配对 bootstrap。区间不含训练、搜索和模型选择变异，且未作多重比较校正。阈值为 0.5。}}'+'\n'
    body+=fig('holdout_extension',r'Adult 后续扩展测试。a，全部二十一个条目按训练数据中的 CV 均值排序，青色菱形为测试前预选模型，误差线为 95\% 区间。b，预选模型减去六项事先指定的提升树和网络参照；若参照与预选模型相同则略去自身。两面板均按指纹组配对重抽样，正差表示预选模型较高。原测试文件已用于前期研究，图中区间不能消除此前查看和模型设计选择的影响。','fig:expanded-holdout')
    return body


def methods():
    body=r'''\subsection{扩展比较与搜索协议}\label{sec:new-protocol}
四个原任务保持既定字段和合格记录，外层三折、内层三折的行号与前期嵌套验证一致。房价和学生随机分折；Adult 仅用原训练文件，按全部十四个非目标原始字段指纹分组并分层；NBA 按完整赛季向前分折，内层沿用同一规则。插补、标准化和独热编码只在当前训练折拟合。为使线性、核和神经网络的回归训练有可比的数值尺度，所有回归模型均用当前训练目标的均值和标准差缩放目标，预测再反变换。该协议及候选域与前期十三流程不同，不合并报告其均值。

二十一个表格模型条目包括常数、无惩罚线性、L2、弹性网、决策树、随机森林、Extra Trees、直方图梯度提升、近邻、AdaBoost、线性 SVM、XGBoost、LightGBM、CatBoost，以及 MLP、残差 MLP、Deep \& Cross、门控 MLP、字段 Transformer、ELM 和 RBF 网络。XGBoost、LightGBM 与 CatBoost 分别依据其官方实现\citep{xgboost2016,lightgbm2017,catboost2018}。为控制编码差异，均接收同一训练折的独热编码输入；本轮没有检验原生类别处理的优势。分类的无惩罚项使用 Logistic，L2 和弹性网使用相应惩罚的 Logistic；线性 SVM 在当前训练数据内分三折分组做 sigmoid 校准，外层验证不参与校准。

每项可训练模型分别完成网格和 TPE 搜索。大部分模型的域为八点，无惩罚线性模型仅比较是否保留截距的两点。两种搜索均评估完整域大小的次数；TPE 前三次为启动采样，两点域仅作两次启动采样，不进入自适应阶段，种子为 42 加外层折号的 100 倍。相同候选使用相同内层训练种子，重复配置在本次搜索内复用分数，并保留重复轨迹。主比较采用网格流程，TPE 的外层效果另行报告，不按外层结果替每个模型选择搜索方法。分类按内层平均 AUC、回归按平均 RMSE 选参数；阈值固定为 0.5。

\inlinetab{\captionof{table}{扩展实验的参数域与固定训练量}\label{tab:expanded-spaces}
\begin{tabularx}{\linewidth}{@{}lY@{}}\toprule
模型或组 & 参数域与固定配置\\\midrule
无惩罚线性 & 截距：有或无；分类使用无惩罚 Logistic。\\
L2 & $\alpha\in\{.001,.01,.1,1,10,100,1000,10000\}$；分类取 $C=1/\alpha$。\\
弹性网 & $\alpha\in\{.01,.1,1,10\}$；L1 比例 $.5$ 或 $1$，分类取 $C=1/\alpha$。\\
决策树 & 深度 $2,4,8$ 或不限；叶节点最少 $2$ 或 $10$ 条。\\
森林、Extra Trees & 深度 $6/12$；叶节点最少 $3/10$；候选特征比例 $.7/1$；64 棵树。\\
直方图提升 & 叶节点上限 $7/31$；学习率 $.05/.1$；L2 为 $0/1$；100 次迭代，关闭早停。\\
K 近邻 & 邻居数 $3,7,15,31$；等权或距离权重。\\
AdaBoost & 基树深度 $1/3$；学习率 $.03/.1$；50/100 棵树。\\
线性 SVM & $C\in\{.001,.01,.1,.3,1,3,10,100\}$；回归 $\epsilon=.1$。\\
XGBoost & 深度 $3/6$；学习率 $.03/.1$；L2 为 $1/10$；120 棵树。\\
LightGBM & 叶节点上限 $7/31$；学习率 $.03/.1$；叶节点最少 $10/30$；120 棵树。\\
CatBoost & 深度 $4/6$；学习率 $.03/.1$；L2 为 $1/10$；120 次迭代。\\
端到端网络 & 宽度 $32/64$，学习率 $.003/.01$，权重衰减 $.0001/.01$；两类 Transformer 宽度为 $16/32$。\\
ELM & 隐层 $32/128$，tanh/ReLU，输出层惩罚 $.1/10$。\\
RBF 网络 & 中心 $16/64$，宽度参数 $.1/1$，输出层惩罚 $.1/10$。\\\bottomrule
\end{tabularx}\tabnote{除常数和沿用上季得分外均调参。候选域有限，不代表各算法在更大预算下的最佳性能。}} 

\subsection{网络训练与结构控制}
端到端网络使用本地明确实现，固定四十轮、每批 2,048 条、AdamW、梯度范数上限 5，dropout 模块的概率为 0.1。残差 MLP 含两个残差块；门控 MLP 含两个带 sigmoid 门的残差块；Deep \& Cross 包含三层显式交叉与两层深层分支\citep{dcn2017}。字段 Transformer 将同一原始字段的编码投影为一个 token，使用单层、两头自注意力与均值汇总\citep{transformer2017}。这些是小规模本地实现，不声称复现 FT-Transformer、TabNet 或同名论文的完整实验。ELM 使用固定种子的随机隐层，RBF 网络在全部当前训练样本上学习中心，再拟合带惩罚的输出层。

固定轮数使外层验证不参与早停。训练与验证曲线仅在第 1、5、10、20、30、40 轮记录；不按曲线重新选择轮数。结构控制分别关闭 MLP 激活、残差相加、自注意力，或将交叉分支输出置零、门控置为一。每项控制重新进行内层网格搜索。特征组消融也为每个可训练模型重新拟合预处理和网格搜索，不沿用完整模型的参数。

另以固定提升树参数比较四种训练样本比例，并以叶节点最少两条的决策树比较深度 1、2、4、8 和不限深度；合计 108 次辅助拟合。每次保存训练与完整外层验证预测，再独立复算指标。学习曲线的参数取自完整外层训练集的内层搜索，较小子样本不再调参，因此该控制只检查固定容量下的数据量变化。

原四任务的消融组与前期定义一致：房价面积组为 \code{GrLivArea} 与 \code{TotalBsmtSF}，另设仅保留 \code{GrLivArea} 与 \code{OverallQual} 的两字段配置；Adult 工作组为工时、用工类别和职业；NBA 身体属性组为身高与体重；其余变体逐项删去指定字段。完整模型与变体使用同一外层样本，按同折指标差评价。

\subsection{NBA 历史窗口的构造}
从原 NBA 表构造三个连续历史赛季预测下一赛季的样本。按姓名、国家和选秀年份组成的哈希键连接记录，排除该键与赛季不唯一的记录；要求相邻年龄增量在 0 至 2 岁之间、身高变化不超过 5 厘米。该键不是已核实的球员编号。原 12,844 条记录中，2 条键与赛季重复，6,512 条缺少完整连续历史，102 条未通过衔接检查，保留 6,228 个目标窗口；原始文件不改写。连接与排除不依据得分大小或预测成绩。

每个历史赛季输入年龄、身高、体重、场均得分、篮板、助攻和出场数，共二十一列；不输入目标赛季技术统计。所有模型使用同一窗口，平面模型将三个赛季展平，序列模型保留时间轴。比较 RNN、GRU、BiGRU、LSTM、BiLSTM、双层因果卷积与单层序列 Transformer\citep{torchlstm,tcn2018,transformer2017}，以及七项共享基线和固定沿用上季得分参照。循环网络为单层，双向编码仅覆盖预测时已经知道的历史窗口。因果卷积核宽为 2，膨胀率为 1、2；序列 Transformer 使用可学习历史位置。

外层与内层均按目标赛季向前分折。验证块按每次前进一步的下一季预测解释，因此后来时点可以使用该时点以前已观测的赛季；模型在整个验证块中保持固定。额外删除年龄、身体属性、历史产出、出场数，或改为只保留最近两个赛季，且所有变体保持同一目标样本。该任务沿用已分析的 NBA 数据，不能称独立新队列测试。

\subsection{解释、精度与复算}
SHAP 事先固定解释网格提升树，使用训练树路径计数为背景，覆盖四任务全部外层验证记录。回归先将基准与贡献反变换回目标尺度，分类解释原始对数几率；类别编码先有符号合并，再计算原始字段绝对贡献。逐条检查贡献加和与保存预测一致。结构消融、特征消融及 SHAP 均描述预测依赖，不直接识别因果效应。

部分估计器原生输出为单精度。模型训练、搜索分数与参数选择保持不变，导出时将预测转换为双精度后再反变换，统一用双精度复算外层指标；原精度记录另行归档，模型哈希保持一致。该步骤只修正数值舍入，没有重训或按评价成绩调整参数。独立脚本按稳定行号复核每项模型及消融的目标值、预测指标、内层分组、时间顺序、训练预处理和文件哈希。三折标准差同时受样本与固定训练种子影响，没有独立估计初始化变异。

前期 Adult 预留测试的冻结协议与结果单独保留；新增模型不以该文件选择参数，后续扩展评价另按冻结协议统一测试。完整扩展入口为 \code{Run-Expanded.command}，结果、参数、模型、逐条预测及协议位于 \code{artifacts/expanded}。需要先准备项目 Python 环境；本轮使用 Python 3.13、CPU 多进程、每个模型单线程。依赖版本见 \code{codes/requirements-expanded.txt} 与冻结协议。批量实验不调用 DeepSeek API。
'''
    body+=r'''\subsection{Adult 后续扩展测试的冻结与复算}
本轮额外测试二十项网格流程和常数参照。每项模型在全部 32,561 条原训练记录内，重新进行同一参数域的三折分层分组网格搜索；不使用测试标签选择参数。所有最终模型锁定后，根据此前嵌套验证的网格 AUC 预选模型，精确相同时按模型名排序，再统一计算测试预测。TPE 的搜索方法比较仍只按外层 CV 报告。

测试文件与前期完全相同：排除与训练文件十四个预测字段指纹重合的 26 条，保留 16,255 条、16,249 个指纹组。主指标为 AUC，另报 AP、Brier、对数损失和固定 0.5 阈值 F1。所有模型共用 2,000 次指纹组 bootstrap，种子为 20261102，报告 95\% 百分位区间和预选模型相对其他条目的配对差值区间，不作多重比较校正。原测试已用于前期评价，设计模型时也已知道前期表现，因此将本轮测试明确称为后续扩展；区间只反映给定模型的测试样本变化，不覆盖这种设计选择及训练变异。原有文件不覆盖，新结果单独保存于 \code{artifacts/expanded/holdout_extension}。

'''
    from adult_holdout_docs import methods as prior_methods
    body+=prior_methods().replace(r'\subsection{Adult 预留测试协议}',r'\subsection{前期 Adult 十三流程的预留测试协议}')
    body+=r'''上述预留测试的十三流程属于前期协议：普通、强 L2 与 L1 Logistic 的 $C$ 分别为 1、0.1、0.1；完整树不限制深度；森林和 Extra Trees 为 64 棵、深度 12、叶节点最少 5 条；预设提升树为 100 次迭代，近邻数为 15。森林的十二点域为深度 $4,8$ 或不限、叶节点最少 $3/10$、特征比例 $.7/1$；提升树十二点域为叶节点上限 $7,15,31$、L2 为 $0/1$、学习率 $.05/.1$。这些参数与本轮扩展的八点域分别保存，未混作同一实验。
'''
    return body


def report():
    summary,metrics,abl=load()
    checks=json.loads((ROOT/'codes/verification/research/expanded/checks.json').read_text())
    shap=json.loads((DATA/'shap_audit.json').read_text())
    out='# TableLab 扩展模型实验\n\n## 研究问题\n\n在相同字段和样本划分下，树模型、线性方法与神经网络的预测表现怎样？网格与 TPE 搜索有多大差别？删去字段或关闭网络部件后，结果是否保持？NBA 加入历史赛季后，序列网络是否优于使用同样历史信息的平面模型？\n\n## 结果\n\n'
    out+='四个原任务各比较 **21 个模型条目**，其中 20 项分别做网格和 TPE；NBA 历史任务比较 15 项，其中 13 项调参。常数与沿用上季得分为固定参照，不设置无意义的超参数。\n\n'
    out+='| 任务 | 网格流程中较优条目 | 三折均值 ± 标准差 |\n| --- | --- | --- |\n'
    for ds in DS+['nba_sequence']:
        r=best(summary,ds)
        unit='AUC ' if ds=='adult' else 'RMSE '
        suffix='' if ds=='adult' else ' 美元' if ds=='house' else ' 分'
        out+=f'| {CN[ds]} | {NAMES[r.model].replace(chr(92),"")} | {unit}{score(r)}{suffix} |\n'
    out+='\nNBA 历史任务保留 6,228 个连续三赛季窗口，与原四字段任务的信息和样本不同，两个任务的 RMSE 不能直接用来说明网络收益。\n\n'
    out+='| 任务 | TPE 相对网格改善 / 相同 / 退化的条目数 |\n| --- | --- |\n'
    for r in search_counts():out+=f'| {CN[r["dataset"]]} | {r["better"]} / {r["same"]} / {r["worse"]} |\n'
    out+=f'\n共完成 {checks["jobs"]:,} 项流程、外层折与消融组合。所有可训练条目均完成特征组消融，并在每个变体内重新网格调参；五种端到端表格网络另做结构控制。SHAP 固定解释网格提升树，覆盖 {shap["total_explained"]:,} 条外层验证记录，贡献加和最大误差为 {shap["max_additivity_error"]:.2g}。\n\n'
    out+='## 意义\n\n房价的 RBF 网络、字段 Transformer 和 XGBoost，以及学生任务的线性 SVM、RBF 网络和 L2 模型，其均值差异相对折间变化较小，暂不能确定稳定优劣。增加模型使部分三折均值排序发生变化，结论仍受候选域和固定训练预算影响。\n\n'
    out+='训练与验证误差共同用于检查过拟合或拟合不足；网络固定四十轮，不能把本轮较差结果解释为该架构在充分训练下也一定较差。字段消融检查其他输入能否替代被删信息，结构控制检查本地实现的部件贡献，SHAP 解释既定提升树，三者不直接估计因果效应。\n\n'
    primary,hs,hp=holdout_data();h=hs.loc[primary]
    out+=f'Adult 另对二十一个网格或固定条目做后续扩展测试。按 CV 预选的 {NAMES[primary]} 测试 AUC 为 {h.auc:.4f}，95% 区间为 [{h.auc_ci_low:.4f}, {h.auc_ci_high:.4f}]。前期十三流程的网格提升树 AUC 0.8922 及原记录保留。测试文件已用于前期研究，这次属于后续扩展，不是新的盲测；测试成绩没有用于回选参数或阈值。房价、NBA 和学生仍无独立新样本。NBA 缺少已核实的球员编号，连续历史筛选也排除了新秀及缺少连续记录的球员；学生真实采样过程未确认。问答检索仍仅有十六道固定题，真实 API 方案质量未评价。\n\n'
    out+='CatBoost 相对 XGBoost 和直方图梯度提升的 AUC 配对差值区间均包含零，现有测试未能确定三者的优劣。\n\n'
    out+='## 方法\n\n外层三折评价，内层三折选择参数。四个原任务沿用原外层行号；Adult 按十四个原始预测字段指纹分组分层，NBA 按完整赛季向前验证，内层遵循同一规则。插补、编码、缩放与回归目标标准化均只在当前训练数据拟合。\n\n'
    out+='每项大多搜索八种配置，无惩罚线性项为是否带截距的两种配置；网格与 TPE 使用同一域及评价次数，TPE 的重复配置评分复用并记录。主比较使用网格，TPE 单列比较。端到端网络使用 AdamW、四十轮、每批 2,048 条，搜索宽度、学习率和权重衰减；所有外层验证曲线只记录，不用于轮数选择或早停。\n\n'
    out+='NBA 将姓名、国家、选秀年份组成的哈希键作为保守连接依据，排除 2 条键与赛季重复、6,512 条缺少三段连续历史、102 条年龄身高衔接异常的记录。输入仅含三个历史赛季的年龄、身高、体重、得分、篮板、助攻和出场数；两赛季消融仍保留相同目标记录。BiLSTM、BiGRU 只在已知历史内双向编码。\n\n'
    out+='Adult 扩展测试先在全部原训练记录内重新网格选择参数，锁定所有最终模型，再按 CV 均值预选模型并统一测试。按与前期相同规则排除 26 条跨文件重复，保留测试内部重复组；用 2,000 次指纹组配对 bootstrap 给出区间。区间未作多重比较校正，也不包含训练、搜索或此前查看测试表现造成的选择影响。\n\n'
    out+=f'独立复算 {checks["validation_metrics_reconstructed"]:,} 项外层指标，检查 15 个外层、45 个内层分折；最大指标差为 {checks["max_absolute_metric_difference"]:.2g}。实际内层候选拟合调用 {checks["actual_inner_candidate_fits"]:,} 次，最终拟合 {checks["final_fitted_models"]:,} 个模型；SVM 内部校准与基函数网络内部拟合另有子步骤。原始数据与冻结训练代码的哈希一致。预测统一双精度导出，原精度记录保留，未改训练参数。\n\n'
    out+='补充测试的 105 项指标、147 项重抽样校验及拟合诊断的 756 项指标另行复核。检查 260 个被选中求解器组件，未发现触及迭代上限；固定四十轮的神经网络不据此声称已充分收敛。\n\n'
    out+='准备网页环境后运行 `Run-Expanded.command` 可复现扩展训练、复算、绘图与编译，已完成的实验会跳过。结果保存于 `artifacts/expanded`。图形使用指定 Nature Skills 的 Python 流程，展示稿使用 Madrid；批量实验不调用 API。\n\n[课程论文](paper.pdf) · [Madrid 展示稿](presentation.pdf) · [项目仓库](https://github.com/Karcen/TableLab_DeepSeek)\n'
    return out


def apply(paper):
    summary,_,_=load()
    (SRC/'expanded_results.tex').write_text(results())
    (SRC/'expanded_methods.tex').write_text(methods())
    paper=paper.replace(r'\input{research_results}',r'\input{expanded_results}').replace(r'\input{research_methods}',r'\input{expanded_methods}')
    primary,hs,_=holdout_data()
    abstract='TableLab 在本地检索字段说明，通过 DeepSeek API 生成分析方案，再由本机完成统计和模型训练。字段检索仅检查 16 道固定题，其中 15 道含英文字段名，真实 API 方案质量尚未评价。本文在房价、收入分类、NBA 得分和学生成绩四任务中比较二十一个模型条目，对二十项可训练模型分别做网格与 TPE 搜索，并在所有特征消融中重新调参。外层、内层均为三折，预处理限定在训练折；神经网络另做结构控制与固定轮数的拟合诊断。'+findings(summary)+f'另以 6,228 个历史窗口比较十五项 NBA 下一赛季预测条目，所有模型使用同样历史信息。按 CV 预选模型的 Adult 扩展测试 AUC 为 {hs.loc[primary,"auc"]:.4f}；测试文件曾用于前期评价，本轮作为后续扩展报告。SHAP 解释固定提升树并逐条核对。结论限于当前数据、候选域与训练预算，其余任务仍无独立新样本。'
    paper=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',paper,flags=re.S)
    discussion=findings(summary)+'\n\n在共同分折上比较有助于排除划分差异，但增加模型数本身不构成准确率改进。网格与 TPE 的差异来自本轮有限候选域、重复访问和内层评分，仍以各自外层表现评价。训练曲线与网络部件控制说明，预测表现需要连同训练预算和结构一起阅读。SHAP 与重新训练的字段消融分别描述既定贡献及信息替代，均不直接估计因果效应。\n\nNBA 历史窗口任务提供了序列网络适用的输入结构，平面模型也接收相同历史。与原 NBA 四字段任务相比，输入和样本均改变，不能把误差之差解释为 LSTM 的独立收益。缺少稳定球员编号、连续历史筛选以及已分析过的原数据，均限制结论的推广。\n\nAdult 先在原训练文件内确定参数和预选模型，再统一做后续扩展测试。该文件已用于前期研究，本轮区间不覆盖此前查看表现、模型设计及训练变异的影响；其余任务仍无独立新样本。神经网络仅采用固定四十轮、小型网络和八点候选域，未独立估计初始化变异。真实 DeepSeek API 方案质量未评价，中文检索的检查也仅覆盖十六道固定题。\n'
    paper=re.sub(r'(\\section\{意义\}\n).*?(\n\\section\{方法\})',lambda m:m.group(1)+discussion+m.group(2),paper,flags=re.S)
    paper=paper.replace('本文在原固定参数复核之外，比较多类模型及两种超参数搜索，检查训练误差与验证误差的分离，并用字段贡献和重新训练的消融分析预测依据。','本文在原固定参数复核之外，比较二十一个表格模型条目及十五项 NBA 历史窗口条目，统一完成训练折内调参、特征消融和网络结构控制。')
    paper=paper.replace(r'\subsection{预测模型与评价方法}',r'\subsection{原单次划分的预测模型与评价方法}')
    start=paper.index(r'\subsection{代码公开与复现}')
    end=paper.index(r'\clearpage',start)
    reproduction=r'''\subsection{代码公开与复现}\label{sec:reproducibility}
\faGithub\enspace 项目仓库：\href{https://github.com/Karcen/TableLab_DeepSeek}{\nolinkurl{github.com/Karcen/TableLab_DeepSeek}}。中英文 README 说明启动与实验入口。复现时记录提交编号，核对 \code{artifacts/expanded/protocol.json} 中的数据、代码哈希、随机种子与依赖版本。

\faTerminal\enspace 在 macOS 项目根目录依次运行：
\begin{lstlisting}
bash TableLab-DeepSeek.command
bash Run-Expanded.command
\end{lstlisting}
第一个入口准备 Python 环境并启动网页。网页就绪后按 Control-C 停止服务，再运行第二个入口，完成扩展搜索、消融、补充测试、指标复算、绘图及文档编译。已完成的实验按冻结协议复用，修改训练代码前需归档不兼容结果。批量实验不调用 DeepSeek API；网页生成新方案使用个人密钥，默认存入本机 Mac 钥匙串。

绘图使用指定 Nature Skills 的 \code{nature-figure}，默认目录为 \code{~/Downloads/SKILLS/nature-skills-main/skills/nature-figure}，可用 \code{NATURE_FIGURE_SKILL} 更改位置。编译需要 XeLaTeX、BibTeX 与 Font Awesome 5；仅重建文档可运行 \code{bash docs/build_docs.sh}。

\faCode\enspace 扩展结果位于 \code{artifacts/expanded}，后续 Adult 测试另存于其中的 \code{holdout_extension}。可分别运行：
\begin{lstlisting}
codes/.venv/bin/python codes/tools/verify_expanded.py
codes/.venv/bin/python codes/tools/verify_expanded_holdout.py
codes/.venv/bin/python codes/tools/verify_expanded_shap.py
\end{lstlisting}
核对入口从逐条预测复算指标，并检查分折、参数搜索、模型冻结及 SHAP 加和。交付文件清单为 \code{codes/verification/research/files.sha256}。前期十三流程及其测试保留在 \code{artifacts/research} 和 \code{artifacts/holdout/adult}，由 \code{Run-Research.command} 和原 Adult 入口复现；原单次划分结果也分别保存。数据来源见附录 C、D。

'''
    paper=paper[:start]+reproduction+paper[end:]
    appendix=SRC/'appendices/04_implementation.tex'
    text=appendix.read_text()
    note='本轮扩展另外在同一文件上统一评价二十项网格模型和常数参照，结果明确标为后续扩展；前期十三流程的文件与数值保留。'
    if note not in text:
        text=text.replace('测试标签没有用于选择参数。','测试标签没有用于选择参数。'+note)
        appendix.write_text(text)
    return paper


def update_readmes():
    summary,_,_=load()
    primary,hs,_=holdout_data();h=hs.loc[primary]
    english={'Mean':'Constant','Linear':'Unpenalized linear / logistic','Ridge':'L2 linear / logistic','Sparse':'Elastic net','Tree':'Decision tree','RF':'Random forest','HGB':'Histogram boosting','SVM':'Linear SVM','XGB':'XGBoost','FeatureTransformer':'Feature Transformer'}
    en='## Experiments\n\nThe four original tasks compare 21 model entries: linear methods, tree ensembles, neighbors, SVM, XGBoost, LightGBM, CatBoost, MLP, residual and gated MLPs, Deep & Cross, a feature Transformer, ELM, and an RBF network. A separate NBA task compares 15 entries on the same three-season history windows, including RNN, GRU, BiGRU, LSTM, BiLSTM, causal TCN, and a sequence Transformer. These are batch experiments; the web app retains its existing interactive methods.\n\n'
    en+='Each trainable entry has grid and TPE searches. Most domains have eight configurations; unpenalized linear models compare two intercept settings. Three inner folds select parameters and three outer folds evaluate the procedure. Adult uses stratified groups of full-predictor fingerprints. NBA validation advances through complete seasons. Preprocessing and regression target scaling are fitted only on training data. Each feature ablation repeats grid search; five tabular networks also have structural controls. The gradient-trained networks run for 40 epochs.\n\n'
    en+='| Task | Better grid result in this comparison | Outer-fold mean ± SD |\n| --- | --- | --- |\n'
    zh='## 扩展模型实验\n\n四个原任务各比较 21 个模型条目，包括线性方法、树集成、近邻、SVM、XGBoost、LightGBM、CatBoost、MLP、残差与门控 MLP、Deep & Cross、字段 Transformer、ELM 和 RBF 网络。NBA 另设三季历史预测下一季的任务，共 15 项对照，加入 RNN、GRU、BiGRU、LSTM、BiLSTM、因果 TCN 和序列 Transformer。扩展模型通过批量入口运行，网页保留原有交互算法。\n\n'
    zh+='每项可训练模型分别做网格与 TPE。大部分候选域为八点，无惩罚线性项只比较两种截距配置。外层三折评价、内层三折选参数；Adult 按全部原始预测字段指纹分组分层，NBA 按完整赛季向前验证。预处理和回归目标缩放均只在训练数据拟合。每项特征消融重新网格调参，五项表格网络另做结构控制；端到端网络固定训练四十轮。\n\n'
    zh+='| 任务 | 本轮网格流程中的较优条目 | 外层均值 ± 标准差 |\n| --- | --- | --- |\n'
    etask={'house':'House prices','adult':'Adult income','nba':'NBA points','students':'Student mathematics','nba_sequence':'NBA next-season points'}
    for ds in DS+['nba_sequence']:
        r=best(summary,ds);metric='AUC' if ds=='adult' else 'RMSE';unit='' if ds=='adult' else ' USD' if ds=='house' else ' points'
        en+=f'| {etask[ds]} | {english.get(r.model,r.model)} | {metric} {score(r)}{unit} |\n'
        unit='' if ds=='adult' else ' 美元' if ds=='house' else ' 分'
        zh+=f'| {CN[ds]} | {NAMES[r.model].replace(chr(92),"")} | {metric} {score(r)}{unit} |\n'
    en+=f'\nAdult also has a subsequent test extension for the 21 grid or fixed entries. The model selected from CV before testing is {english.get(primary,primary)}: test AUC {h.auc:.4f}, with a 95% cluster-bootstrap interval of [{h.auc_ci_low:.4f}, {h.auc_ci_high:.4f}]. The original test had already been used for the earlier 13 procedures, so this is not a new blind test. Test scores did not change parameters or thresholds. The other tasks still lack new independent test samples. The NBA history task uses different inputs and records from the original NBA task; their errors do not directly measure an architecture gain.\n\n'
    en+='SHAP explains a boosting reference fixed before the expanded results, on all outer-validation records. Ablation measures performance after retraining with less information or a disabled component. Neither field attribution nor feature removal identifies real-world causal effects. Retrieval is still checked on 16 fixed questions, 15 containing English field names; real DeepSeek plan quality has not been evaluated. [Results and methods, in Chinese](docs/research_summary.md).\n\n'
    zh+=f'\nAdult 对 21 项网格或固定条目另做后续扩展测试。按 CV 事先预选的 {NAMES[primary].replace(chr(92),"")} 测试 AUC 为 {h.auc:.4f}，95% 分组 bootstrap 区间为 [{h.auc_ci_low:.4f}, {h.auc_ci_high:.4f}]。原测试已用于前期十三流程，因此不称新的盲测；测试成绩没有用于回调参数或阈值。其余任务仍缺独立新样本。NBA 历史任务改变了输入和样本，不能用它与原 NBA 任务的误差差值直接说明网络收益。\n\n'
    zh+='SHAP 固定解释提升树，覆盖全部外层验证记录；消融删除输入或关闭部件后重新训练。字段贡献与删减不直接识别现实中的因果效应。检索仍只检查十六道固定题，其中十五道含英文字段名，真实 DeepSeek API 方案质量尚未评价。完整数字、解释和复现办法见 [实验说明](docs/research_summary.md)。\n\n'
    en_repro='''## Reproduce and check results

Launch the app once to prepare the Python environment, stop the web service, then run `Run-Expanded.command`. It installs the pinned research dependencies, runs nested search and ablations, exports predictions in float64, checks the results, generates figures, and rebuilds the paper and Madrid slides. Batch experiments make no DeepSeek API calls. Each model uses one CPU thread; completed jobs are reused under the frozen protocol.

Expanded results are in `artifacts/expanded`. The earlier thirteen-procedure comparison is in `artifacts/research`, and its Adult test is in `artifacts/holdout/adult`; those records remain unchanged. The new Adult test extension is stored separately under `artifacts/expanded/holdout_extension`. Archive incompatible results before changing a frozen training implementation.

Figures use the designated Nature Skills directory, `~/Downloads/SKILLS/nature-skills-main/skills/nature-figure`. Set `NATURE_FIGURE_SKILL` to use another location. Document compilation needs XeLaTeX, BibTeX, and Font Awesome 5.

```bash
# Recompute expanded metrics and check every split and search
codes/.venv/bin/python codes/tools/verify_expanded.py
codes/.venv/bin/python codes/tools/verify_expanded_holdout.py
codes/.venv/bin/python codes/tools/verify_expanded_shap.py

# Rebuild the paper and slides
bash docs/build_docs.sh

# Check the delivered files before modifying them
shasum -a 256 -c codes/verification/research/files.sha256
```

'''
    zh_repro='''## 复现与核对

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

'''
    for name,section,reproduction,start,end,files in [('README.md',en,en_repro,'## Four-task experiments','## Reproduce and check results','## Files'),('README.zh-CN.md',zh,zh_repro,'## 四任务实验','## 复现与核对','## 目录')]:
        p=ROOT/name;text=p.read_text();current='## Experiments' if name=='README.md' else '## 扩展模型实验'
        first=text.index(start) if start in text else text.index(current)
        last=text.index(end,first)
        text=text[:first]+section+text[last:]
        first=text.index(end);last=text.index(files,first)
        text=text[:first]+reproduction+text[last:]
        marker='| `artifacts/research`'
        row='| `artifacts/expanded` | Expanded search, ablations, models, predictions, and test extension |\n' if name=='README.md' else '| `artifacts/expanded` | 扩展搜索、消融、模型、预测与后续测试 |\n'
        if '| `artifacts/expanded`' not in text:text=text.replace(marker,row+marker)
        p.write_text(text)
    (ROOT/'README.txt').write_text((ROOT/'README.zh-CN.md').read_text())
