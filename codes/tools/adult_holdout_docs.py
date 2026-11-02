"""Build the reserved-test additions from verified Adult outputs."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'artifacts/holdout/adult'
NAMES = {'Mean':'常数基线', 'Linear':'逻辑回归', 'Ridge':'强 L2 逻辑回归', 'Sparse':'L1 逻辑回归',
         'Tree':'单棵决策树', 'RF':'预设随机森林', 'ExtraTrees':'Extra Trees', 'HGB':'预设提升树',
         'KNN':'K 近邻', 'RF_grid':'森林＋网格', 'RF_tpe':'森林＋TPE',
         'HGB_grid':'提升树＋网格', 'HGB_tpe':'提升树＋TPE'}


def data():
    assert json.loads((ROOT/'codes/verification/research/adult_holdout/checks.json').read_text())['status'] == 'passed'
    scores = pd.read_csv(DATA/'metrics.csv').set_index('model')
    pairs = pd.read_csv(DATA/'paired_differences.csv').set_index('comparator')
    return scores, pairs


def brief():
    scores, _ = data()
    s = scores.loc['HGB_grid']
    return f'Adult 另以 16,255 条预留记录测试，事先选定的提升树＋网格取得 AUC {s.auc:.4f}（95% 区间 {s.auc_ci_low:.4f}--{s.auc_ci_high:.4f}）。'


def results():
    scores, pairs = data()
    s = scores.loc['HGB_grid']
    body = r'\subsection{Adult 预留测试}\label{sec:adult-holdout}'+'\n'
    body += '在原训练文件内完成上述模型比较后，另用原测试文件评价全部十三条流程。原测试文件有 16,281 条，排除与训练文件全部十四个非目标原始字段指纹重合的 26 条，实际评价 16,255 条。筛选不使用收入标签。测试集内部的六条额外重复指纹记录保留，不确定性估计按指纹组重抽样。\n\n'
    body += brief().replace('%', r'\%')+'测试指标与原三折均值分别列于表~\\ref{tab:adult-holdout}。两者训练规模和评价样本不同，不能把数值之差解释为调参收益。\n\n'
    body += r'\inlinetab{\captionof{table}{Adult 原训练文件内的嵌套验证与预留测试}\label{tab:adult-holdout}'+'\n'
    body += r'\begin{tabularx}{\linewidth}{@{}Yrrrr@{}}\toprule'+'\n'
    body += r'流程 & 三折 AUC & 测试 AUC & 测试 AUC 95\% 区间 & 测试 F1\\\midrule'+'\n'
    for model, row in scores.sort_values('auc', ascending=False, kind='stable').iterrows():
        body += f'{NAMES[model]} & {row.cv_auc:.4f} & {row.auc:.4f} & [{row.auc_ci_low:.4f}, {row.auc_ci_high:.4f}] & {row.f1:.4f}'+r'\\'+'\n'
    body += r'\bottomrule\end{tabularx}\tabnote{三折列为外层均值；测试列为在全部 32,561 条原训练记录上重拟合后，对 16,255 条预留记录的评价。区间来自按 16,249 个指纹组进行的 2,000 次配对 bootstrap，描述给定模型下测试样本的不确定性。阈值固定为 0.5；AP、Brier 与对数损失及全部配对差值随逐条预测保存。}}'+'\n'
    for model in ['HGB', 'RF', 'HGB_tpe']:
        r = pairs.loc[model]
        body += f'相对{NAMES[model]}，事先选定流程的测试 AUC 差为 {r.auc_difference:+.5f}，配对 95\\% 区间为 [{r.ci_low:+.5f}, {r.ci_high:+.5f}]。'
    body += '\n\n'+interpretation()+'\n'
    body += r'''\begin{figure}[!htbp]\centering
\includegraphics[width=\linewidth]{figures/research/adult_holdout.pdf}
\caption{Adult 预留测试与配对比较。a，全部十三条流程的测试 AUC；青色菱形为测试前选定的提升树＋网格，误差线为 95\% 区间，排序仅用于展示。b，选定流程减去其余五条森林或提升树流程的 AUC，正值表示选定流程较高，横轴以 0.001 AUC 为单位。两面板均使用 16,255 条记录、16,249 个指纹组的 2,000 次配对分组 bootstrap；区间未作多重比较校正，且不包含训练及搜索过程的变异。全部十二项配对比较保存在结果文件中。}\label{fig:adult-holdout}
\end{figure}
'''
    return body


def interpretation():
    scores, pairs = data()
    s = scores.loc['HGB_grid']
    better = [NAMES[m] for m, r in pairs.iterrows() if r.ci_high < 0]
    tied = [NAMES[m] for m, r in pairs.iterrows() if r.ci_low <= 0 <= r.ci_high and m in ['HGB','HGB_tpe','RF']]
    text = f'提升树＋网格在本次测试中排第 {int(s.test_rank)}，测试前的选择保持不变。'
    if tied:
        text += '与'+ '、'.join(tied)+' 的差值区间包含零，现有测试未分清这些流程的优劣。'
    if better:
        text += '与'+ '、'.join(better)+'相比，选定流程的差值区间低于零。'
    text += '这些配对区间用于描述预定比较，未作多重比较校正。房价、NBA 和学生任务仍缺少未参与前期分析的新样本；Adult 的测试结果不能确认其余任务的排名。'
    return text


def methods():
    parameters = {m: json.loads((DATA/f'{m}_fit.json').read_text())['selected_params'] for m in ['HGB_grid', 'HGB_tpe', 'RF_grid', 'RF_tpe']}
    assert parameters['HGB_grid'] == {'model__l2_regularization': 0., 'model__learning_rate': .05, 'model__max_leaf_nodes': 31}
    assert parameters['HGB_tpe'] == {'model__l2_regularization': 1., 'model__learning_rate': .1, 'model__max_leaf_nodes': 15}
    assert parameters['RF_grid'] == parameters['RF_tpe'] == {'model__max_depth': 8, 'model__min_samples_leaf': 3, 'model__max_features': .7}
    return r'''\subsection{Adult 预留测试协议}
预留测试在开始拟合前写入 \code{artifacts/holdout/adult/protocol.json}，固定输入字段、十三条训练流程、原始数据和代码哈希、样本排除规则及指标。主评价流程根据此前嵌套验证选为提升树＋网格；TPE 与其三折结果相同，仍保留为单独流程。全部 32,561 条原训练记录用于最终拟合。四条调参流程仅在原训练文件内进行三折分层分组验证，沿用十二点参数域与十二次评估，TPE 种子为 42；预处理仍在每次训练子集拟合。本次新增 144 次内层拟合和 13 次最终拟合。全部模型及参数保存并锁定哈希后，才计算测试预测；测试成绩未用于重选模型、参数或阈值。

在完整原训练文件的内层验证中，网格提升树选出叶节点上限 31、学习率 0.05、L2 惩罚 0；TPE 选出叶节点上限 15、学习率 0.1、L2 惩罚 1。森林两种搜索均选出最大深度 8、叶节点最少 3 条、候选特征比例 0.7。各参数与外层三折的参数分别保存，因此原先三折均值相同的两条流程，最终重拟合的参数仍可不同。

原测试文件排除与训练文件十四个非目标原始字段指纹重合的 26 条，保留 16,255 条、16,249 个指纹组。指纹包括 \code{fnlwgt}，仅用于分组核查；模型仍只输入六个字段，指标不按人口权重加权。保留测试集内部重复组，按全部指纹组有放回重抽样 2,000 次，种子为 20261102，同一次抽样用于全部模型。以百分位法给出 AUC 和事先选定流程与各比较流程 AUC 差的 95\% 区间；若一个组被抽中多次，其全部记录获得相应重数。比较区间未作多重比较校正，不用于事后挑选新模型。区间以最终模型为条件，只反映测试抽样变化，不包含重新训练和搜索的不确定性。原测试文件曾参与数据内容核查，本文称为预留测试，不称从未查看的盲测。

主指标为 ROC AUC，另报告 AP、Brier、对数损失和固定 0.5 阈值的 F1。通过稳定行号对齐原始标签，独立复算 65 项指标，重新检查三折内层的指纹组互斥、跨文件排除规则以及模型冻结时间；另用独立加权 AUC 实现复核 91 项 bootstrap 分数。原始数据文件保持不变。
'''


def write_report():
    scores, pairs = data()
    s = scores.loc['HGB_grid']
    out = '# Adult 预留测试补测\n\n2026年11月2日完成。\n\n## 研究问题\n\n在原训练文件上选出的模型，能否在此前未用于训练、调参和性能比较的 Adult 原测试文件上保持表现？原三折中的微小排名差异能否区分？\n\n## 结果\n\n'
    out += brief().replace('--','–')+'所有十三条流程均在 32,561 条原训练记录上完成拟合后统一测试，没有根据测试成绩回调参数。\n\n'
    out += '| 流程 | 原三折 AUC | 测试 AUC | 测试 AUC 95% 区间 | 测试排名 |\n| --- | ---: | ---: | --- | ---: |\n'
    for model, r in scores.sort_values('auc', ascending=False, kind='stable').iterrows():
        out += f'| {NAMES[model]} | {r.cv_auc:.4f} | {r.auc:.4f} | [{r.auc_ci_low:.4f}, {r.auc_ci_high:.4f}] | {int(r.test_rank)} |\n'
    out += '\n选定流程的 AP、Brier、对数损失、F1 分别为 '+f'{s.ap:.4f}、{s.brier:.4f}、{s.log_loss:.4f}、{s.f1:.4f}。阈值为 0.5。\n\n'
    out += '| 比较对象 | 提升树＋网格减去比较对象的 AUC | 配对 95% 区间 |\n| --- | ---: | --- |\n'
    for model, r in pairs.iterrows():
        out += f'| {NAMES[model]} | {r.auc_difference:+.5f} | [{r.ci_low:+.5f}, {r.ci_high:+.5f}] |\n'
    out += '\n## 意义\n\n'+interpretation()+'原三折均值与这次测试使用不同的训练规模及评价样本，两列之差不代表调参收益。\n\n'
    out += '本次补上了 Adult 的预留测试证据。原测试文件曾做过内容核查，因此不称为从未查看的盲测。区间只描述给定最终模型的测试样本变化，不包含训练或参数搜索的变异。真实 DeepSeek API 的方案生成质量仍未评价；这项测试只涉及本地预测模型。\n\n'
    out += '## 方法\n\n先保存协议，固定六个输入字段、十三条流程、指标与 0.5 阈值。四条调参流程只在原训练文件内进行三折分层分组验证；指纹取全部十四个非目标原始字段，网格和 TPE 沿用原十二点参数域、十二次评估。新增 144 次内层拟合和 13 次最终拟合，全部拟合对象锁定后才计算测试成绩。\n\n'
    out += '原测试文件有 16,281 条。按训练数据的十四字段指纹排除 26 条跨文件重合记录，筛选不使用测试标签；余下 16,255 条对应 16,249 个指纹组，保留内部重复记录。六字段背景相同不能证明是同一个人，因此没有据此删除样本。`fnlwgt` 只参加指纹核查，不作为输入或评价权重。\n\n'
    out += '按指纹组做 2,000 次配对 bootstrap（种子 20261102），以百分位法计算 95% 区间。所有模型共用同一次组抽样，组内记录保留，并随组一起重复。比较区间未作多重比较校正，用于描述预定比较。65 项指标从逐条预测独立复算；3 个内层划分、跨文件指纹互斥、冻结时间均通过核查，另以独立实现复核 91 项重抽样 AUC。\n\n'
    out += '协议、模型、参数、搜索记录、逐条预测和区间保存在 `artifacts/holdout/adult/`。复算入口为 `codes/tools/verify_adult_holdout.py`；主运行入口为 `codes/tools/run_adult_holdout.py`，完成后再次运行会保留现有测试，不重新训练。绘图沿用指定 Nature Skills 的 Python 流程。\n\n'
    out += '[课程论文]('+str(ROOT/'docs/paper.pdf')+') · [Madrid 展示稿]('+str(ROOT/'docs/presentation.pdf')+') · [项目仓库](https://github.com/Karcen/TableLab_DeepSeek)\n'
    (ROOT/'docs/adult_holdout_results.md').write_text(out)


if __name__ == '__main__':
    write_report()
