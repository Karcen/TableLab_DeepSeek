"""单图单 PDF；原始计算表另存 CSV。不导出像素化的论文图。"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
import numpy as np
from functools import lru_cache
from fontTools.ttLib import TTFont


@lru_cache(maxsize=1)
def pdf_chinese_family():
    """Type 42 输出选择真正的 TrueType 轮廓，避免 CFF 集合字体跨阅读器乱码。"""
    preferred=['PingFang SC','Heiti SC','STHeiti','Microsoft YaHei','Arial Unicode MS',
               'Noto Sans CJK SC','Noto Sans CJK JP','Noto Sans SC','Songti SC','AR PL UMing CN','AR PL SungtiL GB','SimHei','SimSun']
    for name in preferred:
        matches=[f for f in font_manager.fontManager.ttflist if f.name==name and f.style=='normal']
        for candidate in matches:
            with TTFont(candidate.fname,fontNumber=0,lazy=True) as font:
                if 'glyf' in font and all(ord(c) in font.getBestCmap() for c in '中文数据收入'):
                    return name
    raise RuntimeError('PDF 绘图缺少可用的中文 TrueType 字体。请安装带 TrueType 轮廓的中文字体并重启；当前不以缺字图替代。')

def setup():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[pdf_chinese_family(),'DejaVu Sans'],
        'pdf.fonttype':42,'font.size':9,'axes.labelsize':10,'axes.titlesize':11,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.8,'legend.frameon':False,'axes.unicode_minus':False,'savefig.bbox':'tight'})

def new_plot(title,source_kind):
    setup(); fig,ax=plt.subplots(figsize=(7.2,4.5))
    ax.set_title(title,loc='left',pad=12)
    if source_kind=='synthetic_test': fig.text(.01,.01,'合成数据 · 仅软件测试，不是实证结果',fontsize=8)
    return fig,ax

def save(fig,path):
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(path,format='pdf',metadata={'Creator':'Offline TableLab'});plt.close(fig)

def data_plot(table,plan,path,source_kind):
    fig,ax=new_plot({'histogram':'数值分布','scatter':'样本关系','bar':'分组比较'}[plan.chart],source_kind)
    if plan.chart=='histogram':
        ax.hist(table[plan.columns[0]],bins=24,edgecolor='white',linewidth=.5)
        ax.set_xlabel(plan.columns[0]);ax.set_ylabel('记录数')
    elif plan.chart=='scatter':
        ax.scatter(table[plan.columns[0]],table[plan.columns[1]],s=9,alpha=.4,linewidths=0)
        ax.set_xlabel(plan.columns[0]);ax.set_ylabel(plan.columns[1])
    else:
        view=table.head(plan.limit).iloc[::-1]
        labels=view[plan.group_by].astype(str).agg(' / '.join,axis=1)
        ax.barh(labels,view['value']);ax.set_xlabel(plan.metric+' '+(plan.columns[0] if plan.columns else 'records'))
    save(fig,path)

def diagnostic_plot(predictions,classification,path,source_kind):
    if classification:
        from sklearn.metrics import roc_curve
        fig,ax=new_plot('交互验证集 ROC',source_kind)
        fpr,tpr,_=roc_curve(predictions['actual'],predictions['probability_1'])
        ax.plot(fpr,tpr,linewidth=1.6,label='Model');ax.plot([0,1],[0,1],linestyle='--',linewidth=.8,label='Random ranking')
        ax.set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1));ax.legend()
    else:
        fig,ax=new_plot('交互验证集：实际值与预测值',source_kind)
        ax.scatter(predictions['actual'],predictions['predicted'],s=11,alpha=.45,linewidths=0)
        lo=min(predictions['actual'].min(),predictions['predicted'].min());hi=max(predictions['actual'].max(),predictions['predicted'].max())
        ax.plot([lo,hi],[lo,hi],linestyle='--',linewidth=.8)
        ax.set(xlabel='Actual target',ylabel='Predicted target',xlim=(lo,hi),ylim=(lo,hi))
    save(fig,path)
