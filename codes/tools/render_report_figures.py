"""仅从已核验 CSV 和验证集预测生成独立 PDF；不抽样、不绘制模拟结果。"""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tablelab.plotting import setup
from sklearn.metrics import roc_curve
ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
TABLE=ROOT/'tables'

def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),format='pdf',metadata={'Title':name,'Author':'Karcen Zheng','Subject':'上传数据固定划分的验证结果；非盲测'})
    plt.close(fig)
    print(FIG/(name+'.pdf'))

def main():
    setup()
    models=pd.read_csv(TABLE/'model_comparison.csv')
    report=json.loads((ROOT/'verification/real_data_benchmark.json').read_text())
    cases={c['case']:c for c in report['cases']}
    meta=[]
    for dataset,title,unit,factor in [('house','房价：固定随机划分','RMSE（千美元）',1000),('nba','NBA：后续赛季留出','RMSE（场均得分）',1),('students','学生成绩：不使用同期考试分数','RMSE（数学分数）',1)]:
        selected=models.loc[(models.dataset==dataset)&(models.role=='primary')].set_index('algorithm')
        vals=np.array([selected.loc['linear','rmse'],selected.loc['random_forest','rmse'],selected.loc['linear','baseline_rmse']])/factor
        labels=['线性回归','随机森林','训练均值基线'];n=int(selected.loc['linear','n_validation']);nt=int(selected.loc['linear','n_train'])
        data=pd.DataFrame({'algorithm':labels,'rmse_display':vals,'display_units':unit,'n_train':nt,'n_validation':n,'divisor_from_original_units':factor})
        data.to_csv(TABLE/f'figure_{dataset}_rmse.csv',index=False,encoding='utf-8-sig')
        fig,ax=plt.subplots(figsize=(7.2,3.7))
        bars=ax.barh(labels,vals,height=.55)
        bars[-1].set_hatch('///')
        ax.invert_yaxis();ax.set_xlim(0,vals.max()*1.22);ax.set_xlabel(unit)
        ax.set_title(title,loc='left',pad=16)
        for bar,value in zip(bars,vals):ax.text(value+vals.max()*.025,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontsize=9)
        fig.text(.24,.09,f'训练 n={nt:,}；验证 n={n:,}。数值越低越好。',fontsize=8)
        fig.text(.24,.035,'单个固定划分；无置信区间。指标不是最终盲测成绩。',fontsize=8)
        fig.subplots_adjust(left=.24,right=.97,top=.82,bottom=.29)
        save(fig,dataset+'_rmse')
        meta.append({'figure':dataset+'_rmse.pdf','source':f'figure_{dataset}_rmse.csv','n':n,'uncertainty':'single fixed split; no confidence interval','omitted_rows':0,'units':unit})
        fig,ax=plt.subplots(figsize=(7.2,2.65))
        bars=ax.barh(labels,vals,height=.55);bars[-1].set_hatch('///')
        ax.invert_yaxis();ax.set_xlim(0,vals.max()*1.22);ax.set_xlabel(unit,fontsize=12)
        ax.tick_params(axis='both',labelsize=11)
        for bar,value in zip(bars,vals):ax.text(value+vals.max()*.025,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center',fontsize=12)
        fig.text(.24,.045,f'训练 {nt:,}；验证 {n:,}。固定划分，无置信区间。',fontsize=10)
        fig.subplots_adjust(left=.24,right=.97,top=.97,bottom=.34)
        save(fig,dataset+'_rmse_slide')
        meta.append({'figure':dataset+'_rmse_slide.pdf','source':f'figure_{dataset}_rmse.csv','n':n,'uncertainty':'single fixed split; no confidence interval','omitted_rows':0,'units':unit,'purpose':'same data, slide typography'})
    fig,ax=plt.subplots(figsize=(7.2,4.7));curves=[]
    for algorithm,label,style in [('logistic','逻辑回归','-'),('random_forest','随机森林','--')]:
        case=cases['adult_primary_'+algorithm]
        pred=pd.read_csv(ROOT/'artifacts/benchmarks'/case['run_id']/'validation_predictions.csv')
        fpr,tpr,thr=roc_curve(pred.actual,pred.probability_1)
        ax.plot(fpr,tpr,linestyle=style,linewidth=1.5,label=f'{label}（AUC={case["metrics"]["roc_auc"]:.4f}）')
        curves.extend({'algorithm':algorithm,'fpr':float(a),'tpr':float(b)} for a,b in zip(fpr,tpr))
    ax.plot([0,1],[0,1],linestyle=':',linewidth=.8,label='随机排序')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='假阳性率',ylabel='真阳性率')
    ax.set_title('Adult：非目标字段指纹分组后的验证集',loc='left',pad=15)
    ax.legend(loc='lower right',fontsize=9)
    fig.text(.14,.08,'训练 n=26,049；验证 n=6,512。原始 adult.test 不参与拟合。',fontsize=8)
    fig.text(.14,.03,'单次固定验证；AUC 衡量排序，不等于阈值分类质量或概率校准。',fontsize=8)
    fig.subplots_adjust(left=.14,right=.96,top=.84,bottom=.25)
    pd.DataFrame(curves).to_csv(TABLE/'figure_adult_roc.csv',index=False)
    save(fig,'adult_roc')
    meta.append({'figure':'adult_roc.pdf','source':'figure_adult_roc.csv','n':6512,'uncertainty':'single fixed split; no confidence interval','omitted_rows':0,'units':'rates'})
    quality=pd.read_csv(TABLE/'column_quality.csv')
    blanks=quality.loc[(quality.dataset=='house')&(quality.missing>0)].sort_values('missing')
    pd.DataFrame(blanks).to_csv(TABLE/'figure_house_blanks.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(7.2,6.1))
    ax.barh(blanks.column,blanks.missing_fraction*100,height=.72)
    ax.set(xlim=(0,110),xlabel='空白单元格比例（%）')
    for i,row in enumerate(blanks.itertuples()):ax.text(row.missing_fraction*100+1,i,str(row.missing),va='center',fontsize=7.5)
    ax.set_title('房价：空白不一律等于未知测量',loc='left',pad=15)
    fig.text(.25,.065,'n=1,460；展示所有含空白的字段。标签为记录数。',fontsize=8)
    fig.text(.25,.025,'保留全部原始行；“None”类别未转成缺失；不作全表删缺失。',fontsize=8)
    fig.subplots_adjust(left=.25,right=.96,top=.89,bottom=.16)
    save(fig,'house_blank_profile')
    meta.append({'figure':'house_blank_profile.pdf','source':'figure_house_blanks.csv','n':1460,'uncertainty':'descriptive full uploaded sample; no inferential interval','omitted_rows':0,'omitted_fields':'columns with zero blank cells only','units':'percent of rows'})
    (ROOT/'verification/report_figure_sources.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
