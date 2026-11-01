"""Nature-style source-backed figures for the four-task nested experiment."""
from pathlib import Path
import argparse
import sys
import json
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/research'
OUT=ROOT/'docs/source/figures/research'
DS=['house','adult','nba','students']
TITLES={'house':'House price','adult':'Adult income','nba':'NBA points','students':'Student mathematics'}
BLUE='#3B739B';ORANGE='#BD7547';GREY='#727982';PALE='#A9BFCF'
LABELS={'OverallQual':'Quality','GrLivArea':'Living area','YearBuilt':'Year built','TotalBsmtSF':'Basement area','GarageCars':'Garage spaces','Neighborhood':'Neighborhood','age':'Age','education-num':'Education','hours-per-week':'Weekly hours','workclass':'Work class','occupation':'Occupation','marital-status':'Marital status','player_height':'Height','player_weight':'Weight','draft_year':'Draft year','gender':'Gender','parental level of education':'Parental education','lunch':'Lunch','test preparation course':'Preparation'}
DISPLAY={'Mean':'Constant','Linear':'Logistic / OLS','Ridge':'Strong L2 / Ridge','Sparse':'L1 / Elastic Net','Tree':'Decision tree','RF':'Random forest','ExtraTrees':'Extra Trees','HGB':'Hist. gradient boosting','KNN':'15-nearest neighbors','RF_grid':'Forest + Grid','RF_tpe':'Forest + TPE','HGB_grid':'Boosting + Grid','HGB_tpe':'Boosting + TPE'}
CONTRACT={
'benchmark':'Compare sample-out performance across predefined model families and tuning flows.',
'search':'Compare the search trajectories of grid enumeration and TPE under the same finite domain and candidate budget.',
'learning':'Check training and held-out errors as the amount of available training information increases.',
'complexity':'Check whether increasing decision-tree depth lowers training error at the cost of validation error.',
'ablation':'Measure paired loss after removing feature groups and repeating inner tuning.',
'shap':'Describe field-level prediction contributions on every outer validation observation.',
'dependence':'Inspect how the contribution of a prespecified field varies across observed values.',
'local':'Decompose one median-residual validation prediction into field-level contributions.'}

def readall(name):return pd.concat([pd.read_csv(p) for p in DATA.glob(f'*/fold*/{name}')],ignore_index=True)
def metric(ds,diagnostic=False):return ('log_loss' if diagnostic else 'auc') if ds=='adult' else 'rmse'
def axislabel(ds,diagnostic=False):return ('Log loss' if diagnostic else 'ROC AUC') if ds=='adult' else ('RMSE (USD)' if ds=='house' else 'RMSE (points)')
def panelgrid():
 fig,axes=plt.subplots(2,2,figsize=(7.2,6.6),gridspec_kw={'wspace':.65,'hspace':.65})
 fig.subplots_adjust(left=.18,right=.97,bottom=.10,top=.92)
 for i,(ax,ds) in enumerate(zip(axes.flat,DS)):
  ax.set_title(TITLES[ds],loc='left',fontsize=8,pad=12)
  ax.text(0,1,string.ascii_lowercase[i],transform=ax.transAxes+ScaledTranslation(-18/72,12/72,fig.dpi_scale_trans),fontweight='bold',fontsize=8,ha='left',va='bottom')
 return fig,list(axes.flat)

def save(fig,name):
 from audit_panel_alignment import require_matplotlib_panel_alignment
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=str(OUT/f'{name}.alignment.json'),overlay_svg=str(OUT/f'{name}.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
 fig.savefig(OUT/f'{name}.pdf')
 fig.savefig(OUT/f'{name}.svg')
 fig.savefig(OUT/f'{name}.png',dpi=600)
 if len(fig.axes)==4:
  renderer=fig.canvas.get_renderer()
  for ax,ds in zip(fig.axes,DS):
   for other in fig.axes:other.set_visible(True)
   bounds=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(.08)
   for other in fig.axes:other.set_visible(other is ax)
   fig.savefig(OUT/f'{name}_{ds}.pdf',bbox_inches=bounds)
  for ax in fig.axes:ax.set_visible(True)
 if len(fig.axes)==4 and name!='benchmark':
  fig.set_size_inches(12,5)
  for artist in fig.findobj(matplotlib.text.Text):artist.set_fontsize(artist.get_fontsize()*2)
  for ax in fig.axes:
   if not isinstance(ax.xaxis.get_major_locator(),matplotlib.ticker.FixedLocator):
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
  if name=='complexity':
   for ax in fig.axes:
    lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+(hi-lo)*.4);ax.legend(loc='upper center',ncol=2,fontsize=12)
  fig.canvas.draw()
  require_matplotlib_panel_alignment(fig,json_out=str(OUT/f'{name}_slide.alignment.json'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
  fig.savefig(OUT/f'{name}_slide.pdf')
 plt.close(fig)

def meanline(ax,data,x,y,part,color,label):
 v=data.loc[data.partition==part].groupby(x)[y].agg(['mean','std']).sort_index()
 ax.errorbar(v.index,v['mean'],yerr=v['std'].fillna(0),color=color,marker='o',ms=3,lw=1,capsize=2,label=label)

def main(skill):
 sys.path.insert(0,str(Path(skill)/'scripts'));OUT.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7,'legend.frameon':False,'legend.fontsize':6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
 metrics=readall('metrics.csv');diags=readall('diagnostics.csv');ablations=readall('ablations.csv')
 metrics.to_csv(DATA/'metrics.csv',index=False);diags.to_csv(DATA/'diagnostics.csv',index=False);ablations.to_csv(DATA/'ablations.csv',index=False)
 fig,axs=panelgrid()
 for ax,ds in zip(axs,DS):
  q=metrics.query('dataset==@ds and partition=="validation"');m=metric(ds)
  summ=q.groupby('model')[m].agg(['mean','std']).sort_values('mean',ascending=ds!='adult');order=list(summ.index)
  for k,name in enumerate(order):
   vals=q.loc[q.model==name,m]
   color=BLUE if '_' in name else GREY
   ax.plot(vals,np.repeat(k,len(vals)),'.',color=color,ms=3,alpha=.7)
   ax.errorbar(summ.loc[name,'mean'],k,xerr=summ.loc[name,'std'],color=color,fmt='o',ms=3,capsize=2,lw=.8)
  ax.set_yticks(range(len(order)),[DISPLAY[n].replace('Logistic / OLS','Logistic' if ds=='adult' else 'OLS').replace('Strong L2 / Ridge','Strong L2' if ds=='adult' else 'Ridge').replace('L1 / Elastic Net','L1 logistic' if ds=='adult' else 'Elastic Net') for n in order]);ax.invert_yaxis();ax.set_xlabel(axislabel(ds));ax.tick_params(axis='y',length=0)
 save(fig,'benchmark')
 for ds in DS:
  q=metrics.query('dataset==@ds and partition=="validation"');m=metric(ds);summary=q.groupby('model')[m].agg(['mean','std']).sort_values('mean',ascending=ds!='adult');order=list(summary.index)
  fig,ax=plt.subplots(figsize=(7.2,3.0));fig.subplots_adjust(left=.28,right=.98,bottom=.20,top=.97)
  for j,name in enumerate(order):
   values=q.loc[q.model==name,m];color=BLUE if '_' in name else GREY
   ax.plot(values,np.repeat(j,len(values)),'.',color=color,ms=3)
   ax.errorbar(summary.loc[name,'mean'],j,xerr=summary.loc[name,'std'],fmt='o',color=color,ms=3,capsize=2)
  ax.set_yticks(range(len(order)),[DISPLAY[n].replace('Logistic / OLS','Logistic' if ds=='adult' else 'OLS').replace('Strong L2 / Ridge','Strong L2' if ds=='adult' else 'Ridge').replace('L1 / Elastic Net','L1 logistic' if ds=='adult' else 'Elastic Net') for n in order],fontsize=9)
  ax.invert_yaxis();ax.set_xlabel(axislabel(ds),fontsize=10);ax.tick_params(axis='x',labelsize=9);ax.tick_params(axis='y',length=0)
  save(fig,f'benchmark_{ds}_slide')
 fig,axs=panelgrid()
 for ax,ds in zip(axs,DS):
  for method,color in [('grid',BLUE),('tpe',ORANGE)]:
   runs=[]
   for p in sorted((DATA/ds).glob(f'fold*/HGB_{method}_search.json')):
    history=json.loads(p.read_text());v=np.array([r['score'] for r in history['rows']]);v=np.maximum.accumulate(v);runs.append(v if ds=='adult' else -v)
   a=np.array(runs);ax.errorbar(np.arange(1,a.shape[1]+1),a.mean(0),yerr=a.std(0,ddof=1),color=color,label='Grid' if method=='grid' else 'TPE',lw=1,ms=2,marker='o',capsize=1)
  ax.set_xlabel('Candidate evaluations');ax.set_ylabel('Inner CV '+axislabel(ds));ax.legend(loc='upper center',bbox_to_anchor=(.5,1.01),ncol=2)
  lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+(hi-lo)*.22)
 save(fig,'search')
 for kind in ['learning','complexity']:
  fig,axs=panelgrid()
  for ax,ds in zip(axs,DS):
   q=diags.query('dataset==@ds and kind==@kind').copy();m=metric(ds,True)
   x='fraction' if kind=='learning' else 'depth'
   if kind=='complexity':q.loc[q.depth==-1,'depth']=12
   meanline(ax,q,x,m,'train',GREY,'Train');meanline(ax,q,x,m,'validation',BLUE,'Validation')
   ax.set_ylabel(axislabel(ds,True));ax.set_xlabel('Training fraction' if kind=='learning' else 'Maximum tree depth');ax.legend(loc='best')
   if kind=='complexity':ax.set_xticks([1,2,4,8,12],['1','2','4','8','Full'])
  save(fig,kind)
 fig,axs=panelgrid()
 deltas=[]
 for ax,ds in zip(axs,DS):
  m=metric(ds);q=ablations.query('dataset==@ds').copy();full=metrics.query('dataset==@ds and model=="HGB_grid" and partition=="validation"').set_index('fold')[m]
  q['delta']=[full.loc[f]-v if ds=='adult' else v-full.loc[f] for f,v in zip(q.fold,q[m])]
  deltas.append(q)
  order=list(q.groupby('variant').delta.mean().sort_values().index)
  for j,v in enumerate(order):
   values=q.loc[q.variant==v,'delta'];ax.plot(values,np.repeat(j,len(values)),'.',color=GREY,ms=3);ax.errorbar(values.mean(),j,xerr=values.std(),fmt='o',color=BLUE,ms=4,capsize=2)
  ax.axvline(0,color=GREY,lw=.6,ls='--');ax.set_yticks(range(len(order)),['Only area + quality' if v=='Area + quality' else v for v in order]);ax.invert_yaxis();ax.set_xlabel('AUC decrease' if ds=='adult' else ('RMSE increase (USD)' if ds=='house' else 'RMSE increase (points)'));ax.tick_params(axis='y',length=0)
 pd.concat(deltas).to_csv(DATA/'ablation_deltas.csv',index=False);save(fig,'ablation')
 # All outer-validation observations are retained in the explanation panels.
 shap_data={};raw_data={}
 for ds in DS:
  shap_data[ds]=pd.concat([pd.read_csv(p) for p in (DATA/ds).glob('fold*/shap.csv')],ignore_index=True)
  raw_data[ds]=pd.concat([pd.read_csv(p) for p in (DATA/ds).glob('fold*/shap_features.csv')],ignore_index=True)
 fig,axs=panelgrid();ranks=[]
 for ax,ds in zip(axs,DS):
  q=shap_data[ds];cols=json.loads((DATA/ds/'data_audit.json').read_text())['features'];order=sorted(cols,key=lambda c:q[c].abs().mean())
  means=q.groupby('fold')[cols].agg(lambda z:z.abs().mean())
  for j,c in enumerate(order):
   ax.errorbar(means[c].mean(),j,xerr=means[c].std(),fmt='o',color=BLUE,capsize=2,ms=4);ranks.append(dict(dataset=ds,feature=c,mean_abs_shap=q[c].abs().mean(),fold_mean_abs_shap=means[c].mean(),fold_sd=means[c].std()))
  ax.set_yticks(range(len(order)),[LABELS[c] for c in order]);ax.set_xlabel('Mean absolute SHAP (log odds)' if ds=='adult' else 'Mean absolute SHAP (target units)');ax.tick_params(axis='y',length=0)
 pd.DataFrame(ranks).to_csv(DATA/'shap_importance.csv',index=False);save(fig,'shap')
 fig,axs=panelgrid()
 selected={'house':'GrLivArea','adult':'education-num','nba':'age','students':'lunch'}
 for ax,ds in zip(axs,DS):
  c=selected[ds];q=shap_data[ds].merge(raw_data[ds][['row_id',c]],on='row_id',validate='one_to_one',suffixes=('_shap','_value'))
  x=q[c+'_value'];y=q[c+'_shap']
  if not pd.api.types.is_numeric_dtype(x):
   cats=sorted(x.unique());x=pd.Categorical(x,categories=cats).codes.astype(float);x+=.15*np.sin(np.arange(len(x))*2.399963);ax.set_xticks(range(len(cats)),cats)
  ax.scatter(x,y,s=2,color=BLUE,alpha=.16,rasterized=True,linewidths=0);ax.axhline(0,color=GREY,ls='--',lw=.6);ax.set_xlabel(LABELS[c]);ax.set_ylabel('SHAP (log odds)' if ds=='adult' else 'SHAP (target units)')
 save(fig,'dependence')
 fig,axs=panelgrid()
 for ax,ds in zip(axs,DS):
  q=shap_data[ds];cols=json.loads((DATA/ds/'data_audit.json').read_text())['features'];order=sorted(cols,key=lambda c:q[c].abs().mean())
  for j,c in enumerate(order):
   jitter=.09*np.sin(np.arange(len(q))*2.399963+j)
   ax.scatter(q[c],j+jitter,s=2,alpha=.12,color=BLUE,linewidths=0,rasterized=True)
  ax.axvline(0,color=GREY,ls='--',lw=.6);ax.set_yticks(range(len(order)),[LABELS[c] for c in order]);ax.set_xlabel('SHAP contribution (log odds)' if ds=='adult' else 'SHAP contribution (target units)');ax.tick_params(axis='y',length=0)
 save(fig,'beeswarm')
 for ds in DS:
  q=shap_data[ds];cols=json.loads((DATA/ds/'data_audit.json').read_text())['features'];order=sorted(cols,key=lambda c:q[c].abs().mean())
  fig,ax=plt.subplots(figsize=(7.2,3.3));fig.subplots_adjust(left=.23,right=.97,bottom=.20,top=.93)
  for j,c in enumerate(order):
   jit=.09*np.sin(np.arange(len(q))*2.399963+j);ax.scatter(q[c],j+jit,s=2,alpha=.12,color=BLUE,linewidths=0,rasterized=True)
  ax.axvline(0,color=GREY,ls='--',lw=.6);ax.set_yticks(range(len(order)),[LABELS[c] for c in order]);ax.set_xlabel('SHAP contribution (log odds)' if ds=='adult' else 'SHAP contribution (target units)');ax.tick_params(axis='y',length=0);save(fig,f'{ds}_beeswarm')
 fig,axs=panelgrid();cases=[]
 for ax,ds in zip(axs,DS):
  preds=pd.concat([pd.read_csv(p) for p in (DATA/ds).glob('fold*/predictions.csv')]);preds=preds.query('model=="HGB_grid"').copy();preds['error']=np.abs(preds.actual-preds.predicted);p=preds.sort_values(['error','row_id']).iloc[len(preds)//2];q=shap_data[ds].set_index('row_id').loc[p.row_id];cols=json.loads((DATA/ds/'data_audit.json').read_text())['features'];cols=sorted(cols,key=lambda c:abs(q[c]),reverse=True)
  current=q['base']
  for j,c in enumerate(cols):
   value=q[c];ax.barh(j,abs(value),left=min(current,current+value),height=.6,color=ORANGE if value>0 else BLUE);current+=value
  ax.set_yticks(range(len(cols)),[LABELS[c] for c in cols]);ax.invert_yaxis();ax.axvline(q['base'],color=GREY,lw=.6,ls='--');ax.set_xlabel('Prediction scale (log odds)' if ds=='adult' else 'Prediction scale (target units)');ax.tick_params(axis='y',length=0);cases.append(dict(dataset=ds,row_id=p.row_id,actual=p.actual,predicted=p.predicted,base=q['base'],reconstructed_raw=current))
 pd.DataFrame(cases).to_csv(DATA/'local_cases.csv',index=False);save(fig,'local')
 (OUT/'figure_contract.json').write_text(json.dumps({'archetype':'quantitative grid','figures':CONTRACT,'dimensions_inches':[7.2,6.6],'uncertainty':'mean ± sample SD of three outer folds; folds are not independent replicates','points':'individual outer folds; dependence and beeswarm use every outer-validation row','sample_exclusions':'Adult original test excluded; NBA initial training-only seasons not explained; missing target counts in data audits','tests':'descriptive paired comparisons; no significance tests','source_data':'artifacts/research/*.csv'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--skill-root',required=True);main(p.parse_args().skill_root)
