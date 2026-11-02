"""Source-backed grouped figures for the four-task course experiments."""
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator, FuncFormatter

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/research'
OUT=ROOT/'docs/source/figures/research'
DS=['house','adult','nba','students']
TITLES={'house':'House price','adult':'Adult income','nba':'NBA points','students':'Student mathematics'}
NAVY='#42658B';TEAL='#368F92';VIOLET='#8D77A9';ORANGE='#CF886B';GREY='#9AA1AA';INK='#34404B';LIGHT='#E7EBEF'
LABELS={'OverallQual':'Quality','GrLivArea':'Living area','YearBuilt':'Year built','TotalBsmtSF':'Basement area','GarageCars':'Garage spaces','Neighborhood':'Neighborhood','age':'Age','education-num':'Education','hours-per-week':'Weekly hours','workclass':'Work class','occupation':'Occupation','marital-status':'Marital status','player_height':'Height','player_weight':'Weight','draft_year':'Draft year','gender':'Gender','parental level of education':'Parental education','lunch':'Lunch','test preparation course':'Preparation'}
DISPLAY={'Mean':'Constant','Linear':'Logistic / OLS','Ridge':'Strong L2 / Ridge','Sparse':'L1 / Elastic Net','Tree':'Decision tree','RF':'Random forest','ExtraTrees':'Extra Trees','HGB':'Hist. boosting','KNN':'15-nearest neighbors','RF_grid':'Forest + Grid','RF_tpe':'Forest + TPE','HGB_grid':'Boosting + Grid','HGB_tpe':'Boosting + TPE'}
ABL={'No neighborhood':'No neighborhood','No size':'No area group','No quality':'No quality','No year':'No year','Area + quality':'Only area + quality','No education':'No education','No work':'No work group','No marital':'No marital status','No age':'No age','No physique':'No physique','No draft':'No draft year','No gender':'No gender','No parental education':'No parental education','No lunch':'No lunch','No preparation':'No preparation'}
CMAP=LinearSegmentedColormap.from_list('Contribution',[NAVY,'#EEF0F2',ORANGE])
DENSITY=LinearSegmentedColormap.from_list('Density',['#E8F2F2','#A8CDCE',TEAL,'#244E69'])


def metric(ds,diagnostic=False):return ('log_loss' if diagnostic else 'auc') if ds=='adult' else 'rmse'
def scale(ds):return 1000 if ds=='house' else 1
def units(ds):return 'kUSD' if ds=='house' else 'points'
def ylabel(ds,diagnostic=False):return ('Log loss' if diagnostic else 'ROC AUC') if ds=='adult' else f'RMSE ({units(ds)})'
def shaplabel(ds):return 'SHAP (log odds)' if ds=='adult' else f'SHAP ({units(ds)})'
def display(model,ds):return DISPLAY[model].replace('Logistic / OLS','Logistic' if ds=='adult' else 'OLS').replace('Strong L2 / Ridge','Strong L2' if ds=='adult' else 'Ridge').replace('L1 / Elastic Net','L1 logistic' if ds=='adult' else 'Elastic Net')
def family(model):
 if model=='Mean':return GREY
 if model in ['Linear','Ridge','Sparse']:return VIOLET
 if '_' in model:return TEAL
 if model=='KNN':return ORANGE
 return NAVY

def clean(ax,grid='x'):
 ax.spines[['top','right']].set_visible(False)
 ax.spines[['left','bottom']].set_color('#BDC5CD')
 ax.tick_params(color='#BDC5CD',length=2.5,pad=2)
 ax.set_axisbelow(True)
 if grid:ax.grid(axis=grid,color='#E6E9ED',lw=.5)
 ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
 ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

def paneltitle(ax,text,label=None):
 ax.set_title(text,loc='left',fontsize=13 if ax.figure.get_figwidth()>10 else 8,fontweight='bold',pad=8,color=INK)
 if label:
  ax.text(0,1,label,transform=ax.transAxes+ScaledTranslation(-20/72,9/72,ax.figure.dpi_scale_trans),ha='left',va='bottom',fontsize=13 if ax.figure.get_figwidth()>10 else 9,fontweight='bold',color=INK)

def grid(rows=2,cols=2,slide=False,header=None):
 figsize=(12,4.35) if slide else ((7.2,6.5) if rows==2 else (7.2,8.6))
 fig,axes=plt.subplots(rows,cols,figsize=figsize,squeeze=False)
 fig.subplots_adjust(left=.16 if not slide else .12,right=.975,bottom=.09 if rows==2 else .065,top=.895 if header else .94,wspace=.65 if not slide else .55,hspace=.66 if not slide else .30)
 if header:fig.legend(handles=header,loc='upper center',bbox_to_anchor=(.54,.995),ncol=len(header),frameon=False,handlelength=1.5,columnspacing=1.8)
 return fig,axes

def export(fig,name):
 from audit_panel_alignment import require_matplotlib_panel_alignment
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=str(OUT/f'{name}.alignment.json'),overlay_svg=str(OUT/f'{name}.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
 fig.savefig(OUT/f'{name}.pdf',dpi=600)
 fig.savefig(OUT/f'{name}.svg',dpi=600)
 fig.savefig(OUT/f'{name}.png',dpi=600)
 plt.close(fig)

def prepare_derived_tables():
 """Reconstruct display tables on a fresh run; reuse existing verified tables."""
 for stem in ['metrics','diagnostics','ablations']:
  path=DATA/(stem+'.csv')
  if not path.exists():
   pd.concat([pd.read_csv(p) for p in DATA.glob('*/fold*/'+stem+'.csv')],ignore_index=True).to_csv(path,index=False)
 metrics=pd.read_csv(DATA/'metrics.csv')
 if not (DATA/'ablation_deltas.csv').exists():
  rows=[]
  ablations=pd.read_csv(DATA/'ablations.csv')
  for ds in DS:
   m=metric(ds);q=ablations.query('dataset==@ds').copy()
   full=metrics.query('dataset==@ds and model=="HGB_grid" and partition=="validation"').set_index('fold')[m]
   q['delta']=[full.loc[f]-v if ds=='adult' else v-full.loc[f] for f,v in zip(q.fold,q[m])]
   rows.append(q)
  pd.concat(rows,ignore_index=True).to_csv(DATA/'ablation_deltas.csv',index=False)
 ranks=[];cases=[]
 for ds in DS:
  shap=pd.concat([pd.read_csv(p) for p in (DATA/ds).glob('fold*/shap.csv')],ignore_index=True)
  cols=json.loads((DATA/ds/'data_audit.json').read_text())['features']
  folds=shap.groupby('fold')[cols].agg(lambda z:z.abs().mean())
  for c in cols:ranks.append(dict(dataset=ds,feature=c,mean_abs_shap=shap[c].abs().mean(),fold_mean_abs_shap=folds[c].mean(),fold_sd=folds[c].std()))
  preds=pd.concat([pd.read_csv(p) for p in (DATA/ds).glob('fold*/predictions.csv')],ignore_index=True)
  preds=preds.query('model=="HGB_grid"').copy();preds['error']=np.abs(preds.actual-preds.predicted)
  case=preds.sort_values(['error','row_id']).iloc[len(preds)//2];values=shap.set_index('row_id').loc[case.row_id]
  cases.append(dict(dataset=ds,row_id=case.row_id,actual=case.actual,predicted=case.predicted,base=values['base'],reconstructed_raw=values['base']+sum(values[c] for c in cols)))
 if not (DATA/'shap_importance.csv').exists():pd.DataFrame(ranks).to_csv(DATA/'shap_importance.csv',index=False)
 if not (DATA/'local_cases.csv').exists():pd.DataFrame(cases).to_csv(DATA/'local_cases.csv',index=False)

class Figures:
 def __init__(self):
  self.metrics=pd.read_csv(DATA/'metrics.csv');self.diags=pd.read_csv(DATA/'diagnostics.csv');self.abl=pd.read_csv(DATA/'ablation_deltas.csv');self.shap={};self.raw={};self.cols={}
  for ds in DS:
   self.shap[ds]=pd.concat([pd.read_csv(p) for p in sorted((DATA/ds).glob('fold*/shap.csv'))],ignore_index=True)
   self.raw[ds]=pd.concat([pd.read_csv(p) for p in sorted((DATA/ds).glob('fold*/shap_features.csv'))],ignore_index=True)
   self.cols[ds]=json.loads((DATA/ds/'data_audit.json').read_text())['features']
  self.cases=pd.read_csv(DATA/'local_cases.csv')
  self.density_counts={}

 def benchmark(self,ax,ds):
  q=self.metrics.query('dataset==@ds and partition=="validation"');m=metric(ds)
  summ=q.groupby('model')[m].agg(['mean','std']).sort_values('mean',ascending=ds!='adult')
  clean(ax)
  for j,(name,row) in enumerate(summ.iterrows()):
   color=family(name);mean=row['mean']/scale(ds);sd=row['std']/scale(ds)
   if np.isclose(row['mean'],summ['mean'].iloc[0],rtol=0,atol=1e-9):ax.axhspan(j-.43,j+.43,color='#EAF3F2',zorder=0)
   ax.errorbar(mean,j,xerr=sd,fmt='D',mfc='white',mec=color,mew=1,ecolor=color,ms=4,capsize=2,lw=1.4,zorder=4)
   for f,(marker,offset) in enumerate([('o',-.13),('s',0),('^',.13)]):
    val=q.loc[(q.model==name)&(q.fold==f),m].iloc[0]/scale(ds)
    ax.plot(val,j+offset,marker=marker,color=color,ms=2.5,alpha=.85,lw=0,zorder=5)
  ax.set_yticks(range(len(summ)),[display(n,ds) for n in summ.index]);ax.tick_params(axis='y',length=0)
  ax.set_ylim(len(summ)-.35,-.65);ax.set_xlabel(ylabel(ds));ax.margins(x=.1)

 def search(self,ax,ds):
  clean(ax,grid='y')
  for method,color,marker in [('grid',TEAL,'o'),('tpe',VIOLET,'s')]:
   runs=[]
   for p in sorted((DATA/ds).glob(f'fold*/HGB_{method}_search.json')):
    hist=json.loads(p.read_text());v=np.maximum.accumulate([r['score'] for r in hist['rows']]);runs.append(v if ds=='adult' else -v/scale(ds))
   a=np.array(runs);x=np.arange(1,a.shape[1]+1);mean=a.mean(0);sd=a.std(0,ddof=1)
   ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.13,step='post',lw=0)
   ax.step(x,mean,where='post',color=color,lw=1.4)
   ax.plot(x,mean,marker=marker,color=color,ms=2.5,lw=0)
  ax.set_xticks([1,4,8,12]);ax.set_xlabel('Candidate evaluations');ax.set_ylabel('Inner '+ylabel(ds));ax.margins(y=.18)

 def gains(self,ax,ds):
  clean(ax)
  base=self.metrics.query('dataset==@ds and partition=="validation"');m=metric(ds)
  for j,(model,label) in enumerate([('HGB_grid','Boosting + Grid'),('HGB_tpe','Boosting + TPE'),('RF_grid','Forest + Grid'),('RF_tpe','Forest + TPE')]):
   tuned=base.loc[base.model==model].set_index('fold')[m];preset=base.loc[base.model==model.split('_')[0]].set_index('fold')[m]
   vals=(tuned-preset if ds=='adult' else preset-tuned)/scale(ds)
   if ds=='adult':vals=vals*1000
   color=TEAL if vals.mean()>=0 else ORANGE
   ax.barh(j,vals.mean(),height=.5,color=color,alpha=.35,zorder=2)
   ax.errorbar(vals.mean(),j,xerr=vals.std(),fmt='D',color=color,mfc='white',ms=4,capsize=2,zorder=4,lw=1.1)
   ax.scatter(vals,np.array([j-.12,j,j+.12]),c=color,s=9,zorder=5)
  ax.axvline(0,color=INK,lw=.65,ls='--');ax.set_yticks(range(4),['Boosting + Grid','Boosting + TPE','Forest + Grid','Forest + TPE']);ax.set_ylim(3.6,-.6);ax.tick_params(axis='y',length=0)
  ax.set_xlabel('AUC gain (×0.001)' if ds=='adult' else f'RMSE reduction ({units(ds)})');ax.margins(x=.2)

 def diagnostic(self,ax,ds,kind):
  clean(ax,grid='y');q=self.diags.query('dataset==@ds and kind==@kind').copy();m=metric(ds,True);x='fraction' if kind=='learning' else 'depth'
  if kind=='complexity':q.loc[q.depth==-1,'depth']=12
  for part,color,marker in [('train',GREY,'o'),('validation',NAVY,'s')]:
   v=q.loc[q.partition==part].groupby(x)[m].agg(['mean','std']).sort_index();a=v['mean']/scale(ds);sd=v['std'].fillna(0)/scale(ds)
   ax.fill_between(v.index,a-sd,a+sd,color=color,alpha=.12,lw=0)
   ax.plot(v.index,a,color=color,lw=1.4,marker=marker,ms=3,mfc='white' if part=='train' else color)
  ax.set_ylabel(ylabel(ds,True));ax.set_xlabel('Training fraction' if kind=='learning' else 'Maximum tree depth');ax.margins(y=.15)
  if kind=='complexity':ax.set_xticks([1,2,4,8,12],['1','2','4','8','Full'])
  else:ax.set_xticks([.25,.5,.75,1])

 def importance(self,ax,ds):
  clean(ax);q=self.shap[ds];cols=self.cols[ds];folds=q.groupby('fold')[cols].agg(lambda z:z.abs().mean())/scale(ds);order=folds.mean().sort_values(ascending=False).index
  for j,c in enumerate(order):
   color=TEAL if j==0 else NAVY;mean=folds[c].mean();sd=folds[c].std()
   ax.barh(j,mean,height=.58,color=color,alpha=.8,zorder=2)
   ax.errorbar(mean,j,xerr=sd,fmt='none',ecolor=INK,capsize=2,lw=.8,zorder=3)
  ax.set_yticks(range(len(order)),[LABELS[c] for c in order]);ax.set_ylim(len(order)-.35,-.65);ax.tick_params(axis='y',length=0);ax.set_xlabel('Mean |SHAP| (log odds)' if ds=='adult' else f'Mean |SHAP| ({units(ds)})');ax.set_xlim(left=0);ax.margins(x=.2)

 def ablation(self,ax,ds):
  clean(ax);q=self.abl.query('dataset==@ds');order=q.groupby('variant').delta.mean().sort_values(ascending=False).index
  for j,v in enumerate(order):
   values=q.loc[q.variant==v].sort_values('fold').delta/scale(ds);color=ORANGE if values.mean()>=0 else TEAL
   ax.barh(j,values.mean(),height=.55,color=color,alpha=.5,zorder=2)
   ax.errorbar(values.mean(),j,xerr=values.std(),fmt='D',ms=3.5,mfc='white',color=INK,capsize=2,lw=.8,zorder=3)
   ax.scatter(values,j+np.linspace(-.12,.12,len(values)),s=9,color=color,zorder=4)
  ax.axvline(0,ls='--',color=INK,lw=.6);ax.set_yticks(range(len(order)),[ABL[v] for v in order]);ax.set_ylim(len(order)-.35,-.65);ax.tick_params(axis='y',length=0)
  ax.set_xlabel('AUC decrease' if ds=='adult' else f'RMSE increase ({units(ds)})');ax.margins(x=.18)

 def beeswarm(self,ax,ds):
  clean(ax);q=self.shap[ds];order=sorted(self.cols[ds],key=lambda c:q[c].abs().mean(),reverse=True)
  for j,c in enumerate(order):
   vals=q[c].to_numpy()/scale(ds);width=max(abs(vals.min()),abs(vals.max()))
   if np.ptp(vals)>1e-12:
    violin=ax.violinplot([vals],positions=[j],vert=False,widths=.74,showextrema=False,points=100)
    for body in violin['bodies']:body.set_facecolor('#DCE5EB');body.set_edgecolor('none');body.set_alpha(.65)
   # Deterministic jitter changes display positions only; every SHAP value remains.
   jit=.18*np.sin(np.arange(len(vals))*2.399963+j)
   ax.scatter(vals,j+jit,c=vals,cmap=CMAP,vmin=-width if width else -1,vmax=width if width else 1,s=2.1,alpha=.45,rasterized=True,linewidths=0)
  ax.axvline(0,color=INK,lw=.65,ls='--');ax.set_yticks(range(len(order)),[LABELS[c] for c in order]);ax.set_ylim(len(order)-.35,-.65);ax.tick_params(axis='y',length=0);ax.set_xlabel(shaplabel(ds));ax.margins(x=.08)

 def dependence(self,ax,ds):
  clean(ax,grid='y');c={'house':'GrLivArea','adult':'education-num','nba':'age','students':'lunch'}[ds]
  q=self.shap[ds][['row_id',c]].merge(self.raw[ds][['row_id',c]],on='row_id',validate='one_to_one',suffixes=('_shap','_value'));x=q[c+'_value'];y=q[c+'_shap']/scale(ds)
  if pd.api.types.is_numeric_dtype(x):
   # mincnt=1 and logarithmic bin colour retain every record, including rare bins.
   h=ax.hexbin(x,y,gridsize=35,mincnt=1,bins='log',cmap=DENSITY,linewidths=0,rasterized=True)
   assert int(np.sum(h.get_array()))==len(q)
   self.density_counts[ds]={'input_rows':len(q),'hexbin_count_sum':int(np.sum(h.get_array())),'gridsize':35,'colour':'log count'}
  else:
   cats=sorted(x.unique());groups=[y[x==cat].to_numpy() for cat in cats]
   vio=ax.violinplot(groups,positions=range(len(cats)),widths=.7,showextrema=False,points=100)
   for body,color in zip(vio['bodies'],[TEAL,VIOLET]):body.set_facecolor(color);body.set_alpha(.22);body.set_edgecolor('none')
   for j,(cat,vals) in enumerate(zip(cats,groups)):
    ax.scatter(j+.12*np.sin(np.arange(len(vals))*2.399963),vals,s=2.5,alpha=.24,color=[TEAL,VIOLET][j],rasterized=True,linewidths=0)
    quart=np.quantile(vals,[.25,.5,.75]);ax.plot([j,j],quart[[0,2]],color=INK,lw=2);ax.plot(j,quart[1],'o',color='white',mec=INK,ms=3)
   ax.set_xticks(range(len(cats)),cats)
   self.density_counts[ds]={'input_rows':len(q),'displayed_points':sum(map(len,groups)),'categorical':'all records + median and IQR'}
  ax.axhline(0,ls='--',lw=.6,color=GREY);ax.set_xlabel(LABELS[c]);ax.set_ylabel(shaplabel(ds));ax.margins(y=.1)

 def local(self,ax,ds):
  clean(ax);case=self.cases.query('dataset==@ds').iloc[0];q=self.shap[ds].set_index('row_id').loc[case.row_id];cols=sorted(self.cols[ds],key=lambda c:abs(q[c]),reverse=True)
  base=q['base']/scale(ds);current=base
  ax.barh(0,base,left=0,height=.55,color=GREY,alpha=.65,zorder=3)
  for j,c in enumerate(cols,1):
   value=q[c]/scale(ds);colour=ORANGE if value>=0 else NAVY
   ax.plot([current,current],[j-1+.28,j-.28],ls=':',color=GREY,lw=.8,zorder=2)
   ax.barh(j,abs(value),left=min(current,current+value),height=.55,color=colour,zorder=3)
   current+=value
  j=len(cols)+1;ax.plot([current,current],[j-1+.28,j-.28],ls=':',color=GREY,lw=.8);ax.barh(j,current,left=0,height=.55,color=TEAL,zorder=3)
  ax.set_yticks(range(len(cols)+2),['Base']+[LABELS[c] for c in cols]+['Prediction']);ax.set_ylim(len(cols)+1.65,-.65);ax.tick_params(axis='y',length=0);ax.set_xlabel('Raw output (log odds)' if ds=='adult' else f'Prediction ({units(ds)})');ax.set_xlim(left=min(0,ax.get_xlim()[0]));ax.margins(x=.12)
  assert abs(current-case.reconstructed_raw/scale(ds))<1e-6

 def draw_topics(self):
  families=[Patch(color=VIOLET,label='Linear / regularized'),Patch(color=NAVY,label='Trees'),Patch(color=TEAL,label='Tuned trees'),Patch(color=ORANGE,label='Neighbors')]
  legends={'benchmark':families,'search':[Line2D([],[],color=TEAL,marker='o',label='Grid'),Line2D([],[],color=VIOLET,marker='s',label='TPE')],
   'learning':[Line2D([],[],color=GREY,marker='o',mfc='white',label='Train'),Line2D([],[],color=NAVY,marker='s',label='Validation')],
   'complexity':[Line2D([],[],color=GREY,marker='o',mfc='white',label='Train'),Line2D([],[],color=NAVY,marker='s',label='Validation')],
   'beeswarm':[Patch(color=NAVY,label='Lower prediction'),Patch(color=ORANGE,label='Higher prediction')],
   'local':[Patch(color=NAVY,label='Negative contribution'),Patch(color=ORANGE,label='Positive contribution'),Patch(color=TEAL,label='Prediction')]}
  funcs={'benchmark':self.benchmark,'search':self.search,'learning':lambda a,d:self.diagnostic(a,d,'learning'),'complexity':lambda a,d:self.diagnostic(a,d,'complexity'),'shap':self.importance,'ablation':self.ablation,'beeswarm':self.beeswarm,'dependence':self.dependence,'local':self.local}
  for name,func in funcs.items():
   with plt.rc_context({'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':7}):
    fig,axs=grid(header=legends.get(name))
    for i,(ax,ds) in enumerate(zip(axs.flat,DS)):func(ax,ds);paneltitle(ax,TITLES[ds],string.ascii_lowercase[i])
    export(fig,name)
   with plt.rc_context({'font.size':14,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12}):
    if name=='benchmark':
     for ds in DS:
      fig,ax=plt.subplots(figsize=(8.8,4.4));fig.subplots_adjust(left=.25,right=.97,bottom=.13,top=.9)
      self.benchmark(ax,ds);paneltitle(ax,TITLES[ds]);export(fig,f'benchmark_{ds}_slide')
     continue
    fig,axs=plt.subplots(2,2,figsize=(12,5.0))
    fig.subplots_adjust(left=.15,right=.98,bottom=.16,top=.82,wspace=.65,hspace=.85)
    if legends.get(name):fig.legend(handles=legends[name],loc='upper center',bbox_to_anchor=(.55,.995),ncol=len(legends[name]))
    for i,(ax,ds) in enumerate(zip(axs.flat,DS)):
     func(ax,ds);paneltitle(ax,TITLES[ds],string.ascii_lowercase[i])
     if name=='local':ax.tick_params(axis='y',labelsize=10)
    export(fig,name+'_slide')

 def draw_groups(self):
  pairs=[('search_validation',self.search,self.gains,'Inner search','Outer validation'),('fit_diagnostics',lambda a,d:self.diagnostic(a,d,'learning'),lambda a,d:self.diagnostic(a,d,'complexity'),'Learning curve','Model complexity'),('explanation_ablation',self.importance,self.ablation,'Fixed-model contribution','Retrain after removing fields'),('shap_structure',self.beeswarm,self.dependence,'Contribution distribution','Dependence in held-out records')]
  for name,left,right,lt,rt in pairs:
   with plt.rc_context({'font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5}):
    fig,axs=grid(rows=4,cols=2)
    fig.subplots_adjust(top=.90,hspace=.78)
    fig.text(.31,.982,lt,ha='center',va='top',weight='bold',fontsize=9,color=INK);fig.text(.805,.982,rt,ha='center',va='top',weight='bold',fontsize=9,color=INK)
    if name=='search_validation':fig.legend(handles=[Line2D([],[],color=TEAL,marker='o',label='Grid'),Line2D([],[],color=VIOLET,marker='s',label='TPE'),Patch(color=TEAL,alpha=.5,label='Gain'),Patch(color=ORANGE,alpha=.5,label='Loss')],loc='upper center',bbox_to_anchor=(.55,.955),ncol=4,fontsize=7)
    if name=='fit_diagnostics':fig.legend(handles=[Line2D([],[],color=GREY,marker='o',mfc='white',label='Train'),Line2D([],[],color=NAVY,marker='s',label='Validation')],loc='upper center',bbox_to_anchor=(.55,.955),ncol=2,fontsize=7)
    if name=='shap_structure':fig.legend(handles=[Patch(color=NAVY,label='Negative SHAP'),Patch(color=ORANGE,label='Positive SHAP'),Patch(color=TEAL,label='Darker density = more records')],loc='upper center',bbox_to_anchor=(.55,.955),ncol=3,fontsize=6.5)
    for i,ds in enumerate(DS):
     left(axs[i,0],ds);right(axs[i,1],ds)
     for j in range(2):paneltitle(axs[i,j],TITLES[ds],string.ascii_lowercase[2*i+j])
    export(fig,name)
  (OUT/'density_integrity.json').write_text(json.dumps(self.density_counts,indent=2))

 def legacy(self):
  table=pd.read_csv(ROOT/'tables/model_comparison.csv')
  roc=pd.read_csv(ROOT/'tables/figure_adult_roc.csv') if (ROOT/'tables/figure_adult_roc.csv').exists() else pd.read_csv(ROOT/'codes/verification/figure_adult_roc.csv')
  def bars(ax,ds):
   q=table.query('dataset==@ds and role=="primary"');base=q.baseline_rmse.iloc[0]/scale(ds);vals=[base]+[q.loc[q.algorithm==m,'rmse'].iloc[0]/scale(ds) for m in ['linear','random_forest']]
   clean(ax,grid=None);colors=[GREY,VIOLET,TEAL];ax.barh(range(3),vals,color=colors,height=.56)
   ax.set_yticks(range(3),['Train mean','OLS','Random forest']);ax.set_ylim(2.65,-.65);ax.tick_params(axis='y',length=0);ax.set_xlabel(ylabel(ds));ax.set_xlim(0,max(vals)*1.22)
   for j,v in enumerate(vals):ax.text(v+max(vals)*.025,j,f'{v:.2f}',va='center',fontsize=11 if ax.figure.get_figwidth()>7.5 else 7,color=INK)
  def curve(ax,ds):
   clean(ax,grid=None)
   for algo,color in [('logistic',VIOLET),('random_forest',TEAL)]:
    q=roc.loc[roc.algorithm==algo];ax.plot(q.fpr,q.tpr,color=color,lw=1.5,label='Logistic' if algo=='logistic' else 'Random forest')
   ax.plot([0,1],[0,1],color=GREY,ls='--',lw=.7);ax.set(xlim=(0,1),ylim=(0,1),xlabel='False positive rate',ylabel='True positive rate');ax.set_yticks([.25,.5,.75,1]);ax.legend(loc='lower right',fontsize=8 if ax.figure.get_figwidth()>7.5 else 6.5)
  fig,axs=grid();funcs=[bars,curve,bars,bars]
  for i,(ax,ds,func) in enumerate(zip(axs.flat,DS,funcs)):func(ax,ds);paneltitle(ax,TITLES[ds],string.ascii_lowercase[i])
  export(fig,'legacy_overview')
  for ds in DS:
   with plt.rc_context({'font.size':12,'xtick.labelsize':11,'ytick.labelsize':12,'axes.labelsize':12}):
    fig,ax=plt.subplots(figsize=(8,3.5));fig.subplots_adjust(left=.23,right=.97,bottom=.19,top=.92)
    (curve if ds=='adult' else bars)(ax,ds);export(fig,'legacy_'+ds+'_slide')


def main(skill):
 sys.path.insert(0,str(Path(skill)/'scripts'))
 OUT.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'svg.fonttype':'none','font.size':7,'axes.labelsize':7,'axes.linewidth':.7,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
 prepare_derived_tables()
 figures=Figures();figures.draw_topics();figures.draw_groups();figures.legacy()
 (OUT/'figure_contract.json').write_text((ROOT/'codes/verification/research/figure_refresh/design_contract.json').read_text())
 print('Redesigned component figures, paired composites, and separate slide views from saved results.')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--skill-root',required=True);main(parser.parse_args().skill_root)
