"""Grouped figures for the expanded benchmark, retaining every fitted flow."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'artifacts/expanded'
OUT=ROOT/'docs/source/figures/expanded'
DS=['house','adult','nba','students']
TITLE={'house':'House price','adult':'Adult income','nba':'NBA points','students':'Student mathematics','nba_sequence':'NBA: next season'}
LABEL={'Mean':'Constant','Persistence':'Last-season points','Linear':'Unpenalized linear','Ridge':'L2 linear','Sparse':'L1 / elastic net','Tree':'Decision tree','RF':'Random forest','ExtraTrees':'Extra Trees','HGB':'Hist. boosting','KNN':'Nearest neighbors','AdaBoost':'AdaBoost','SVM':'Linear SVM','XGB':'XGBoost','LightGBM':'LightGBM','CatBoost':'CatBoost','MLP':'MLP','ResMLP':'Residual MLP','DCN':'Deep & Cross','GatedMLP':'Gated MLP','FeatureTransformer':'Feature Transformer','ELM':'ELM','RBFNet':'RBF network','RNN':'RNN','GRU':'GRU','BiGRU':'BiGRU','LSTM':'LSTM','BiLSTM':'BiLSTM','TCN':'Causal TCN','SeqTransformer':'Sequence Transformer'}
FIELDS={'OverallQual':'Quality','GrLivArea':'Living area','YearBuilt':'Year built','TotalBsmtSF':'Basement area','GarageCars':'Garage spaces','Neighborhood':'Neighborhood','age':'Age','education-num':'Education','hours-per-week':'Weekly hours','workclass':'Work class','occupation':'Occupation','marital-status':'Marital status','player_height':'Height','player_weight':'Weight','draft_year':'Draft year','gender':'Gender','parental level of education':'Parental education','lunch':'Lunch','test preparation course':'Preparation'}
VARIANTS={'house':['No neighborhood','No size','No quality','No year','Area + quality'],
          'adult':['No education','No work','No marital','No age'],'nba':['No age','No physique','No draft'],
          'students':['No gender','No parental education','No lunch','No preparation'],
          'nba_sequence':['No age','No physique','No past production','No availability','No oldest season']}
VLABEL={'No neighborhood':'No neighborhood','No size':'No area group','No quality':'No quality','No year':'No year','Area + quality':'Only area + quality','No education':'No education','No work':'No work group','No marital':'No marital status','No age':'No age','No physique':'No height / weight','No draft':'No draft year','No gender':'No gender','No parental education':'No parental education','No lunch':'No lunch','No preparation':'No preparation','No past production':'No points / rebounds / assists','No availability':'No games played','No oldest season':'Two-season window'}
TEAL,VIOLET,NAVY,ORANGE,GREY,INK='#368F92','#8D77A9','#42658B','#CF886B','#9AA1AA','#34404B'
CMAP=LinearSegmentedColormap.from_list('Paired change',[NAVY,'#FAFBFC',ORANGE])


def label(model,ds):
    if ds=='adult':
        return {'Linear':'Unpenalized logit','Ridge':'L2 logistic','Sparse':'Elastic-net logistic'}.get(model,LABEL[model])
    return LABEL[model]


def save(fig,name,axes):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=list(np.asarray(axes,dtype=object).flat),
        json_out=str(OUT/f'{name}.alignment.json'),overlay_svg=str(OUT/f'{name}.alignment.svg'),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(OUT/f'{name}.pdf',dpi=600)
    fig.savefig(OUT/f'{name}.svg',dpi=600)
    fig.savefig(OUT/f'{name}.png',dpi=600)
    plt.close(fig)


def clean(ax):
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    ax.tick_params(axis='y',length=0,pad=5)
    ax.grid(axis='x',color='#E7EBEF',lw=.6,zorder=0)


class Figures:
    def __init__(self):
        self.summary=pd.read_csv(DATA/'summary.csv')
        self.metrics=pd.read_csv(DATA/'metrics.csv')
        self.abl=pd.read_csv(DATA/'ablation_deltas.csv')
        self.shap=pd.read_csv(DATA/'shap_importance.csv')
        self.curves=pd.read_csv(DATA/'training_curves.csv')
        self.diagnostics=pd.read_csv(DATA/'diagnostics.csv')

    def order(self,ds):
        q=self.summary.query('dataset == @ds and variant == "Full" and method in ["grid","fixed"]')
        return q.sort_values('mean',ascending=ds!='adult',kind='stable').model.tolist()

    def benchmark(self,ax,ds,short=False):
        order=self.order(ds)
        if short:
            order=[m for m in order if m not in ['Mean','Persistence']][:7]+['Mean']
        scale=1000 if ds=='house' else 1
        q=self.summary.query('dataset == @ds and variant == "Full"')
        for method,color,offset,marker in [('grid',TEAL,-.14,'o'),('tpe',VIOLET,.14,'s'),('fixed',GREY,0,'D')]:
            g=q.query('method == @method').set_index('model')
            for i,model in enumerate(order):
                if model not in g.index:continue
                row=g.loc[model]
                ratio=float(plt.rcParams['font.size'])/10
                ax.errorbar(row['mean']/scale,i+offset,xerr=row.sd/scale,fmt=marker,color=color,ms=3.5*ratio,capsize=2*ratio,elinewidth=.8,lw=.8,zorder=3)
        ax.set_yticks(np.arange(len(order)),[label(m,ds) for m in order])
        ax.set_ylim(len(order)-.5,-.5)
        ax.set_xlabel('ROC AUC' if ds=='adult' else 'RMSE (kUSD)' if ds=='house' else 'RMSE (points)')
        clean(ax)

    def ablation(self,ax,ds,models=None):
        order=models if models is not None else [m for m in self.order(ds) if m not in ['Mean','Persistence']]
        variants=VARIANTS[ds]
        q=self.abl.query('dataset == @ds').set_index(['model','variant','fold'])
        values=np.array([[q.loc[(model,variant,fold),'loss_percent'] for variant in variants for fold in [0,1,2]] for model in order])
        vmax=max(float(np.abs(values).max()),.01)
        image=ax.imshow(values,cmap=CMAP,norm=TwoSlopeNorm(0,vmin=-vmax,vmax=vmax),aspect='auto',interpolation='nearest')
        ax.set_yticks(np.arange(len(order)),[label(m,ds) for m in order])
        compact={'No neighborhood':'No\nneighb.','No size':'No\narea','No quality':'No\nquality','No year':'No\nyear','Area + quality':'Only area\n+ quality','No education':'No\neducation','No work':'No\nwork','No marital':'No marital\nstatus','No age':'No\nage','No physique':'No height\n/ weight','No draft':'No draft\nyear','No gender':'No\ngender','No parental education':'No parental\neducation','No lunch':'No\nlunch','No preparation':'No\npreparation'}
        ax.set_xticks(np.arange(len(variants))*3+1,[compact.get(v,VLABEL[v]) for v in variants],rotation=0,ha='center',rotation_mode='anchor')
        ax.tick_params(length=0)
        for edge in np.arange(len(variants)-1)*3+2.5:ax.axvline(edge,color='white',lw=1.8)
        for edge in np.arange(len(order)-1)+.5:ax.axhline(edge,color='white',lw=.5)
        for spine in ax.spines.values():spine.set_visible(False)
        return image

    def local(self,ax,ds):
        frames=[]
        for fold in range(3):
            folder=DATA/ds/f'fold{fold}'/'Full/HGB_grid'
            q=pd.read_csv(folder/'shap.csv').merge(pd.read_csv(folder/'predictions.csv'),on='row_id',validate='one_to_one')
            q['error']=abs(q.actual-q.predicted);frames.append(q)
        q=pd.concat(frames,ignore_index=True).sort_values(['error','row_id'],kind='stable')
        row=q.iloc[len(q)//2]
        cols=[c for c in q.columns if c in FIELDS]
        cols=sorted(cols,key=lambda col:abs(row[col]),reverse=True)
        scale=1000 if ds=='house' else 1
        current=row['base']/scale
        values=[('Base',0,current,GREY)]
        for col in cols:
            effect=row[col]/scale
            values.append((FIELDS[col],min(current,current+effect),abs(effect),ORANGE if effect>=0 else NAVY))
            current+=effect
        values.append(('Prediction',min(0,current),abs(current),TEAL))
        for i,(name,start,width,color) in enumerate(values):
            ax.barh(i,width,left=start,color=color,height=.6)
        ax.set_yticks(range(len(values)),[v[0] for v in values]);ax.invert_yaxis()
        ax.axvline(0,color=GREY,lw=.7)
        ax.set_xlabel('Model output (log odds)' if ds=='adult' else 'Predicted price (kUSD)')
        clean(ax)
        return {'dataset':ds,'row_id':row.row_id,'actual':float(row.actual),'predicted':float(row.predicted),'base':float(row['base']),'raw_prediction':float(row.raw_prediction),'selection':'median absolute prediction error across all three outer validation folds'}

    def holdout(self,axes,short=False):
        folder=DATA/'holdout_extension'
        scores=pd.read_csv(folder/'metrics.csv').set_index('model').sort_values('cv_auc',ascending=False,kind='stable')
        primary=json.loads((folder/'selection.json').read_text())['chosen_before_test_prediction']
        pairs=pd.read_csv(folder/'paired_differences.csv').set_index('comparator')
        references=[m for m in ['HGB','XGB','LightGBM','CatBoost','MLP','FeatureTransformer'] if m!=primary]
        order=scores.index.tolist()
        if short:order=list(dict.fromkeys([primary]+references+['Mean']))
        ax,bx=axes
        for i,model in enumerate(order):
            r=scores.loc[model]
            ax.errorbar(r.auc,i,xerr=[[r.auc-r.auc_ci_low],[r.auc_ci_high-r.auc]],fmt='D' if model==primary else 'o',
                        color=TEAL if model==primary else GREY,ms=3.5,capsize=2,elinewidth=.8)
        ax.set_yticks(range(len(order)),[label(m,'adult') for m in order]);ax.invert_yaxis()
        ax.set_xlabel('Test ROC AUC');ax.set_xlim(.48,.93);clean(ax)
        for i,model in enumerate(references):
            r=pairs.loc[model]
            bx.errorbar(r.auc_difference*1000,i,xerr=[[(r.auc_difference-r.ci_low)*1000],[(r.ci_high-r.auc_difference)*1000]],fmt='o',color=TEAL,ms=4,capsize=3,elinewidth=1)
        bx.set_yticks(range(len(references)),[label(m,'adult') for m in references]);bx.set_ylim(len(references)-.5,-.5)
        bx.axvline(0,color=GREY,ls='--',lw=.8);bx.set_xlabel('AUC difference (0.001 units)');bx.margins(x=.15);clean(bx)

    def architecture(self,ax):
        controls={'MLP':'No activations','ResMLP':'No residuals','DCN':'No cross branch','GatedMLP':'No gates','FeatureTransformer':'No attention'}
        q=self.abl.set_index(['dataset','model','variant','fold'])
        values=np.array([[q.loc[(ds,model,variant,fold),'loss_percent'] for ds in DS for fold in [0,1,2]] for model,variant in controls.items()])
        vmax=max(np.abs(values).max(),.01)
        image=ax.imshow(values,cmap=CMAP,norm=TwoSlopeNorm(0,vmin=-vmax,vmax=vmax),aspect='auto',interpolation='nearest')
        ax.set_yticks(range(len(controls)),list(controls.values()))
        ax.set_xticks(np.arange(4)*3+1,[TITLE[d] for d in DS])
        ax.tick_params(length=0)
        for edge in [2.5,5.5,8.5]:ax.axvline(edge,color='white',lw=2)
        for edge in np.arange(4)+.5:ax.axhline(edge,color='white',lw=1)
        for spine in ax.spines.values():spine.set_visible(False)
        return image

    def training(self,ax,ds):
        for model,color in [('ResMLP',TEAL),('FeatureTransformer',VIOLET)]:
            q=self.curves.query('dataset == @ds and model == @model and variant == "Full" and method == "grid"')
            for part,style in [('train','--'),('validation','-')]:
                col=part+'_display'
                g=q.loc[q[col].notna()].groupby('epoch')[col].agg(['mean','std'])
                scale=1000 if ds=='house' else 1
                ax.plot(g.index,g['mean']/scale,ls=style,color=color,lw=1.2)
                ax.fill_between(g.index,(g['mean']-g['std'])/scale,(g['mean']+g['std'])/scale,color=color,alpha=.08)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Log loss' if ds=='adult' else 'RMSE (kUSD)' if ds=='house' else 'RMSE (points)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=4))
        clean(ax)

    def diagnostic(self,ax,ds,kind):
        metric='log_loss' if ds=='adult' else 'rmse'
        scale=1000 if ds=='house' else 1
        order=[.25,.5,.75,1.] if kind=='learning' else [1.,2.,4.,8.,-1.]
        for part,color,marker in [('train',GREY,'o'),('validation',TEAL,'s')]:
            q=self.diagnostics.query('dataset == @ds and kind == @kind and partition == @part').groupby('value')[metric].agg(['mean','std']).loc[order]
            x=np.array(order) if kind=='learning' else np.arange(5)
            ax.plot(x,q['mean']/scale,color=color,marker=marker,ms=2.7,lw=1,markerfacecolor='white' if part=='train' else color)
            ax.fill_between(x,(q['mean']-q['std'])/scale,(q['mean']+q['std'])/scale,color=color,alpha=.10)
        ax.set_ylabel('Log loss' if ds=='adult' else 'RMSE (kUSD)' if ds=='house' else 'RMSE (points)')
        if kind=='learning':ax.set_xticks(order);ax.set_xlabel('Training fraction')
        else:ax.set_xticks(range(5),['1','2','4','8','Full']);ax.set_xlabel('Tree depth')
        clean(ax)

    def importance(self,ax,ds):
        q=self.shap.query('dataset == @ds').groupby('field').mean_absolute_shap.agg(['mean','std']).sort_values('mean')
        scale=1000 if ds=='house' else 1
        ax.barh(np.arange(len(q)),q['mean']/scale,xerr=q['std']/scale,color=NAVY,height=.62,error_kw={'lw':.8,'capsize':2})
        ax.set_yticks(np.arange(len(q)),[FIELDS.get(f,f) for f in q.index])
        ax.set_xlabel('Mean |SHAP| (log odds)' if ds=='adult' else 'Mean |SHAP| (kUSD)' if ds=='house' else 'Mean |SHAP| (points)')
        clean(ax)


def build(skill):
    sys.path.insert(0,str(Path(skill)/'scripts'))
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],
                         'font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,
                         'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7,
                         'legend.frameon':False,'savefig.facecolor':'white','text.color':INK,'axes.labelcolor':INK})
    figures=Figures()
    legend=[Line2D([],[],color=TEAL,marker='o',ls='',label='Grid'),Line2D([],[],color=VIOLET,marker='s',ls='',label='TPE'),Line2D([],[],color=GREY,marker='D',ls='',label='Fixed reference')]
    for name,size,font,short in [('benchmark',(7.2,6.65),7.2,False),('deck_benchmark',(12,4.8),11,True)]:
        with plt.rc_context({'font.size':font,'axes.labelsize':font,'xtick.labelsize':font,'ytick.labelsize':font,'legend.fontsize':font+1}):
            fig,axes=plt.subplots(2,2,figsize=size)
            fig.subplots_adjust(left=.20,right=.98,bottom=.12,top=.84 if short else .88,wspace=.72,hspace=.72 if short else .50)
            fig.legend(handles=legend,loc='upper center',ncol=3,bbox_to_anchor=(.56,.99))
            for i,(ax,ds) in enumerate(zip(axes.flat,DS)):
                figures.benchmark(ax,ds,short)
                ax.set_title(f'{chr(97+i)}   {TITLE[ds]}',loc='left',fontweight='bold',fontsize=font+1,pad=11)
            save(fig,name,axes)
    with plt.rc_context({'font.size':7.2,'xtick.labelsize':6.5,'ytick.labelsize':7.2}):
        fig,axes=plt.subplots(2,2,figsize=(7.2,7.06))
        fig.subplots_adjust(left=.20,right=.96,bottom=.16,top=.94,wspace=.72,hspace=.70)
        for i,(ax,ds) in enumerate(zip(axes.flat,DS)):
            image=figures.ablation(ax,ds)
            ax.set_title(f'{chr(97+i)}   {TITLE[ds]}',loc='left',fontweight='bold',fontsize=8,pad=10)
            bounds=ax.get_position();cax=fig.add_axes([bounds.x0,bounds.y0-.095,bounds.width,.009])
            colorbar=fig.colorbar(image,cax=cax,orientation='horizontal');colorbar.set_label('Paired deterioration (%)',fontsize=6.5)
            colorbar.ax.tick_params(labelsize=6.5)
        save(fig,'ablation',axes)
    for name,size,font in [('architecture',(7.2,3.4),9),('deck_architecture',(12,4.0),16)]:
        with plt.rc_context({'font.size':font,'xtick.labelsize':font,'ytick.labelsize':font}):
            fig,ax=plt.subplots(figsize=size)
            fig.subplots_adjust(left=.25,right=.97,bottom=.38 if font>12 else .33,top=.89)
            image=figures.architecture(ax)
            bounds=ax.get_position();cax=fig.add_axes([bounds.x0,.17 if font>12 else .14,bounds.width,.025])
            colorbar=fig.colorbar(image,cax=cax,orientation='horizontal');colorbar.set_label('Paired deterioration (%)',fontsize=font)
            save(fig,name,[ax])
    for name,size,font in [('neural_training',(7.2,5.1),9),('deck_neural_training',(12,5.0),13)]:
        with plt.rc_context({'font.size':font,'axes.labelsize':font,'xtick.labelsize':font,'ytick.labelsize':font}):
            fig,axes=plt.subplots(2,2,figsize=size)
            fig.subplots_adjust(left=.13,right=.98,bottom=.12,top=.79,wspace=.40,hspace=.70)
            handles=[Line2D([],[],color=TEAL,label='Residual MLP'),Line2D([],[],color=VIOLET,label='Feature Transformer'),Line2D([],[],color=GREY,ls='--',label='Train'),Line2D([],[],color=GREY,ls='-',label='Validation')]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.99),ncol=2,fontsize=font)
            for i,(ax,ds) in enumerate(zip(axes.flat,DS)):
                figures.training(ax,ds);ax.set_title(f'{chr(97+i)}   {TITLE[ds]}',loc='left',fontweight='bold',fontsize=font+1,pad=10)
            save(fig,name,axes)
    with plt.rc_context({'font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5}):
        fig,axes=plt.subplots(4,2,figsize=(7.2,8.7))
        fig.subplots_adjust(left=.13,right=.98,bottom=.07,top=.92,wspace=.40,hspace=.85)
        fig.legend(handles=[Line2D([],[],color=GREY,marker='o',mfc='white',label='Train'),Line2D([],[],color=TEAL,marker='s',label='Validation')],loc='upper center',ncol=2,bbox_to_anchor=(.55,.99))
        for row,ds in enumerate(DS):
            for col,kind in enumerate(['learning','complexity']):
                ax=axes[row,col];figures.diagnostic(ax,ds,kind)
                ax.set_title(f'{chr(97+row*2+col)}   {TITLE[ds]}',loc='left',fontweight='bold',fontsize=8.5,pad=8)
        save(fig,'fit_diagnostics',axes)
    with plt.rc_context({'font.size':13,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':13}):
        fig,axes=plt.subplots(1,2,figsize=(12,4.2))
        fig.subplots_adjust(left=.20,right=.96,bottom=.37,top=.85,wspace=.74)
        models=['HGB','XGB','LightGBM','CatBoost','ResMLP','FeatureTransformer']
        for i,(ax,ds) in enumerate(zip(axes,['house','adult'])):
            image=figures.ablation(ax,ds,models)
            short={'No neighborhood':'Neighborhood','No size':'Area','No quality':'Quality','No year':'Year','Area + quality':'Only area\n+ quality','No education':'Education','No work':'Work','No marital':'Marital','No age':'Age'}
            ax.set_xticklabels([short[v] for v in VARIANTS[ds]],rotation=0,rotation_mode='anchor',ha='center',fontsize=12)
            ax.set_title(f'{chr(97+i)}   {TITLE[ds]}',loc='left',fontweight='bold',fontsize=14,pad=11)
            bounds=ax.get_position();cax=fig.add_axes([bounds.x0,.16,bounds.width,.02])
            bar=fig.colorbar(image,cax=cax,orientation='horizontal');bar.set_label('Paired deterioration (%)',fontsize=13)
        save(fig,'deck_ablation',axes)
    local_examples=[]
    for name,size,font in [('local',(7.2,3.6),9),('deck_local',(12,3.8),13)]:
        with plt.rc_context({'font.size':font,'axes.labelsize':font,'xtick.labelsize':font,'ytick.labelsize':font}):
            fig,axes=plt.subplots(1,2,figsize=size)
            fig.subplots_adjust(left=.18,right=.98,bottom=.18,top=.85,wspace=.75)
            for i,(ax,ds) in enumerate(zip(axes,['house','adult'])):
                example=figures.local(ax,ds)
                if name=='local':local_examples.append(example)
                ax.set_title(f'{chr(97+i)}   {TITLE[ds]}',loc='left',fontweight='bold',fontsize=font+1,pad=10)
            save(fig,name,axes)
    (OUT/'local_examples.json').write_text(json.dumps(local_examples,ensure_ascii=False,indent=2))
    for name,size,font in [('shap',(7.2,5.0),9),('deck_shap',(12,5.0),13)]:
        with plt.rc_context({'font.size':font,'axes.labelsize':font,'xtick.labelsize':font,'ytick.labelsize':font}):
            fig,axes=plt.subplots(2,2,figsize=size)
            fig.subplots_adjust(left=.21,right=.98,bottom=.13,top=.89,wspace=.78,hspace=.70)
            for i,(ax,ds) in enumerate(zip(axes.flat,DS)):
                figures.importance(ax,ds);ax.set_title(f'{chr(97+i)}   {TITLE[ds]}',loc='left',fontweight='bold',fontsize=font+1,pad=10)
            save(fig,name,axes)
    for name,size,font in [('sequence',(7.2,4.15),7.7),('deck_sequence',(12,5.2),13)]:
        with plt.rc_context({'font.size':font,'axes.labelsize':font,'xtick.labelsize':font,'ytick.labelsize':font}):
            fig,axes=plt.subplots(1,2,figsize=size)
            fig.subplots_adjust(left=.19,right=.97,bottom=.32,top=.86,wspace=.75)
            figures.benchmark(axes[0],'nba_sequence')
            axes[0].set_title('a   Shared history: next-season error',loc='left',fontweight='bold',fontsize=font,pad=13)
            image=figures.ablation(axes[1],'nba_sequence')
            axes[1].set_xticklabels(['No\nage','No\nbody','No past\noutput','No\ngames','Two\nseasons'],rotation=0,ha='center',rotation_mode='anchor')
            axes[1].set_title('b   History and feature ablations',loc='left',fontweight='bold',fontsize=font,pad=13)
            bounds=axes[1].get_position();cax=fig.add_axes([bounds.x0,.12,bounds.width,.014])
            colorbar=fig.colorbar(image,cax=cax,orientation='horizontal');colorbar.set_label('Paired deterioration (%)',fontsize=font)
            fig.legend(handles=legend,loc='upper center',ncol=3,bbox_to_anchor=(.56,.99),fontsize=font)
            save(fig,name,axes)
    for name,size,font,short in [('holdout_extension',(7.2,5.2),8,False),('deck_holdout',(12,4.0),13,True)]:
        with plt.rc_context({'font.size':font,'axes.labelsize':font,'xtick.labelsize':font,'ytick.labelsize':font}):
            fig,axes=plt.subplots(1,2,figsize=size)
            fig.subplots_adjust(left=.20,right=.98,bottom=.18,top=.88,wspace=.90)
            figures.holdout(axes,short)
            axes[0].set_title('a   Subsequent test extension',loc='left',fontweight='bold',fontsize=font+1,pad=13)
            axes[1].set_title('b   CV-selected minus reference',loc='left',fontweight='bold',fontsize=font+1,pad=13)
            save(fig,name,axes)
    (OUT/'source_data.json').write_text(json.dumps({'source_directory':str(DATA),'record_exclusion':'No extra exclusions for plotting; master-window exclusions documented in sequence_audit.json','uncertainty':'Means and three-fold SD in benchmarks/curves/SHAP; heatmaps retain all three paired fold changes as vertical strips','colourbar_alignment':'Colourbar axes excluded from comparable scientific-panel alignment groups','slide_selection':'Benchmark only: seven highest grid mean entries excluding fixed references, plus constant; full 21-entry paper figure and CSV retained'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--skill-root',required=True);build(parser.parse_args().skill_root)
