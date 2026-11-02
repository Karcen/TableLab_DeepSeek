"""Export figure variants whose labels remain readable inside the oral slides."""
from pathlib import Path
import argparse,sys,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_research import Figures,OUT,DS,TITLES,TEAL,VIOLET,NAVY,ORANGE,GREY,INK,paneltitle


def export(fig,name):
 from audit_panel_alignment import require_matplotlib_panel_alignment
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=str(OUT/f'{name}.alignment.json'),overlay_svg=str(OUT/f'{name}.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
 fig.savefig(OUT/f'{name}.pdf',dpi=600)
 fig.savefig(OUT/f'{name}.svg',dpi=600)
 fig.savefig(OUT/f'{name}.png',dpi=600)
 plt.close(fig)


def build_talk_assets(skill):
 sys.path.insert(0,str(Path(skill)/'scripts'))
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans','Liberation Sans'],'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7,'legend.frameon':False,'savefig.facecolor':'white'})
 figures=Figures()
 definitions=[('search',figures.search,[Line2D([],[],color=TEAL,marker='o',label='Grid'),Line2D([],[],color=VIOLET,marker='s',label='TPE')]),('complexity',lambda a,d:figures.diagnostic(a,d,'complexity'),[Line2D([],[],color=GREY,marker='o',mfc='white',label='Train'),Line2D([],[],color=NAVY,marker='s',label='Validation')]),('learning',lambda a,d:figures.diagnostic(a,d,'learning'),[Line2D([],[],color=GREY,marker='o',mfc='white',label='Train'),Line2D([],[],color=NAVY,marker='s',label='Validation')]),('shap',figures.importance,[]),('ablation',figures.ablation,[])]
 for name,draw,legend in definitions:
  tick=16 if name=='search' else 13.5
  with plt.rc_context({'font.size':14,'axes.labelsize':16 if name=='search' else 14,'xtick.labelsize':tick,'ytick.labelsize':tick,'legend.fontsize':16 if name=='search' else 14}):
   fig,axes=plt.subplots(2,2,figsize=(12,5))
   fig.subplots_adjust(left=.15,right=.98,bottom=.16,top=.82,wspace=.65,hspace=.85)
   if legend:fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.55,.995),ncol=len(legend))
   for i,(ax,ds) in enumerate(zip(axes.flat,DS)):
    draw(ax,ds);paneltitle(ax,TITLES[ds],chr(97+i));ax.set_title(TITLES[ds],loc='left',fontsize=16 if name=='search' else 14,fontweight='bold',color=INK,pad=8)
    for label in ax.texts:
     if label.get_text() in ['a','b','c','d']:label.set_fontsize(16 if name=='search' else 14)
   export(fig,'deck_'+name)
 with plt.rc_context({'font.size':13,'axes.labelsize':13,'xtick.labelsize':12.5,'ytick.labelsize':12.5}):
  fig,ax=plt.subplots(figsize=(8.8,4.4));fig.subplots_adjust(left=.25,right=.97,bottom=.13,top=.90)
  figures.benchmark(ax,'house');ax.set_title(TITLES['house'],loc='left',fontsize=13,fontweight='bold',color=INK,pad=8);export(fig,'deck_benchmark_house')
 with plt.rc_context({'font.size':14,'axes.labelsize':14,'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':13}):
  fig,axes=plt.subplots(1,2,figsize=(12,3.7));fig.subplots_adjust(left=.15,right=.98,bottom=.18,top=.80,wspace=.65)
  fig.legend(handles=[Patch(color=NAVY,label='Negative contribution'),Patch(color=ORANGE,label='Positive contribution'),Patch(color=TEAL,label='Prediction')],loc='upper center',bbox_to_anchor=(.55,.995),ncol=3)
  for i,(ax,ds) in enumerate(zip(axes,['house','adult'])):
   figures.local(ax,ds);paneltitle(ax,TITLES[ds],chr(97+i));ax.set_title(TITLES[ds],loc='left',fontsize=14,fontweight='bold',color=INK,pad=8)
  export(fig,'deck_local')
 (OUT/'deck_figure_contract.json').write_text(json.dumps({'backend':'Python','intent':'Larger labels in a short oral presentation','full_task_panels':['benchmark summary','search','complexity','learning','shap','ablation'],'local_examples':['house','adult'],'selection_reason':'Two prespecified median-error cases illustrate regression and classification output scales. Full four-task cases remain in the paper.','record_sampling':False,'geometry':'Unchanged values, mean, three-fold SD and original preprocessing; new typography and layout only.','minimum_effective_glyph_pt':5},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--skill-root',required=True);build_talk_assets(p.parse_args().skill_root)
