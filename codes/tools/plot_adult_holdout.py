"""Reserved Adult test: absolute performance and paired AUC differences."""
from pathlib import Path
import argparse
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'artifacts/holdout/adult'
OUT = ROOT/'docs/source/figures/research'
LABELS = {'Mean': 'Constant', 'Linear': 'Logistic', 'Ridge': 'Strong L2', 'Sparse': 'L1 logistic',
          'Tree': 'Decision tree', 'RF': 'Random forest', 'ExtraTrees': 'Extra Trees', 'HGB': 'Hist. boosting',
          'KNN': '15-nearest neighbors', 'RF_grid': 'Forest + Grid', 'RF_tpe': 'Forest + TPE',
          'HGB_grid': 'Boosting + Grid', 'HGB_tpe': 'Boosting + TPE'}
TEAL, GREY, INK = '#368F92', '#9AA1AA', '#34404B'


def render(skill):
    sys.path.insert(0, str(Path(skill)/'scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    scores = pd.read_csv(DATA/'metrics.csv').set_index('model').sort_values('auc', ascending=False, kind='stable')
    pairs = pd.read_csv(DATA/'paired_differences.csv').set_index('comparator')
    focus = [m for m in scores.index if m in ['HGB', 'HGB_tpe', 'RF', 'RF_grid', 'RF_tpe']]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
                         'font.size': 8, 'legend.frameon': False, 'savefig.facecolor': 'white', 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
    for name, size, font in [('adult_holdout', (7.2, 4.5), 8), ('deck_adult_holdout', (12, 5), 13.5)]:
        with plt.rc_context({'font.size': font, 'axes.labelsize': font, 'xtick.labelsize': font, 'ytick.labelsize': font}):
            fig, axes = plt.subplots(1, 2, figsize=size)
            fig.subplots_adjust(left=.20, right=.98, bottom=.18, top=.88, wspace=.90)
            ax, bx = axes
            for i, (model, row) in enumerate(scores.iterrows()):
                color = TEAL if model == 'HGB_grid' else GREY
                ax.errorbar(row.auc, i, xerr=[[row.auc-row.auc_ci_low], [row.auc_ci_high-row.auc]],
                            fmt='D' if model == 'HGB_grid' else 'o', color=color, markersize=4.5,
                            capsize=2, elinewidth=1, zorder=3)
            ax.set_yticks(np.arange(len(scores)), [LABELS[m] for m in scores.index])
            ax.invert_yaxis()
            ax.set_xlim(.48, .93)
            ax.set_xticks([.5, .6, .7, .8, .9])
            ax.set_xlabel('Test ROC AUC')
            ax.axvline(.5, color='#D7DCE0', lw=.7, ls='--', zorder=0)
            ax.set_title('a   Reserved test', loc='left', fontweight='bold', fontsize=font+1, pad=14)
            for i, model in enumerate(focus):
                row = pairs.loc[model]
                bx.errorbar(row.auc_difference*1000, i,
                            xerr=[[(row.auc_difference-row.ci_low)*1000], [(row.ci_high-row.auc_difference)*1000]],
                            fmt='o', color=TEAL, markersize=4.5, capsize=3, elinewidth=1)
            bx.set_yticks(np.arange(len(focus)), [LABELS[m] for m in focus])
            bx.set_ylim(len(focus)-.5, -.5)
            bx.axvline(0, color=GREY, lw=.8, ls='--', zorder=0)
            bx.set_xlabel('AUC difference (0.001 units)')
            bx.set_title('b   Grid boosting minus', loc='left', fontweight='bold', fontsize=font+1, pad=14)
            bx.margins(x=.16)
            for panel in axes:
                panel.tick_params(axis='y', length=0, pad=5)
                panel.spines['left'].set_visible(False)
                panel.grid(axis='x', color='#EDF0F2', lw=.6, zorder=0)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig, json_out=str(OUT/f'{name}.alignment.json'),
                                                overlay_svg=str(OUT/f'{name}.alignment.svg'), tolerance_pt=1.5,
                                                gutter_tolerance_pt=1.5, strict=True)
            fig.savefig(OUT/f'{name}.pdf', dpi=600)
            fig.savefig(OUT/f'{name}.svg', dpi=600)
            fig.savefig(OUT/f'{name}.png', dpi=600)
            plt.close(fig)
    (OUT/'adult_holdout_source.json').write_text(json.dumps({'metrics': str(DATA/'metrics.csv'),
         'paired_differences': str(DATA/'paired_differences.csv'), 'n_rows': 16255, 'n_groups': 16249,
         'bootstrap': '2,000 paired cluster bootstrap resamples; 95% percentile CI',
         'primary_flow_fixed_before_test': 'HGB_grid', 'multiplicity': 'Unadjusted descriptive intervals'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill-root', required=True)
    render(parser.parse_args().skill_root)
