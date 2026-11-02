"""Check the compiled delivery against the separately verified experiment."""
from pathlib import Path
import json,re
import pandas as pd
import pymupdf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'codes/verification/research/expanded'


def main():
    for name in ['checks','shap_checks','holdout_extension_checks','diagnostic_checks','optimization_checks']:
        assert json.loads((OUT/f'{name}.json').read_text())['status']=='passed',name
    figures=json.loads((OUT/'figures/summary.json').read_text())
    assert len(figures)==17
    assert all(r['text_exit']==r['collision_exit']==0 for r in figures)
    paper=(ROOT/'docs/source/paper.tex').read_text()
    deck=(ROOT/'docs/source/presentation.tex').read_text()
    assert r'\usetheme{Madrid}' in deck and '边界' not in deck
    assert len(re.findall(r'\\begin\{frame\}',deck))==20
    assert r'\input{expanded_results}' in paper and r'\input{expanded_methods}' in paper
    sources=[paper,deck]+[(ROOT/'docs/source'/p).read_text() for p in ['expanded_results.tex','expanded_methods.tex']]
    for text in sources:
        assert not re.search(r'\bnan\b|端到端端到端',text,re.I)
        assert not re.search(r'\\begin\{(?:table|figure)\}\[H\]',text)
    report=(ROOT/'docs/research_summary.md').read_text()
    assert re.findall(r'^## (.+)$',report,re.M)==['研究问题','结果','意义','方法']
    summary=pd.read_csv(ROOT/'artifacts/expanded/summary.csv')
    expected={'house':'RBFNet','adult':'CatBoost','nba':'RF','students':'SVM','nba_sequence':'BiGRU'}
    for ds,winner in expected.items():
        q=summary.query('dataset==@ds and variant=="Full" and method in ["grid","fixed"]')
        assert len(q)==(15 if ds=='nba_sequence' else 21)
        row=q.sort_values('mean',ascending=ds!='adult',kind='stable').iloc[0]
        assert row.model==winner
        digits=4 if ds=='adult' else 2 if ds=='house' else 3
        assert f'{row["mean"]:,.{digits}f}' in report
    holdout=pd.read_csv(ROOT/'artifacts/expanded/holdout_extension/metrics.csv').set_index('model').loc['CatBoost']
    for name in ['README.md','README.zh-CN.md','docs/research_summary.md']:
        text=(ROOT/name).read_text()
        assert f'{holdout.auc:.4f}' in text and 'Run-Expanded.command' in text
    assert 'github.com/Karcen/TableLab_DeepSeek' in report
    pdfs={}
    for name in ['paper','presentation']:
        doc=pymupdf.open(ROOT/f'docs/{name}.pdf')
        sizes=[];outside=[]
        for i,page in enumerate(doc):
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        if not span['text'].strip():continue
                        sizes.append(span['size'])
                        x0,y0,x1,y1=span['bbox']
                        if x0<-.5 or y0<-.5 or x1>page.rect.width+.5 or y1>page.rect.height+.5:
                            outside.append({'page':i+1,'text':span['text']})
        assert min(sizes)>=5 and not outside,(name,min(sizes),outside)
        assert any('github.com/Karcen/TableLab_DeepSeek' in link.get('uri','') for p in doc for link in p.get_links())
        if name=='presentation':assert len(doc)==20
        pdfs[name]={'pages':len(doc),'minimum_font_pt':min(sizes),'clipped_text_count':len(outside)}
    result={'status':'passed','figures':len(figures),'pdfs':pdfs,'numbers_checked_against_csv':True,
            'readmes_and_summary_synchronized':True,'review':'Visual and prose review recorded in delivery_review.md'}
    (OUT/'delivery_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
