"""Run the user's designated Nature figure gates on the expanded outputs."""
from pathlib import Path
import argparse,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--skill-root',required=True);args=parser.parse_args()
    scripts=Path(args.skill_root)/'scripts';out=ROOT/'codes/verification/research/expanded/figures';out.mkdir(parents=True,exist_ok=True)
    source=subprocess.run([sys.executable,str(scripts/'validate_figure.py'),str(ROOT/'codes/tools/plot_expanded.py'),'--json'],capture_output=True,text=True)
    (out/'source_audit.json').write_text(source.stdout)
    if source.returncode:raise SystemExit(source.stdout+source.stderr)
    rows=[]
    for pdf in sorted((ROOT/'docs/source/figures/expanded').glob('*.pdf')):
        text=subprocess.run([sys.executable,str(scripts/'audit_pdf_text.py'),str(pdf),'--min-pt','5','--json'],capture_output=True,text=True)
        (out/f'{pdf.stem}.text.json').write_text(text.stdout)
        collision=subprocess.run([sys.executable,str(scripts/'audit_figure_collisions.py'),str(pdf),'--json-out',str(out/f'{pdf.stem}.collision.json'),'--overlay-pdf',str(out/f'{pdf.stem}.collision.pdf')],capture_output=True,text=True)
        (out/f'{pdf.stem}.collision.txt').write_text(collision.stdout+collision.stderr)
        rows.append({'figure':pdf.name,'text_exit':text.returncode,'collision_exit':collision.returncode})
        print(pdf.name,text.returncode,collision.returncode,flush=True)
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
    if any(r['text_exit'] or r['collision_exit'] for r in rows):raise SystemExit('Resolve rendered figure QA findings before delivery.')
