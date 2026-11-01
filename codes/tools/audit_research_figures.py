"""Run the designated figure skill's source, text, alignment and collision audits."""
from pathlib import Path
import argparse,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--skill-root',required=True);a=p.parse_args();scripts=Path(a.skill_root)/'scripts'
 out=ROOT/'docs/source/figures/research';qa=ROOT/'codes/verification/research/figures';qa.mkdir(parents=True,exist_ok=True)
 rows=[]
 for pdf in sorted(out.glob('*.pdf')):
  if 'audit' in pdf.name:continue
  text=subprocess.run([sys.executable,str(scripts/'audit_pdf_text.py'),str(pdf),'--min-pt','5','--json'],capture_output=True,text=True)
  (qa/f'{pdf.stem}.text.json').write_text(text.stdout)
  collision=subprocess.run([sys.executable,str(scripts/'audit_figure_collisions.py'),str(pdf),'--json-out',str(qa/f'{pdf.stem}.collision.json'),'--overlay-pdf',str(qa/f'{pdf.stem}.collision.pdf')],capture_output=True,text=True)
  (qa/f'{pdf.stem}.collision.txt').write_text(collision.stdout+collision.stderr)
  rows.append({'figure':pdf.name,'text_exit':text.returncode,'collision_exit':collision.returncode})
  print(pdf.name,'text',text.returncode,'collision',collision.returncode,flush=True)
 (qa/'summary.json').write_text(json.dumps(rows,indent=2))
 if any(x['text_exit'] or x['collision_exit'] for x in rows):raise SystemExit('Figure audit requires correction; inspect saved reports.')
