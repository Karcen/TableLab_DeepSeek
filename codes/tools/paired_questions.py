"""Paired local evaluation. References are used by the human grader, never by Ollama."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import secrets
import sys
import time
from uuid import uuid4
import requests
from sklearn.exceptions import ConvergenceWarning

CODES = Path(__file__).resolve().parents[1]
ROOT = CODES.parent
sys.path.insert(0, str(CODES))
from tablelab.catalog import Catalog, SPECS
from tablelab.engine import Engine, runtime_fingerprint
from tablelab.llm import Ollama
from tablelab.offline import install_loopback_guard
from tablelab.retrieval import Retriever

BANK = CODES / 'examples/paired'
RESULTS = ROOT / 'tables/paired_runs'
ARMS = ('schema', 'schema_retrieval')
INPUT = ('id', 'dataset', 'previous_id', 'question', 'reference')
RATING = ('id','dataset','variant','question','reference','answer_file','initial_ok','why','corrected_ok','correction_record')
BAD = {'plan_error','execution_error','upstream_failed','service_error'}

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),encoding='utf8')

def read_json(path): return json.loads(path.read_text(encoding='utf8'))

def read_csv(path, fields):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f)
        if tuple(reader.fieldnames or ())!=tuple(fields): raise ValueError(f'{path.name} 列名应为：{", ".join(fields)}')
        out=[]
        for row in reader:
            if None in row or None in row.values(): raise ValueError(f'{path.name} 第 {reader.line_num} 行列数不符，请用 CSV 编辑器保存。')
            out.append({k:v.strip() for k,v in row.items()})
        return out

def write_csv(path, fields, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def validate_bank(path, kind):
    rows=read_csv(path,INPUT); seen={}; texts=set(); expected=6 if kind=='class' else 2
    if Counter(r['dataset'] for r in rows) != {d:expected for d in SPECS}:
        raise ValueError(f'每个主题应有 {expected} 题，数据集用 house、adult、nba、students。')
    for r in rows:
        q=r['id']; parent=r['previous_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,36}',q) or q in seen: raise ValueError(f'题号为空、重复或字符非法：{q}')
        if not r['question'] or not r['reference']: raise ValueError(f'{q}：请先填写 question 和 reference。随包练习在菜单 4。')
        if len(r['question'])>1500: raise ValueError(f'{q}：问题超过网页的 1500 字限制。')
        if parent and seen.get(parent)!=r['dataset']: raise ValueError(f'{q}：previous_id 必须是同一主题中已写在前面的题号。')
        item=(r['dataset'],parent,r['question'])
        if item in texts: raise ValueError(f'{q}：重复题。')
        texts.add(item); seen[q]=r['dataset']
    if kind=='class':
        existing={x['question'].strip() for x in read_csv(BANK/'practice.csv',INPUT)}|{x['question'].strip() for x in read_json(CODES/'examples/acceptance_questions.json')}
        if existing & {r['question'] for r in rows}: raise ValueError('正式题与练习题逐字相同，请更换题目。')
    return rows

class Recorder(Ollama):
    """Use the application client unchanged, including its schema, system prompt and seed."""
    trace:Path
    def _request(self,path,body=None,timeout=10):
        if path=='/api/chat': dump(self.trace/'request.json',body)
        obj=super()._request(path,body,timeout)
        if path=='/api/chat': dump(self.trace/'response.json',obj)
        return obj

def make_assignments(rows, seed):
    rng=random.Random(seed); assignment={};orders={}
    for dataset in SPECS:
        ids=[r['id'] for r in rows if r['dataset']==dataset]
        for mapping in (assignment,orders):
            balanced=[0,1]*(len(ids)//2); rng.shuffle(balanced)
            for k,i in zip(ids,balanced,strict=True): mapping[k]=dict(zip(('A','B'),ARMS[::1 if i==0 else -1],strict=True)) if mapping is assignment else ARMS[::1 if i==0 else -1]
    return assignment,orders

def run(bank,kind,client=None,outbase=RESULTS):
    rows=validate_bank(bank,kind);cat=Catalog(ROOT/'data');schemas={d:cat.schema(d) for d in SPECS}
    if any(x['source_kind'] not in {'course_dataset','original'} for x in schemas.values()): raise ValueError('对照实验需要四份课程数据。')
    if client is None: client=Recorder(ROOT/'models/runtime/model.lock.json')
    model=client.check_lock(); seed=secrets.randbits(32); assignment,orders=make_assignments(rows,seed)
    now=datetime.now(timezone.utc); out=outbase/(kind+'_'+now.strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
    (out/'answers').mkdir(); (out/'audit').mkdir(); (out/'questions.csv').write_bytes(bank.read_bytes())
    dump(out/'protocol.json',{'set':kind,'questions_sha256':hashlib.sha256(bank.read_bytes()).hexdigest(),'model':model['name'],'digest':model['digest'],'time_utc':now.isoformat(),'seed':seed,'data_sha256':{d:s['sha256'] for d,s in schemas.items()},'runtime':runtime_fingerprint(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'n_questions':len(rows),'intervention':'Same schema, instructions, local model, parameters and tools; only retrieved_documents differs. Follow-ups inherit their own branch. No gold in prompts.'})
    dump(out/'audit/assignment.json',assignment);dump(out/'audit/order.json',orders)
    engines={a:Engine(cat,out/'audit'/a/'executions') for a in ARMS}; retrieved={d:Retriever(cat,d) for d in SPECS}
    states={};review=[]; print(f'本次实验目录：{out}',flush=True)
    for pos,r in enumerate(rows,1):
        q,d,parent=r['id'],r['dataset'],r['previous_id']; paired={}
        for arm in orders[q]:
            v=next(k for k,a in assignment[q].items() if arm==a); tr=out/'audit'/arm/q;tr.mkdir(parents=True)
            prior,handle=states.get((parent,arm),(None,None)); evidence=[]; plan=None;stage='plan_error'; t=time.perf_counter();ans={'id':q,'variant':v}
            if parent and prior is None:
                ans.update(status='upstream_failed',error=f'前题 {parent} 在该条件下没有可继续的方案。');states[q,arm]=(None,None)
            else:
                query=r['question']+(' '+' '.join(prior.features+prior.columns) if prior else '')
                if arm=='schema_retrieval': evidence=retrieved[d].search(query)
                client.trace=tr
                try:
                    plan,meta=client.plan(r['question'],schemas[d],evidence,prior,handle)
                    if meta['model_digest']!=model['digest']: raise RuntimeError('两次请求间模型指纹改变，实验停止。')
                    ans['plan']=plan.model_dump(); dump(tr/'model_meta.json',meta);stage='execution_error'
                    result=engines[arm].execute(plan,r['question'],evidence)
                    ans.update({k:result[k] for k in ('status','summary','table','metrics','explanation','notes','table_rows_total','run_id') if k in result})
                    states[q,arm]=(plan,result.get('model_id',handle));
                except (requests.Timeout,requests.ConnectionError,requests.HTTPError) as e:
                    ans.update(status='service_error',error=f'{type(e).__name__}: {e}');states[q,arm]=(None,None)
                except (ValueError,ConvergenceWarning) as e:
                    ans.update(status=stage,error=f'{type(e).__name__}: {e}');states[q,arm]=(None,None)
            dump(tr/'timing.json',{'elapsed_seconds':time.perf_counter()-t});dump(tr/'evidence.json',evidence)
            f=f'answers/{q}-{v}.json';dump(out/f,ans)
            paired[v]={k:r[k] for k in ('id','dataset','question','reference')}|{'variant':v,'answer_file':f,'initial_ok':'0' if ans['status'] in BAD else '', 'why':ans['status'] if ans['status'] in BAD else '', 'corrected_ok':'','correction_record':''}
        review.extend(paired[v] for v in ('A','B'));print(f'{pos}/{len(rows)}  {q} 已保存两份输出',flush=True)
    write_csv(out/'review.csv',RATING,review);dump(out/'complete.json',{'n_answers':len(review)})
    print('请核对 answers 内的结果，在 review.csv 填 1 或 0，再用菜单 5 汇总。',flush=True);return out

def score(out):
    proto=read_json(out/'protocol.json'); complete=read_json(out/'complete.json');bank=read_csv(out/'questions.csv',INPUT)
    if hashlib.sha256((out/'questions.csv').read_bytes()).hexdigest()!=proto['questions_sha256']: raise ValueError('题表在运行后被更改。')
    assignments=read_json(out/'audit/assignment.json');expected={(r['id'],v) for r in bank for v in ('A','B')};byid={r['id']:r for r in bank}
    scores=read_csv(out/'review.csv',RATING)
    if len(scores)!=complete['n_answers'] or len(scores)!=len(expected) or {(r['id'],r['variant']) for r in scores}!=expected: raise ValueError('评分表有重复行或缺少题目。')
    marks={};corrections={a:{'attempted':0,'successful':0} for a in ARMS}
    for r in scores:
        q=r['id'];v=r['variant']; ref=byid[q]; arm=assignments[q][v]; a=read_json(out/'answers'/f'{q}-{v}.json')
        if any(r[k]!=ref[k] for k in ('dataset','question','reference')): raise ValueError(f'{q}-{v}：题目或评分依据被更改。')
        if r['initial_ok'] not in {'0','1'}: raise ValueError(f'{q}-{v}：initial_ok 还没填完，只能填 0 或 1。')
        if a['status'] in BAD and r['initial_ok']=='1': raise ValueError(f'{q}-{v}：原始调用失败，不能记为首次答对。')
        if r['initial_ok']=='0' and not r['why']: raise ValueError(f'{q}-{v}：请在 why 写明错误。')
        edited=r['corrected_ok'];note=r['correction_record']
        if edited not in {'','0','1'} or (bool(edited)!=bool(note)) or (edited and r['initial_ok']!='0'): raise ValueError(f'{q}-{v}：只为首次答错且实际尝试纠正的题填写 corrected_ok=0/1 和 correction_record。')
        marks[q,arm]=int(r['initial_ok']);corrections[arm]['attempted']+=int(bool(edited));corrections[arm]['successful']+=int(edited=='1')
    counts=Counter((marks[r['id'],ARMS[0]],marks[r['id'],ARMS[1]]) for r in bank);n=len(bank)
    summary={'bank':proto['set'],'n_questions':n,'schema_correct':counts[1,0]+counts[1,1],'with_retrieval_correct':counts[0,1]+counts[1,1],'both_correct':counts[1,1],'only_with_retrieval':counts[0,1],'only_without_retrieval':counts[1,0],'both_wrong':counts[0,0],'difference_percentage_points':100*(counts[0,1]-counts[1,0])/n,'review_sha256':hashlib.sha256((out/'review.csv').read_bytes()).hexdigest(),'self_reported_edits':corrections}
    dump(out/'summary.json',summary); write_csv(out/'paired_by_question.csv',('id','dataset','schema_correct','with_retrieval_correct'),[{'id':r['id'],'dataset':r['dataset'],'schema_correct':marks[r['id'],ARMS[0]],'with_retrieval_correct':marks[r['id'],ARMS[1]]} for r in bank]);print(f"仅表结构：{summary['schema_correct']}/{n}；加检索：{summary['with_retrieval_correct']}/{n}；仅加检索答对 {counts[0,1]} 题；仅不加检索答对 {counts[1,0]} 题。\n结果：{out/'summary.json'}");return summary

def main():
    app=argparse.ArgumentParser(description=__doc__);sub=app.add_subparsers(dest='cmd',required=True)
    for name in ('validate','run'):
        p=sub.add_parser(name);p.add_argument('--set',choices=['practice','class'],required=True);p.add_argument('--bank',type=Path)
    p=sub.add_parser('score');p.add_argument('folder',type=Path,nargs='?');args=app.parse_args();install_loopback_guard()
    if args.cmd=='score':
        folder=args.folder
        if folder is None:
            choices=sorted([p.parent for p in RESULTS.glob('*/complete.json')],key=lambda p:read_json(p/'protocol.json')['time_utc'],reverse=True)
            if not choices:app.error('还没有已完成的本机对照实验。')
            for i,p in enumerate(choices,1):print(i,p.name)
            num=int(input('输入待汇总的运行编号：'))
            if not 1<=num<=len(choices):app.error('编号超出范围。')
            folder=choices[num-1]
        score(folder);return
    bank=args.bank or BANK/('practice.csv' if args.set=='practice' else 'class_questions.csv')
    if args.cmd=='validate': print(f'题表检查通过：{len(validate_bank(bank,args.set))} 题。');return
    run(bank,args.set)

if __name__=='__main__':
    try: main()
    except (ValueError,FileNotFoundError,requests.RequestException) as e: raise SystemExit(f'未完成：{e}') from e
