"""DeepSeek proposes a plan; the existing local engine validates and executes it."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from . import credentials
from pydantic import ValidationError
from ..catalog import Catalog, DataError, METHOD_DOCS, SPECS
from ..contracts import Plan
from .deepseek_transport import MODEL
from ..llm import SYSTEM

SECRET = re.compile(r'sk-[A-Za-z0-9_-]{12,}', re.IGNORECASE)
METHODS = {key:(title,text) for key,title,text in METHOD_DOCS}
PROMPT = SYSTEM.replace('离线中文表格分析助手','TableLab 中文表格分析助手').removesuffix('/no_think') + '''
返回一个 JSON 对象，仅填写与问题有关的字段；未填写的可选字段使用 schema 默认值。
示例（仅示范格式；数字和字段不要照搬）：{"dataset":"students","operation":"aggregate","columns":["math score"],"group_by":["lunch"],"metric":"mean"}。
若当前问题明确要求“全部记录”，清除上一轮的筛选条件。只改算法时，保留先前样本、目标、变量和划分。'''

def cloud_fingerprint():
    digest=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode());digest.update(path.read_bytes())
    return digest.hexdigest()

class RemoteFailure(RuntimeError):
    """User-safe error, without request headers, raw response or key."""

def safe_evidence(schema, docs):
    """Rebuild excerpts from the static dictionary; retrieved text contains sample values."""
    ds = schema['id']; spec = SPECS[ds]; out = []
    known = {c['column']:c['dtype'] for c in schema['columns']}
    for doc in docs:
        key = doc['id']
        if key.startswith(ds + ':field:'):
            name = key.removeprefix(ds + ':field:')
            title, text = name, spec.descriptions.get(name,'数据字典未收录中文释义。') + ' 类型：' + known[name]
        elif key == ds + ':scope':
            title, text = spec.label, spec.warning
        elif key in METHODS:
            title, text = METHODS[key]
        else:
            raise DataError('检索结果含有未登记的说明。没有发出请求。')
        out.append({'id':key, 'title':title, 'text':text})
    return out

def make_payload(question, schema, docs, previous, model_id):
    ds=schema['id']; spec=SPECS[ds]
    prev=previous.model_dump(exclude={'record','message','model_id'},exclude_defaults=True) if previous else None
    ctx={'question':question, 'dataset':ds, 'columns':[(c['column'],c['dtype']) for c in schema['columns']],
         'targets':spec.targets,'default_features':schema['default_features'],
         'retrieved_documents':safe_evidence(schema,docs),'previous_plan':prev,
         'model_id':('0'*32 if model_id else None)}  # local handles are never disclosed
    text=json.dumps(ctx,ensure_ascii=False,separators=(',',':'))
    if SECRET.search(text):
        raise DataError('问题或上一轮筛选条件中疑似有密钥。请删除凭据或清空上下文，没有发出请求。')
    return {'model':MODEL, 'stream':False,'thinking':{'type':'disabled'},'temperature':0,'max_tokens':1900,
            'response_format':{'type':'json_object'},'messages':[
                {'role':'system','content':PROMPT+'\nJSON schema:\n'+json.dumps(Plan.model_json_schema(),ensure_ascii=False,separators=(',',':'))},
                {'role':'user','content':text}]}

def via_worker(key, payload):
    packet=json.dumps({'key':key, 'payload':payload},ensure_ascii=True).encode('ascii')
    try:
        p=subprocess.run([sys.executable,'-m','tablelab.cloud.deepseek_transport'],input=packet,
                         stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=85,check=True)
    except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
        raise RemoteFailure('联网进程未正常完成。没有自动重试，请检查网络。') from exc
    return json.loads(p.stdout)

class DeepSeek:
    provider='deepseek'
    def __init__(self, key:str, transport=via_worker):
        if not key or not key.isascii() or any(c.isspace() or not c.isprintable() for c in key):
            raise DataError('密钥为空或含空格、换行。请重新启动，在终端输入新密钥。')
        self._key=key; self._send=transport; self.verified=False

    def status(self):
        return {'provider':self.provider,'name':MODEL,'ready':True,'verified_by_request':self.verified}

    def plan(self, question, schema, evidence, previous=None, model_id=None):
        payload=make_payload(question,schema,evidence,previous,model_id)
        start=time.perf_counter(); reply=self._send(self._key,payload)
        if 'error' in reply:
            self.verified=False
            raise RemoteFailure(reply['error'])
        raw=reply['plan']
        if raw.get('operation')=='predict' and model_id and raw.get('model_id')=='0'*32:
            raw={**raw,'model_id':model_id}
        try:
            plan=Plan.model_validate(raw)
        except ValidationError:
            raise DataError('DeepSeek 返回的参数不符合 TableLab 的方案格式。请改写问题。没有执行计算。') from None
        if plan.dataset!=schema['id']:
            raise DataError('DeepSeek 选错了数据集。请重新提问。没有执行计算。')
        known={c['column'] for c in schema['columns']}
        used=plan.columns+plan.features+plan.group_by+[f.column for f in plan.filters]+([plan.target] if plan.target else [])+list(plan.record)
        if set(used)-known: raise DataError('DeepSeek 提到了当前表中没有的字段。没有执行计算。')
        self.verified=True
        return plan,{'provider':'deepseek','planner_code_sha256':cloud_fingerprint(),'model':MODEL,'reported_model':reply['model_reported'],
                     'elapsed_seconds':time.perf_counter()-start,'usage':reply['usage'],
                     'sent_dictionary':json.loads(payload['messages'][1]['content'])['retrieved_documents']}

def main(*, session_only: bool = False):
    import uvicorn
    from .web import create_app
    from ..offline import install_loopback_guard
    for ds in SPECS:
        _, entry=Catalog(Path('data')).load(ds)
        if entry['source_kind'] not in ('original','course_dataset'):
            raise SystemExit(f'{ds} 不是登记的课程数据，停止启动。')
    key = (credentials.validated(credentials.hidden_input()) if session_only
           else credentials.acquire(credentials.native_store()))
    client=DeepSeek(key); del key
    install_loopback_guard()
    print('浏览器打开 http://127.0.0.1:8765 。第一次生成方案时才联系 DeepSeek。按 Control-C 退出。',flush=True)
    uvicorn.run(create_app(llm=client),host='127.0.0.1',port=8765,workers=1,access_log=False)
