"""本机单用户界面，无 CDN、遥测和远程前端依赖。"""
from pathlib import Path
import json
import os
import re
import threading
import uuid
import requests
from fastapi import FastAPI,Request
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from sklearn.exceptions import ConvergenceWarning
from .catalog import Catalog,SPECS,DataError
from .contracts import Question,ExecutionRequest
from .engine import Engine,write_json
from .retrieval import Retriever
from .llm import Ollama,MODEL

STATIC=Path(__file__).parent/'static'

def create_app(data_root=None,artifact_root=None,llm=None):
    cat=Catalog(Path(data_root or os.environ.get('TABLELAB_DATA_ROOT','data')))
    engine=Engine(cat,Path(artifact_root or os.environ.get('TABLELAB_ARTIFACT_ROOT','artifacts/runs')))
    client=llm or Ollama()
    lock=threading.Lock();model_lock=threading.Lock();proposals={}
    app=FastAPI(title='离线分析台',docs_url=None,redoc_url=None,openapi_url=None)
    app.state.engine=engine;app.state.catalog=cat
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])

    @app.middleware('http')
    async def boundary(request:Request,call_next):
        if request.method=='POST':
            if request.headers.get('X-TableLab')!='local-ui': return JSONResponse({'error':'缺少本地界面请求标记。'},403)
            origin=request.headers.get('origin')
            if origin and origin not in ('http://127.0.0.1:8765','http://localhost:8765'):
                return JSONResponse({'error':'拒绝跨来源写请求。'},403)
            if len(await request.body())>100_000: return JSONResponse({'error':'请求超过 100 KB。'},413)
        response=await call_next(request)
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-src 'self'; object-src 'self'; base-uri 'none'; frame-ancestors 'self'"
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        return response

    @app.exception_handler(ValueError)
    async def invalid(request,exc): return JSONResponse({'error':str(exc),'status':'not_executed_or_failed'},422)
    @app.exception_handler(requests.RequestException)
    async def model_failure(request,exc): return JSONResponse({'error':'本地 Ollama 调用失败：'+str(exc),'status':'not_executed'},503)
    @app.exception_handler(ConvergenceWarning)
    async def convergence(request,exc): return JSONResponse({'error':'优化未收敛；本次模型不作为成功结果。'+str(exc)},422)

    @app.get('/')
    def index():return FileResponse(STATIC/'index.html')

    @app.get('/api/status')
    def status():
        datasets=cat.available()
        for item in datasets:
            if item['present']:
                try:
                    df,entry=cat.load(item['id']);item.update(rows=len(df),source_kind=entry['source_kind'],verified=True)
                except (ValueError,KeyError) as e:item.update(verified=False,error=str(e))
        try: info=client.check_lock();model={'ready':True,'name':MODEL,'digest':info['digest']}
        except (ValueError,requests.RequestException,FileNotFoundError) as e:model={'ready':False,'name':MODEL,'error':str(e)}
        return {'datasets':datasets,'model':model,'boundary':'应用进程仅允许回环连接；正式验收仍需关闭外部网络。'}

    @app.get('/api/schema/{dataset}')
    def schema(dataset:str):return cat.schema(dataset)

    @app.post('/api/propose')
    def propose(request:Question):
        schema=cat.schema(request.dataset)
        previous=request.previous
        if previous and previous.dataset!=request.dataset: raise DataError('上一方案属于另一个数据集，请清空上下文。')
        search_text=request.question+(' '+' '.join(previous.features+previous.columns) if previous else '')
        evidence=Retriever(cat,request.dataset).search(search_text)
        with model_lock:plan,model_meta=client.plan(request.question,schema,evidence,previous,request.model_id)
        proposal_id=uuid.uuid4().hex
        proposals[proposal_id]={'question':request.question,'dataset':request.dataset,'evidence':evidence,'proposed_plan':plan.model_dump(),'model':model_meta}
        if len(proposals)>100:del proposals[next(iter(proposals))]
        return {'proposal_id':proposal_id,'plan':plan.model_dump(),'evidence':evidence,'model':model_meta}

    @app.post('/api/execute')
    def execute(request:ExecutionRequest):
        if request.evidence:raise DataError('检索证据由服务器保存，不接受客户端自报证据。')
        if request.proposal_id:
            if request.proposal_id not in proposals:raise DataError('方案会话已失效，请重新生成。')
            proposal=proposals[request.proposal_id]
            if proposal['dataset']!=request.plan.dataset:raise DataError('方案与检索证据的数据集不一致。')
            evidence=proposal['evidence'];question=proposal['question']
            mode='natural_language_edited' if proposal['proposed_plan']!=request.plan.model_dump() else 'natural_language'
        else:
            proposal=None;question=request.question
            evidence=Retriever(cat,request.plan.dataset).search(question or (' '.join(request.plan.columns+request.plan.features) or '数据边界'))
            mode='manual_parameters'
        with lock: result=engine.execute(request.plan,question,evidence)
        result['interaction_mode']=mode
        result['evidence']=evidence
        if 'run_id' in result:
            folder=engine.out/result['run_id']
            if proposal is not None:
                write_json(folder/'proposal.json',proposal);result['files'].append('proposal.json')
            write_json(folder/'result.json',result)
        return result

    @app.get('/api/files/{run_id}/{filename}')
    def artifact(run_id:str,filename:str):
        if not re.fullmatch(r'[a-f0-9]{32}',run_id):raise DataError('无效运行编号。')
        allowed={'table.csv','plan.json','evidence.json','executor_source.py','provenance.json','reproduce.py','result.json','proposal.json','figure.pdf','diagnostic.pdf','metrics.csv','validation_predictions.csv','split.csv','explanation.csv','confusion.csv','transformation_audit.json'}
        if filename not in allowed:raise DataError('此文件不允许通过界面读取。')
        path=engine.out/run_id/filename
        if not path.is_file():return JSONResponse({'error':'文件不存在。'},404)
        return FileResponse(path,media_type='application/pdf' if filename.endswith('.pdf') else None)

    app.mount('/static',StaticFiles(directory=STATIC),name='static')
    return app


def main():
    from .offline import install_loopback_guard
    install_loopback_guard()
    import uvicorn
    uvicorn.run(create_app(),host='127.0.0.1',port=8765,workers=1,access_log=False)

if __name__=='__main__':main()
