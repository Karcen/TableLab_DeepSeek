"""No network: mocked provider response and real local calculations."""
import json
from pathlib import Path
from unittest.mock import Mock
import requests
import pytest
from fastapi.testclient import TestClient
from tablelab.catalog import Catalog,DataError
from tablelab.contracts import Plan
from tablelab.cloud.deepseek_online import DeepSeek,make_payload
from tablelab.cloud.deepseek_transport import send_once,URL
from tablelab.cloud.web import create_app

DUMMY='test-only-placeholder'; HEAD={'X-TableLab':'local-ui'}; DATA=Path(__file__).resolve().parents[2]/'data'
def result(plan): return {'plan':plan,'model_reported':'contract-test-double','usage':{}}

def test_no_records_sent():
    s=Catalog(DATA).schema('nba')
    for c in s['columns']:c['examples']=['LOCAL_SAMPLE'];c['missing']=83938
    prev=Plan(dataset='nba',operation='predict',model_id='a'*32,record={'age':'LOCAL_RECORD'},message='LOCAL_FREE_TEXT')
    p=make_payload('这个字段代表什么？',s,[{'id':'nba:field:player_name','text':'LOCAL_SAMPLE','source':'LOCAL_PATH'}],prev,'a'*32);wire=json.dumps(p)
    for forbidden in ['LOCAL_SAMPLE','LOCAL_RECORD','LOCAL_PATH','LOCAL_FREE_TEXT','a'*32,s['sha256'],'83938']: assert forbidden not in wire
    assert p['response_format']=={'type':'json_object'} and p['stream'] is False
    assert json.loads(p['messages'][1]['content'])['retrieved_documents'][0]['title']=='player_name'

@pytest.mark.parametrize('question,prev',[('密钥sk-'+'X'*25,None),('继续',Plan(dataset='house',operation='describe',filters=[{'column':'Neighborhood','op':'eq','value':'sk-'+'Y'*25}]))])
def test_no_accidentally_pasted_key(question,prev):
    spy=Mock(); client=DeepSeek(DUMMY,spy)
    with pytest.raises(DataError,match='疑似有密钥'):client.plan(question,Catalog(DATA).schema('house'),[],prev)
    spy.assert_not_called()

@pytest.mark.parametrize('code',[301,400,401,402,403,404,422,429,500,503])
def test_failed_http(code):
    s=Mock();s.post.return_value.status_code=code;s.post.return_value.text='SENSITIVE_DIAGNOSTIC'
    e=send_once(DUMMY,{},s);assert str(code) in e['error'] and 'SENSITIVE_DIAGNOSTIC' not in str(e)
    s.post.assert_called_once();a,k=s.post.call_args;assert a==(URL,) and k['allow_redirects'] is False

@pytest.mark.parametrize('val,finish',[('  ','stop'),('{','length'),('[]','stop'),('bad json','stop')])
def test_bad_json(val,finish):
    s=Mock();s.post.return_value.status_code=200;s.post.return_value.json.return_value={'choices':[{'message':{'content':val},'finish_reason':finish}]}
    assert 'error' in send_once(DUMMY,{},s)

def test_timeout_and_strip_extra():
    s=Mock();s.post.side_effect=requests.Timeout('TEST_DIAGNOSTIC');assert '超时' in send_once(DUMMY,{},s)['error']
    s=Mock();s.post.return_value.status_code=200;s.post.return_value.json.return_value={'model':'test','usage':{'total_tokens':11,'private':'discard'},'choices':[{'finish_reason':'stop','message':{'content':'{"dataset":"house"}','reasoning_content':'PRIVATE_TRACE'}}]}
    assert send_once(DUMMY,{},s)=={'model_reported':'test','usage':{'total_tokens':11},'plan':{'dataset':'house'}}

@pytest.mark.parametrize('plan',[{'dataset':'nba','operation':'describe'},{'dataset':'house','operation':'shell'},{'dataset':'house','operation':'aggregate','columns':['invented_column']}])
def test_wrong_plan(plan):
    c=DeepSeek(DUMMY,lambda k,p:result(plan))
    with pytest.raises(DataError):c.plan('示例',Catalog(DATA).schema('house'),[])

@pytest.mark.parametrize('ds,col',[('house','SalePrice'),('adult','income_gt_50k'),('nba','pts'),('students','math score')])
def test_real_data_mock_cloud(ds,col,tmp_path):
    df,_=Catalog(DATA).load(ds);filters=[{'column':'Neighborhood','op':'eq','value':'CollgCr'}] if ds=='house' else []
    spy=Mock(return_value=result(dict(dataset=ds,operation='aggregate',metric='mean',columns=[col],filters=filters)))
    app=TestClient(create_app(DATA,tmp_path,DeepSeek(DUMMY,spy)));state=app.get('/api/status').json()['model'];assert state['provider']=='deepseek' and not state['verified_by_request']
    for hdr in ({},{**HEAD,'Origin':'https://example.invalid'}):assert app.post('/api/propose',headers=hdr,json={'dataset':ds,'question':'均价？'}).status_code==403
    spy.assert_not_called();r=app.post('/api/propose',headers=HEAD,json={'dataset':ds,'question':col+' 的均值？'});assert r.status_code==200,r.text
    a=r.json();assert list(tmp_path.iterdir())==[] and '示例值' not in str(a['evidence']);assert DUMMY not in r.text
    b=app.post('/api/execute',headers=HEAD,json={'plan':a['plan'],'proposal_id':a['proposal_id']});assert b.status_code==200,b.text
    row=b.json()['table'][0];ref=df.loc[df.Neighborhood=='CollgCr'] if ds=='house' else df
    assert row['n_valid']==len(ref) and row['value']==pytest.approx(ref[col].mean());assert spy.call_count==1
    if ds=='house':assert row['n_valid']==150 and round(row['value'],2)==197965.77
    saved=(tmp_path/b.json()['run_id']/'proposal.json').read_text();assert DUMMY not in saved and json.loads(saved)['model']['provider']=='deepseek'
    assert app.get('/api/status').json()['model']['verified_by_request']

def test_remote_unavailable_never_executes(tmp_path):
    c=TestClient(create_app(DATA,tmp_path,DeepSeek(DUMMY,lambda key,body:{'error':'DeepSeek HTTP 401：测试用错误'})))
    r=c.post('/api/propose',headers=HEAD,json={'dataset':'house','question':'平均成交价？'});assert r.status_code==502 and '401' in r.json()['error'] and not list(tmp_path.iterdir())
