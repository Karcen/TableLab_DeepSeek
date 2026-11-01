import json,subprocess,sys
import pytest
from fastapi.testclient import TestClient
from tablelab.web import create_app
from tablelab.contracts import Plan
from tablelab.llm import Ollama

class MockOllama:
    """仅验证本地协议，不验证语言模型的理解能力。"""
    def check_lock(self):return {'digest':'TEST_ONLY_NO_MODEL'}
    def plan(self,question,schema,evidence,previous=None,model_id=None):
        return Plan(dataset='house',operation='aggregate',columns=['SalePrice'],metric='mean'),{'model':'MOCK_TEST_ONLY','model_digest':'TEST_ONLY_NO_MODEL','elapsed_seconds':0}

@pytest.fixture
def client(root,tmp_path):
    return TestClient(create_app(root,tmp_path/'runs',MockOllama()))


def test_status_exposes_synthetic(client):
    data=client.get('/api/status').json()
    assert all(x['source_kind']=='synthetic_test' for x in data['datasets'])


def test_ui_static_no_cdn(client):
    text=client.get('/').text
    assert 'http://' not in text and 'https://' not in text
    assert 'cdn' not in text
    assert client.get('/static/app.js').status_code==200


def test_proposal_execution_and_export_protocol(client):
    headers={'X-TableLab':'local-ui'}
    proposed=client.post('/api/propose',headers=headers,json={'dataset':'house','question':'平均房价是多少？'}).json()
    assert proposed['model']['model']=='MOCK_TEST_ONLY'
    result=client.post('/api/execute',headers=headers,json={'plan':proposed['plan'],'proposal_id':proposed['proposal_id']}).json()
    assert result['status']=='ok' and result['interaction_mode']=='natural_language'
    assert client.get(f'/api/files/{result["run_id"]}/table.csv').status_code==200
    assert client.get(f'/api/files/{result["run_id"]}/secrets.txt').status_code==422


def test_cross_origin_write_blocked(client):
    r=client.post('/api/execute',headers={'X-TableLab':'local-ui','Origin':'https://attacker.example'},json={'plan':{'dataset':'house','operation':'describe'}})
    assert r.status_code==403


def test_missing_header_blocked(client):
    assert client.post('/api/execute',json={'plan':{'dataset':'house','operation':'describe'}}).status_code==403


def test_forged_evidence_blocked(client):
    r=client.post('/api/execute',headers={'X-TableLab':'local-ui'},json={'plan':{'dataset':'house','operation':'describe'},'evidence':[{'source':'fabricated'}]})
    assert r.status_code==422


def test_invalid_algorithm_blocked(client):
    r=client.post('/api/execute',headers={'X-TableLab':'local-ui'},json={'plan':{'dataset':'house','operation':'train','target':'SalePrice','features':['GrLivArea'],'algorithm':'xgboost'}})
    assert r.status_code==422


def test_local_only_client_does_not_inherit_proxy():
    assert Ollama().session.trust_env is False


def test_network_guard_rejects_external_before_connection():
    code='from tablelab.offline import install_loopback_guard\nimport socket\ninstall_loopback_guard()\nsocket.create_connection(("8.8.8.8",443))'
    r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert r.returncode!=0 and 'PermissionError' in r.stderr and '外部 DNS' in r.stderr


def test_network_guard_allows_loopback():
    code='from tablelab.offline import install_loopback_guard\nimport socket\ninstall_loopback_guard()\ns=socket.socket();s.bind(("127.0.0.1",0));s.listen(1)\nc=socket.create_connection(s.getsockname());a,_=s.accept();c.close();a.close();s.close()'
    subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
