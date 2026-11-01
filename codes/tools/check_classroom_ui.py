"""Exercise the interface in Chromium's memory, with actual FastAPI and course data.

The test transport connects DOM fetch calls to TestClient; it does not test browser
navigation to loopback, the operating system's offline state, or a real Qwen model.
The deterministic planner is used only to test presentation of a model proposal.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from playwright.sync_api import sync_playwright
from tablelab.web import create_app
from tablelab.contracts import Plan
from tablelab.engine import write_json

class TestPlanner:
    def check_lock(self): return {'digest':'interface-test-only'}
    def plan(self,question,schema,evidence,previous=None,model_id=None):
        if schema['id']!='house': raise ValueError('该接口替身只生成房价例题。')
        plan=Plan(dataset='house',operation='aggregate',columns=['SalePrice'],metric='mean',filters=[{'column':'Neighborhood','op':'eq','value':'CollgCr'}]);return plan,{'model':'TEST_ONLY','model_digest':'interface-test-only'}

def main():
    cli=argparse.ArgumentParser();cli.add_argument('--browser-path',required=True);args=cli.parse_args();os.chdir(ROOT)
    checks=[];external=[];errors=[]
    def check(label,test):
        if not test:raise AssertionError(label)
        checks.append(label)
    def wait(page):page.wait_for_function("document.querySelector('#progress').hidden && !document.querySelector('#propose').disabled")
    static=ROOT/'tablelab/static'
    with TemporaryDirectory() as tmp,TestClient(create_app(ROOT/'data',Path(tmp),llm=TestPlanner())) as client,sync_playwright() as pw:
        chrome=pw.chromium.launch(executable_path=args.browser_path,headless=True);page=chrome.new_page(viewport={'width':1360,'height':880});page.on('pageerror',lambda e:errors.append(str(e)))
        def traffic(r):
            if urlsplit(r.url).scheme in ('http','https'):external.append(r.url)
        page.on('request',traffic)
        def transport(path,body):
            if not path.startswith('/api/') or '://' in path:raise ValueError('Only application API paths allowed')
            response=client.get(path) if body is None else client.post(path,json=json.loads(body),headers={'X-TableLab':'local-ui','Origin':'http://127.0.0.1:8765'})
            return {'status':response.status_code,'body':response.text}
        page.expose_function('qaApi',transport)
        dom=(static/'index.html').read_text().replace('<script defer src="/static/app.js"></script>','').replace('<link rel="stylesheet" href="/static/style.css">','<style>'+(static/'style.css').read_text()+'</style>');page.set_content(dom)
        page.evaluate("() => { window.fetch=async (path,init={})=>{let x=await window.qaApi(path,init.body||null);return new Response(x.body,{status:x.status,headers:{'Content-Type':'application/json'}})}; }")
        page.add_script_tag(content=(static/'app.js').read_text());wait(page)
        check('实表：四个标签；房价行数 1,460',page.locator('.dataset').count()==4 and page.locator('#nrows').inner_text()=='1,460')
        page.locator('#field-search').fill('居住面积');check('按中文查英文字段',page.locator('#schema').inner_text().find('GrLivArea')>=0)
        page.locator('#field-search').fill('没有这种字段199');check('搜索空状态',page.locator('#field-count').inner_text()=='找到 0 个字段')
        page.locator('#examples button').first.click();check('示例仅填入，不自动执行',page.locator('#execute').is_disabled() and 'CollgCr' in page.locator('#question').input_value())
        page.locator('#propose').click();wait(page);check('协议替身方案：字段和筛选清晰显示','Neighborhood ＝ CollgCr' in page.locator('#plan-summary').inner_text());page.locator('#evidence-card summary').click();check('检索依据可展开',page.locator('.evidence-item').first.is_visible());page.locator('#execute').click();wait(page)
        check('真实执行：150 条与 197,965.77 美元','197,965.77' in page.locator('#headline').inner_text() and '150' in page.locator('#table').inner_text());check('可查看本轮执行器代码',page.locator('#downloads a',has_text='executor_source.py').count()==1)
        page.locator('#question').fill('问一个新问题');check('改题即撤下旧结果和执行按钮',page.locator('#result-card').is_hidden() and page.locator('#execute').is_disabled());page.locator('#clear').click();check('新会话清空问题',page.locator('#question').input_value()=='')
        for i,d in enumerate(('house','adult','nba','students')):
            page.locator('.dataset').nth(i).click();wait(page);form=json.loads((ROOT/'examples'/f'{d}_plan.json').read_text());page.locator('#editor-details summary').click();page.locator('#plan').fill(json.dumps(form));page.locator('#execute').click();wait(page);check(d+'：真数据人工参数重新训练',page.locator('#metrics .metric').count()>=3 and page.locator('#result-origin').inner_text()=='人工参数');check(d+'：单条预测表单',page.locator('#prediction-card').is_visible());check(d+'：诊断图 PDF',page.locator('#figures a').first.get_attribute('href').endswith('diagnostic.pdf'))
            if d=='house':
                inputs=page.locator('#prediction-fields input');inputs.nth(0).fill('1500');inputs.nth(1).fill('6');inputs.nth(2).fill('CollgCr');page.locator('#predict').click();wait(page);check('真实单条估价返回',page.locator('#table tbody tr').count()==1)
        page.locator('#projector').click();check('投影模式收起词典',page.locator('.sidebar').is_hidden() and page.locator('#projector').get_attribute('aria-pressed')=='true');page.locator('#projector').click();page.set_viewport_size({'width':390,'height':844});check('窄屏无横向溢出',page.evaluate('document.documentElement.scrollWidth===innerWidth'))
        replay=chrome.new_page();replay.on('request',traffic);replay.on('pageerror',lambda e:errors.append(str(e)));replay.set_content((ROOT/'docs/classroom_replay.html').read_text());wait(replay)
        for i in range(4):
            replay.locator('.dataset').nth(i).click();wait(replay)
            for j in range(replay.locator('#examples button').count()):
                replay.locator('#examples button').nth(j).click();replay.locator('#propose').click();wait(replay);replay.locator('#execute').click();wait(replay);check(f'回放 {i+1}.{j+1}：只读、有结果、可导出',replay.locator('#question').get_attribute('readonly') is not None and replay.locator('#table tbody tr').count()>0 and replay.locator('#downloads a').count()>=3)
        check('两种页面的内存测试无 JS 异常',not errors);check('内存测试未发出 HTTP 或 HTTPS 页面请求',not external);chrome.close()
    write_json(ROOT/'verification/interface/browser_memory_checks.json',{'test':'Chromium in-memory DOM, real ASGI TestClient and course data; simulated proposal only. No browser-to-localhost or Qwen/Mac validation.','checks':checks,'errors':errors,'observed_http_requests':external});print(len(checks),'checks passed')
if __name__=='__main__':main()
