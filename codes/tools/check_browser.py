"""已启动本机测试服务器后的浏览器组件验收；只使用明确标注的合成夹具。"""
import json
import argparse
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'verification'
parser=argparse.ArgumentParser();parser.add_argument('--browser-path');parser.add_argument('--screens',default=str(ROOT/'verification/rendered'))
args=parser.parse_args()
SCREEN=Path(args.screens);SCREEN.mkdir(parents=True,exist_ok=True)
checks=[];external=[];errors=[]
def record(name,passed):
    checks.append({'check':name,'passed':bool(passed)})
    assert passed,name

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path=args.browser_path,args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
    page.on('pageerror',lambda e: errors.append(str(e)))
    page.on('request',lambda r: external.append(r.url) if urlparse(r.url).hostname not in ('127.0.0.1','localhost',None) else None)
    page.goto('http://127.0.0.1:8765',wait_until='networkidle')
    record('四个数据集入口',page.locator('.dataset').count()==4)
    record('合成夹具警示', '合成测试' in page.locator('#source-banner').inner_text())
    record('缺失模型明确报告', '未就绪' in page.locator('#model-state').inner_text())
    page.locator('#question').fill('按社区比较平均成交价。')
    page.locator('#propose').click()
    record('不伪装自然语言模型', '模型未就绪' in page.locator('#error').inner_text())
    page.locator('#editor-details').evaluate('(e)=>e.open=true')
    plan={'dataset':'house','operation':'aggregate','columns':['SalePrice'],'group_by':['Neighborhood'],'metric':'mean'}
    page.locator('#plan').fill(json.dumps(plan))
    page.locator('#execute').click()
    page.locator('#result-card').wait_for(state='visible')
    record('实际汇总结果出现',page.locator('#table tbody tr').count()==3)
    record('服务端检索依据出现',page.locator('#evidence-card').is_visible())
    page.screenshot(path=str(SCREEN/'desktop.png'),full_page=True)
    # 每个数据集实际运行一次预先限定的分析模型。
    for dataset,idx in [('house',0),('adult',1),('nba',2),('students',3)]:
        page.locator('.dataset').nth(idx).click();page.wait_for_timeout(300)
        page.locator('#editor-details').evaluate('(e)=>e.open=true')
        plan=json.loads((ROOT/'examples'/f'{dataset}_plan.json').read_text())
        page.locator('#plan').fill(json.dumps(plan))
        page.locator('#execute').click()
        page.wait_for_function("document.querySelector('#result-card').hidden===false && document.querySelector('#progress').hidden===true",timeout=45000)
        record(dataset+'：训练得到验证指标',page.locator('#metrics .metric').count()>=3)
        record(dataset+'：预测表单存在',page.locator('#prediction-card').is_visible())
        record(dataset+'：PDF 导出存在',page.locator('#downloads a').filter(has_text='diagnostic.pdf').count()==1)
    # 新生预测：空值是界面允许的显式缺失，按训练集拟合的填补器处理。
    page.locator('#predict').click()
    page.wait_for_function("document.querySelector('#progress').hidden===true",timeout=45000)
    record('新记录预测成功','预测' in page.locator('#answer').inner_text())
    # 缩窄布局仅验证重排和页面溢出，不声称真实手机浏览器验收。
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(SCREEN/'narrow.png'),full_page=True)
    record('窄屏不横向溢出',page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'))
    record('无 JavaScript 未捕获错误',not errors)
    record('页面请求均为本机来源',not external)
    browser.close()
(OUT/'browser_checks.json').write_text(json.dumps({'scope':'Chromium/Linux；合成数据；人工参数执行；未运行真实 LLM','checks':checks,'external_requests':external,'page_errors':errors},ensure_ascii=False,indent=2))
print(json.dumps({'passed':len(checks),'scope':'synthetic UI component checks'},ensure_ascii=False))
