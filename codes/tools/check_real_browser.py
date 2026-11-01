"""浏览器人工参数验收；不调用或模拟语言模型，不代替 Mac 断网验收。"""
from pathlib import Path
import json,os,sys,subprocess,time,urllib.request
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, Error as BrowserError
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'verification'; SCREEN=Path(os.environ.get('TABLELAB_BROWSER_SCREENSHOTS','/tmp/tablelab_browser_screens'))
SCREEN.mkdir(parents=True,exist_ok=True)
checks=[];external=[];errors=[]
def record(name,condition):
    checks.append({'check':name,'passed':bool(condition)})
    assert condition,name

def main():
    env=os.environ.copy();env['TABLELAB_DATA_ROOT']=str(ROOT/'data');env['TABLELAB_ARTIFACT_ROOT']=str(ROOT/'artifacts/ui_checks')
    log=(OUT/'browser_server.log').open('w')
    proc=subprocess.Popen([sys.executable,'-m','tablelab.web'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    result={'scope':'Chromium/Linux；上传课程数据；人工参数；真实自然语言、Mac 和物理断网均不在检查范围','status':'not_completed','checks':checks}
    try:
        time.sleep(3)
        assert proc.poll() is None,'测试服务器启动失败；查看 browser_server.log'
        with urllib.request.urlopen('http://127.0.0.1:8765/api/status',timeout=15) as response:
            record('本机 HTTP 状态接口可达',response.status==200)
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1440,'height':1050},device_scale_factor=1)
            page.on('pageerror',lambda e: errors.append(str(e)))
            page.on('request',lambda r: external.append(r.url) if urlparse(r.url).hostname not in ('127.0.0.1','localhost',None) else None)
            page.goto('http://127.0.0.1:8765',wait_until='networkidle',timeout=20000)
            record('四个数据集入口',page.locator('.dataset').count()==4)
            record('课程数据来源提示','合成测试' not in page.locator('#source-banner').inner_text())
            record('模型未就绪明确提示','未就绪' in page.locator('#model-state').inner_text())
            page.locator('#question').fill('按社区比较平均成交价。');page.locator('#propose').click()
            page.wait_for_timeout(500)
            record('缺少模型时不伪装回答',bool(page.locator('#error').inner_text()))
            page.locator('#editor-details').evaluate('(e)=>e.open=true')
            page.locator('#plan').fill(json.dumps({'dataset':'house','operation':'aggregate','columns':['SalePrice'],'group_by':['Neighborhood'],'metric':'mean'}))
            page.locator('#execute').click();page.locator('#result-card').wait_for(state='visible')
            record('上传房价汇总结果显示',page.locator('#table tbody tr').count()>0)
            record('本地检索证据显示',page.locator('#evidence-card').is_visible())
            page.screenshot(path=str(SCREEN/'desktop.png'),full_page=True)
            for dataset,idx in [('house',0),('adult',1),('nba',2),('students',3)]:
                page.locator('.dataset').nth(idx).click();page.wait_for_timeout(300)
                page.locator('#editor-details').evaluate('(e)=>e.open=true')
                page.locator('#plan').fill((ROOT/'examples'/f'{dataset}_plan.json').read_text())
                page.locator('#execute').click()
                page.wait_for_function("document.querySelector('#result-card').hidden===false && document.querySelector('#progress').hidden===true",timeout=60000)
                record(dataset+'：指标显示',page.locator('#metrics .metric').count()>=3)
                record(dataset+'：预测表单',page.locator('#prediction-card').is_visible())
                record(dataset+'：PDF 导出',page.locator('#downloads a').filter(has_text='diagnostic.pdf').count()==1)
            page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(SCREEN/'narrow.png'),full_page=True)
            record('窄屏无横向页面溢出',page.evaluate('document.documentElement.scrollWidth <= innerWidth+1'))
            record('未捕获 JS 错误为零',not errors);record('未观察到页面外部请求',not external)
            browser.close();result['status']='passed'
    except (BrowserError,AssertionError) as error:
        result['status']='blocked_or_failed';result['error']=str(error)
    finally:
        proc.terminate();proc.wait(timeout=10);log.close()
        result['checks']=checks;result['external_requests']=external;result['page_errors']=errors
        (OUT/'real_browser_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
