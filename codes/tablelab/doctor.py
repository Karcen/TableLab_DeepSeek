"""验收本地文件和模型锁；不负责联网安装。"""
import argparse
import hashlib
from pathlib import Path
import json
import platform
import sys
import requests
from datetime import datetime,timezone
from .catalog import Catalog,SPECS
from .llm import Ollama,MODEL
from .engine import write_json
from .offline import install_loopback_guard


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='data');ap.add_argument('--lock-model',action='store_true');ap.add_argument('--require-all',action='store_true')
    args=ap.parse_args();install_loopback_guard();client=Ollama()
    if args.lock_model:
        info=client.model_info();Path('runtime').mkdir(exist_ok=True)
        write_json('runtime/model.lock.json',{'name':MODEL,'digest':info['digest'],'size_bytes':info['size'],
            'ollama_version':client._request('/api/version')['version'],'locked_utc':datetime.now(timezone.utc).isoformat()})
        print('本地模型指纹已记录。')
    checks=[];cat=Catalog(Path(args.root))
    for name in SPECS:
        try:
            df,entry=cat.load(name)
            if entry['source_kind']=='synthetic_test': raise ValueError('合成测试数据不能通过正式验收。')
            checks.append({'dataset':name,'status':'pass','rows':len(df),'sha256':entry['sha256']})
        except (ValueError,KeyError,FileNotFoundError) as e:
            checks.append({'dataset':name,'status':'fail','error':str(e)})
    try:
        model=client.check_lock();model_check={'status':'pass','name':MODEL,'digest':model['digest']}
    except (ValueError,FileNotFoundError) as e: model_check={'status':'fail','error':str(e)}
    except requests.RequestException as e: model_check={'status':'fail','error':str(e)}
    report={'python':sys.version,'platform':platform.platform(),'datasets':checks,'model':model_check,
            'scope':'静态准备检查，不代替真实断网自然语言端到端验收。'}
    print(json.dumps(report,ensure_ascii=False,indent=2));Path('verification').mkdir(exist_ok=True)
    write_json('verification/local_readiness.json',report)
    if args.require_all and (any(x['status']!='pass' for x in checks) or model_check['status']!='pass'): raise SystemExit(1)

if __name__=='__main__': main()
