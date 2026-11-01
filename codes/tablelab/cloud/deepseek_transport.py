"""Single outbound destination. This worker receives JSON on stdin, never key arguments."""
import json
import sys
import requests

URL = 'https://api.deepseek.com/chat/completions'
MODEL = 'deepseek-flash'
ERRORS = {400:'请求格式有误，请核对接口配置。', 401:'密钥无效。请在控制台检查新密钥，然后重启联网演示。',
          402:'账户余额不足，请检查 DeepSeek 账户余额。', 403:'请求未获授权，请检查账户权限。',
          404:'接口或模型不可用，请核对官方模型名称。', 422:'服务未接受请求参数。',
          429:'请求过于频繁，请稍后再点生成。', 500:'服务出错，请稍后再试。', 503:'服务繁忙，请稍后再试。'}

def send_once(key, payload, session):
    try:
        r = session.post(URL, headers={'Authorization':'Bearer ' + key}, json=payload,
                         timeout=(8, 65), allow_redirects=False)
    except requests.Timeout:
        return {'error':'请求超时，服务端可能已处理请求。程序没有自动重发。'}
    except requests.RequestException:
        return {'error':'未能通过 HTTPS 连接 DeepSeek。请检查网络和系统时间。'}
    if r.status_code != 200:
        return {'error':f'DeepSeek HTTP {r.status_code}：' + ERRORS.get(r.status_code,'请求未成功；没有跟随重定向或自动重试。')}
    try:
        obj = r.json()
        choice = obj['choices'][0]
        text = choice['message']['content']
        if choice['finish_reason'] != 'stop' or not isinstance(text, str) or not text.strip():
            return {'error':'服务没有返回完整方案。没有执行计算。'}
        # Do not forward provider diagnostics, response headers or reasoning traces.
        ans = json.loads(text)
        if not isinstance(ans, dict):
            return {'error':'模型返回的 JSON 不是单个方案。'}
        usage = obj.get('usage') or {}
        return {'plan':ans, 'model_reported':obj.get('model'),
                'usage':{k:v for k,v in usage.items() if k in ('prompt_tokens','completion_tokens','total_tokens') and type(v) is int}}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return {'error':'服务返回了无法解析的响应。没有执行计算。'}

def main():
    req = json.load(sys.stdin)
    with requests.Session() as session:
        # Never inherit proxy settings, .netrc credentials or alternate CA path settings.
        session.trust_env = False
        reply = send_once(req['key'], req['payload'], session)
    json.dump(reply,sys.stdout,ensure_ascii=True,allow_nan=False)

if __name__ == '__main__':
    main()
