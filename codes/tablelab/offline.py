"""限制本 Python 进程联网；不是操作系统沙箱，也不控制独立 Ollama 进程。"""
import ipaddress
import sys

_INSTALLED=False

def _local(host):
    if host=='localhost': return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

def install_loopback_guard():
    global _INSTALLED
    if _INSTALLED: return
    def audit(event, args):
        if event=='socket.connect':
            address=args[1]
            if isinstance(address,tuple) and not _local(address[0]):
                raise PermissionError('离线模式禁止非回环地址连接。')
        if event=='socket.getaddrinfo' and not _local(args[0]):
            raise PermissionError('离线模式禁止外部 DNS 查询。')
    sys.addaudithook(audit)
    _INSTALLED=True
