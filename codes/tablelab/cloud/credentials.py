"""One app-specific credential in macOS Keychain. No file or environment fallback."""
from __future__ import annotations

import getpass
import sys
import warnings
from typing import Callable, Protocol

SERVICE = 'TableLab.DeepSeek'
ACCOUNT = 'api.deepseek.com'


class Store(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...


def native_store() -> Store:
    if sys.platform != 'darwin':
        raise SystemExit('此入口使用 macOS 钥匙串。仅临时输入可运行 python -m tablelab.cloud --session-only。')
    try:
        from keyring.backends.macOS import Keyring
    except ModuleNotFoundError as exc:
        raise SystemExit('钥匙串依赖未装好。请在运行入口选 8，再重试。') from exc
    _ = Keyring.priority  # Require the native Security framework before prompting.
    return Keyring()


def hidden_input() -> str:
    if not sys.stdin.isatty():
        raise SystemExit('请从本机终端打开；不接受从文件或管道导入密钥。')
    with warnings.catch_warnings():
        warnings.simplefilter('error', getpass.GetPassWarning)
        return getpass.getpass('粘贴新的 DeepSeek API Key（输入不会显示），按回车：')


def validated(value: str) -> str:
    if not value or not value.isascii() or not value.isprintable() or any(c.isspace() for c in value):
        raise ValueError('密钥为空或含有空格、换行或非法字符。请重新复制。')
    return value


def acquire(store: Store, *, replace: bool = False, prompt: Callable[[], str] = hidden_input) -> str:
    if not replace:
        saved = store.get_password(SERVICE, ACCOUNT)
        if saved is not None:
            return validated(saved)
    print('按回车会将密钥存入当前用户的 Mac 钥匙串（TableLab.DeepSeek）；按 Control-C 取消。')
    key = validated(prompt())
    store.set_password(SERVICE, ACCOUNT, key)
    return key


def manage(action: str) -> None:
    if action not in ('replace', 'forget'):
        raise ValueError('未知的密钥管理操作。')
    store = native_store()
    if action == 'replace':
        acquire(store, replace=True)
        print('新密钥已保存。关闭已运行的演示服务，重新选 9 后生效。')
    else:
        if store.get_password(SERVICE, ACCOUNT) is None:
            print('当前用户的钥匙串中没有 TableLab 密钥。')
            return
        if input('仅删除本机保存的 TableLab 密钥？输入 DELETE 确认：') != 'DELETE':
            print('已取消。')
            return
        store.delete_password(SERVICE, ACCOUNT)
        print('本机副本已删除。请关闭已运行的服务；撤销密钥须到 DeepSeek 平台操作。')
