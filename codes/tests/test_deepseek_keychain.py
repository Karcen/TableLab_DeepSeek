"""Test the local credential lifecycle without real credentials, network or Keychain."""
from unittest.mock import Mock, patch
import pytest
from tablelab.cloud import credentials as c


def test_first_start_then_reuse(capsys):
    store, prompt = Mock(), Mock(return_value='DUMMY_TEST_VALUE')
    store.get_password.side_effect = (None, 'DUMMY_TEST_VALUE')
    assert c.acquire(store, prompt=prompt) == c.acquire(store, prompt=prompt) == 'DUMMY_TEST_VALUE'
    assert prompt.call_count == 1
    store.set_password.assert_called_once_with(c.SERVICE, c.ACCOUNT, 'DUMMY_TEST_VALUE')
    assert 'DUMMY_TEST_VALUE' not in capsys.readouterr().out


def test_replace_does_not_require_previous_secret():
    store = Mock()
    assert c.acquire(store, replace=True, prompt=lambda:'DUMMY_NEW') == 'DUMMY_NEW'
    store.get_password.assert_not_called()
    store.set_password.assert_called_once_with(c.SERVICE, c.ACCOUNT, 'DUMMY_NEW')


@pytest.mark.parametrize('value',['','a b','abc\n','abc\x1b','中文'])
def test_invalid_input_is_never_persisted(value):
    store = Mock(); store.get_password.return_value=None
    with pytest.raises(ValueError): c.acquire(store,prompt=lambda:value)
    store.set_password.assert_not_called()


def test_locked_keychain_does_not_prompt_or_write():
    store=Mock();store.get_password.side_effect=PermissionError('locked');prompt=Mock()
    with pytest.raises(PermissionError): c.acquire(store,prompt=prompt)
    prompt.assert_not_called();store.set_password.assert_not_called()


def test_store_error_stops_startup():
    store=Mock();store.get_password.return_value=None;store.set_password.side_effect=PermissionError('locked')
    with pytest.raises(PermissionError):c.acquire(store,prompt=lambda:'DUMMY')


@pytest.mark.parametrize('answer,should_delete',[('DELETE',True),('',False)])
def test_forget_requires_confirmation(answer,should_delete):
    store=Mock();store.get_password.return_value='DUMMY'
    with patch.object(c,'native_store',return_value=store),patch('builtins.input',return_value=answer):c.manage('forget')
    assert store.delete_password.called is should_delete
    if should_delete:store.delete_password.assert_called_once_with(c.SERVICE,c.ACCOUNT)


def test_no_unsecured_store_on_other_operating_system():
    with patch.object(c.sys,'platform','linux'),pytest.raises(SystemExit,match='macOS'):c.native_store()


def test_noninteractive_secret_input_rejected():
    with patch.object(c.sys.stdin,'isatty',return_value=False),pytest.raises(SystemExit,match='终端'):c.hidden_input()
