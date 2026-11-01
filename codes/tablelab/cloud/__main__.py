import argparse

from . import credentials
from .deepseek_online import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TableLab DeepSeek 联网模式')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--replace-key', action='store_true')
    modes.add_argument('--forget-key', action='store_true')
    modes.add_argument('--session-only', action='store_true')
    args = parser.parse_args()
    if args.replace_key or args.forget_key:
        credentials.manage('replace' if args.replace_key else 'forget')
    else:
        main(session_only=args.session_only)
