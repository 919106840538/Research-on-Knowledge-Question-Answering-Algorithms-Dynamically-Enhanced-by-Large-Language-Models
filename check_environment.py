import importlib.util
import os


def has_module(name):
    return importlib.util.find_spec(name) is not None


def main():
    report = {
        'openai': has_module('openai'),
        'SPARQLWrapper': has_module('SPARQLWrapper'),
        'dotenv': has_module('dotenv'),
        'OPENAI_KEY': bool(os.getenv('OPENAI_KEY')),
    }
    print(report)
    if not all(report.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
