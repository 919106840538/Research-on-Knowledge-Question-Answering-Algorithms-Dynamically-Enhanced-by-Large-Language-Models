import argparse
import csv
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--operators', nargs='+', default=['gpt-4o-mini', 'gpt-4o'])
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'operator', 'status'])
        for operator in args.operators:
            cmd = [sys.executable, 'r2kg_chatbot.py', '--dataset', args.dataset, '--operator', operator]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            status = 'ok' if completed.returncode == 0 else f'fail:{completed.stderr[:120]}'
            writer.writerow([args.dataset, operator, status])
            print([args.dataset, operator, status])


if __name__ == '__main__':
    main()
