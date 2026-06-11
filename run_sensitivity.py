import argparse
import csv
import os
import subprocess
import sys


def run_command(command):
    completed = subprocess.run(command, shell=False, capture_output=True, text=True)
    return completed.returncode, completed.stdout, completed.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--operators', nargs='+', default=['gpt-4o-mini'])
    parser.add_argument('--iter_limits', nargs='+', type=int, default=[8, 12, 15])
    parser.add_argument('--budgets', nargs='+', type=int, default=[20, 40, 60])
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'operator', 'iter_limit', 'budget', 'status'])
        for operator in args.operators:
            for iter_limit in args.iter_limits:
                for budget in args.budgets:
                    cmd = [sys.executable, 'r2kg_chatbot.py', '--dataset', args.dataset, '--operator', operator, '--iter_limit', str(iter_limit)]
                    code, stdout, stderr = run_command(cmd)
                    status = 'ok' if code == 0 else f'fail:{stderr[:120]}'
                    writer.writerow([args.dataset, operator, iter_limit, budget, status])
                    print([args.dataset, operator, iter_limit, budget, status])


if __name__ == '__main__':
    main()
