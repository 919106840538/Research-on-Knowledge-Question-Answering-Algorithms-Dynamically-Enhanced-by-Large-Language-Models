import argparse
import csv
import os
import subprocess
import sys


ABLATIONS = [
    ('full', []),
    ('w_o_critic_agent', ['--ablation', 'no_critic']),
    ('w_o_gamma', ['--ablation', 'no_gamma']),
    ('w_o_adaptive_constraint_adjustment', ['--ablation', 'no_adaptive']),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--operator', default='gpt-4o-mini')
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'operator', 'setting', 'status'])
        for name, extra in ABLATIONS:
            cmd = [sys.executable, 'r2kg_chatbot.py', '--dataset', args.dataset, '--operator', args.operator] + extra
            completed = subprocess.run(cmd, capture_output=True, text=True)
            status = 'ok' if completed.returncode == 0 else f'fail:{completed.stderr[:120]}'
            writer.writerow([args.dataset, args.operator, name, status])
            print([args.dataset, args.operator, name, status])


if __name__ == '__main__':
    main()
