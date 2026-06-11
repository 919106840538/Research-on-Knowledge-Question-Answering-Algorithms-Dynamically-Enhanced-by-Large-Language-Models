import argparse
import csv
import os
import random
import subprocess
import sys


DEFAULT_WEIGHTS = [0.3, 0.4, 0.1, 0.1, 0.1]
WEIGHT_NAMES = ['sem', 'str', 'cov', 'conf', 'eff']


def run(cmd):
    completed = subprocess.run(cmd, capture_output=True, text=True)
    return 'ok' if completed.returncode == 0 else f'fail:{completed.stderr[:160]}'


def weight_string(values):
    return ','.join(str(round(x, 4)) for x in values)


def rescale_single(index, value):
    out = [0.0] * 5
    out[index] = value
    rest_sum = sum(DEFAULT_WEIGHTS[j] for j in range(5) if j != index)
    for j in range(5):
        if j != index:
            out[j] = (1 - value) * DEFAULT_WEIGHTS[j] / rest_sum
    return out


def perturb(epsilon):
    vals = [max(0.0, w + random.uniform(-epsilon, epsilon)) for w in DEFAULT_WEIGHTS]
    total = sum(vals) or 1.0
    return [x / total for x in vals]


def base_cmd(args, setting_name, extra):
    log_path = os.path.join(args.log_dir, f'{args.dataset}_{setting_name}.jsonl')
    return [
        sys.executable, 'r2kg_chatbot.py',
        '--dataset', args.dataset,
        '--operator', args.operator,
        '--supervisor', args.supervisor,
        '--iter_limit', str(args.iter_limit),
        '--budget', str(args.budget),
        '--c4_log_path', log_path,
    ] + extra, log_path


def settings_for_mode(mode):
    if mode == 'incremental':
        return [('c3_base', ['--c4_ranker', 'base']), ('c4_hybrid', ['--c4_ranker', 'hybrid'])]
    if mode == 'ablation':
        return [
            ('full', ['--c4_ranker', 'hybrid']),
            ('w_o_sem', ['--c4_ranker', 'hybrid', '--ablation', 'no_sem']),
            ('w_o_str', ['--c4_ranker', 'hybrid', '--ablation', 'no_str']),
            ('w_o_cov', ['--c4_ranker', 'hybrid', '--ablation', 'no_cov']),
            ('w_o_conf', ['--c4_ranker', 'hybrid', '--ablation', 'no_conf']),
            ('w_o_eff', ['--c4_ranker', 'hybrid', '--ablation', 'no_eff']),
        ]
    if mode == 'baselines':
        return [(x, ['--c4_ranker', x]) for x in ['bm25', 'sentence_bert', 'transe', 'tog', 'hybrid']]
    return []


def sensitivity_settings(samples_per_epsilon=20):
    settings = []
    for i, name in enumerate(WEIGHT_NAMES):
        for step in range(11):
            value = step / 10
            settings.append((f'sweep_{name}_{value:.1f}', ['--c4_ranker', 'hybrid', '--c4_weights', weight_string(rescale_single(i, value))]))
    for eps in [0.05, 0.10, 0.15, 0.20]:
        for sample_idx in range(samples_per_epsilon):
            settings.append((f'perturb_{eps:.2f}_{sample_idx}', ['--c4_ranker', 'hybrid', '--c4_weights', weight_string(perturb(eps))]))
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['incremental', 'ablation', 'baselines', 'sensitivity', 'all'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--operator', default='gpt-4o-mini')
    parser.add_argument('--supervisor', default='gpt-4o')
    parser.add_argument('--iter_limit', type=int, default=15)
    parser.add_argument('--budget', type=int, default=40)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--log_dir', default='results/c4_logs')
    parser.add_argument('--samples_per_epsilon', type=int, default=20)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    modes = ['incremental', 'ablation', 'baselines', 'sensitivity'] if args.mode == 'all' else [args.mode]
    rows = []
    for mode in modes:
        settings = sensitivity_settings(args.samples_per_epsilon) if mode == 'sensitivity' else settings_for_mode(mode)
        for name, extra in settings:
            cmd, log_path = base_cmd(args, name, extra)
            status = run(cmd)
            row = [mode, args.dataset, args.operator, name, ' '.join(extra), log_path, status]
            rows.append(row)
            print(row)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'dataset', 'operator', 'setting', 'extra_args', 'c4_log_path', 'status'])
        writer.writerows(rows)


if __name__ == '__main__':
    main()
