import argparse
import csv
import os
from c4_local_ranking_eval import compute_local_ranking_metrics, load_jsonl
from evaluation_utils import compute_metrics
from run_evaluation import load_csv_records


def safe_e2e(path):
    if not path or not os.path.exists(path):
        return {'Hits@1': 0.0, 'F1': 0.0}
    return compute_metrics(load_csv_records(path))


def safe_local(path):
    if not path or not os.path.exists(path):
        return {'MRR': 0.0, 'Hits@1': 0.0, 'Hits@3': 0.0, 'valid_steps': 0}
    return compute_local_ranking_metrics(load_jsonl(path), ks=(1, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_csv', required=True)
    parser.add_argument('--prediction_dir', default='')
    parser.add_argument('--output_csv', required=True)
    args = parser.parse_args()
    rows = []
    with open(args.run_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            local = safe_local(row.get('c4_log_path', ''))
            pred_path = row.get('prediction_csv', '')
            if not pred_path and args.prediction_dir:
                candidates = [x for x in os.listdir(args.prediction_dir) if x.endswith('.csv')]
                pred_path = os.path.join(args.prediction_dir, candidates[-1]) if candidates else ''
            e2e = safe_e2e(pred_path)
            rows.append({**row, **{f'local_{k}': v for k, v in local.items()}, **{f'e2e_{k}': v for k, v in e2e.items()}})
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
