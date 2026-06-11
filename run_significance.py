import argparse
import csv
from evaluation_utils import mcnemar_pvalue, bootstrap_pvalue


def load_rows(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] == 'qid' or len(row) < 3:
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system_a', required=True)
    parser.add_argument('--system_b', required=True)
    args = parser.parse_args()

    a_rows = load_rows(args.system_a)
    b_rows = load_rows(args.system_b)
    gold = [row[2] for row in a_rows]
    a_pred = [row[1] for row in a_rows]
    b_pred = [row[1] for row in b_rows]

    print({
        'mcnemar_pvalue': mcnemar_pvalue(a_pred, b_pred, gold),
        'bootstrap_pvalue': bootstrap_pvalue(a_pred, b_pred, gold),
    })


if __name__ == '__main__':
    main()
