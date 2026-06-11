import argparse
import csv
import os
from evaluation_utils import compute_metrics, summarize_predictions


def load_csv_records(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == 'qid':
                continue
            if len(row) < 3:
                continue
            records.append({'qid': row[0], 'prediction': row[1], 'gt_label': row[2]})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_txt', required=False, default='')
    args = parser.parse_args()

    records = load_csv_records(args.input_csv)
    summary = summarize_predictions(records)
    text = str(summary)
    print(text)
    if args.output_txt:
        os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
        with open(args.output_txt, 'w', encoding='utf-8') as f:
            f.write(text)


if __name__ == '__main__':
    main()
