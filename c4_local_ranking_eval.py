import argparse
import csv
import json
from collections import deque
from evaluation_utils import normalize_label_set


def action_reaches_answer(adapter, target_entity, answers, remaining_depth):
    answers = normalize_label_set(answers)
    if not answers:
        return False
    start_key = str(target_entity).strip().lower().replace(' ', '_')
    if start_key in answers:
        return True
    if adapter is None or remaining_depth <= 0:
        return False
    visited = {target_entity}
    queue = deque([(target_entity, 0)])
    while queue:
        entity, depth = queue.popleft()
        if depth >= remaining_depth:
            continue
        for action in adapter.get_actions(entity):
            nxt = action.get('target_entity')
            if nxt in visited:
                continue
            if str(nxt).strip().lower().replace(' ', '_') in answers:
                return True
            visited.add(nxt)
            queue.append((nxt, depth + 1))
    return False


def annotate_correct_actions(ranked_actions, adapter, answers, remaining_depth):
    out = []
    for item in ranked_actions:
        action = item.get('action', item)
        x = dict(item)
        x['is_correct_action'] = action_reaches_answer(adapter, action.get('target_entity'), answers, remaining_depth)
        out.append(x)
    return out


def reciprocal_rank(ranked_items):
    for idx, item in enumerate(ranked_items, start=1):
        if item.get('is_correct_action'):
            return 1.0 / idx
    return None


def hits_at_k(ranked_items, k):
    if not any(x.get('is_correct_action') for x in ranked_items):
        return None
    return 1.0 if any(x.get('is_correct_action') for x in ranked_items[:k]) else 0.0


def compute_local_ranking_metrics(records, ks=(1, 3)):
    rr_values = []
    hits = {k: [] for k in ks}
    for record in records:
        ranked = record.get('ranked_candidates', [])
        if not any(x.get('is_correct_action') for x in ranked):
            continue
        rr = reciprocal_rank(ranked)
        if rr is not None:
            rr_values.append(rr)
        for k in ks:
            h = hits_at_k(ranked, k)
            if h is not None:
                hits[k].append(h)
    metrics = {'MRR': round(sum(rr_values) / len(rr_values) * 100, 3) if rr_values else 0.0}
    for k, values in hits.items():
        metrics[f'Hits@{k}'] = round(sum(values) / len(values) * 100, 3) if values else 0.0
    metrics['valid_steps'] = len(rr_values)
    return metrics


def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_csv', default='')
    args = parser.parse_args()
    metrics = compute_local_ranking_metrics(load_jsonl(args.input_jsonl))
    print(metrics)
    if args.output_csv:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(metrics.keys())
            writer.writerow(metrics.values())


if __name__ == '__main__':
    main()
