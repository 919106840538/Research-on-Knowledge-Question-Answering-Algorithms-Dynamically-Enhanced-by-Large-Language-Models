import math
from ast import literal_eval
from collections import Counter


def normalize_label_set(label_obj):
    try:
        if isinstance(label_obj, str):
            values = literal_eval(label_obj)
        else:
            values = label_obj
    except Exception:
        values = [label_obj] if label_obj is not None else []
    if not isinstance(values, (list, set, tuple)):
        values = [values]
    return set(str(v).strip().lower().replace("'", '').replace(' ', '_') for v in values if str(v).strip())


def sample_f1(true_set, pred_set):
    if not true_set and not pred_set:
        return 1.0
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def hits_at_1(true_set, pred_set):
    return 1.0 if (true_set & pred_set) else 0.0


def compute_metrics(records):
    normalized = []
    for rec in records:
        pred = normalize_label_set(rec['prediction'])
        true = normalize_label_set(rec['gt_label'])
        normalized.append((true, pred))
    if not normalized:
        return {'Hits@1': 0.0, 'F1': 0.0}
    hits = sum(hits_at_1(t, p) for t, p in normalized) / len(normalized)
    f1 = sum(sample_f1(t, p) for t, p in normalized) / len(normalized)
    return {'Hits@1': round(hits * 100, 3), 'F1': round(f1 * 100, 3)}


def mcnemar_pvalue(system_a, system_b, gold):
    b = c = 0
    for a_pred, b_pred, g in zip(system_a, system_b, gold):
        g_set = normalize_label_set(g)
        a_ok = hits_at_1(g_set, normalize_label_set(a_pred))
        b_ok = hits_at_1(g_set, normalize_label_set(b_pred))
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return math.exp(-0.5 * chi2)


def bootstrap_pvalue(system_a, system_b, gold):
    # deterministic approximation without random dependency
    diffs = []
    for a_pred, b_pred, g in zip(system_a, system_b, gold):
        g_set = normalize_label_set(g)
        diffs.append(sample_f1(g_set, normalize_label_set(a_pred)) - sample_f1(g_set, normalize_label_set(b_pred)))
    if not diffs:
        return 1.0
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((x - mean_diff) ** 2 for x in diffs) / max(len(diffs) - 1, 1)
    if variance == 0:
        return 1.0 if mean_diff == 0 else 0.0
    z = abs(mean_diff) / math.sqrt(variance / len(diffs))
    return math.exp(-0.717 * z - 0.416 * z * z)


def summarize_predictions(records):
    metrics = compute_metrics(records)
    answer_sizes = Counter(len(normalize_label_set(r['prediction'])) for r in records)
    return {'metrics': metrics, 'answer_size_hist': dict(answer_sizes)}
