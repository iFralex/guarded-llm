"""Metrics for binary moderation decisions, with explicit abstentions."""


def calibration_threshold(examples, minimum_recall=0.8):
    """Select maximum precision at the recall floor, breaking ties by recall."""
    positives = sum(label == 1 for label, _ in examples)
    if not positives or positives == len(examples):
        raise ValueError('Calibration requires both classes')
    ranked = sorted(((score, label) for label, score in examples if score is not None), reverse=True)
    best = None
    tp = 0
    for i, (score, label) in enumerate(ranked, start=1):
        tp += label
        if (i == len(ranked) or ranked[i][0] != score) and tp / positives >= minimum_recall:
            candidate = (tp / i, tp / positives, score)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise ValueError('No threshold reaches the calibration recall floor')
    return best[2]


def summarize(examples, prevalences=(0.001, 0.01, 0.05)):
    counts = dict.fromkeys(('tp', 'fp', 'tn', 'fn', 'abstain_positive', 'abstain_negative'), 0)
    keys = {(1, True): 'tp', (1, False): 'fn', (1, None): 'abstain_positive',
            (0, True): 'fp', (0, False): 'tn', (0, None): 'abstain_negative'}
    for label, decision in examples:
        counts[keys[label, decision]] += 1
    tp, fp, tn, fn, ap, an = counts.values()
    recall = tp / (tp + fn + ap)
    false_alarm_rate = fp / (fp + tn + an)
    return {
        'counts': counts,
        'precision': tp / (tp + fp) if tp + fp else None,
        'recall': recall,
        'false_alarm_rate': false_alarm_rate,
        'coverage': (tp + fp + tn + fn) / len(examples),
        'prevalence_scenarios': [
            {'assumed_prevalence': p, 'true_alerts_per_1000': 1000 * p * recall,
             'false_alerts_per_1000': 1000 * (1 - p) * false_alarm_rate}
            for p in prevalences
        ],
    }
