"""Metrics from supplied annotations, never inferred ground truth."""

from collections import Counter


def classification_metrics(truth, prediction):
    if len(truth) != len(prediction) or not truth:
        raise ValueError("Paired nonempty labels required")
    labels = sorted(set(truth) | set(prediction))
    confusion = Counter(zip(truth, prediction))
    per_class = {}
    for label in labels:
        tp = confusion[label, label]
        fp = sum(v for (a, b), v in confusion.items() if b == label and a != label)
        fn = sum(v for (a, b), v in confusion.items() if a == label and b != label)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "support": sum(a == label for a in truth),
        }
    return {
        "sample_count": len(truth),
        "accuracy": sum(a == b for a, b in zip(truth, prediction)) / len(truth),
        "per_class": per_class,
        "confusion_matrix": {a: {b: confusion[a, b] for b in labels} for a in labels},
    }
