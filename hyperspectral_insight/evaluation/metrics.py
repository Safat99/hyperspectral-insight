import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)

def overall_accuracy(y_true, y_pred):
    """
    Compute Overall Accuracy (OA).
    Equivalent to accuracy_score.
    """
    return float(accuracy_score(y_true, y_pred))

def precision_recall_f1(y_true, y_pred, average="macro"):
    """
    Compute precision, recall, f1-score.

    Args:
        average: "macro", "micro", "weighted", or None (per-class)

    Returns:
        precision, recall, f1
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return float(precision), float(recall), float(f1)

def kappa(y_true, y_pred):
    """
    Cohen's Kappa coefficient.
    """
    return float(cohen_kappa_score(y_true, y_pred))

def compute_metrics(y_true, y_pred, average="macro"):
    """
    Compute all key metrics in one call.

    Returns:
        dict with keys:
            "oa", "precision", "recall", "f1", "kappa"
    """
    metrics = {}

    metrics["oa"] = overall_accuracy(y_true, y_pred)

    p, r, f = precision_recall_f1(y_true, y_pred, average=average)
    metrics["precision"] = p
    metrics["recall"] = r
    metrics["f1"] = f

    metrics["kappa"] = kappa(y_true, y_pred)

    return metrics