import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate evaluation metrics.
    Args:
        y_true: Ground truth labels (numpy array).
        y_pred: Predicted labels (numpy array).
        y_prob: Predicted probabilities (numpy array, optional for AUC).
    Returns:
        A dictionary containing ACC, SEN, SPE, PRE, AUC, and F1-score.
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall)
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    f1 = f1_score(y_true, y_pred)

    return {
        "ACC": acc,
        "SEN": sen,
        "SPE": spe,
        "PRE": pre,
        "AUC": auc,
        "F1": f1
    }