"""Classification metrics in plain numpy.

Macro-F1 is the headline because the classes are heavily imbalanced; accuracy would mostly report
the ten biggest classes. It is averaged over the classes the split can actually validate -- a
class with no val images has no recall to measure -- and also reported over every class that has
val support, for comparison.
"""

import numpy as np
import pandas as pd


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns predictions."""
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class(cm: np.ndarray) -> pd.DataFrame:
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    f1 = np.where(support > 0, f1, np.nan)
    return pd.DataFrame(dict(class_idx=np.arange(len(cm)), support=support, predicted=predicted,
                             precision=precision, recall=recall, f1=f1))


def macro_f1(cm: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean F1 over classes in `mask` that have val support (all such classes when mask is None)."""
    pc = per_class(cm)
    keep = pc.support.to_numpy() > 0
    if mask is not None:
        keep &= mask
    return float(pc.f1.to_numpy()[keep].mean()) if keep.any() else float("nan")


def top_confused(cm: np.ndarray, k: int = 15) -> pd.DataFrame:
    """Largest off-diagonal cells: (true, predicted, count, share of the true class)."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    t, p = np.unravel_index(flat, cm.shape)
    n = off[t, p]
    keep = n > 0
    t, p, n = t[keep], p[keep], n[keep]
    return pd.DataFrame(dict(true_idx=t, pred_idx=p, count=n, share_of_true=n / cm.sum(1)[t]))
