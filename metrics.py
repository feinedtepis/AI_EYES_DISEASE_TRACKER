"""
Evaluation metrics. Accuracy is deliberately NOT the headline number.

Multi-label classification: per-class precision, recall (sensitivity),
specificity, F1, ROC-AUC, PR-AUC (+ bootstrap 95% CI for ROC-AUC), 2x2
confusion counts, and macro / micro averages. Unknown labels (NaN) are masked.

Regression: MAE, MSE, RMSE, R², mean bias, % within ±0.50 / ±1.00 / ±2.00 D.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             roc_auc_score)

SMALL_N = 30   # fewer positives than this -> flag the class metric as unstable


def _safe_auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def _safe_ap(y, p):
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def bootstrap_ci(y, p, fn, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if 0 < y[s].sum() < len(s):
            vals.append(fn(y[s], p[s]))
    if len(vals) < 20:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def tune_thresholds(y_true, y_prob, mask, strategy="f1"):
    """Per-class decision threshold chosen on the VALIDATION set (never on test)."""
    th = []
    for k in range(y_true.shape[1]):
        m = mask[:, k] > 0
        y, p = y_true[m, k], y_prob[m, k]
        if y.sum() == 0 or y.sum() == len(y):
            th.append(0.5)
            continue
        if strategy == "youden":
            from sklearn.metrics import roc_curve
            fpr, tpr, t = roc_curve(y, p)
            th.append(float(np.clip(t[np.argmax(tpr - fpr)], 0.01, 0.99)))
        else:
            prec, rec, t = precision_recall_curve(y, p)
            f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
            th.append(float(np.clip(t[np.argmax(f1[:-1])], 0.01, 0.99)) if len(t) else 0.5)
    return th


def multilabel_metrics(y_true, y_prob, mask, thresholds, names, ci=True):
    y_true, y_prob, mask = map(np.asarray, (y_true, y_prob, mask))
    per = {}
    all_y, all_p, all_pred = [], [], []
    for k, name in enumerate(names):
        m = mask[:, k] > 0
        y, p = y_true[m, k].astype(int), y_prob[m, k]
        pred = (p >= thresholds[k]).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        if tp + fn == 0:
            f1 = float("nan")          # no positives in this subset -> F1 undefined
        elif tp == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        per[name] = {
            "n": int(m.sum()), "n_pos": int(y.sum()), "prevalence": float(y.mean()) if len(y) else float("nan"),
            "threshold": float(thresholds[k]), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall_sensitivity": rec, "specificity": spec, "f1": f1,
            "roc_auc": _safe_auc(y, p), "pr_auc": _safe_ap(y, p),
            "roc_auc_ci95": bootstrap_ci(y, p, _safe_auc) if ci and len(y) else [float("nan")] * 2,
            "small_sample": bool(y.sum() < SMALL_N),
        }
        all_y.append(y); all_p.append(p); all_pred.append(pred)

    valid = [n for n in names if per[n]["n_pos"] > 0 and per[n]["n_pos"] < per[n]["n"]]

    def macro(key):
        v = [per[n][key] for n in valid if not np.isnan(per[n][key])]
        return float(np.mean(v)) if v else float("nan")

    fy, fp_, fpred = np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_pred)
    tp = int(((fpred == 1) & (fy == 1)).sum()); fpc = int(((fpred == 1) & (fy == 0)).sum())
    fn = int(((fpred == 0) & (fy == 1)).sum()); tn = int(((fpred == 0) & (fy == 0)).sum())
    micro_p = tp / (tp + fpc) if tp + fpc else float("nan")
    micro_r = tp / (tp + fn) if tp + fn else float("nan")
    summary = {
        "macro_roc_auc": macro("roc_auc"), "macro_pr_auc": macro("pr_auc"), "macro_f1": macro("f1"),
        "macro_precision": macro("precision"), "macro_recall": macro("recall_sensitivity"),
        "macro_specificity": macro("specificity"),
        "micro_roc_auc": _safe_auc(fy, fp_), "micro_pr_auc": _safe_ap(fy, fp_),
        "micro_precision": micro_p, "micro_recall": micro_r,
        "micro_specificity": tn / (tn + fpc) if tn + fpc else float("nan"),
        "micro_f1": 2 * micro_p * micro_r / (micro_p + micro_r) if tp else 0.0,
        "n_images": int(len(y_true)), "classes_in_macro": valid,
    }
    return {"summary": summary, "per_class": per}


def regression_metrics(y, yhat, ci=True):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = yhat - y
    n = len(y)
    if n == 0:
        return {"n": 0}
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out = {
        "n": int(n), "mae": float(np.mean(np.abs(err))), "mse": mse, "rmse": float(np.sqrt(mse)),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "bias_mean_error": float(err.mean()), "residual_std": float(err.std(ddof=1)) if n > 1 else float("nan"),
        "within_0.50D": float(np.mean(np.abs(err) <= 0.5)),
        "within_1.00D": float(np.mean(np.abs(err) <= 1.0)),
        "within_2.00D": float(np.mean(np.abs(err) <= 2.0)),
        "small_sample": bool(n < SMALL_N),
    }
    if ci and n >= 10:
        rng = np.random.default_rng(0)
        maes = [np.mean(np.abs(err[rng.integers(0, n, n)])) for _ in range(500)]
        out["mae_ci95"] = [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))]
    return out


def age_group(a):
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return "unknown"
    a = float(a)
    return "<40" if a < 40 else "40-59" if a < 60 else "60+"
