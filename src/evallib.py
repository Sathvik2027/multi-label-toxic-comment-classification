"""Evaluation helpers: bootstrap confidence intervals, ensemble averaging, subgroup AUCs."""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def avg_probs(proj, runs, split):
    """Average the saved probabilities of several runs (the ensemble)."""
    from toxlib import load_probs
    return np.mean([load_probs(proj, r, split) for r in runs], axis=0)


def ci95(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_macro_ap(Y, probs, n_boot=300, seed=42):
    """probs = {run: P}. The same resampled rows are used for every run, so differences are meaningful."""
    rng = np.random.RandomState(seed)
    out = {k: [] for k in probs}
    for _ in range(n_boot):
        idx = rng.randint(0, len(Y), len(Y))
        for k, P in probs.items():
            out[k].append(np.mean([average_precision_score(Y[idx, i], P[idx, i]) for i in range(len(LABELS))]))
    return {k: np.array(v) for k, v in out.items()}


def bootstrap_metrics(Y, P, thr, n_boot=500, seed=42):
    """Per-label precision, recall, F1, AP with 95% bootstrap intervals at fixed thresholds, plus macro F1 / macro AP."""
    t = np.array([thr[l] for l in LABELS])

    def metrics(y, p):
        pred = p >= t
        tp = (pred & (y == 1)).sum(0); fp = (pred & (y == 0)).sum(0); fn = ((~pred) & (y == 1)).sum(0)
        prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp + fn, 1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        ap = np.array([average_precision_score(y[:, i], p[:, i]) for i in range(len(LABELS))])
        return np.vstack([prec, rec, f1, ap])

    base = metrics(Y, P)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(Y), len(Y))
        boots.append(metrics(Y[idx], P[idx]))
    boots = np.array(boots)
    names = ['precision', 'recall', 'F1', 'AP']
    rows = []
    for i, l in enumerate(LABELS):
        r = {'label': l, 'n_pos': int(Y[:, i].sum()), 'threshold': t[i]}
        for j, m in enumerate(names):
            lo, hi = np.percentile(boots[:, j, i], [2.5, 97.5])
            r[m] = round(float(base[j, i]), 3)
            r[m + '_95CI'] = f'{lo:.3f}-{hi:.3f}'
        rows.append(r)
    macro = {'macro_F1': round(float(base[2].mean()), 4), 'macro_F1_95CI': [round(x, 4) for x in ci95(boots[:, 2, :].mean(1))],
             'macro_AP': round(float(base[3].mean()), 4), 'macro_AP_95CI': [round(x, 4) for x in ci95(boots[:, 3, :].mean(1))]}
    return pd.DataFrame(rows), macro


def auc_safe(y, s):
    y = np.asarray(y)
    if len(y) < 2 or y.min() == y.max():
        return np.nan
    return roc_auc_score(y, s)


def subgroup_aucs(y, s, mask):
    """Subgroup AUC, BPSN AUC and BNSP AUC for one identity mask (y = 0/1 labels, s = scores)."""
    sub, bg = mask, ~mask
    sub_auc = auc_safe(y[sub], s[sub])
    sel = (sub & (y == 0)) | (bg & (y == 1)); bpsn = auc_safe(y[sel], s[sel])
    sel = (sub & (y == 1)) | (bg & (y == 0)); bnsp = auc_safe(y[sel], s[sel])
    return sub_auc, bpsn, bnsp


def bpsn_ci(y, s, mask, n_boot=200, seed=42):
    """95% interval for BPSN AUC, resampling the two groups separately (fast)."""
    rng = np.random.RandomState(seed)
    neg = s[mask & (y == 0)]; pos = s[(~mask) & (y == 1)]
    if len(neg) < 5 or len(pos) < 5:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        a = rng.choice(neg, len(neg)); b = rng.choice(pos, len(pos))
        vals.append(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
    return ci95(vals)
