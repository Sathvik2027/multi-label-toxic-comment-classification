"""Shared helpers for the NLP-04 toxic comment project. Every notebook imports this file."""
import os, time
import numpy as np, pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_score, recall_score)

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
SPLITS = ['val', 'test_internal', 'test_external']


def norm_text(s):
    """Key for spotting identical comments (lowercase, trimmed, single spaces)."""
    return s.astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()


def load_split(proj, name):
    return pd.read_csv(f'{proj}/data/processed/{name}.csv')


def thr_vec(thr):
    if np.isscalar(thr):
        return np.full(len(LABELS), float(thr))
    if isinstance(thr, dict):
        return np.array([thr[l] for l in LABELS], dtype=float)
    return np.array(thr, dtype=float)


def macro_ap(Y, P):
    """Macro Average Precision (our main selection metric)."""
    return float(np.mean([average_precision_score(Y[:, i], P[:, i]) for i in range(len(LABELS))]))


def tune_thresholds(Y, P, grid=None):
    """Best-F1 threshold per label. VALIDATION data only."""
    grid = np.arange(0.05, 0.951, 0.01) if grid is None else grid
    best = {}
    for i, l in enumerate(LABELS):
        f1s = [f1_score(Y[:, i], P[:, i] >= t, zero_division=0) for t in grid]
        best[l] = round(float(grid[int(np.argmax(f1s))]), 2)
    return best


def per_label_report(Y, P, thr=0.5):
    t = thr_vec(thr)
    rows = []
    for i, l in enumerate(LABELS):
        pred = P[:, i] >= t[i]
        rows.append(dict(label=l, n_pos=int(Y[:, i].sum()), threshold=round(t[i], 2),
                         precision=precision_score(Y[:, i], pred, zero_division=0),
                         recall=recall_score(Y[:, i], pred, zero_division=0),
                         F1=f1_score(Y[:, i], pred, zero_division=0),
                         AP=average_precision_score(Y[:, i], P[:, i]),
                         ROC_AUC=roc_auc_score(Y[:, i], P[:, i])))
    return pd.DataFrame(rows).round(4)


def save_probs(proj, run_id, split, P):
    d = f'{proj}/results/predictions'
    os.makedirs(d, exist_ok=True)
    np.save(f'{d}/{run_id}_{split}.npy', np.asarray(P, dtype='float32'))


def load_probs(proj, run_id, split):
    return np.load(f'{proj}/results/predictions/{run_id}_{split}.npy')


def log_run(proj, run_id, **info):
    """One row per experiment in results/experiment_logs/registry.csv (replaces the row if re-run)."""
    path = f'{proj}/results/experiment_logs/registry.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = pd.DataFrame([dict(run_id=run_id, time=time.strftime('%Y-%m-%d %H:%M'), **info)])
    if os.path.exists(path):
        old = pd.read_csv(path)
        row = pd.concat([old[old.run_id != run_id], row], ignore_index=True)
    row.to_csv(path, index=False)
    return row
