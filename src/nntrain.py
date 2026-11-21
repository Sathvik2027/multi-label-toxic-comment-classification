"""Training function for the loss and model experiments (one call = one experiment)."""
import os, json, time, random
import numpy as np, pandas as pd


def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """Focal loss: shrinks the penalty on examples the model already gets right."""
    import torch
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    a = alpha * targets + (1 - alpha) * (1 - targets)
    return (a * (1 - pt) ** gamma * bce).mean()


def train_run(proj, run_id, model_name, loss_name, max_len, epochs=4, lr=2e-5,
              batch_size=32, patience=2, seed=42, notes=''):
    import torch, torch.nn as nn
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    from toxlib import LABELS, load_split, save_probs, macro_ap, tune_thresholds, per_label_report, log_run
    from nnlib import predict_with

    assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'
    t0 = time.time()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = 'cuda'

    tr, va = load_split(proj, 'train_sub'), load_split(proj, 'val')
    Ytr = tr[LABELS].values.astype('float32'); Yva = va[LABELS].values
    tok = AutoTokenizer.from_pretrained(model_name)
    tr_ids = tok(tr.comment_text.fillna('').astype(str).tolist(), truncation=True, max_length=max_len)['input_ids']
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), problem_type='multi_label_classification').to(device)

    if loss_name == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_name == 'weighted':
        pos = Ytr.sum(axis=0); neg = len(Ytr) - pos
        pw = torch.tensor(np.sqrt(neg / pos), dtype=torch.float32, device=device)
        print('pos_weight:', dict(zip(LABELS, pw.cpu().numpy().round(2))))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    elif loss_name == 'focal':
        loss_fn = lambda lg, y: focal_loss(lg, y, gamma=2.0, alpha=0.25)
    else:
        raise ValueError('loss_name must be bce, weighted or focal')

    steps_per_epoch = int(np.ceil(len(tr_ids) / batch_size)); total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler('cuda')

    ckpt = f'{proj}/models/saved_checkpoints/{run_id}_best'
    log_path = f'{proj}/results/experiment_logs/{run_id}.json'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    history, best, best_ep, bad = [], -1.0, 0, 0
    print(f'{run_id}: {model_name} | loss={loss_name} | max_len={max_len} | batch={batch_size} | '
          f'lr={lr} | {steps_per_epoch} steps per epoch')

    for ep in range(1, epochs + 1):
        model.train(); run_loss = 0.0; n = 0
        perm = np.random.RandomState(seed + ep).permutation(len(tr_ids))
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            b = tok.pad({'input_ids': [tr_ids[i] for i in idx]}, return_tensors='pt')
            b = {k: v.to(device) for k, v in b.items()}
            y = torch.tensor(Ytr[idx], device=device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(**b).logits
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item(); n += 1
            if n % 300 == 0:
                print(f'  epoch {ep} step {n}/{steps_per_epoch} loss {run_loss / n:.4f}')
        Pva = predict_with(model, tok, va.comment_text.tolist(), max_len)
        ap = macro_ap(Yva, Pva)
        history.append({'epoch': ep, 'train_loss': round(run_loss / n, 5), 'val_macro_ap': round(ap, 5),
                        'minutes': round((time.time() - t0) / 60, 1)})
        print(f'== {run_id} epoch {ep}: train loss {run_loss / n:.4f} | val Macro AP {ap:.4f} | '
              f'{(time.time() - t0) / 60:.1f} min')
        if ap > best:
            best, best_ep, bad = ap, ep, 0
            model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
            wf = [f for f in os.listdir(ckpt) if f.endswith('.safetensors') or f.endswith('.bin')]
            mb = sum(os.path.getsize(f'{ckpt}/{f}') for f in wf) / 1e6
            assert mb > 100, 'weights file missing or too small - do not continue'
            save_probs(proj, run_id, 'val', Pva)
            print(f'   saved best checkpoint ({mb:.0f} MB)')
        else:
            bad += 1
        json.dump({'run_id': run_id, 'model': model_name, 'loss': loss_name, 'max_len': max_len,
                   'batch_size': batch_size, 'lr': lr, 'seed': seed, 'best_epoch': best_ep,
                   'best_val_macro_ap': round(best, 5), 'history': history, 'notes': notes},
                  open(log_path, 'w'), indent=2)
        if bad >= patience:
            print(f'Early stopping: no improvement for {patience} epochs.')
            break

    # reload the best epoch and save predictions for all three evaluation sets
    del model; torch.cuda.empty_cache(); time.sleep(15)
    best_model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    btok = AutoTokenizer.from_pretrained(ckpt)
    for split in ['val', 'test_internal', 'test_external']:
        df = load_split(proj, split)
        P = predict_with(best_model, btok, df.comment_text.tolist(), max_len)
        save_probs(proj, run_id, split, P)          # test predictions saved, NOT opened until the freeze
        if split == 'val':
            Pva = P
    thr = tune_thresholds(Yva, Pva)
    rep = per_label_report(Yva, Pva, thr)
    log_run(proj, run_id, model=f'{model_name} | {loss_name} loss', max_len=max_len, loss=loss_name, seed=seed,
            val_macro_AP=round(macro_ap(Yva, Pva), 4), val_tuned_macro_F1=round(rep.F1.mean(), 4),
            best_epoch=best_ep, runtime_s=round(time.time() - t0), notes=notes)
    print(f'\n{run_id} DONE | best epoch {best_ep} | val Macro AP {macro_ap(Yva, Pva):.4f} | '
          f'tuned macro F1 {rep.F1.mean():.4f} | {(time.time() - t0) / 60:.1f} min')
    return {'run_id': run_id, 'val_macro_ap': macro_ap(Yva, Pva), 'best_epoch': best_ep, 'history': history}
