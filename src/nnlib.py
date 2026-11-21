"""Transformer helpers shared by every neural notebook (fast prediction)."""
import os
import numpy as np


def predict_with(model, tok, texts, max_len, batch_size=64):
    """Probabilities for a list of texts. Texts are sorted by length, so short comments are not padded to max_len."""
    import torch
    device = next(model.parameters()).device
    use_amp = device.type == 'cuda'
    texts = [t if isinstance(t, str) else '' for t in texts]
    ids = tok(texts, truncation=True, max_length=max_len)['input_ids']
    order = np.argsort([len(x) for x in ids], kind='stable')
    out = np.zeros((len(texts), model.config.num_labels), dtype='float32')
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            b = tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt')
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**b).logits
            out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    if was_training:
        model.train()
    return out


def predict_probs(model_dir, texts, max_len, batch_size=64, fallback_tokenizer=None):
    """Load a saved checkpoint folder and return probabilities for the texts."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    try:
        tok = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(fallback_tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    P = predict_with(model, tok, texts, max_len, batch_size)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return P
