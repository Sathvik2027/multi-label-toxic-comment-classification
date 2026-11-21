"""Load the frozen model (any list of runs) and predict probabilities for new texts."""
import numpy as np, pandas as pd

CKPT_NAMES = {'B1_256': 'B1_distilbert_bce_best', 'B2_256': 'B2_roberta_bce_best'}


class Ensemble:
    def __init__(self, proj, runs, batch_size=64):
        import joblib
        self.runs, self.bs, self.items = list(runs), batch_size, []
        reg = pd.read_csv(f'{proj}/results/experiment_logs/registry.csv').set_index('run_id')
        ck = f'{proj}/models/saved_checkpoints'
        for r in self.runs:
            if r in ('A1', 'A2'):
                word = joblib.load(f'{ck}/word_tfidf_vectorizer.pkl')
                char = joblib.load(f'{ck}/char_tfidf_vectorizer.pkl')
                fn = 'A1_logistic_regression' if r == 'A1' else 'A2_calibrated_linear_svm'
                self.items.append(('classical', r, (word, char, joblib.load(f'{ck}/{fn}.pkl'))))
            else:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                d = f'{ck}/{CKPT_NAMES.get(r, r + "_best")}'
                tok = AutoTokenizer.from_pretrained(d)
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = AutoModelForSequenceClassification.from_pretrained(d).to(dev).eval()
                self.items.append(('neural', r, (tok, model, int(reg.loc[r, 'max_len']))))
        print('Loaded models:', [(k, r) for k, r, _ in self.items])

    def predict(self, texts):
        import scipy.sparse as sp
        texts = [t if isinstance(t, str) else '' for t in texts]
        outs = []
        for kind, r, obj in self.items:
            if kind == 'classical':
                word, char, clf = obj
                parts = []
                for s in range(0, len(texts), 8000):
                    chunk = texts[s:s + 8000]
                    X = sp.hstack([word.transform(chunk), char.transform(chunk)]).tocsr()
                    parts.append(clf.predict_proba(X))
                outs.append(np.vstack(parts))
            else:
                from nnlib import predict_with
                tok, model, ml = obj
                outs.append(predict_with(model, tok, texts, ml, self.bs))
        return np.mean(outs, axis=0)

    def free(self):
        self.items = []
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
