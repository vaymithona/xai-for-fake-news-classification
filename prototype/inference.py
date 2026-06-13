"""
Per-model prediction wrappers and the combined predict_all() function.
Mirrors cell 88 of the notebook exactly so inference paths match training.
"""
import numpy as np
import pandas as pd
import torch

from prototype.model_loader import (
    TFIDF, VOCAB, LSTM_MAX_LEN,
    BERT_MAX_LEN, BERT_BATCH_SIZE,
    TOKENIZER, ENCODER, HEAD,
    MODELS, MODEL_NAMES, DEVICE,
)
from prototype.preprocessing import cleaning, encode


# ── Classical (TF-IDF) wrappers ─────────────────────────────────────────────

def make_classical_proba_fn(model):
    """Return raw-text -> [[P(Real), P(Fake)], ...] for a sklearn/boosting model."""
    def f(texts):
        cleaned = [cleaning(str(t)) for t in texts]
        return model.predict_proba(TFIDF.transform(cleaned))
    return f


# ── LSTM wrapper ─────────────────────────────────────────────────────────────

def _lstm_batch_predict(X_t, batch_size: int = 256):
    lstm_model = MODELS['LSTM']
    lstm_model.train(False)
    all_probs = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            chunk = X_t[i:i + batch_size].to(DEVICE)
            logits = lstm_model(chunk)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs).reshape(-1)


def lstm_proba_fn(texts):
    """Raw text -> cleaning -> encode -> LSTM -> [[P(Real), P(Fake)], ...]."""
    cleaned = [cleaning(str(t)) for t in texts]
    seqs = np.array([encode(t, VOCAB, LSTM_MAX_LEN) for t in cleaned], dtype=np.int64)
    p_fake = _lstm_batch_predict(torch.tensor(seqs, dtype=torch.long))
    return np.column_stack([1.0 - p_fake, p_fake])


# ── DistilBERT wrapper ────────────────────────────────────────────────────────

def bert_proba_fn(texts):
    """Raw text -> DistilBERT [CLS] -> MLP head -> [[P(Real), P(Fake)], ...]."""
    texts = [str(t) for t in texts]
    ENCODER.train(False)
    HEAD.train(False)
    probs = []
    with torch.no_grad():
        for s in range(0, len(texts), BERT_BATCH_SIZE):
            enc = TOKENIZER(
                texts[s:s + BERT_BATCH_SIZE],
                padding=True,
                truncation=True,
                max_length=BERT_MAX_LEN,
                return_tensors='pt',
            ).to(DEVICE)
            cls = ENCODER(**enc).last_hidden_state[:, 0, :].float()
            logit = HEAD(cls).squeeze(-1)
            probs.append(torch.sigmoid(logit).cpu().numpy())
    p_fake = np.concatenate(probs).reshape(-1)
    return np.column_stack([1.0 - p_fake, p_fake])


# ── Combined function exposed to the app ────────────────────────────────────

_CLASSICAL_MODELS = ('Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM')

_PROBA_FN = {
    **{name: make_classical_proba_fn(MODELS[name]) for name in _CLASSICAL_MODELS},
    'LSTM':             lstm_proba_fn,
    'DistilBERT + MLP': bert_proba_fn,
}


def get_proba_fn(model_name: str):
    """Return the raw-text -> probability callable for the named model."""
    return _PROBA_FN[model_name]


def predict_all(text: str) -> pd.DataFrame:
    """Run all 6 models on one article.

    Returns a DataFrame with columns: Model, P(Fake)%, Verdict.
    """
    rows = []
    for name in MODEL_NAMES:
        p_fake = float(get_proba_fn(name)([text])[0, 1])
        verdict = 'FAKE' if p_fake >= 0.5 else 'REAL'
        rows.append({
            'Model':    name,
            'P(Fake)%': round(p_fake * 100, 1),
            'Verdict':  verdict,
        })
    return pd.DataFrame(rows)
