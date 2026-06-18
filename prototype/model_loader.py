"""
Loads all model artifacts once at module import time.
On HuggingFace Spaces this runs once at startup; subsequent requests reuse
the already-loaded objects.

If the artifacts/ folder is missing, ARTIFACTS_MISSING is set to True and all
model objects are None. app.py checks this flag and shows a helpful message
instead of crashing.
"""
import json
import pathlib

import nltk
import torch

nltk.download('stopwords', quiet=True)

ARTIFACTS = pathlib.Path(__file__).parent.parent / 'artifacts'
DEVICE = torch.device('cpu')   # HF Spaces free tier is CPU only

ARTIFACTS_MISSING = not (ARTIFACTS / 'config.json').exists()

if ARTIFACTS_MISSING:
    CONFIG = {}
    LSTM_MAX_LEN = LSTM_MAX_VOCAB = LSTM_EMBED_DIM = LSTM_HIDDEN_DIM = None
    BERT_MODEL_NAME = BERT_MAX_LEN = BERT_BATCH_SIZE = BERT_FINETUNED_DIR = None
    TFIDF = LR_MODEL = RF_MODEL = XGB_MODEL = LGBM_MODEL = None
    VOCAB = LSTM_MODEL = TOKENIZER = BERT_MODEL = None
    MODEL_NAMES = [
        'Logistic Regression', 'Random Forest',
        'XGBoost', 'LightGBM', 'LSTM', 'DistilBERT',
    ]
    MODELS = {}
else:
    import joblib

    # ── Config ─────────────────────────────────────────────────────────────────
    with open(ARTIFACTS / 'config.json') as f:
        CONFIG = json.load(f)

    LSTM_MAX_LEN    = CONFIG['LSTM_MAX_LEN']
    LSTM_MAX_VOCAB  = CONFIG['LSTM_MAX_VOCAB']
    LSTM_EMBED_DIM  = CONFIG['LSTM_EMBED_DIM']
    LSTM_HIDDEN_DIM = CONFIG['LSTM_HIDDEN_DIM']
    BERT_MODEL_NAME = CONFIG['BERT_MODEL_NAME']
    BERT_MAX_LEN    = CONFIG['BERT_MAX_LEN']
    BERT_BATCH_SIZE = CONFIG.get('BERT_BATCH_SIZE', 32)
    BERT_FINETUNED_DIR = CONFIG.get('BERT_FINETUNED_DIR', 'distilbert_finetuned')

    # ── TF-IDF vectorizer ───────────────────────────────────────────────────────
    TFIDF = joblib.load(ARTIFACTS / 'tfidf.joblib')

    # ── Classical models ────────────────────────────────────────────────────────
    LR_MODEL   = joblib.load(ARTIFACTS / 'lr_model.joblib')
    RF_MODEL   = joblib.load(ARTIFACTS / 'rf_model.joblib')
    XGB_MODEL  = joblib.load(ARTIFACTS / 'xgb_model.joblib')
    LGBM_MODEL = joblib.load(ARTIFACTS / 'lgbm_model.joblib')

    # ── LSTM ────────────────────────────────────────────────────────────────────
    with open(ARTIFACTS / 'vocab.json') as f:
        VOCAB = json.load(f)

    from prototype.lstm_model import LSTMClassifier  # noqa: E402

    LSTM_MODEL = LSTMClassifier(
        vocab_size=LSTM_MAX_VOCAB,
        embed_dim=LSTM_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
    )
    LSTM_MODEL.load_state_dict(
        torch.load(ARTIFACTS / 'lstm_model.pt', map_location=DEVICE)
    )
    LSTM_MODEL.train(False)

    # ── DistilBERT (fine-tuned, self-contained model + tokenizer) ───────────────
    from transformers import (  # noqa: E402
        AutoTokenizer, AutoModelForSequenceClassification,
    )

    BERT_DIR   = ARTIFACTS / BERT_FINETUNED_DIR
    TOKENIZER  = AutoTokenizer.from_pretrained(BERT_DIR)
    BERT_MODEL = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(DEVICE)
    BERT_MODEL.train(False)

    # ── Unified model registry ──────────────────────────────────────────────────
    MODEL_NAMES = [
        'Logistic Regression',
        'Random Forest',
        'XGBoost',
        'LightGBM',
        'LSTM',
        'DistilBERT',
    ]

    MODELS = {
        'Logistic Regression':     LR_MODEL,
        'Random Forest':           RF_MODEL,
        'XGBoost':                 XGB_MODEL,
        'LightGBM':                LGBM_MODEL,
        'LSTM':                    LSTM_MODEL,
        'DistilBERT': BERT_MODEL,
    }
