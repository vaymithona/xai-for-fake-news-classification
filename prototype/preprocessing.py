"""
Text cleaning and LSTM sequence encoding.
Copied verbatim from notebook cells 19 and 47 so the inference path
matches training exactly.
"""
import re

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords', quiet=True)

_ps = PorterStemmer()
_STOPWORDS = set(nltk_stopwords.words('english'))


def cleaning(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [_ps.stem(w) for w in words if w not in _STOPWORDS]
    words = [re.sub(r'(.)\1{2,}', r'\1\1', w) for w in words]
    cleaned = ' '.join(words)
    cleaned = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned)
    return cleaned


def encode(text: str, vocab: dict, max_len: int = 100) -> list:
    tokens = text.split()[:max_len]
    ids = [vocab.get(tok, 1) for tok in tokens]
    ids = ids + [0] * (max_len - len(ids))
    return ids
