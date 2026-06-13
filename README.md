---
title: Fake News Classifier XAI Demo
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Six fake-news classifiers with SHAP explanations on WELFake
---

# XAI for Fake News Classification

Interactive demo for a thesis comparing six fake-news classifiers on the
[WELFake](https://doi.org/10.1109/TCSS.2021.3068519) dataset (~72 k articles,
binary Real / Fake).

## Models

| Model | Type | Features |
|-------|------|----------|
| Logistic Regression | Classical | TF-IDF |
| Random Forest | Classical | TF-IDF |
| XGBoost | Boosted trees | TF-IDF |
| LightGBM | Boosted trees | TF-IDF |
| BiLSTM | Recurrent | Learned word embeddings |
| DistilBERT + MLP | Transformer | Frozen [CLS] embeddings |

## Features

- **Predictions tab** — P(Fake) probability for all 6 models + horizontal bar chart
- **SHAP tab** — token-level attributions (Partition SHAP with Text masker) for the selected model

> ⚠️ DistilBERT SHAP runs on CPU and may take up to 45 s. Classical models complete in < 5 s.

## Run locally

```bash
git clone <repo>
cd xai-for-fake-news-classification
pip install -r requirements.txt
python app.py
```

Artifacts (model weights) must be present in `artifacts/` — generate them by
running the notebook (`Fake_News_Classification_Final.ipynb`) end-to-end on
a machine with the WELFake dataset, then running the saving cell.

> Models were trained on Kaggle (2× NVIDIA T4 GPU, ~3 h 10 m full run).

## References

1. Verma et al. (2021) — WELFake dataset
2. Lundberg & Lee (2017) — SHAP
