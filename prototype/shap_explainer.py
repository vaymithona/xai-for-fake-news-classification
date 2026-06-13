"""
SHAP Partition explainer for one model at a time.
Returns an HTML string suitable for gr.HTML in Gradio.
"""
import inspect
import re

import numpy as np
import shap

from prototype.inference import get_proba_fn
from prototype.model_loader import MODEL_NAMES

_MAX_WORDS = 120
_MASKER = shap.maskers.Text(r"\W+")

# Number of masked samples per model family (SHAP's max_evals parameter).
# DistilBERT is slow on CPU, so we cap it to stay under HF Spaces' timeout.
_BUDGET = {
    'Logistic Regression': 500,
    'Random Forest':       500,
    'XGBoost':             500,
    'LightGBM':            500,
    'LSTM':                500,
    'DistilBERT + MLP':    100,
}


# SHAP's text plot only *underlines* the word you hover (in the text or in the
# top force bar). These patterns let us append a matching bold toggle on the same
# token element so the selected word stands out more. SHAP owns the markup, so we
# rewrite its inline onmouseover / onmouseout handlers as a string patch.
_HOVER_ON = re.compile(
    r"(document\.getElementById\('_tp_[^']+'\)\.style)\.textDecoration\s*=\s*'underline';"
)
_HOVER_OFF = re.compile(
    r"(document\.getElementById\('_tp_[^']+'\)\.style)\.textDecoration\s*=\s*'none';"
)


def _bold_on_hover(html: str) -> str:
    """Make the hovered/selected word render bold in addition to underlined."""
    html = _HOVER_ON.sub(r"\1.textDecoration = 'underline';\1.fontWeight = 'bold';", html)
    html = _HOVER_OFF.sub(r"\1.textDecoration = 'none';\1.fontWeight = 'normal';", html)
    return html


def _shap_html(sv_j) -> str:
    """Render one SHAP TextExplanation to an HTML string."""
    sig = inspect.signature(shap.plots.text)
    if 'display' in sig.parameters:          # shap >= 0.41
        return _bold_on_hover(shap.plots.text(sv_j, display=False))
    # Fallback for older shap: intercept IPython.display
    from unittest.mock import patch
    captured = {}

    def _grab(obj, *a, **kw):
        if hasattr(obj, 'data'):
            captured['html'] = obj.data

    with patch('IPython.display.display', side_effect=_grab):
        shap.plots.text(sv_j)
    return _bold_on_hover(captured.get('html', '<p><em>SHAP rendering unavailable.</em></p>'))


def compute_shap(text: str, model_name: str) -> str:
    """Compute Partition SHAP for one model on one article.

    Returns
    -------
    html : str
        Full HTML (including SHAP JS) for rendering in gr.HTML.
    """
    truncated = ' '.join(text.split()[:_MAX_WORDS])
    fn = get_proba_fn(model_name)
    budget = _BUDGET.get(model_name, 500)

    explainer = shap.Explainer(fn, _MASKER, output_names=['Real', 'Fake'])
    sv = explainer([truncated], max_evals=budget)
    sv_j = sv[0, :, 1]

    token_html = _shap_html(sv_j)
    header = (
        f'<p style="margin:0 0 10px"><strong>Model:</strong> {model_name}'
        f'&nbsp;&nbsp;<strong>Explaining:</strong> P(Fake)</p>'
    )
    full_heading = (
        '<div style="font-weight:700;color:#0f172a;margin:8px 0 4px;">'
        'Full article — every word shaded by influence '
        '<span style="font-weight:400;color:#64748b;font-size:0.85em">'
        '(<span style="color:#d73027">red = Fake</span>, '
        '<span style="color:#4575b4">blue = Real</span>; hover for the score)</span></div>'
    )
    html = shap.getjs() + header + full_heading + token_html
    return html
