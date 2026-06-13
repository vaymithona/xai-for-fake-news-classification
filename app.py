"""
Gradio web app for the XAI Fake News Classifier thesis demo.
Deployed to HuggingFace Spaces (CPU Basic tier).

Single-page, one-click design: paste an article, press **Analyze**, and the whole
story renders top-to-bottom on one screen — no tabs to click through.

Two modes, driven by the model dropdown:
  • "All 6 Models (compare)" — run all six, show the consensus verdict + per-model
    confidence bars, and explain the best model (DistilBERT).
  • A specific model         — focus on just that model: its verdict + its SHAP.
"""
import gradio as gr
import pandas as pd

from prototype.model_loader import MODEL_NAMES, ARTIFACTS_MISSING, ARTIFACTS
from prototype.examples import EXAMPLES

if not ARTIFACTS_MISSING:
    from prototype.inference import predict_all, get_proba_fn
    from prototype.shap_explainer import compute_shap

# ── Palette ─────────────────────────────────────────────────────────────────
_FAKE = '#dc2626'   # red  — pushes Fake
_REAL = '#2563eb'   # blue — pushes Real
_SPLIT = '#b45309'  # amber — tie
_FAKE_BG = '#fef2f2'
_REAL_BG = '#eff6ff'
_TRACK = '#f1f5f9'

# Dropdown: "All 6 Models" compares every model and explains the best one (DistilBERT).
_ALL = 'All 6 Models (compare)'
_BEST_MODEL = 'DistilBERT + MLP'   # highest test accuracy (~96%) in the notebook
DROPDOWN_CHOICES = [_ALL] + MODEL_NAMES


# ── HTML builders ────────────────────────────────────────────────────────────

def _verdict_banner(pred_df: pd.DataFrame, target_model: str) -> str:
    n = len(pred_df)
    n_fake = int((pred_df['Verdict'] == 'FAKE').sum())
    n_real = n - n_fake
    avg_fake = float(pred_df['P(Fake)%'].mean())

    if n_fake > n_real:
        label, color, bg, icon, agree = 'LIKELY FAKE', _FAKE, _FAKE_BG, '&#9888;', n_fake
        conf = avg_fake
    elif n_real > n_fake:
        label, color, bg, icon, agree = 'LIKELY REAL', _REAL, _REAL_BG, '&#10003;', n_real
        conf = 100 - avg_fake
    else:
        label, color, bg, icon, agree = 'SPLIT DECISION', _SPLIT, '#fffbeb', '&#8776;', n_fake
        conf = avg_fake

    return (
        f'<div style="background:{bg};border:1px solid {color}33;border-left:6px solid {color};'
        f'border-radius:10px;padding:18px 22px;margin:4px 0 14px;">'
        f'<div style="font-size:1.6em;font-weight:800;color:{color};letter-spacing:.5px;">'
        f'{icon} {label}</div>'
        f'<div style="margin-top:6px;color:#334155;font-size:1.0em;">'
        f'<b>{agree} of {n}</b> models agree &nbsp;&middot;&nbsp; '
        f'average confidence <b>{conf:.0f}%</b></div>'
        f'<div style="margin-top:4px;color:#64748b;font-size:0.88em;">'
        f'Explanation below is for <b>{target_model}</b> '
        f'(the best-performing model, ~96% test accuracy).</div>'
        f'</div>'
    )


def _single_banner(model: str, p_fake: float) -> str:
    """Verdict banner for a single, explicitly chosen model (focus mode)."""
    pct = p_fake * 100
    if p_fake >= 0.5:
        label, color, bg, icon, conf = 'LIKELY FAKE', _FAKE, _FAKE_BG, '&#9888;', pct
    else:
        label, color, bg, icon, conf = 'LIKELY REAL', _REAL, _REAL_BG, '&#10003;', 100 - pct
    return (
        f'<div style="background:{bg};border:1px solid {color}33;border-left:6px solid {color};'
        f'border-radius:10px;padding:18px 22px;margin:4px 0 14px;">'
        f'<div style="font-size:1.6em;font-weight:800;color:{color};letter-spacing:.5px;">'
        f'{icon} {model} &mdash; {label}</div>'
        f'<div style="margin-top:6px;color:#334155;font-size:1.0em;">'
        f'Confidence <b>{conf:.0f}%</b> &nbsp;&middot;&nbsp; P(Fake) = <b>{pct:.0f}%</b></div>'
        f'<div style="margin-top:4px;color:#64748b;font-size:0.88em;">'
        f'Showing this single model&rsquo;s prediction and its SHAP explanation below.</div>'
        f'</div>'
    )


def _pred_chart(pred_df: pd.DataFrame, target_model: str) -> str:
    rows = []
    for _, r in pred_df.iterrows():
        p = float(r['P(Fake)%'])
        is_fake = r['Verdict'] == 'FAKE'
        color = _FAKE if is_fake else _REAL
        badge_bg = _FAKE_BG if is_fake else _REAL_BG
        star = (' <span style="color:#0f172a">&#9679;</span>'
                if r['Model'] == target_model else '')
        rows.append(
            '<div style="display:flex;align-items:center;gap:12px;margin:7px 0;">'
            f'<div style="width:160px;text-align:right;font-size:0.86em;color:#334155;">'
            f'{r["Model"]}{star}</div>'
            f'<div style="flex:1;position:relative;background:{_TRACK};border-radius:6px;height:22px;">'
            f'<div style="width:{p}%;background:{color};height:100%;border-radius:6px;'
            f'transition:width .4s;"></div>'
            '<div style="position:absolute;left:50%;top:-2px;bottom:-2px;'
            'border-left:2px dashed #94a3b8;"></div>'
            '</div>'
            f'<div style="width:46px;text-align:right;font-weight:700;color:{color};">'
            f'{p:.0f}%</div>'
            f'<span style="min-width:46px;text-align:center;font-size:0.72em;font-weight:700;'
            f'color:{color};background:{badge_bg};border:1px solid {color}40;'
            f'border-radius:5px;padding:2px 6px;">{r["Verdict"]}</span>'
            '</div>'
        )
    return (
        '<h3 style="margin:18px 0 2px;">Per-model confidence</h3>'
        '<p style="color:#64748b;font-size:0.84em;margin:0 0 8px;">'
        'Each bar is that model&rsquo;s P(Fake). Past the dashed 50% line &rarr; classified '
        f'<b style="color:{_FAKE}">FAKE</b>; before it &rarr; '
        f'<b style="color:{_REAL}">REAL</b>. &nbsp;&#9679; = model explained below.</p>'
        '<div style="padding:6px 4px;">' + ''.join(rows) + '</div>'
    )


_NO_ARTIFACTS_MSG = (
    f'<div style="padding:1.5em;background:#fef3c7;border:1px solid #f59e0b;border-radius:8px">'
    f'<b>&#9888; Models not loaded</b><br>'
    f'Artifacts folder not found at <code>{ARTIFACTS.resolve()}</code>.<br><br>'
    f'1. Run the notebook end-to-end &nbsp; 2. Download <code>artifacts/</code> &nbsp; '
    f'3. Place it at the repo root and restart.'
    f'</div>'
)

_PLACEHOLDER = (
    '<div style="padding:1.4em;text-align:center;color:#94a3b8;border:1px dashed #cbd5e1;'
    'border-radius:10px;">Paste an article above and press '
    '<b>Analyze</b> to see the verdict, per-model confidence, and the SHAP '
    'explanation — all here.</div>'
)


def _empty(msg_html: str):
    """Standard 3-tuple (banner, chart, shap) for non-result states."""
    return msg_html, '', ''


def analyse(text: str, model_choice: str, progress=gr.Progress(track_tqdm=True)):
    if ARTIFACTS_MISSING:
        return _empty(_NO_ARTIFACTS_MSG)
    if not text or not text.strip():
        return _empty('<p style="color:#b45309">Enter article text above, then press Analyze.</p>')

    if model_choice == _ALL:
        # Compare mode: run all six, explain the best model (DistilBERT).
        progress(0.0, desc='Running all 6 models…')
        pred_df = predict_all(text)
        target_model = _BEST_MODEL
        banner = _verdict_banner(pred_df, target_model)
        chart = _pred_chart(pred_df, target_model)
    else:
        # Focus mode: run only the chosen model, show just its verdict + SHAP.
        target_model = model_choice
        progress(0.0, desc=f'Running {target_model}…')
        p_fake = float(get_proba_fn(target_model)([text])[0, 1])
        banner = _single_banner(target_model, p_fake)
        chart = ''

    note = ('<p style="color:#b45309;font-size:0.82em;margin:6px 0 0;">'
            '&#9888; DistilBERT SHAP can take up to ~45 s on CPU; other models finish in &lt;5 s.</p>'
            if target_model == 'DistilBERT + MLP' else '')

    progress(0.35, desc=f'Computing SHAP for {target_model}…')
    shap_html = compute_shap(text, target_model)

    progress(1.0)
    return banner, chart, note + shap_html


# ── Gradio UI ─────────────────────────────────────────────────────────────

with gr.Blocks(title='Fake News Classifier — XAI Demo',
               theme=gr.themes.Soft(primary_hue='blue')) as demo:

    gr.Markdown(
        '# 📰 Fake News Classifier — XAI Demo\n'
        'Six models trained on **WELFake** (~72 k articles). Paste an article, press '
        '**Analyze**, and see the verdict and *why* the model decided — all on one screen.\n'
        '> _Thesis project: Explainable AI for Fake News Classification_'
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                lines=7,
                label='Article text',
                placeholder='Paste a news article here…',
            )
        with gr.Column(scale=1, min_width=240):
            model_dd = gr.Dropdown(
                choices=DROPDOWN_CHOICES,
                value=_ALL,
                label='Model',
                info='“Compare” runs all six; pick one to focus on its prediction + SHAP.',
            )
            run_btn = gr.Button('🔍 Analyze', variant='primary', size='lg')

    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_input],
        label='Example articles — click to load, then press Analyze',
        examples_per_page=3,
    )

    gr.Markdown('---')

    verdict_out = gr.HTML(value=_PLACEHOLDER)
    chart_out = gr.HTML()
    shap_out = gr.HTML()

    run_btn.click(
        fn=analyse,
        inputs=[text_input, model_dd],
        outputs=[verdict_out, chart_out, shap_out],
    )

if __name__ == '__main__':
    demo.launch()
