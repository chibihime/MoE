"""
app.py
=======
Streamlit Dashboard — DEAP Continuous Emotion Prediction

Run:
    streamlit run src/app.py
    streamlit run src/app.py -- --checkpoint src/checkpoints/fold_01/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deap_emotion_model import DEAPEmotionModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIMENSIONS     = ["valence", "arousal", "dominance", "liking"]
EXPERT_NAMES   = ["Physio", "Audio", "Cross-Modal", "Context"]
AUDIO_FEATURES = ["note_density", "mean_pitch", "tempo"]

PHYSIO_CHANNELS = [
    f"ch{ch}_{stat}"
    for stat in ("mean", "std", "min", "max")
    for ch in range(1, 9)
]

# 6 user-facing grouped signals → slice of the 32-feature physio vector
SIGNAL_GROUPS = {
    "EEG alpha":        (0,  8),
    "EEG beta":         (8,  16),
    "Heart rate":       (16, 20),
    "Skin conductance": (20, 24),
    "Respiration":      (24, 28),
    "Skin temperature": (28, 32),
}

SIGNAL_HELP = {
    "EEG alpha":        "Alpha-band power. Higher = more relaxed / drowsy.",
    "EEG beta":         "Beta-band power. Higher = more alert / stressed.",
    "Heart rate":       "Cardiac activity. Higher = more aroused.",
    "Skin conductance": "Sweat response. Higher = more emotionally activated.",
    "Respiration":      "Breathing rate. Higher = more aroused / anxious.",
    "Skin temperature": "Peripheral blood flow proxy.",
}

# Session-state keys for every slider.
# Writing to these BEFORE the widget renders is the correct Streamlit pattern
# for programmatically updating slider values (the value= param is ignored
# after first render once the key is registered in session_state).
PHYSIO_SKEYS = {g: f"sl_physio_{i}" for i, g in enumerate(SIGNAL_GROUPS)}
AUDIO_SKEYS  = {
    "note_density": "sl_audio_0",
    "mean_pitch":   "sl_audio_1",
    "tempo":        "sl_audio_2",
}

DEFAULT_CKPT = str(SRC_DIR / "checkpoints" / "fold_01" / "best.pt")
DIM_COLORS   = ["#378ADD", "#D85A30", "#1D9E75", "#BA7517"]

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "Calm music": {
        "physio": {"EEG alpha": 1.2,  "EEG beta": -0.5, "Heart rate": -0.8,
                   "Skin conductance": -1.0, "Respiration": -0.6, "Skin temperature":  0.2},
        "audio":  {"note_density": 0.8, "mean_pitch": 55.0, "tempo":  70.0},
    },
    "Energetic track": {
        "physio": {"EEG alpha": -0.8, "EEG beta":  1.5, "Heart rate":  1.2,
                   "Skin conductance":  1.0, "Respiration":  0.8, "Skin temperature":  0.4},
        "audio":  {"note_density": 3.5, "mean_pitch": 72.0, "tempo": 160.0},
    },
    "Tense moment": {
        "physio": {"EEG alpha": -1.5, "EEG beta":  2.0, "Heart rate":  1.8,
                   "Skin conductance":  2.2, "Respiration":  1.2, "Skin temperature": -0.5},
        "audio":  {"note_density": 2.2, "mean_pitch": 68.0, "tempo": 140.0},
    },
    "Melancholic": {
        "physio": {"EEG alpha":  0.5, "EEG beta": -0.3, "Heart rate": -0.5,
                   "Skin conductance": -0.8, "Respiration": -0.4, "Skin temperature": -0.2},
        "audio":  {"note_density": 0.6, "mean_pitch": 48.0, "tempo":  65.0},
    },
    "Neutral": {
        "physio": {g: 0.0 for g in SIGNAL_GROUPS},
        "audio":  {"note_density": 1.0, "mean_pitch": 60.0, "tempo": 120.0},
    },
}

# Seed session_state slider defaults on first load (before any widget renders)
if "sliders_initialised" not in st.session_state:
    for grp in SIGNAL_GROUPS:
        st.session_state[PHYSIO_SKEYS[grp]] = 0.0
    st.session_state[AUDIO_SKEYS["note_density"]] = 1.0
    st.session_state[AUDIO_SKEYS["mean_pitch"]]   = 60.0
    st.session_state[AUDIO_SKEYS["tempo"]]        = 120.0
    st.session_state["sliders_initialised"] = True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
try:
    _cli_args, _ = _parser.parse_known_args()
    CLI_CKPT = _cli_args.checkpoint
except Exception:
    CLI_CKPT = DEFAULT_CKPT

# ---------------------------------------------------------------------------
# Page config — sidebar fully hidden
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DEAP Emotion Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"]  { display: none !important; }

[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 600; }
[data-testid="stMetricDelta"] { font-size: 0.78rem; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading (cached — runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(ckpt_path: str) -> tuple[DEAPEmotionModel | None, bool]:
    if os.path.exists(ckpt_path):
        m = DEAPEmotionModel.load(ckpt_path, map_location="cpu")
        m.eval()
        return m, True
    return None, False

model, ckpt_found = load_model(CLI_CKPT)

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def group_vals_to_physio_vec(group_vals: dict[str, float]) -> np.ndarray:
    """Expand 6 grouped z-scores into the 32-element physio feature vector."""
    vec = np.zeros(32, dtype=np.float32)
    for name, (s, e) in SIGNAL_GROUPS.items():
        vec[s:e] = group_vals[name]
    return vec


def mock_inference(
    group_vals: dict[str, float],
    audio_vals: dict[str, float],
    physio_present: bool,
    audio_present: bool,
) -> dict:
    """Input-sensitive heuristic used when no trained checkpoint is available."""
    alpha = group_vals["EEG alpha"]
    beta  = group_vals["EEG beta"]
    hr    = group_vals["Heart rate"]
    gsr   = group_vals["Skin conductance"]
    resp  = group_vals["Respiration"]

    t_n = (audio_vals["tempo"]        - 120) / 80
    p_n = (audio_vals["mean_pitch"]   -  65) / 35
    d_n = (audio_vals["note_density"] - 2.5) / 2.5

    p_val =  0.55 * alpha - 0.30 * beta  - 0.25 * gsr
    p_aro = -0.20 * alpha + 0.55 * beta  + 0.45 * hr  + 0.30 * gsr + 0.20 * resp
    p_dom = -0.35 * gsr   + 0.30 * alpha + 0.20 * beta
    p_lik =  0.50 * alpha - 0.20 * beta  - 0.20 * gsr

    a_val =  0.55 * p_n - 0.15 * d_n
    a_aro =  0.55 * t_n + 0.30 * d_n
    a_dom =  0.30 * p_n + 0.10 * t_n
    a_lik =  0.50 * p_n - 0.10 * d_n

    if physio_present and audio_present:
        val = 5 + 0.60 * p_val + 0.40 * a_val
        aro = 5 + 0.55 * p_aro + 0.45 * a_aro
        dom = 5 + 0.65 * p_dom + 0.35 * a_dom
        lik = 5 + 0.55 * p_lik + 0.45 * a_lik
        gw  = [0.38, 0.28, 0.22, 0.12]
    elif physio_present:
        val, aro, dom, lik = 5 + p_val, 5 + p_aro, 5 + p_dom, 5 + p_lik
        gw = [0.72, 0.04, 0.14, 0.10]
    elif audio_present:
        val, aro, dom, lik = 5 + a_val, 5 + a_aro, 5 + a_dom, 5 + a_lik
        gw = [0.04, 0.72, 0.14, 0.10]
    else:
        val = aro = dom = lik = 5.0
        gw = [0.25, 0.25, 0.25, 0.25]

    clamp = lambda v: float(np.clip(v, 1.0, 9.0))
    return {
        "valence": clamp(val), "arousal": clamp(aro),
        "dominance": clamp(dom), "liking": clamp(lik),
        "gate_weights": torch.tensor(gw),
    }


def run_inference(
    group_vals: dict[str, float],
    audio_vals: dict[str, float],
    physio_present: bool,
    audio_present: bool,
) -> dict:
    if not ckpt_found or model is None:
        return mock_inference(group_vals, audio_vals, physio_present, audio_present)

    physio_vec = group_vals_to_physio_vec(group_vals)
    audio_vec  = np.array(
        [audio_vals["note_density"], audio_vals["mean_pitch"], audio_vals["tempo"]],
        dtype=np.float32,
    )
    pt = torch.tensor(physio_vec).unsqueeze(0)
    at = torch.tensor(audio_vec).unsqueeze(0)
    mt = torch.tensor([[float(physio_present), float(audio_present)]])

    with torch.no_grad():
        out = model.predict(pt, at, mt, return_gate_weights=True)

    def rescale(x):
        return float((x + 1) * 4 + 1)   # [-1,1] → [1,9]

    return {dim: rescale(out[dim][0, 0]) for dim in DIMENSIONS} | {
    "gate_weights": out["gate_weights"][0]
}


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def score_label(v: float) -> str:
    return "Low" if v < 3.5 else ("Mid" if v < 6.5 else "High")


def state_label(valence: float, arousal: float) -> str:
    hv, ha = valence >= 5, arousal >= 5
    if hv and ha:      return "😄 Excited / Happy"
    if not hv and ha:  return "😤 Stressed / Tense"
    if hv and not ha:  return "😌 Calm / Relaxed"
    return "😔 Sad / Disengaged"


def render_bar(label: str, value: float, colour: str, width_px: int = 90) -> None:
    pct = (value - 1.0) / 8.0 * 100
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:7px'>"
        f"<span style='width:{width_px}px;font-size:0.83rem;color:#9CA3AF'>{label}</span>"
        f"<div style='flex:1;background:#1F2937;border-radius:4px;height:10px'>"
        f"<div style='width:{pct:.1f}%;background:{colour};border-radius:4px;height:10px'></div>"
        f"</div>"
        f"<span style='width:40px;text-align:right;font-size:0.83rem;font-weight:600;"
        f"color:#E5E7EB'>{value:.2f}</span></div>",
        unsafe_allow_html=True,
    )


def render_gate_bar(label: str, weight: float, colour: str) -> None:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>"
        f"<span style='width:100px;font-size:0.82rem;color:#9CA3AF'>{label}</span>"
        f"<div style='flex:1;background:#1F2937;border-radius:4px;height:8px'>"
        f"<div style='width:{weight*100:.1f}%;background:{colour};border-radius:4px;height:8px'></div>"
        f"</div>"
        f"<span style='width:42px;text-align:right;font-size:0.82rem;color:#D1D5DB'>"
        f"{weight*100:.0f}%</span></div>",
        unsafe_allow_html=True,
    )


def render_circumplex(valence: float, arousal: float) -> None:
    cx, cy, r = 160, 160, 125
    px = cx + (valence - 5) / 4 * r
    py = cy - (arousal - 5) / 4 * r
    r2, r3 = r * 2 // 3, r // 3
    st.markdown(
        f"""<svg width="320" height="320" xmlns="http://www.w3.org/2000/svg">
  <rect width="320" height="320" fill="#0D1117" rx="10"/>
  <circle cx="{cx}" cy="{cy}" r="{r}"  stroke="#2D3748" stroke-width="0.5" fill="none"/>
  <circle cx="{cx}" cy="{cy}" r="{r2}" stroke="#2D3748" stroke-width="0.5" fill="none" stroke-dasharray="4,4"/>
  <circle cx="{cx}" cy="{cy}" r="{r3}" stroke="#2D3748" stroke-width="0.5" fill="none" stroke-dasharray="4,4"/>
  <line x1="{cx-r-12}" y1="{cy}" x2="{cx+r+12}" y2="{cy}" stroke="#374151" stroke-width="0.5"/>
  <line x1="{cx}" y1="{cy-r-12}" x2="{cx}" y2="{cy+r+12}" stroke="#374151" stroke-width="0.5"/>
  <text x="{cx+r+6}"  y="{cy+4}"    fill="#6B7280" font-size="10" font-family="system-ui">+Valence</text>
  <text x="{cx-r-50}" y="{cy+4}"    fill="#6B7280" font-size="10" font-family="system-ui">−Valence</text>
  <text x="{cx+4}"    y="{cy-r-6}"  fill="#6B7280" font-size="10" font-family="system-ui">+Arousal</text>
  <text x="{cx+4}"    y="{cy+r+14}" fill="#6B7280" font-size="10" font-family="system-ui">−Arousal</text>
  <line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{cy}"      stroke="#4F8EF7" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>
  <line x1="{px:.1f}" y1="{py:.1f}" x2="{cx}"      y2="{py:.1f}" stroke="#4F8EF7" stroke-width="1" stroke-dasharray="3,3" opacity="0.5"/>
  <circle cx="{px:.1f}" cy="{py:.1f}" r="14" fill="#4F8EF7" opacity="0.15"/>
  <circle cx="{px:.1f}" cy="{py:.1f}" r="6"  fill="#4F8EF7"/>
  <circle cx="{px:.1f}" cy="{py:.1f}" r="2"  fill="white"/>
</svg>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Emotion Predictor")
st.caption(
    "Continuous Valence · Arousal · Dominance · Liking  ·  "
    "Mixture-of-Experts fusion model trained on DEAP."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_manual, tab_batch = st.tabs(["Single prediction", "Batch inference"])


# ===========================================================================
# TAB 1 — SINGLE PREDICTION
# ===========================================================================
with tab_manual:

    # ------------------------------------------------------------------
    # Preset buttons
    #
    # The fix for "same output every time":
    #   Write preset values directly into PHYSIO_SKEYS / AUDIO_SKEYS
    #   in session_state BEFORE the slider widgets are created.
    #   Streamlit reads session_state[key] as the current slider value,
    #   so this correctly repositions every slider on the next rerun.
    #   We also set run_on_rerun=True so inference fires immediately
    #   without the user having to click Predict.
    # ------------------------------------------------------------------
    st.markdown("**Start from a preset**")
    preset_cols = st.columns(len(PRESETS))
    for col, name in zip(preset_cols, PRESETS):
        if col.button(name, use_container_width=True, key=f"btn_preset_{name}"):
            for grp, val in PRESETS[name]["physio"].items():
                st.session_state[PHYSIO_SKEYS[grp]] = float(val)
            for feat, val in PRESETS[name]["audio"].items():
                st.session_state[AUDIO_SKEYS[feat]] = float(val)
            st.session_state["active_preset"]  = name
            st.session_state["run_on_rerun"]   = True
            st.session_state.pop("manual_preds", None)  # clear stale result

    st.divider()

    col_input, col_results = st.columns([1, 1], gap="large")

    # ------------------------------------------------------------------
    # Left column — inputs
    # ------------------------------------------------------------------
    with col_input:
        st.subheader("Inputs")

        mc1, mc2 = st.columns(2)
        physio_present = mc1.toggle("Physiological", value=True, key="tog_physio")
        audio_present  = mc2.toggle("Audio",         value=True, key="tog_audio")

        if not physio_present and not audio_present:
            st.warning("Enable at least one modality.")

        # Physiological sliders — no value= param; driven purely by session_state key
        with st.expander("Physiological signals", expanded=physio_present):
            st.caption(
                "Z-score deviation from training mean  "
                "(−3 = very low · 0 = average · +3 = very high)"
            )
            group_vals: dict[str, float] = {}
            for grp in SIGNAL_GROUPS:
                group_vals[grp] = st.slider(
                    grp,
                    min_value=-3.0, max_value=3.0, step=0.1,
                    help=SIGNAL_HELP[grp],
                    key=PHYSIO_SKEYS[grp],
                    disabled=not physio_present,
                )

        # Audio sliders
        with st.expander("Audio features", expanded=audio_present):
            ac1, ac2, ac3 = st.columns(3)
            note_density = ac1.slider(
                "Note density (n/s)", 0.0, 5.0, step=0.1,
                key=AUDIO_SKEYS["note_density"],
                disabled=not audio_present,
            )
            mean_pitch = ac2.slider(
                "Mean pitch (MIDI)", 30.0, 100.0, step=1.0,
                key=AUDIO_SKEYS["mean_pitch"],
                disabled=not audio_present,
            )
            tempo = ac3.slider(
                "Tempo (BPM)", 40.0, 200.0, step=1.0,
                key=AUDIO_SKEYS["tempo"],
                disabled=not audio_present,
            )

        audio_vals = {
            "note_density": note_density,
            "mean_pitch":   mean_pitch,
            "tempo":        tempo,
        }

        predict_clicked = st.button("Predict →", type="primary", use_container_width=True)

        # Fire inference on explicit button click OR after a preset button sets run_on_rerun
        if predict_clicked or st.session_state.pop("run_on_rerun", False):
            preds = run_inference(group_vals, audio_vals, physio_present, audio_present)
            st.session_state["manual_preds"] = preds

    # ------------------------------------------------------------------
    # Right column — results
    # ------------------------------------------------------------------
    with col_results:
        st.subheader("Results")

        if "manual_preds" not in st.session_state:
            st.info("Click a preset above, or set inputs and press **Predict →**.")
        else:
            preds  = st.session_state["manual_preds"]
            active = st.session_state.get("active_preset", "")

            state = state_label(preds["valence"], preds["arousal"])
            st.markdown(
                f"<div style='font-size:1.4rem;font-weight:600;margin-bottom:4px'>{state}</div>",
                unsafe_allow_html=True,
            )
            if active:
                st.caption(f"Preset: {active}")

            st.divider()

            m1, m2, m3, m4 = st.columns(4)
            for mc, dim in zip([m1, m2, m3, m4], DIMENSIONS):
                mc.metric(dim.capitalize(), f"{preds[dim]:.2f}", score_label(preds[dim]))

            st.divider()

            st.markdown("**Dimension scores**")
            for dim, colour in zip(DIMENSIONS, DIM_COLORS):
                render_bar(dim.capitalize(), preds[dim], colour)

            st.divider()

            circ_col, gate_col = st.columns([1, 1], gap="medium")
            with circ_col:
                st.markdown("**Valence × Arousal**")
                render_circumplex(preds["valence"], preds["arousal"])
            with gate_col:
                st.markdown("**Expert utilisation**")
                gw = preds["gate_weights"]
                for name, w, colour in zip(EXPERT_NAMES, gw.tolist(), DIM_COLORS):
                    render_gate_bar(name, w, colour)
                st.caption(
                    "Gate weights show how much each expert "
                    "contributed to this prediction."
                )


# ===========================================================================
# TAB 2 — BATCH INFERENCE
# ===========================================================================
with tab_batch:
    st.subheader("Batch inference")
    st.markdown(
        "Upload a CSV with 32 physio columns (`ch1_mean` … `ch8_max`) and "
        "3 audio columns (`note_density`, `mean_pitch`, `tempo`).  \n"
        "Optional columns `physio_present` and `audio_present` (0/1) enable per-row masking."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(uploaded_df):,} rows × {len(uploaded_df.columns)} columns.")
            st.dataframe(uploaded_df.head(5), use_container_width=True)
            st.session_state["batch_df"] = uploaded_df
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.divider()
    st.markdown("**Or generate a synthetic test batch**")
    n_demo = st.slider("Number of samples", 5, 200, 40, key="n_demo")

    if st.button("Generate synthetic batch", key="gen_btn"):
        rng          = np.random.default_rng(42)
        physio_batch = rng.normal(0, 1, (n_demo, 32)).astype(np.float32)
        audio_batch  = np.column_stack([
            rng.uniform(0,   5,   n_demo),
            rng.uniform(30, 100,  n_demo),
            rng.uniform(40, 200,  n_demo),
        ]).astype(np.float32)
        synth_df = pd.DataFrame(physio_batch, columns=PHYSIO_CHANNELS)
        for i, col in enumerate(AUDIO_FEATURES):
            synth_df[col] = audio_batch[:, i]
        st.session_state["batch_df"]      = synth_df
        st.session_state["batch_results"] = None
        st.success(f"Generated {n_demo} synthetic samples.")
        st.dataframe(synth_df.head(5), use_container_width=True)

    batch_df = st.session_state.get("batch_df")

    if batch_df is not None:
        if st.button("Run batch inference", type="primary", key="run_batch_btn"):
            missing_p = [c for c in PHYSIO_CHANNELS if c not in batch_df.columns]
            missing_a = [c for c in AUDIO_FEATURES  if c not in batch_df.columns]
            if missing_p or missing_a:
                st.error(
                    f"Missing columns — physio: {missing_p or 'OK'}  audio: {missing_a or 'OK'}"
                )
            else:
                physio_np = batch_df[PHYSIO_CHANNELS].values.astype(np.float32)
                audio_np  = batch_df[AUDIO_FEATURES].values.astype(np.float32)
                mask_np   = (
                    batch_df[["physio_present", "audio_present"]].values.astype(np.float32)
                    if "physio_present" in batch_df.columns and "audio_present" in batch_df.columns
                    else np.ones((len(batch_df), 2), dtype=np.float32)
                )
                with torch.no_grad():
                    raw = model.predict(
                        torch.from_numpy(physio_np),
                        torch.from_numpy(audio_np),
                        torch.from_numpy(mask_np),
                        return_gate_weights=True,
                    )
                result_df = batch_df.copy()
                for dim in DIMENSIONS:
                    result_df[f"pred_{dim}"] = raw[dim].cpu().numpy().flatten()
                gw_np = raw["gate_weights"].cpu().numpy()
                for i, name in enumerate(EXPERT_NAMES):
                    result_df[f"gate_{name.lower().replace('-', '_')}"] = gw_np[:, i]
                st.session_state["batch_results"] = result_df

        results = st.session_state.get("batch_results")
        if results is not None:
            pred_cols = [f"pred_{d}" for d in DIMENSIONS]
            gate_cols = [f"gate_{n.lower().replace('-', '_')}" for n in EXPERT_NAMES]

            st.success(f"Inference complete — {len(results):,} samples.")

            st.markdown("**Summary statistics**")
            summary = results[pred_cols].describe().T.rename(
                index={f"pred_{d}": d.capitalize() for d in DIMENSIONS}
            )
            st.dataframe(summary[["mean", "std", "min", "max"]].round(3), use_container_width=True)

            st.markdown("**Score distributions**")
            chart_df = results[pred_cols].rename(
                columns={f"pred_{d}": d.capitalize() for d in DIMENSIONS}
            )
            st.line_chart(chart_df, height=260, use_container_width=True)

            st.markdown("**Mean expert utilisation**")
            gw_means = results[gate_cols].mean()
            for name, col, colour in zip(EXPERT_NAMES, gate_cols, DIM_COLORS):
                render_gate_bar(name, gw_means[col], colour)

            st.divider()
            st.markdown("**Results table** (first 50 rows)")
            st.dataframe(results[pred_cols + gate_cols].head(50).round(4), use_container_width=True)

            st.download_button(
                label="Download full results CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="deap_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.info("Upload a CSV or generate a synthetic batch, then click **Run batch inference**.")