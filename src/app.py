# app.py (UPDATED)

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
DIMENSIONS = ["valence", "arousal", "dominance", "liking"]
EXPERT_NAMES = ["Physio", "Audio", "Cross-Modal", "Context"]

DEFAULT_CKPT = str(SRC_DIR / "checkpoints" / "lstm_best.pt")

AVAILABLE_MODELS = {
    "LSTM (best CCC)": "checkpoints/lstm_best.pt",
    "RNN": "checkpoints/rnn.pt",
    "Transformer": "checkpoints/transformer.pt",
}

BEST_MODEL = "LSTM (best CCC)"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Emotion Predictor", layout="wide")

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(ckpt_path: str):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model = DEAPEmotionModel.load(ckpt_path, map_location="cpu")
        model.eval()

        config = ckpt.get("config", {})
        metrics = ckpt.get("metrics", {})

        return model, True, config, metrics

    return None, False, {}, {}

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def confidence_from_gates(gw):
    entropy = -torch.sum(gw * torch.log(gw + 1e-8))
    return float(1 - entropy / np.log(len(gw)))

def run_inference(model, group_vals, audio_vals):
    pt = torch.tensor(group_vals).unsqueeze(0)
    at = torch.tensor(audio_vals).unsqueeze(0)
    mt = torch.tensor([[1.0, 1.0]])

    with torch.no_grad():
        out = model.predict(pt, at, mt, return_gate_weights=True)

    def rescale(x):
        return float((x + 1) * 4 + 1)

    preds = {dim: rescale(out[dim][0, 0]) for dim in DIMENSIONS}
    preds["gate_weights"] = out["gate_weights"][0]

    return preds

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Emotion Predictor (MoE + Sequence Models)")

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
selected_model_name = st.selectbox("Select model", list(AVAILABLE_MODELS.keys()))
selected_ckpt = str(SRC_DIR / AVAILABLE_MODELS[selected_model_name])

model, ckpt_found, model_config, model_metrics = load_model(selected_ckpt)

# ---------------------------------------------------------------------------
# Model Info Panel
# ---------------------------------------------------------------------------
if ckpt_found:
    st.success("Model loaded")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", model_config.get("seq_model", "unknown").upper())
    col2.metric("Embedding", model_config.get("embed_dim", "-"))
    col3.metric("Mean CCC", f"{model_metrics.get('mean_ccc', 0):.3f}")

    if selected_model_name == BEST_MODEL:
        st.markdown("🏆 **Best performing model**")

else:
    st.warning("No checkpoint found")

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.subheader("Inputs")

col1, col2 = st.columns(2)

with col1:
    val = st.slider("Valence proxy input", -3.0, 3.0, 0.0)
    aro = st.slider("Arousal proxy input", -3.0, 3.0, 0.0)

with col2:
    pitch = st.slider("Pitch", 30.0, 100.0, 60.0)
    tempo = st.slider("Tempo", 40.0, 200.0, 120.0)

group_vals = np.array([val, aro] + [0]*30, dtype=np.float32)
audio_vals = np.array([1.0, pitch, tempo], dtype=np.float32)

# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------
compare_mode = st.toggle("Compare all models")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict"):
    if not ckpt_found:
        st.error("No model loaded")
    else:
        with st.spinner("Running inference..."):

            if compare_mode:
                results = {}

                for name, path in AVAILABLE_MODELS.items():
                    m, ok, _, _ = load_model(str(SRC_DIR / path))
                    if ok:
                        preds = run_inference(m, group_vals, audio_vals)
                        results[name] = preds

                df = pd.DataFrame({
                    k: [v[d] for d in DIMENSIONS]
                    for k, v in results.items()
                }, index=DIMENSIONS)

                st.dataframe(df.round(2))

            else:
                preds = run_inference(model, group_vals, audio_vals)

                st.subheader("Results")

                cols = st.columns(4)
                for c, d in zip(cols, DIMENSIONS):
                    c.metric(d.capitalize(), f"{preds[d]:.2f}")

                # Confidence
                conf = confidence_from_gates(preds["gate_weights"])
                st.metric("Confidence", f"{conf:.2f}")

                # Gate weights
                st.subheader("Expert Weights")
                for name, w in zip(EXPERT_NAMES, preds["gate_weights"].tolist()):
                    st.progress(float(w), text=f"{name}: {w:.2f}")

# ---------------------------------------------------------------------------
# Batch inference (safe)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Batch Inference")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    if not ckpt_found:
        st.error("Batch inference requires trained model")
    else:
        if st.button("Run batch"):
            physio = torch.tensor(df.iloc[:, :32].values, dtype=torch.float32)
            audio = torch.tensor(df.iloc[:, 32:35].values, dtype=torch.float32)
            mask = torch.ones((len(df), 2))

            with torch.no_grad():
                out = model.predict(physio, audio, mask)

            for dim in DIMENSIONS:
                df[f"pred_{dim}"] = out[dim].numpy().flatten()

            st.dataframe(df.head())
            st.download_button("Download", df.to_csv(index=False))