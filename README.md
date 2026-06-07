# MoE Emotion Predictor

Real-time Japanese emotion prediction from text and audio using a Mixture of Experts (MoE) architecture. Given Japanese speech or text (or both), the system predicts a discrete emotion label and continuous valence/arousal scores — live, from mic input streamed by the client, or from an uploaded audio file.

---

## Current model performance

Best checkpoint: `outputs/checkpoints/best.pt` — macro F1 **0.303** on 7 classes (random baseline = 0.143).

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|----|---------|
| happy | 0.551 | 0.503 | 0.526 | 2,037 |
| sad | 0.490 | 0.284 | 0.359 | 1,632 |
| angry | 0.205 | 0.466 | 0.284 | 279 |
| neutral | 0.418 | 0.294 | 0.345 | 1,406 |
| anxious | 0.154 | 0.263 | 0.195 | 338 |
| surprised | 0.219 | 0.347 | 0.269 | 680 |
| disgusted | 0.106 | 0.212 | 0.141 | 288 |

See `documentation/training_performance_log.md` for the full training history.

---

## What this does

Given Japanese text, audio, or both, the system outputs:

- **Discrete emotion** — one of: `happy`, `sad`, `angry`, `neutral`, `anxious`, `surprised`, `disgusted`
- **Confidence scores** — softmax probability over all 7 classes
- **Valence** — how positive/negative the emotion is (1–9 scale)
- **Arousal** — how activated/calm the emotion is (1–9 scale)

Two inference modes:
- **File mode** — upload a `.wav`, `.mp3`, `.flac`, or `.m4a` clip (up to 30s) via HTTP
- **Stream mode** — browser mic captured via Web Audio API, sent as PCM chunks over WebSocket

---

## Architecture

```
Japanese text  →  JapaneseTextEncoder           Audio (file / stream bytes)
                  bert-base-japanese-v3     →   AudioEncoder
                  768-dim output                wav2vec2-base
                       │                        768-dim output
                       └──────────┬─────────────────────┘
                                  │
                             MoE Router
                        (learned soft gating)
                         text_weight │ audio_weight
                                  │
                           Weighted Fusion
                        (gate × text + gate × audio)
                                  │
                  ┌───────────────┴──────────────┐
                  │                              │
        Classification head             Regression head
         Linear → 7 classes              Linear → 2 outputs
         softmax probabilities           sigmoid → valence, arousal [1–9]
```

### Why these models

| Component | Choice | Reason |
|---|---|---|
| Text encoder | `cl-tohoku/bert-base-japanese-v3` | Best general Japanese BERT; native MeCab tokenisation |
| Audio encoder | `facebook/wav2vec2-base` | Encodes acoustic features (pitch, rhythm, energy) — not transcription |
| Router | Learned soft gate | Adapts during training; not a hardcoded if/else |
| Fusion | Weighted sum | Degrades cleanly when one modality is absent |
| Loss | CrossEntropy + inverse-frequency class weights + label smoothing 0.1 | Corrects for heavy class imbalance in training data |

---

## Deployment architecture

Audio capture happens on the client. The server receives bytes — no mic library needed server-side.

```
CLIENT                                    SERVER
──────────────────────────────────────    ──────────────────────────────────────
File upload: audio file ────────────────► POST /predict/file
                                          soundfile decode → AudioEncoder → model

Stream: Web Audio API (mic)
        PCM Float32 chunks ─────────────► WS /predict/stream
        500ms @ 16kHz                     np.frombuffer → AudioEncoder → model

Japanese text string ───────────────────► POST /predict/text
                                          BERT tokenise → model
```

`pyaudio` and `webrtcvad` are not installed anywhere in this project.

---

## Training datasets

### WRIME — Japanese text (text expert)
~43,000 Japanese social media posts annotated with Plutchik emotion intensities (0–3). Downloaded automatically from GitHub on first run. Covers the text modality end-to-end.

### RAVDESS — English emotional speech (audio expert)
1,440 clips from 24 professional actors. English speech bootstraps the audio encoder — acoustic emotion patterns (pitch, energy, rhythm) transfer across languages. Download once:

```bash
python data/download_ravdess.py   # ~1.6 GB from Zenodo
```

### Emotion label derivation

Both datasets map to our 7 classes via `data/emotion_mappings.py`:

| Our label | From WRIME | From RAVDESS |
|---|---|---|
| happy | joy | happy (03) |
| sad | sadness | sad (04) |
| angry | anger | angry (05) |
| neutral | trust, anticipation | neutral (01), calm (02) |
| anxious | fear | fearful (06) |
| surprised | surprise | surprised (08) |
| disgusted | disgust | disgust (07) |

Valence/arousal targets are approximated using Russell's circumplex model and scaled by per-sample intensity. Neither dataset provides continuous V/A labels directly.

---

## Repository structure

```
MoE/
├── configs/
│   ├── base.yaml              # paths, emotion classes, inference settings
│   ├── model.yaml             # encoder names, hidden sizes, freeze layers
│   └── train.yaml             # epochs, LR, loss weights, dataset paths
│
├── models/
│   ├── text/encoder.py        # JapaneseTextEncoder (BERT)
│   ├── audio/encoder.py       # AudioEncoder (Wav2Vec2, soundfile-based)
│   ├── heads/emotion_head.py  # dual head: 7-class clf + V/A regression
│   ├── fusion/fusion.py       # WeightedFusion with learned gate
│   └── multimodal/
│       ├── router.py          # MoERouter — learned soft gating
│       └── moe.py             # MoEEmotionModel — top-level model
│
├── data/
│   ├── emotion_mappings.py    # label maps, V/A approximations (single source of truth)
│   ├── wrime_dataset.py       # WRIME PyTorch Dataset (auto-downloads from GitHub)
│   ├── ravdess_dataset.py     # RAVDESS PyTorch Dataset (soundfile-based loading)
│   ├── combined_dataset.py    # raw DataLoader factory (used for phase 2 / inference)
│   ├── cached_dataset.py      # fast DataLoader from pre-computed features
│   ├── download_ravdess.py    # one-time RAVDESS download script
│   └── raw/
│       ├── ravdess/           # Actor_01/ … Actor_24/ (not committed)
│       ├── wrime/             # wrime-ver1.tsv (not committed)
│       └── features/          # cached encoder outputs (not committed)
│
├── training/
│   ├── cache_features.py      # run once: encodes all data, saves 768-dim vectors
│   └── train.py               # training loop (phase 1 cached, phase 2 raw)
│
├── utils/
│   └── deap_label_mapper.py   # V/A → discrete (scale reference, not used in training)
│
├── documentation/
│   └── training_performance_log.md
│
├── inference/                 # next: pipeline.py
├── api/                       # next: FastAPI HTTP + WebSocket
├── frontend/                  # next: Web Audio API client
├── notebooks/
├── reporting/
├── personalization/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <your-repo-url>
cd MoE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify Japanese tokeniser
python -c "import fugashi; print('MeCab OK')"

# Download RAVDESS (one-time, ~1.6 GB)
python data/download_ravdess.py
```

---

## Training

```bash
# Step 1 — cache encoder features (run once, ~10 min)
python training/cache_features.py

# Step 2 — train phase 1 (frozen encoders, ~1–2 min/epoch)
python training/train.py

# Resume from checkpoint
python training/train.py --resume outputs/checkpoints/best.pt

# Monitor
tensorboard --logdir logs/
```

---

## Inference (quick usage)

```python
import yaml, torch, soundfile as sf
from models.multimodal.moe import MoEEmotionModel

with open("configs/model.yaml") as f:
    config = yaml.safe_load(f)

model = MoEEmotionModel(config)
model.load_state_dict(torch.load("outputs/checkpoints/best.pt")["model"])

# Text only
result = model.predict(text="今日はとても嬉しい！")
print(result["predicted_label"])   # ["happy"]

# Audio file
waveform, sr = sf.read("clip.wav", dtype="float32")
result = model.predict(audio=waveform)

# Both
result = model.predict(text="悲しいね", audio=waveform)
print(result["valence"].item())    # e.g. 2.4
print(result["arousal"].item())    # e.g. 3.1
```

---

## Dependencies

| Category | Key packages |
|---|---|
| Deep learning | `torch`, `torchaudio`, `transformers`, `accelerate` |
| Japanese NLP | `fugashi`, `ipadic`, `sentencepiece` |
| Audio (server-side) | `librosa`, `soundfile` — no mic capture |
| API | `fastapi`, `uvicorn`, `websockets`, `pydantic` |
| Training | `torchmetrics`, `tensorboard`, `wandb` |
| Utilities | `pyyaml`, `omegaconf`, `tqdm`, `rich` |

Not included: `pyaudio`, `webrtcvad` — mic capture is client-side.

---

## License

MIT