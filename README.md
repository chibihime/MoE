# MoE Emotion Predictor

Real-time Japanese emotion prediction from text and audio using a **Mixture of Experts (MoE)** architecture. Predicts both a discrete emotion label and continuous valence/arousal scores — live, from microphone input (streamed from the client) or uploaded audio files.

---

## What This Does

Given Japanese speech or text (or both), the system outputs:

- **Discrete emotion** — one of: `happy`, `sad`, `angry`, `neutral`, `anxious`, `surprised`, `disgusted`
- **Confidence scores** — softmax probability over all 7 classes
- **Valence score** — how positive/negative the emotion is (1–9 scale)
- **Arousal score** — how activated/calm the emotion is (1–9 scale)

It runs in two modes:
- **File mode** — upload a `.wav`, `.mp3`, `.flac`, or `.m4a` audio clip (up to 30s) via HTTP
- **Stream mode** — real-time mic capture in the browser/client, sent as PCM chunks over WebSocket

---

## Deployment Architecture

Audio capture **always happens on the client** (browser or mobile app). The server never touches a microphone directly. `pyaudio` and `webrtcvad` are not installed or required anywhere in this project.

```
┌─────────────────────────────────┐        ┌──────────────────────────────────┐
│           CLIENT                │        │            SERVER                │
│                                 │        │                                  │
│  Browser / Mobile App           │        │  FastAPI                         │
│                                 │        │                                  │
│  File mode:                     │        │  POST /predict/file              │
│    audio file ──────────────────┼──HTTP──►  soundfile / torchaudio decode   │
│                                 │        │  → AudioEncoder → MoEModel       │
│  Stream mode:                   │        │                                  │
│    Web Audio API (mic)          │        │  WS  /predict/stream             │
│    → PCM Float32 chunks ────────┼──WS────►  np.frombuffer(bytes, float32)   │
│      (500ms @ 16kHz)            │        │  → AudioEncoder → MoEModel       │
│                                 │        │                                  │
│  Text mode (optional):          │        │  POST /predict/text              │
│    Japanese text string ────────┼──HTTP──►  BERT tokenizer → MoEModel       │
└─────────────────────────────────┘        └──────────────────────────────────┘
```

---

## Model Architecture

```
Japanese Text ──► JapaneseTextEncoder          Audio (file / stream bytes)
(string)          [cl-tohoku/bert-base-ja-v3]  ──► AudioEncoder
                  hidden: 768                       [facebook/wav2vec2-base]
                       │                            hidden: 768
                       │                                │
                       └──────────┬─────────────────────┘
                                  │
                             MoERouter
                        (learned soft gating)
                         text_weight │ audio_weight
                                  │
                           WeightedFusion
                           (gate × text_proj
                            + gate × audio_proj)
                           hidden: 512
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
             ClassificationHead          RegressionHead
             Linear → 7 classes          Linear → 2 outputs
             softmax probabilities        sigmoid → [1, 9]
                    │                            │
             predicted_label              valence, arousal
             + confidence scores
```

### Key design choices

| Component | Choice | Reason |
|---|---|---|
| Text encoder | `cl-tohoku/bert-base-japanese-v3` | Best general Japanese BERT; native MeCab tokenization |
| Audio encoder | `facebook/wav2vec2-base` | Acoustic features (not transcription); captures pitch, energy, rhythm |
| Routing | Learned soft gate | Adapts during training; not a hardcoded if/else switch |
| Fusion | Weighted sum with learned gate | Gracefully degrades when one modality is absent |
| Output | Dual head (clf + regression) | Discrete label for UX; continuous scores for interpretability |
| Loss | 0.6 × CE + 0.4 × MSE | Prioritises correct class; continuous scores as auxiliary signal |
| Audio capture | Client-side only | Server never touches mic; no pyaudio/webrtcvad on server |

---

## Training Datasets

### WRIME — Japanese Text (Text Expert)
- ~17,000 Japanese social media posts annotated with Plutchik emotion intensities
- No manual download required — loads via HuggingFace `datasets`
- Covers text modality training end-to-end

### RAVDESS — English Emotional Speech (Audio Expert)
- 1,440 clips from 24 professional actors (12F / 12M)
- 8 emotions, 2 intensity levels, 16-bit WAV
- English speech is used to bootstrap the audio expert — acoustic emotion patterns (pitch, rhythm, energy) transfer across languages
- Download once: `python data/download_ravdess.py` (~1.6 GB from Zenodo)

### Emotion Label Mapping

Both datasets map to our 7-class emotion space:

| Our Label | From WRIME | From RAVDESS |
|---|---|---|
| happy | joy | happy (03) |
| sad | sadness | sad (04) |
| angry | anger | angry (05) |
| neutral | trust, anticipation | neutral (01), calm (02) |
| anxious | fear | fearful (06) |
| surprised | surprise | surprised (08) |
| disgusted | disgust | disgust (07) |

### Valence / Arousal Labels

Neither WRIME nor RAVDESS provides continuous V/A scores directly. They are approximated using Russell's circumplex model — each emotion class is assigned a base (valence, arousal) pair on the 1–9 scale, scaled by the sample's intensity. See `data/emotion_mappings.py`.

---

## Repository Structure

```
MoE/
├── configs/
│   ├── base.yaml              # Project-wide settings (paths, emotion classes, inference)
│   ├── model.yaml             # Architecture (encoders, router, fusion, heads)
│   └── train.yaml             # Training (epochs, LR, loss weights, datasets, augmentation)
│
├── models/
│   ├── text/
│   │   └── encoder.py         # JapaneseTextEncoder (BERT-based)
│   ├── audio/
│   │   └── encoder.py         # AudioEncoder (Wav2Vec2; accepts np.ndarray, no mic dependency)
│   ├── heads/
│   │   └── emotion_head.py    # Dual head: 7-class clf + valence/arousal regression
│   ├── fusion/
│   │   └── fusion.py          # WeightedFusion with learned gate
│   └── multimodal/
│       ├── router.py          # MoERouter — learned soft gating over experts
│       └── moe.py             # MoEEmotionModel — top-level model + predict() API
│
├── data/
│   ├── emotion_mappings.py    # Shared label maps + V/A approximations (single source of truth)
│   ├── wrime_dataset.py       # WRIME PyTorch Dataset (Japanese text)
│   ├── ravdess_dataset.py     # RAVDESS PyTorch Dataset (English audio)
│   ├── combined_dataset.py    # Unified DataLoader factory for both datasets
│   ├── download_ravdess.py    # One-time RAVDESS download script
│   └── raw/
│       └── ravdess/           # Extracted RAVDESS (Actor_01/ … Actor_24/) — not committed
│
├── utils/
│   └── deap_label_mapper.py   # V/A → discrete mapping utility (scale reference only)
│
├── training/                  # Training loop — next step
├── inference/                 # Inference pipeline — next step
├── api/                       # FastAPI endpoints (HTTP + WebSocket) — next step
├── frontend/                  # Web Audio API mic capture + WebSocket client — next step
├── notebooks/                 # EDA, label distribution checks, ablations
├── reporting/                 # Metrics, confusion matrices
├── personalization/           # Per-user calibration (future)
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and create environment

```bash
git clone <your-repo-url>
cd MoE
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Japanese tokenizer

```bash
python -c "import fugashi; print('MeCab OK')"
```

### 4. Download RAVDESS audio data

```bash
python data/download_ravdess.py
# Downloads ~1.6 GB to data/raw/ravdess/
# WRIME loads automatically from HuggingFace — no download needed
```

### 5. Train

```bash
python training/train.py --config configs/
```

---

## Usage

### File upload inference

```python
import yaml, torch, soundfile as sf
from models.multimodal.moe import MoEEmotionModel

with open("configs/model.yaml") as f:
    config = yaml.safe_load(f)

model = MoEEmotionModel(config)
model.load_state_dict(torch.load("outputs/checkpoints/best.pt"))

# Text only
result = model.predict(text="今日はとても嬉しい！")
print(result["predicted_label"])   # ["happy"]

# Audio file (decoded server-side from uploaded bytes)
waveform, sr = sf.read("audio.wav", dtype="float32")
result = model.predict(audio=waveform)

# Both modalities
result = model.predict(text="悲しいね", audio=waveform)
print(result["valence"].item())    # e.g. 2.4
print(result["arousal"].item())    # e.g. 3.1
```

### WebSocket stream inference

```python
# api/ws_stream.py (preview — full implementation in api/ step)
from fastapi import WebSocket
import numpy as np
from models.multimodal.moe import MoEEmotionModel

CHUNK_SAMPLES = 8000  # 500ms at 16kHz

async def stream_endpoint(websocket: WebSocket, model: MoEEmotionModel):
    await websocket.accept()
    while True:
        raw   = await websocket.receive_bytes()
        chunk = np.frombuffer(raw, dtype=np.float32)
        out   = model.predict_stream_chunk(chunk)
        await websocket.send_json({
            "emotion": out["predicted_label"][0],
            "valence": round(out["valence"].item(), 2),
            "arousal": round(out["arousal"].item(), 2),
            "probs":   out["probs"].tolist(),
        })
```

---

## Configuration

| Config | Key params |
|---|---|
| `base.yaml` | emotion classes, audio stream settings, paths |
| `model.yaml` | encoder model names, hidden sizes, freeze layers, fusion type |
| `train.yaml` | epochs, LR, loss weights, dataset paths, augmentation |

---

## Emotion Classes & V/A Approximations (Russell Circumplex)

| Emotion | Valence | Arousal | Notes |
|---|---|---|---|
| happy | 7.5 | 6.5 | High V, High A |
| sad | 2.5 | 3.0 | Low V, Low A |
| angry | 2.0 | 7.5 | Low V, Very High A |
| neutral | 5.0 | 4.0 | Mid V, Mid A |
| anxious | 3.0 | 7.0 | Low V, High A |
| surprised | 6.5 | 7.5 | High V, High A |
| disgusted | 2.0 | 5.0 | Low V, Mid A |

---

## Roadmap

- [x] Configs — `base.yaml`, `model.yaml`, `train.yaml`
- [x] Text encoder — Japanese BERT (`cl-tohoku/bert-base-japanese-v3`)
- [x] Audio encoder — Wav2Vec2, stream + file (deployment-ready, no mic dependency)
- [x] MoE Router — learned soft gating
- [x] Fusion layer — weighted gate
- [x] Dual emotion head — classification + regression
- [x] Emotion mappings — label maps + Russell circumplex V/A approximations
- [x] WRIME dataset — Japanese text emotion (HuggingFace)
- [x] RAVDESS dataset — English emotional speech audio (Zenodo)
- [x] Combined DataLoader — unified train/val/test with weighted sampling
- [x] RAVDESS download script
- [ ] Training loop — `training/train.py`
- [ ] Inference pipeline — `inference/pipeline.py`
- [ ] FastAPI endpoints — `api/` (HTTP file upload + WebSocket stream)
- [ ] Frontend — `frontend/` (Web Audio API + WebSocket client)
- [ ] Per-user calibration — `personalization/`

---

## Dependencies Summary

| Category | Key packages |
|---|---|
| Deep learning | `torch`, `torchaudio`, `transformers`, `accelerate` |
| Japanese NLP | `fugashi`, `ipadic`, `sentencepiece` |
| Audio (server) | `librosa`, `soundfile` — decode + resample only, no mic capture |
| API | `fastapi`, `uvicorn`, `websockets`, `pydantic` |
| Training | `datasets`, `torchmetrics`, `tensorboard`, `wandb` |
| Utilities | `pyyaml`, `omegaconf`, `tqdm`, `rich` |

> **Not included:** `pyaudio`, `webrtcvad` — mic capture is handled client-side.

---

## License

MIT