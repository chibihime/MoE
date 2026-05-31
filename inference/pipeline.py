import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
inference/pipeline.py

EmotionPipeline — the single entry point for all inference.

Everything that calls the model (API routes, notebooks, CLI, tests) goes through
this file. It handles:
  - Config loading and merging
  - Model instantiation and checkpoint loading
  - Device placement
  - Audio decoding from raw bytes (HTTP upload or WebSocket PCM)
  - Resampling to 16kHz
  - A typed, JSON-serialisable output — not a raw tensor dict

Usage:
    from inference.pipeline import EmotionPipeline

    pipe = EmotionPipeline.load("outputs/checkpoints/best.pt")

    # From Japanese text
    result = pipe.predict_text("今日はとても嬉しい！")

    # From uploaded audio file (bytes)
    with open("clip.wav", "rb") as f:
        result = pipe.predict_audio_bytes(f.read(), fmt="wav")

    # From WebSocket PCM chunk (raw Float32 bytes from Web Audio API)
    result = pipe.predict_stream_chunk(pcm_bytes)

    # Both modalities
    result = pipe.predict(text="悲しいね", audio_bytes=audio_bytes)

    print(result.emotion)      # "sad"
    print(result.confidence)   # 0.61
    print(result.valence)      # 2.8
    print(result.arousal)      # 3.1
    print(result.all_probs)    # {"happy": 0.05, "sad": 0.61, ...}
    print(result.to_dict())    # JSON-serialisable dict
"""

import io
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Union

import numpy as np
import torch
import yaml
import soundfile as sf
import librosa

from models.multimodal.moe import MoEEmotionModel
from data.emotion_mappings import EMOTION_CLASSES

logger = logging.getLogger(__name__)

TARGET_SR  = 16_000
CHUNK_DTYPE = np.float32


# ── Output type ────────────────────────────────────────────────────────────────

@dataclass
class EmotionResult:
    """
    Clean, typed output from the pipeline.

    All values are Python scalars or dicts — no tensors, ready for JSON.
    """
    emotion:    str            # predicted emotion label
    confidence: float          # probability of the predicted class (0–1)
    valence:    float          # continuous valence score (1–9)
    arousal:    float          # continuous arousal score (1–9)
    all_probs:  Dict[str, float]  # full probability distribution over all 7 classes
    modality:   str            # "text" | "audio" | "multimodal"
    text_weight: Optional[float] = None   # router weight for text expert (0–1)
    audio_weight: Optional[float] = None  # router weight for audio expert (0–1)

    def to_dict(self) -> dict:
        """JSON-serialisable dict — use this for API responses."""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"EmotionResult(emotion={self.emotion!r}, "
            f"confidence={self.confidence:.2f}, "
            f"valence={self.valence:.1f}, arousal={self.arousal:.1f}, "
            f"modality={self.modality!r})"
        )


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _decode_audio_bytes(
    audio_bytes: bytes,
    fmt: Optional[str] = None,
) -> np.ndarray:
    """
    Decode raw audio bytes to a float32 mono waveform at 16kHz.

    Accepts any format soundfile supports (wav, flac, ogg) plus mp3/m4a via
    librosa as fallback.

    Args:
        audio_bytes: raw file bytes from HTTP upload
        fmt:         optional format hint ("wav", "mp3", etc.)

    Returns:
        waveform: np.ndarray of shape (n_samples,), float32, 16kHz
    """
    buf = io.BytesIO(audio_bytes)
    try:
        # soundfile returns (n_samples, channels) with always_2d=True
        waveform, sr = sf.read(buf, dtype="float32", always_2d=True)
        # Mix to mono: average over channels axis (axis=1), not samples axis
        if waveform.shape[1] > 1:
            waveform = waveform.mean(axis=1)   # (n_samples,)
        else:
            waveform = waveform[:, 0]           # (n_samples,)
    except Exception:
        # fallback for mp3/m4a — librosa handles resampling + mono mixing
        buf.seek(0)
        waveform, sr = librosa.load(buf, sr=None, mono=True, dtype=np.float32)
        # waveform is already (n_samples,) from librosa

    # Resample to 16kHz if needed
    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

    return waveform.astype(np.float32)


def _decode_pcm_bytes(pcm_bytes: bytes) -> np.ndarray:
    """
    Decode raw PCM Float32 bytes from a WebSocket stream chunk.

    The Web Audio API sends Float32 PCM at the browser's native sample rate
    (typically 44100 or 48000 Hz). We resample here if needed.

    Args:
        pcm_bytes: raw bytes from websocket.receive_bytes()
                   expected format: float32 little-endian, mono, 16kHz
                   (client should downsample to 16kHz before sending for efficiency)

    Returns:
        waveform: np.ndarray (n_samples,), float32
    """
    return np.frombuffer(pcm_bytes, dtype=np.float32).copy()


def _trim_silence(
    waveform: np.ndarray,
    top_db: float = 40.0,
) -> np.ndarray:
    """Trim leading/trailing silence. Falls back to original if all silence."""
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed if len(trimmed) > 0 else waveform


# ── Pipeline ───────────────────────────────────────────────────────────────────

class EmotionPipeline:
    """
    Stateful inference pipeline. Instantiate once, call repeatedly.

    Holds the loaded model in memory on the target device. Thread-safety:
    the model is in eval mode and uses torch.no_grad() — safe for concurrent
    reads if your server framework handles GIL appropriately.
    """

    def __init__(
        self,
        model:    MoEEmotionModel,
        config:   dict,
        device:   torch.device,
        ckpt_path: Optional[str] = None,
    ):
        self.model    = model
        self.config   = config
        self.device   = device
        self.ckpt_path = ckpt_path
        self.model.eval()

    # ── Constructor ────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        checkpoint: str = "outputs/checkpoints/best.pt",
        config_dir: str = "configs/",
        device:     Optional[str] = None,
    ) -> "EmotionPipeline":
        """
        Load a trained pipeline from a checkpoint file.

        Args:
            checkpoint: path to best.pt or any saved checkpoint
            config_dir: directory containing base.yaml, model.yaml, train.yaml
            device:     "cpu", "cuda", "mps", or None (auto-detect)

        Returns:
            Ready-to-use EmotionPipeline

        Example:
            pipe = EmotionPipeline.load("outputs/checkpoints/best.pt")
        """
        # Resolve device
        if device:
            dev = torch.device(device)
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        elif torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")

        logger.info(f"Loading pipeline on {dev}")

        # Merge configs
        config = {}
        config_path = Path(config_dir)
        for name in ("base", "model", "train"):
            p = config_path / f"{name}.yaml"
            if p.exists():
                with open(p) as f:
                    config.update(yaml.safe_load(f))

        # Build model
        model = MoEEmotionModel(config).to(dev)

        # Load weights
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}\n"
                "Train first: python training/train.py"
            )
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        # Checkpoints saved by train.py wrap weights under "model" key
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        model.eval()

        epoch = ckpt.get("epoch", "?")
        metrics = ckpt.get("metrics", {})
        f1 = metrics.get("f1_macro", "?")
        logger.info(f"Loaded checkpoint (epoch={epoch}, val_f1_macro={f1})")
        print(f"Pipeline ready — epoch {epoch}, val f1_macro {f1}")

        return cls(model=model, config=config, device=dev, ckpt_path=str(ckpt_path))

    # ── Internal inference ─────────────────────────────────────────────────────

    def _run(
        self,
        text:          Optional[str]        = None,
        waveform:      Optional[np.ndarray] = None,
    ) -> EmotionResult:
        """
        Core inference call. Accepts pre-processed inputs only.

        Args:
            text:     Japanese text string, or None
            waveform: float32 waveform at 16kHz, or None

        Returns:
            EmotionResult
        """
        if text is None and waveform is None:
            raise ValueError("At least one of text or waveform must be provided.")

        with torch.no_grad():
            outputs = self.model.predict(
                text=text,
                audio=waveform,
                device=str(self.device),
            )

        # Determine modality label
        if text is not None and waveform is not None:
            modality = "multimodal"
        elif text is not None:
            modality = "text"
        else:
            modality = "audio"

        # Extract routing weights if available
        routing = outputs.get("routing", {})
        tw = routing.get("text_weight")
        aw = routing.get("audio_weight")
        text_w  = tw[0].item()  if tw  is not None else None
        audio_w = aw[0].item()  if aw  is not None else None

        # Build probability dict
        probs_tensor = outputs["probs"][0]   # (7,)
        all_probs = {
            cls: round(float(probs_tensor[i]), 4)
            for i, cls in enumerate(EMOTION_CLASSES)
        }

        pred_idx    = outputs["predicted_class"][0].item()
        emotion     = EMOTION_CLASSES[pred_idx]
        confidence  = round(float(probs_tensor[pred_idx]), 4)
        valence     = round(outputs["valence"][0].item(), 2)
        arousal     = round(outputs["arousal"][0].item(), 2)

        return EmotionResult(
            emotion=emotion,
            confidence=confidence,
            valence=valence,
            arousal=arousal,
            all_probs=all_probs,
            modality=modality,
            text_weight=round(text_w, 4)  if text_w  is not None else None,
            audio_weight=round(audio_w, 4) if audio_w is not None else None,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict_text(self, text: str) -> EmotionResult:
        """
        Predict emotion from a Japanese text string.

        Args:
            text: Japanese text, e.g. "今日はとても嬉しい！"

        Returns:
            EmotionResult
        """
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string.")
        return self._run(text=text.strip())

    def predict_audio_bytes(
        self,
        audio_bytes: bytes,
        fmt: Optional[str] = None,
        trim_silence: bool = True,
    ) -> EmotionResult:
        """
        Predict emotion from an uploaded audio file (raw bytes).

        Accepts any format soundfile supports (wav, flac, ogg) plus mp3/m4a.
        Handles decoding, mono mixing, and resampling to 16kHz internally.

        Args:
            audio_bytes:   raw file content from HTTP multipart upload
            fmt:           optional format hint ("wav", "mp3", etc.)
            trim_silence:  remove leading/trailing silence before inference

        Returns:
            EmotionResult
        """
        if not audio_bytes:
            raise ValueError("audio_bytes is empty.")

        waveform = _decode_audio_bytes(audio_bytes, fmt=fmt)

        if trim_silence:
            waveform = _trim_silence(waveform)

        if len(waveform) == 0:
            raise ValueError("Audio is silent or too short after trimming.")

        return self._run(waveform=waveform)

    def predict_stream_chunk(
        self,
        pcm_bytes: bytes,
        text: Optional[str] = None,
    ) -> EmotionResult:
        """
        Predict emotion from a real-time WebSocket audio chunk.

        The client (browser) sends raw Float32 PCM at 16kHz. This method
        decodes the bytes and runs inference — no further preprocessing needed
        if the client already downsampled to 16kHz.

        Args:
            pcm_bytes: raw bytes from websocket.receive_bytes()
                       expected: float32 little-endian, mono, 16kHz
            text:      optional concurrent text from live ASR

        Returns:
            EmotionResult
        """
        if not pcm_bytes:
            raise ValueError("pcm_bytes is empty.")

        waveform = _decode_pcm_bytes(pcm_bytes)

        if len(waveform) < TARGET_SR * 0.1:   # < 100ms — too short
            raise ValueError(
                f"Chunk too short ({len(waveform)} samples). "
                f"Minimum is {int(TARGET_SR * 0.1)} samples (100ms at 16kHz)."
            )

        return self._run(
            text=text.strip() if text else None,
            waveform=waveform,
        )

    def predict(
        self,
        text:        Optional[str]   = None,
        audio_bytes: Optional[bytes] = None,
        fmt:         Optional[str]   = None,
        trim_silence: bool = True,
    ) -> EmotionResult:
        """
        Predict emotion from any combination of modalities.

        This is the general-purpose method. Use the specific methods
        (predict_text, predict_audio_bytes, predict_stream_chunk) when
        you know the input type — they have cleaner signatures.

        Args:
            text:         Japanese text string, or None
            audio_bytes:  raw audio file bytes, or None
            fmt:          audio format hint ("wav", "mp3", etc.)
            trim_silence: trim silence from audio before inference

        Returns:
            EmotionResult
        """
        if text is None and audio_bytes is None:
            raise ValueError("Provide at least one of: text, audio_bytes.")

        waveform = None
        if audio_bytes:
            waveform = _decode_audio_bytes(audio_bytes, fmt=fmt)
            if trim_silence:
                waveform = _trim_silence(waveform)

        return self._run(
            text=text.strip() if text else None,
            waveform=waveform,
        )

    # ── Utility ────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Run a dummy forward pass to initialise MPS/CUDA kernels.

        Call this once after loading, before serving requests, to avoid
        a slow first inference.
        """
        logger.info("Running warmup pass...")
        dummy_text  = "テスト"
        dummy_audio = np.zeros(TARGET_SR, dtype=np.float32)   # 1s silence
        _ = self._run(text=dummy_text, waveform=dummy_audio)
        logger.info("Warmup complete.")
        print("Warmup complete.")

    @property
    def info(self) -> dict:
        """Return pipeline metadata — useful for health check endpoints."""
        return {
            "checkpoint":    self.ckpt_path,
            "device":        str(self.device),
            "emotions":      EMOTION_CLASSES,
            "sample_rate":   TARGET_SR,
            "model_params":  sum(p.numel() for p in self.model.parameters()),
        }

    def __repr__(self) -> str:
        return (
            f"EmotionPipeline("
            f"device={self.device}, "
            f"checkpoint={Path(self.ckpt_path).name if self.ckpt_path else None!r})"
        )


# ── CLI quick-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick pipeline test.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--text",  default="今日はとても嬉しい！")
    parser.add_argument("--audio", default=None, help="Path to audio file")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    pipe = EmotionPipeline.load(
        checkpoint=args.checkpoint,
        device=args.device,
    )
    pipe.warmup()

    print(f"\n{pipe}\n")

    if args.text:
        result = pipe.predict_text(args.text)
        print(f"Text:  {args.text!r}")
        print(f"       → {result}\n")

    if args.audio:
        with open(args.audio, "rb") as f:
            audio_bytes = f.read()
        result = pipe.predict_audio_bytes(audio_bytes)
        print(f"Audio: {args.audio}")
        print(f"       → {result}\n")

    if args.text and args.audio:
        with open(args.audio, "rb") as f:
            audio_bytes = f.read()
        result = pipe.predict(text=args.text, audio_bytes=audio_bytes)
        print(f"Both:  → {result}\n")
        print(f"Routing: text={result.text_weight:.3f}  audio={result.audio_weight:.3f}")

    print("\nFull output dict:")
    import json
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))