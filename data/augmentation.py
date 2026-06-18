"""
data/augmentation.py

Audio augmentation transforms for the RAVDESS training set.

All functions operate on a 1-D numpy float32 waveform at 16kHz and return
the same shape. They are designed to be cheap on CPU — no GPU dependency.

Usage:
    from data.augmentation import augment_audio
    waveform = augment_audio(waveform, sample_rate=16000, p=0.5)
"""

import random
import numpy as np

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False


def _add_noise(waveform: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """Add white Gaussian noise at the given signal-to-noise ratio."""
    signal_rms = np.sqrt(np.mean(waveform ** 2)) + 1e-9
    noise_rms  = signal_rms / (10 ** (snr_db / 20.0))
    noise      = np.random.randn(len(waveform)).astype(np.float32) * noise_rms
    return np.clip(waveform + noise, -1.0, 1.0)


def _time_stretch(waveform: np.ndarray, rate: float) -> np.ndarray:
    """Stretch/compress time without changing pitch (requires librosa)."""
    if not _LIBROSA_OK:
        return waveform
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    return stretched.astype(np.float32)


def _pitch_shift(waveform: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch up/down by n_steps semitones (requires librosa)."""
    if not _LIBROSA_OK:
        return waveform
    shifted = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
    return shifted.astype(np.float32)


def _random_gain(waveform: np.ndarray, low: float = 0.75, high: float = 1.25) -> np.ndarray:
    gain = random.uniform(low, high)
    return np.clip(waveform * gain, -1.0, 1.0)


def augment_audio(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    p: float = 0.5,
) -> np.ndarray:
    """
    Apply a random subset of augmentations to a waveform.

    Each individual transform fires independently with probability p.
    At most one of {time_stretch, pitch_shift} is applied per sample
    (they are mutually exclusive to keep runtime predictable).

    Args:
        waveform:    float32 numpy array, shape (n_samples,)
        sample_rate: sample rate of the waveform (Hz)
        p:           probability each transform is applied

    Returns:
        Augmented waveform, same shape and dtype as input.
    """
    out = waveform.copy()

    # Additive noise
    if random.random() < p:
        snr = random.uniform(20.0, 40.0)   # 20–40 dB SNR
        out = _add_noise(out, snr_db=snr)

    # Random gain
    if random.random() < p:
        out = _random_gain(out)

    # Time stretch XOR pitch shift (mutually exclusive for speed)
    if _LIBROSA_OK:
        roll = random.random()
        if roll < p / 2:
            rate = random.uniform(0.85, 1.15)
            out  = _time_stretch(out, rate=rate)
        elif roll < p:
            steps = random.uniform(-2.0, 2.0)
            out   = _pitch_shift(out, sr=sample_rate, n_steps=steps)

    return out
