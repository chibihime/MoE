"""
data/emotion_mappings.py

Single source of truth for all emotion label mappings across datasets.

Our 7-class emotion space:
    0: happy
    1: sad
    2: angry
    3: neutral
    4: anxious
    5: surprised
    6: disgusted

Valence/Arousal approximations use Russell's circumplex model (1980).
These are used to supervise the regression head when a dataset only
provides discrete labels (WRIME, RAVDESS) and not continuous V/A scores.
"""

from typing import Dict, Tuple

# ── Core label space ──────────────────────────────────────────────────────────

EMOTION_CLASSES = ["happy", "sad", "angry", "neutral", "anxious", "surprised", "disgusted"]
EMOTION_TO_IDX:  Dict[str, int] = {e: i for i, e in enumerate(EMOTION_CLASSES)}
NUM_CLASSES = len(EMOTION_CLASSES)

# ── Approximate V/A values per emotion (Russell circumplex, 1–9 scale) ────────
# These are used as regression targets when only discrete labels are available.
# Values are midpoints; intensity modifiers are applied per-dataset below.

EMOTION_TO_VA: Dict[str, Tuple[float, float]] = {
    #           valence  arousal
    "happy":    (7.5,    6.5),
    "sad":      (2.5,    3.0),
    "angry":    (2.0,    7.5),
    "neutral":  (5.0,    4.0),
    "anxious":  (3.0,    7.0),
    "surprised":(6.5,    7.5),
    "disgusted":(2.0,    5.0),
}

def emotion_to_va(emotion: str, intensity: float = 1.0) -> Tuple[float, float]:
    """
    Return (valence, arousal) for an emotion label, scaled by intensity.

    Args:
        emotion:   one of EMOTION_CLASSES
        intensity: 0.0–1.0 scale factor; moves scores toward neutral (5, 4)
                   at 0.0, away from neutral at 1.0. Default 1.0 (no scaling).

    Returns:
        (valence, arousal) both in [1, 9]
    """
    v, a = EMOTION_TO_VA[emotion]
    # Interpolate between neutral and full emotion value
    v = 5.0 + (v - 5.0) * intensity
    a = 4.0 + (a - 4.0) * intensity
    return round(v, 2), round(a, 2)


# ── WRIME → our 7 classes ─────────────────────────────────────────────────────
# WRIME uses Plutchik's 8 emotions with intensity 0–3 per label.
# We map each Plutchik label to the closest of our 7 classes.
# 'anticipation' and 'trust' both map to neutral (calm expectation / acceptance).

WRIME_TO_EMOTION: Dict[str, str] = {
    "joy":          "happy",
    "sadness":      "sad",
    "anger":        "angry",
    "trust":        "neutral",
    "anticipation": "neutral",
    "fear":         "anxious",
    "surprise":     "surprised",
    "disgust":      "disgusted",
}

# WRIME intensity: 0 = none, 1 = weak, 2 = medium, 3 = strong → normalise to [0, 1]
WRIME_INTENSITY_SCALE = 3.0


# ── RAVDESS → our 7 classes ───────────────────────────────────────────────────
# RAVDESS filename emotion codes (3rd field in filename, 2-digit int).
# 'calm' is treated as a subdued form of neutral.

RAVDESS_CODE_TO_EMOTION: Dict[int, str] = {
    1: "neutral",
    2: "neutral",    # calm → neutral
    3: "happy",
    4: "sad",
    5: "angry",
    6: "anxious",    # fearful → anxious
    7: "disgusted",
    8: "surprised",
}

# RAVDESS intensity codes: 1 = normal, 2 = strong → normalise to [0.6, 1.0]
RAVDESS_INTENSITY_MAP: Dict[int, float] = {
    1: 0.6,
    2: 1.0,
}