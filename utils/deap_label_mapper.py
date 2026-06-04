"""
utils/deap_label_mapper.py

Maps DEAP continuous valence/arousal scores to discrete emotion labels.

DEAP scores are on a 1–9 scale. We split at the midpoint (5.0) to assign
quadrants, then map each quadrant to one or more of our 7 emotion classes.

Quadrant map:
  High V + High A → happy, surprised
  High V + Low A  → neutral
  Low V  + High A → angry, anxious
  Low V  + Low A  → sad, disgusted

Within each quadrant, finer-grained assignment uses the distance from
the quadrant boundaries (e.g., very high arousal in low-V quadrant → angry
over anxious).
"""

import numpy as np
from typing import Union, List


EMOTION_CLASSES = ["happy", "sad", "angry", "neutral", "anxious", "surprised", "disgusted"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_CLASSES)}
MIDPOINT = 5.0


def valence_arousal_to_label(
    valence: float,
    arousal: float,
    midpoint: float = MIDPOINT
) -> str:
    """
    Map a single (valence, arousal) pair to a discrete emotion label.

    Args:
        valence: DEAP valence score, 1–9
        arousal: DEAP arousal score, 1–9
        midpoint: threshold separating high/low (default 5.0)

    Returns:
        Emotion label string.
    """
    high_v = valence >= midpoint
    high_a = arousal >= midpoint

    if high_v and high_a:
        # Distinguish happy vs surprised by arousal intensity
        # Higher arousal → more surprised/excited
        return "surprised" if arousal >= 7.0 else "happy"

    elif high_v and not high_a:
        return "neutral"

    elif not high_v and high_a:
        # Distinguish angry vs anxious by valence depth
        # Very low valence + high arousal → angry; moderate → anxious
        return "angry" if valence <= 3.0 else "anxious"

    else:  # not high_v and not high_a
        # Distinguish sad vs disgusted by arousal depth
        # Very low arousal → sad; moderate-low → disgusted
        return "sad" if arousal <= 3.0 else "disgusted"


def batch_map(
    valences: Union[np.ndarray, List[float]],
    arousals: Union[np.ndarray, List[float]],
    midpoint: float = MIDPOINT
) -> np.ndarray:
    """
    Map arrays of valence/arousal scores to integer class indices.

    Args:
        valences: array of shape (N,)
        arousals: array of shape (N,)
        midpoint: quadrant threshold

    Returns:
        Integer label array of shape (N,)
    """
    valences = np.asarray(valences, dtype=float)
    arousals = np.asarray(arousals, dtype=float)
    assert valences.shape == arousals.shape, "valences and arousals must have same shape"

    labels = np.array([
        EMOTION_TO_IDX[valence_arousal_to_label(v, a, midpoint)]
        for v, a in zip(valences, arousals)
    ], dtype=np.int64)

    return labels


def label_distribution(labels: np.ndarray) -> dict:
    """Return class counts for a label array — useful for imbalance checks."""
    counts = {}
    for idx, name in enumerate(EMOTION_CLASSES):
        counts[name] = int((labels == idx).sum())
    return counts


if __name__ == "__main__":
    # Quick sanity check
    test_cases = [
        (8.0, 8.0, "happy or surprised"),
        (8.0, 8.5, "surprised"),
        (7.0, 3.0, "neutral"),
        (2.0, 8.0, "angry"),
        (3.5, 7.0, "anxious"),
        (2.0, 2.0, "sad"),
        (3.0, 4.0, "disgusted"),
    ]
    print("DEAP Label Mapper — sanity check\n")
    for v, a, expected in test_cases:
        label = valence_arousal_to_label(v, a)
        status = "✓" if expected.startswith(label) or label in expected else "?"
        print(f"  V={v}, A={a} → {label:12s}  (expected: {expected}) {status}")