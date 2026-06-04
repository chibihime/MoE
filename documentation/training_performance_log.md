# Training Performance Log

All results on the same held-out test set (6,660 samples: 6,480 WRIME + 180 RAVDESS).
Macro F1 is the primary metric — it weights all 7 classes equally.

---

## Summary table

| Run | Change made | Macro F1 | Accuracy | Notes |
|-----|-------------|----------|----------|-------|
| Baseline | CrossEntropy, 20 epochs, no class weights | 0.236 | 0.443 | happy/sad dominated; 5 classes F1 < 0.10 |
| +Class weights | Inverse-frequency weighting on loss | 0.266 | 0.333 | minority classes recover; angry 0.09→0.23 |
| +More epochs | Extended to 40 epochs, resumed from checkpoint | 0.295 | 0.360 | all classes improved; early stop at ep 35 |
| +Fresh retrain | Retrained from scratch for 40 epochs | 0.303 | 0.363 | best result; label smoothing 0.1 active |
| Phase 2 attempt 1 | Unfroze 2 layers, 25 steps, LR 4e-6 | 0.166 | 0.189 | catastrophic forgetting; abandoned |
| Phase 2 attempt 2 | Unfroze 1 layer, 75 steps, LR 5e-7 | 0.166 | 0.216 | same regression pattern; abandoned |
| Focal loss | Replaced CrossEntropy with focal gamma=2 | 0.288 | 0.347 | worse than CE; double-penalised minority |
| **Final** | **CrossEntropy + class weights + smoothing** | **0.303** | **0.363** | **current best.pt** |

---

## Per-class F1 progression

| Class | Baseline | +Weights | +Epochs | Final |
|-------|----------|----------|---------|-------|
| happy | 0.581 | 0.526 | 0.542 | 0.526 |
| sad | 0.490 | 0.314 | 0.332 | 0.359 |
| angry | 0.092 | 0.225 | 0.254 | 0.284 |
| neutral | 0.305 | 0.289 | 0.340 | 0.345 |
| anxious | 0.067 | 0.176 | 0.203 | 0.195 |
| surprised | 0.057 | 0.210 | 0.252 | 0.269 |
| disgusted | 0.059 | 0.122 | 0.144 | 0.141 |

---

## Dataset decisions

| Decision | Replaced | Reason |
|----------|----------|--------|
| WRIME (Japanese text) | DEAP | DEAP contains EEG signals, not text or audio |
| RAVDESS (English audio) | DEAP | DEAP has no speech recordings |
| Cached encoder features | Raw forward pass | 79M param forward pass = 8+ min/epoch on MPS |
| WeightedRandomSampler | No sampling | WRIME has 7x class imbalance (happy vs angry) |

---

## What did not work and why

**Phase 2 fine-tuning:** WeightedRandomSampler balances classes to ~4,400 each, but
phase 1 was trained on imbalanced data. Unfreezing encoders destabilises the learned
distributions immediately — val F1 drops to ~0.10 at epoch 1 and never recovers.

**Focal loss:** Class weights + focal loss double-penalise minority examples. The
combined signal is too aggressive; training destabilises and early stops 6 epochs
earlier than CrossEntropy.

---

## Known ceilings and future fixes

| Ceiling | Future fix |
|---------|------------|
| Frozen encoders; 1.5M trainable params | Phase 2 with non-balanced dataloader |
| RAVDESS is English, not Japanese | Replace with Japanese speech corpus |
| No paired text+audio training samples | Collect or synthesise paired Japanese data |
| Neutral = trust + anticipation (noisy) | Separate neutral mapping in emotion_mappings.py |