# Results — curated (report artifact)

> Machine log: `results.csv` (one row per run). Curated tables below.
> Test sets: **A** = v3.2-test hin+eng subset (1,200) · **B** = TTS Hinglish
> (33-clip pilot) · **C** = MUCS real speech (pending P2). Dev split pending P2.

## Thesis table v1 (P1 baselines, 2026-08-29)

| Model | test-A acc | test-A F1 | test-B acc | test-B F1 | test-C | p50 ms (CPU) |
|---|---|---|---|---|---|---|
| Silence-threshold (T=0.05 s*) | 0.537 | 0.560 | 0.485 | 0.622 | — | ~0 |
| Smart Turn v3.2 int8 (zero-shot) | **0.927** | **0.930** | 0.788 | 0.759 | — | 15 |

\* best F1 on the T sweep; acc-best T=0.20 s → test-A acc 0.578 / F1 0.534. Near-chance either way.

**Thesis confirmed**: strong in-distribution (A), 14-point acc drop on TTS
Hinglish (B pilot); trailing silence alone is uninformative (tail_sil median
0.18 s complete vs 0.06 s incomplete — distributions overlap heavily).

## Zero-shot slices (test-A, n=1,200)

| Slice | n | acc | F1 |
|---|---|---|---|
| eng | 600 | 0.955 | 0.956 |
| hin | 600 | 0.900 | 0.906 |
| midfiller=False | 729 | 0.973 | 0.973 |
| midfiller=True | 471 | 0.858 | 0.868 |
| endfiller=False | 942 | 0.921 | 0.940 |
| endfiller=True (all incomplete) | 258 | 0.950 | n/a¹ |

¹ Single-class slice — positive-class F1 undefined; accuracy is the meaningful number there.

**Observations**
- Hindi is the weak language even in-distribution (−5.5 acc vs eng) — code-mixing
  and Hindi prosody matter; our Hinglish data targets exactly this.
- Midfiller clips cost 11.5 acc points — pause-vs-end disambiguation is the hard core.
- test-B pilot (33 clips) weak as expected; real test-B (~600) arrives in P2.

## Reproduce

```powershell
uv run python scripts/eval_baselines.py --splits test_a test_b   # per-clip preds -> results/
```

## Data appendix (dataset card, in progress)

| Split | Composition | Status |
|---|---|---|
| train_pool (pre-dev) | 8,800 FLAC / 1.10 GB — hin 6000, eng 2000, mar 400, ben 400 (50:50 labels) | ✅ |
| dev_v1 | 881 clips = 10% of train_pool, stratified lang × label-folder (`data/splits/dev_v1.csv`) — also int8 calibration | ✅ |
| test-A | 1,200 FLAC / 167 MB — hin 600, eng 600 | ✅ |
| test-B | TTS Hinglish held-out scripts, ~600 target | ⏳ after listening gate |
| TTS train | target 3–4k | ⏳ **pilot 100 ✅ — listening gate pending** |
| test-C | MUCS 2021 subset 1–2 h (`dianavdavidson/MUCS-Hinglish`, 1 test parquet) | ⏳ scout done |

TTS pilot: 25 script pairs × 4 voices (Swara/Madhur hi-IN, Neerja/Prabhat en-IN),
rate +8%, FLAC 16 kHz mono, +250 ms tail, 0 failures, 96/100 ≤ 8 s.
Disk: 1.30 GB / 2 GB cap.
