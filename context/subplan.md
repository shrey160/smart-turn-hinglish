# Subplan — Phase 4 (P3: Two-stage reproduction)

> Mutable task breakdown for the CURRENT phase only. Previous phases:
> phase1/P0 ✅, phase2/P1 ✅, phase3/P2 ✅ (closed per pivot: TTS scale-up,
> test-B 600, listening gate cancelled; test-C build carried into this phase).

## SP1 — turn_v2 skeleton
- [ ] `turn_v2/data/dataset.py`: folder-layout loading, label/filler parsing,
      8 s contract via `smart_turn_reference/audio_utils`, dev-manifest support
      (`data/splits/dev_v1.csv`), replay-sampler support for s2
- [ ] `turn_v2/data/augment.py`: telephony bandpass / noise 10–20 dB / speed
      0.9–1.1×, p≈0.5, applied in `__getitem__`
- [ ] `turn_v2/models/model.py` + `pooling.py`: whisper-tiny frozen encoder +
      attention-mean head
- [ ] `turn_v2/train.py`: `--stage s1|s2`, `--init-ckpt`, `--overfit 100`,
      batch pos_weight BCE, cosine schedule, seeds 42, results.csv logging
      (schema v2: stage + init_ckpt columns; backfill p1 rows as baseline)
- [ ] `turn_v2/evaluate.py`: dev + test-A/B(/C), acc/F1 + per-lang/filler slices

## SP2 — Sanity gate
- [ ] `--overfit 100`: loss → ~0 on 100 samples (blocks all full runs)

## SP3 — Stage 1 run
- [ ] Train s1: hin+eng train_pool minus dev_v1 (~7,200), lr 5e-5, 4 ep,
      frozen encoder → `ckpt_s1` (select on dev_v1 hin+eng F1)
- [ ] Eval ckpt_s1 on dev + test-A + test-B → results.csv row `s1-001`

## SP4 — Stage 2 run
- [ ] Stats-QC admission of 100 pilot clips (labels/durations/16 kHz mono)
- [ ] TTS dev carve: ~23 clips (10%, seed 42) → `data/splits/dev_tts_v1.csv`
- [ ] Train s2: init `ckpt_s1`, 232 TTS ×2 upsample + 50:50 hin/eng replay
      (fresh seeded draw each epoch), lr 1e-5, 2 ep → `ckpt_s2`
- [ ] Eval ckpt_s2 on dev + A/B → thesis table v2 in `turn_v2/results.md`
      (rows: zero-shot vs s1 vs s2)

## SP5 — test-C build (parallel workstream)
- [ ] Download 1 MUCS test parquet (~295 MB) → extract 1–2 h utterance subset
      → delete parquet
- [ ] Incomplete clips via torchaudio forced alignment; fallback = complete-heavy
- [ ] Eval ckpt_s1 + ckpt_s2 on test-C → add testC columns to s1/s2 rows
