# Phase 4 — P3: Two-stage reproduction

> Maps to masterplan §5 P3 (rewritten 2026-08-29, two-stage pivot). Progress
> trace: completed phases stay in `context/phases/`.

## Objective

Build the `turn_v2/` training pipeline and run the two-stage transfer recipe
(masterplan §4): Stage 1 pretrain on hin+eng, Stage 2 finetune on the existing
232 TTS Hinglish clips + 50:50 replay. Produce thesis table v2 (s1, s2 rows vs
the zero-shot baseline).

## Tasks

1. `turn_v2/` skeleton: `data/dataset.py` (folder-layout loading, 8 s contract
   imported from `smart_turn_reference/audio_utils`, dev-manifest support), `data/augment.py`
   (telephony / noise 10–20 dB / speed 0.9–1.1×, p≈0.5, on-the-fly),
   `models/model.py` + `models/pooling.py` (whisper-tiny frozen encoder +
   attention-mean head), `train.py` (`--stage s1|s2`, `--init-ckpt`,
   `--overfit 100`, seeds 42), `evaluate.py` (dev + A/B/C, per-lang/filler slices).
2. `--overfit 100` sanity gate — loss → ~0 before any full run.
3. Pilot stats-QC admission (labels 50:50, durations, 16 kHz mono — listening
   gate waived by user) + TTS dev carve (~23 clips, 10%, seed 42).
4. Stage 1 run: hin+eng train_pool minus dev_v1 (~7,200), lr 5e-5, 4 ep →
   `ckpt_s1`; eval dev + test-A/B.
5. Stage 2 run: init `ckpt_s1`, 232 TTS ×2 upsample + 50:50 hin/eng replay,
   lr 1e-5, 2 ep → `ckpt_s2`; eval dev + A/B(/C); thesis table v2 in
   `turn_v2/results.md`.
6. test-C build (parallel workstream): 1 MUCS test parquet → 1–2 h utterance
   subset → delete parquet; forced-alignment cuts or complete-heavy fallback.

## Exit criteria (masterplan P3)

- [ ] `ckpt_s2` beats Smart Turn zero-shot on test-B (0.788 acc / 0.759 F1)
- [ ] test-A regression of `ckpt_s2` vs `ckpt_s1` ≤ ~1–2%
- [ ] s1 + s2 rows in `turn_v2/results.csv` (schema v2: stage + init_ckpt),
      curated in `results.md`
- [ ] test-C built (or complete-heavy fallback invoked) and both ckpts evaluated

## Constraints

- One change per row; seeds 42; 8 s contract imported from `smart_turn_reference/` — never re-implemented
- No new TTS generation (pivot); disk ≤ 2 GB cap
- mar/ben excluded from all training (user decision)
