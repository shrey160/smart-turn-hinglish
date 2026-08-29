# Phase 2 — P1: Baselines (the thesis table v1)

> Maps to masterplan §5 P1 (playbook Day 2). Progress trace: completed phases
> stay in `context/phases/`.

## Objective

Produce the thesis table: baseline × test-set → acc/F1, establishing that
in-distribution performance is strong while target-domain (TTS Hinglish)
performance is weak — the gap our `turn_v2` model must close.

## Tasks

1. `scripts/eval_baselines.py` — batch evaluator reusing the read-only
   `smart-turn/` inference path and the shared 8 s contract
   (`truncate_audio_to_last_n_seconds`); per-clip predictions saved to
   `results/` (gitignored, regenerable).
2. Smart Turn v3.2 (int8) zero-shot on test-A (full 1,200: hin 600 / eng 600)
   + test-B pilot (33 TTS Hinglish clips).
3. Silence-threshold baseline: trailing-silence duration, sweep T on test-A,
   transfer best T to test-B (unsupervised baseline — no training).
4. Thesis table v1 in `turn_v2/results.md`; machine rows in
   `turn_v2/results.csv` (schema per masterplan §4); per-language and
   per-filler slices reported.

## Exit criteria (masterplan P1)

- [ ] Table: model × test-set → acc/F1 exists, expected pattern visible
      (strong A, weaker B)
- test-C pilot is pending P2 (MUCS not yet downloaded) — noted, not blocking

## Constraints

- Import from `smart-turn/`, never modify; 8 s contract via shared code only.
- Latency: record per-clip p50 inference ms (CPU, int8) for results.csv.
