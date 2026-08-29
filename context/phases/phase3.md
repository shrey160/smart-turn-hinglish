# Phase 3 — P2: Data completion

> Maps to masterplan §5 P2 (playbook Day 2 evening). Progress trace: completed
> phases stay in `context/phases/`.

## Objective

Complete the target-domain data: TTS Hinglish train set + full test-B, the
dev/quantization-calibration split, and scout test-C (MUCS 2021).

## Tasks

1. `scripts/generate_tts_hinglish.py` — edge-tts, 4 voices
   (hi-IN-Swara/hi-IN-Madhur, en-IN-Neerja/en-IN-Prabhat), logistics/
   customer-care scripts, complete/incomplete **pairs from shared templates**
   (marker-based: cut at `|`, filler at `^`), FLAC 16 kHz mono ≤ 8 s,
   +250 ms tail, Smart Turn folder layout, full manifest CSV.
2. **Pilot batch (100 clips)** — 50 script pairs × 2 variants, voice rotation;
   stats verification; pilot report for human listening BEFORE scaling
   (masterplan risk rule: TTS quality gate).
3. Dev split: 10% of train_pool stratified lang × label → `data/splits/dev_v1.csv`
   (manifest-based — files stay in place so the downloader stays resumable);
   doubles as int8 quantization calibration set (held-out!).
4. test-C scouting: locate MUCS 2021 on HF Hub, assess size/timestamps for
   word-boundary cutting — **no bulk download** (data cap).

## Exit criteria (masterplan P2)

- [ ] Pilot generated + verified; user listening gate passed (full 3–4k +
      test-B 600 generation follows in the next work block)
- [ ] Dev split carved + verified
- [ ] MUCS plan decided (subset path or fallback)
- [ ] Dataset card drafted; disk ≤ 2 GB

## Constraints

- Split scripts train vs test-B BEFORE rendering (zero overlap).
- Never cut mid-word; fillers per Smart Turn sub-labels
  (midfiller/endfiller in folder names).
