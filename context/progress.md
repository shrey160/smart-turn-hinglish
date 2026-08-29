# Progress Log — Hinglish Turn Detection Project

## Session 1 — 2026-08-22: Setup + data download prep

### Done
1. **Read the full context**: playbook (`Hinglish-Turn-Detection-Playbook.md`) and baseline
   code report (`CODE_REPORT.md`). Project = Shiprocket challenge: build a Hinglish
   turn detection model; Smart Turn v3.2 is the read-only reference baseline.
2. **Created uv-managed environment**:
   - Root `pyproject.toml` (Python 3.11, `uv`, package=false).
   - `.venv/` created; `uv sync` done. Installed: datasets 5.0.1, huggingface-hub,
     soundfile, librosa, numpy 2.1.3, pandas, torch (CPU wheel), torchcodec, python-dotenv.
   - Verified `import datasets/soundfile/librosa` works.
3. **Wrote `AGENTS.md`** — repo layout, uv commands, secrets policy (.env has
   SARVAM_API_KEY / HUGGINGFACE_API_KEY; never print contents), data strategy table,
   input-contract conventions, agent working rules.
4. **Probed HF datasets-server API** for `pipecat-ai/smart-turn-data-v3.2-train`:
   - `/info` works: features = audio, id, language, endpoint_bool, midfiller,
     endfiller, synthetic, spoken_text(null), dataset.
   - `/rows` works and returns per-row audio URLs.
   - `/filter` endpoint: quoted-column where clause accepted but index kept loading /
     erroring → **not reliable**, abandoned that route.
5. **Wrote `scripts/download_data.py`** (streaming subset downloader):
   - train_pool quotas: hin 3000+3000, eng 1000+1000, mar 200+200, ben 200+200
     (complete/incomplete), test_a: hin/eng × 300+300 from v3.2-test (~1.9 GB total).
   - Saves as `data/{split}/{lang}/{label}-{midfiller}-{endfiller}/{id}.flac`.
   - Resumable (counts existing .flac files), interrupt-safe, progress prints,
     final summary table.

### Pending / next session
- [ ] **User runs** `uv run python scripts/download_data.py` (several hours; network
      traffic > disk usage due to row-group streaming — expected).
      - Fix applied after first failure: `Audio(decode=False)` cast so raw FLAC bytes
        stream without torchcodec/FFmpeg; symlink warning silenced via env var.
      - Smoke-tested: streaming yields raw bytes dict correctly.
- [ ] Verify download stats: counts per language × label, filler distribution,
      duration histogram, disk size vs ~1.9 GB budget.
- [ ] Day-1 listening log: sample 50 clips (esp. hin — pure Hindi or code-mixed?).
- [ ] Baselines: silence-threshold sweep + Smart Turn v3 zero-shot on test-A/B/C
      (needs `uv add onnxruntime` for `smart-turn/predict.py`).
- [ ] TTS Hinglish pilot batch (edge-tts, 100 clips) → then full 3–4k generation.
- [ ] Heavy ML deps still not installed (transformers, torchaudio GPU/onnxruntime/onnx) —
      add when needed via `uv add`.

### Key decisions
- Data acquisition via HF streaming (playbook Option A) since datasets-server `/filter`
  was unstable; resumability added to tolerate long/interrupted downloads.
- turn_v2/ still empty; all our code goes to root `scripts/` + `turn_v2/`.

## Session 2 — 2026-08-29: P0 complete (env, baseline weights, zero-shot, listening log)

### Done
1. **Wrote `context/phases/phase1.md` (P0 plan) + `context/subplan.md`** — context
   workflow files now exist and are up to date.
2. **Installed heavy deps** via `uv add transformers torchaudio onnx onnxruntime edge-tts`:
   transformers 5.16.1, torchaudio 2.11.0 (torch 2.13.0+cpu — version skew but
   torchaudio ops verified working), onnx 1.22.0, onnxruntime 1.29.0, edge-tts 7.2.8.
3. **Downloaded Smart Turn v3 int8 ONNX** → `models_ref/smart-turn-v3.2-cpu.onnx`
   (8.68 MB; repo also has v3.0/v3.1 and 32 MB fp32 gpu variants). Hardlinked at
   repo root as `smart-turn-v3.1.onnx` because `smart-turn/inference.py:7`
   hardcodes that CWD-relative filename.
4. **Zero-shot smoke test (6 clips)** — all in-distribution correct with confident
   probabilities; the TTS Hinglish incomplete clip was MISCLASSIFIED:

   | clip | true | p(complete) | verdict |
   |---|---|---|---|
   | hin complete | complete | 0.9875 | ✓ |
   | hin incomplete | incomplete | 0.0055 | ✓ |
   | eng complete | complete | 0.9901 | ✓ |
   | eng incomplete | incomplete | 0.0575 | ✓ |
   | tts-hinglish complete | complete | 0.6067 | ✓ (weak) |
   | tts-hinglish incomplete | incomplete | 0.5588 | ✗ |

   Exactly the playbook thesis: strong in-distribution, weak on TTS Hinglish.
5. **Listening log created**: `scripts/sample_listening_log.py` (seed 42,
   label-balanced stratified sample) → `context/listening_log.md` with 60 clips
   (hin 30 / eng 15 / mar 5 / ben 5 / tts 5) + duration/RMS/peak/tail-silence stats.
6. Created `.gitignore` (`.venv/`, `.env`, `data/`, `models_ref/`, root ONNX).

### Pending / next session (P1)
- [ ] **User task**: annotate `context/listening_log.md` notes column (50+ clips,
      hin first: pure Hindi vs code-mixed, filler realism).
- [ ] P1 silence-threshold baseline: sweep T on dev/test sets.
- [ ] P1 Smart Turn zero-shot on test-A (full 1,200) + test-B (33-clip pilot)
      — batch script needed (predict.py is single-file).
- [ ] P2 data completion: TTS Hinglish pilot (100 clips, edge-tts) then full 3–4k.
- [ ] torchaudio/torch version skew: if any torchaudio issue appears, pin
      `torchaudio==2.13.*` to match torch 2.13.0.

### Key decisions
- Baseline = `smart-turn-v3.2-cpu.onnx` (int8) — matches the v3.2 data pool and
  test-A provenance; root hardlink name `smart-turn-v3.1.onnx` kept for
  compatibility with the read-only `smart-turn/inference.py`.

## Session 3 — 2026-08-29: P1 complete (thesis table v1)

### Done
1. **Wrote `context/phases/phase2.md` (P1 plan) + rewrote `context/subplan.md`**.
2. **Built `scripts/eval_baselines.py`** — batch evaluator: iterates split
   folders, loads 16 kHz, applies the shared 8 s contract (imported from
   `smart-turn/audio_utils`), runs zero-shot and/or trailing-silence baseline;
   per-clip predictions → `results/` (gitignored), silence sweep →
   `results/silence_sweep.csv`. p50 15 ms/clip on CPU (int8 + feature extraction).
3. **Zero-shot Smart Turn v3.2 int8**:
   - test-A (1,200): acc 0.927 / F1 0.930 — eng 0.955 > hin 0.900;
     midfiller=True slice drops to 0.858 (pause-vs-end is the hard core)
   - test-B pilot (33 TTS Hinglish): acc 0.788 / F1 0.759
   - Slice note: endfiller=True clips are ALL incomplete → positive-class F1
     undefined there (acc 0.950 is the meaningful number)
4. **Silence-threshold baseline** (30 ms frames, RMS floor 0.01): sweep
   T∈[0.05,1.5] on test-A → near-chance (best acc 0.578 @ T=0.20 s). Tail-silence
   medians: 0.18 s complete vs 0.06 s incomplete → trailing silence alone is
   uninformative in this data. Report-worthy finding.
5. **Thesis table v1** seeded: `turn_v2/results.csv` (rows p1-001/p1-002) +
   `turn_v2/results.md` (curated table, slices, reproduce commands).

### Pending / next session (P2 — data completion)
- [ ] `scripts/generate_tts_hinglish.py`: edge-tts, 4 voices, complete/incomplete
      pairs from shared templates; pilot 100 clips → listen → scale to 3–4k
- [ ] test-B full ~600 (scripts split train vs test-B BEFORE rendering)
- [ ] test-C: MUCS 2021 subset (1–2 h) with word-boundary cutting
- [ ] Carve dev split (10%, stratified lang × label) — doubles as quantization
      calibration set
- [ ] User: annotate `context/listening_log.md` notes column

### Key decisions
- Silence baseline frozen at RMS floor 0.01 (single knob T swept) — weak is the
  point; further tuning would be baseline overfitting.
- F1 reported per slice with single-class-slice caveat documented in results.md.

## Session 4 — 2026-08-29: P2 progress (TTS pilot, dev split, MUCS scout)

### Done
1. **Wrote `context/phases/phase3.md` (P2 plan) + rewrote `context/subplan.md`**.
2. **Built `scripts/generate_tts_hinglish.py`** — 25 handwritten Hinglish script
   pairs (logistics/customer-care; hi Devanagari-mixed + en Roman renderings),
   marker-based complete/incomplete derivation (`|` final-clause cut, `^`
   midfiller), template slots ({oid},{n}) for combinatorial scale-up, 4 voices
   rotated, edge-tts rate +8% → mp3 → 16 kHz mono FLAC + 250 ms tail, manifest
   CSV, resumable. Fixed along the way: markers leaked into text (carets after
   cut pipes), double commas, cp1252 console encoding, 10 clips >8s → uniform
   +8% rate → 4 remain at 8.0–8.65 s (loader contract applies).
3. **Pilot 100 clips generated** — 0 failures; labels 50:50 (37+13 complete-FF/TF
   vs 17+17+16 incomplete cut/endfiller/midfiller → after rate fix 35+12/17+17+15);
   durations mean 6.25 s; 16 kHz mono verified. `scripts/tts_report.py` →
   `data/tts_hinglish/pilot_report.md` (listening gate artifact).
4. **Dev split carved** — `scripts/carve_dev.py` → `data/splits/dev_v1.csv`:
   881 clips (10% of train_pool), stratified lang × label-folder, seed 42;
   manifest-based so files stay in place (downloader resumable); doubles as
   int8 quantization calibration set.
5. **MUCS scout** — `dianavdavidson/MUCS-Hinglish` (public): train ~9 GB
   (never bulk), test 2 parquets ~563 MB total; plan = one test parquet →
   1–2 h utterance subset → delete parquet; incomplete clips via torchaudio
   forced alignment, fallback complete-heavy (masterplan risk rule).
6. Dataset-card data appendix drafted in `turn_v2/results.md`.

### Pending / next session
- [ ] **USER: listening gate** on `data/tts_hinglish/pilot_report.md` (100 clips,
      verdict per clip: keep / re-voice / re-script / drop) → then scale TTS
      generation to 3–4k train + ~600 test-B (add 30+ test-B scripts first)
- [ ] test-C build: download 1 MUCS test parquet, cut 1–2 h subset, decide
      forced-alignment vs complete-heavy fallback
- [ ] Extend dev split with TTS-train 10% after scale-up
- [ ] P3 can start in parallel: turn_v2 data pipeline + model skeleton
      (train_pool + dev_v1 are ready; TTS-full can join later)

### Key decisions
- Pilot quality gate BEFORE full TTS scale-up (masterplan TTS-quality risk rule).
- Dev split = manifest (no file moves) → download resumability + calibration
  set stays held-out.
- MUCS: one test parquet only; train parquets off-limits (data cap).
- too_long TTS clips stay unmodified on disk; loader applies last-8s contract.

## Session 4 close-out — STRATEGY PIVOT directive (2026-08-29, user decision)

**User decision:** change the training strategy from "single-stage training on
the full mixed pool" to a **two-stage transfer recipe**:

1. **Stage 1 — pretrain**: train on the ORIGINAL Smart Turn data only —
   `hin` + `eng` from train_pool (6,000 + 2,000; minus dev_v1).
2. **Stage 2 — finetune**: fine-tune that checkpoint on the TTS Hinglish data
   using a SMALLER amount than the 3–4k originally budgeted (use what was
   generated / is feasible; quality-gated by the pilot listening).

**Consequences to rewire NEXT SESSION (do not start coding):**
- Rewrite masterplan §4 (training recipe → two-stage: per-stage data, lr,
  epochs, freeze policy) and §5 P2/P3 accordingly; re-derive TTS data budget
  (3–4k target likely drops; test-B ~600 unchanged as eval set).
- Decide where `mar`/`ben` 800 clips go (stage 1? drop? stage 2 mix?).
- P3 ablation arms gain a stage dimension; results.csv needs a stage column.
- P5 KD teacher should follow the same two-stage recipe.
- Still open from Session 4: pilot listening gate, test-C build, dev-v1 TTS
  extension (all secondary to the rewiring).
- NEXT SESSION SCOPE: rewire masterplan + phase plan + subplan only (planning
  pass), then proceed to implementation in the following work blocks.

## Session 5 — 2026-08-29: Strategy rewire complete (two-stage transfer)

### Done — rewired per Session 4 close-out directive (planning pass only, no code)
1. **User decisions locked** (6): mar/ben **dropped from training**; Stage-2
   **replay ON** (50:50 per-epoch mix, TTS×2 upsample + equal hin/eng replay);
   TTS scale-up **cancelled** — Stage 2 = existing **232 clips** (132 example-train
   + 100 pilot); test-B **stays at 33**; Stage-2 hyper **lr 1e-5, 2 epochs,
   same freeze**; pilot listening gate **waived** (stats-QC admission only).
2. **Masterplan rewired**: §2.1/§2.2 state updated; **§4 fully rewritten** as
   two-stage recipe (s1: hin+eng ~7,200, lr 5e-5, 4 ep → ckpt_s1; s2: 232 TTS +
   replay, lr 1e-5, 2 ep → ckpt_s2); results.csv schema v2 (+`stage`,
   +`init_ckpt`); P2 closed per pivot (test-C carried to P3); **P3 rewritten**
   (two-stage reproduction, new exit: s2 beats zero-shot on test-B AND test-A
   regression ≤ ~1–2%); P4 stage dimension (architecture arms = full s1+s2
   rerun, training-only arms = cheap s2-only; added ARM-5 replay-ratio sweep);
   P5 KD teacher follows same two-stage recipe, KD applied at student s2;
   §7 risks updated (tiny-s2-set probe framing).
3. **`context/phases/phase4.md` written** (P3 plan) + **`context/subplan.md`
   rewritten** (SP1 skeleton → SP2 overfit gate → SP3 s1 run → SP4 s2 run →
   SP5 test-C build parallel).
4. **`AGENTS.md` updated**: data strategy table (TTS 3–4k → ~232 existing;
   mar/ben excluded note) + two-stage recipe pointer.

### Pending / next session (P3 implementation — Phase 4)
- [ ] SP1: build `turn_v2/` skeleton (dataset/augment/model/pooling/train/evaluate)
- [ ] SP2: `--overfit 100` sanity gate
- [ ] SP3: Stage-1 run → ckpt_s1, eval dev + A/B
- [ ] SP4: pilot stats-QC + TTS dev carve (~23 clips) → Stage-2 run → ckpt_s2,
      thesis table v2
- [ ] SP5: test-C build (MUCS 1 parquet → 1–2 h subset)
- [ ] Backfill `turn_v2/results.csv` p1 rows with stage=baseline, init_ckpt=none

### Key decisions
- Replay ratio default **50:50 per epoch** (TTS upsampled ×2 vs equal replay):
  with only 232 TTS clips, a 10–20%-of-s1 replay (~1,080) would swamp the TTS
  signal 5:1. Ratio is a cheap P4 arm (ARM-5: 0/25%/50%).
- TTS dev carve ~23 clips (10%, seed 42) gives s2 an in-domain early-stop signal.
- Pivot framed as an approach **probe**: invest in TTS scale-up only if s2 shows
  test-B gains.
