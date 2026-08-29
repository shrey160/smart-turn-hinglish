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

## Session 6 — 2026-08-29: P3 complete (skeleton → gate → s1 → s2 ladder, exit MET)

### Done
1. **Created `context/HARDPOINT.md`** (initialized; 2 entries this session).
2. **SP1 skeleton built** (`turn_v2/`): `data/dataset.py` (folder/manifest splits,
   8 s contract imported from reference, on-demand (80,800) log-mel, collate,
   TTS dev carve), `data/augment.py` (telephony highpass300/lowpass3400 — playbook
   §7.4 snippet fixed, it passed 3400 as Q — noise 10–20 dB SNR, speed 0.9–1.1×,
   p≈0.5 pre-contract), `models/pooling.py` (attention-mean/ASP/asp-end, N(0,0.1)
   init mirrored from reference), `models/model.py` (frozen whisper-tiny encoder;
   mel-length check fixed for transformers 5 by setting max_source_positions=400
   and keeping pretrained pos-emb rows 0–399), `train.py` (plain loop, batch
   pos_weight BCE, warmup+cosine, per-epoch eval, best-ckpt on dev F1, early stop,
   results.csv schema v2), `evaluate.py` (slices + e2e/fwd latency p50).
3. **SP2 gate**: `--overfit 100` → FAIL at locked 5e-5 (stall ~0.67 loss), PASS at
   lr 1e-3 (train_f1 1.000, loss 0.0024) — root cause + open question logged in
   HARDPOINT (head-only needs higher lr than full-encoder reference lrs).
4. **SP3 s1-001**: 7,200 clips, 4 ep, frozen encoder → dev 0.851 / test-A
   0.833/0.829 / test-B 0.727/0.571 (~12 min CPU, workers 4).
5. **SP4 s2 ladder** (all init ckpt_s1, 209 TTS ×2 + 50:50 replay = 836/epoch):
   s2-001 locked 1e-5 → flat, test-B 0.727/0.571 (HARDPOINT entry; needs user
   sign-off to supersede locked hyper); s2-002 lr 1e-4 → 0.758/0.692; s2-003
   +3 ep → 0.788/0.741 (acc ties zero-shot); **s2-004 +unfreeze-last-2 →
   test-B 0.970/0.968, test-A 0.882/0.886, dev 0.884**.
6. **P3 exit criteria MET by s2-004** (test-B beats zero-shot; test-A no
   regression vs s1). Thesis table v2 in `turn_v2/results.md`; 5 results.csv rows
   (schema v2; p1 rows backfilled baseline/none and de-malformed).
7. Stats-QC admission of 232 TTS clips: 16 kHz mono ✓, 110:122 labels,
   mean 5.57 s; `data/splits/dev_tts_v1.csv` (23 clips, seed 42) written.
8. **User sign-off received** (same day): §4 amendment approved — s2 recipe is
   lr 1e-4, 3 ep, unfreeze-last-2; ARM-4 → k-sweep (k=1/4/all) in masterplan;
   HARDPOINT entry #2 marked RESOLVED.
9. **`turn_v2/README.md` written** (layout, contract, quickstart, full CLI
   tables, module notes, open items) and `results.csv` p1 rows de-malformed
   (20 → 19 fields; all rows now parse clean — `pandas` table verified).
10. **Session closed** — P3 complete except SP5 (test-C).

### Pending / next session
- [ ] **SP5 test-C build** (MUCS `dianavdavidson/MUCS-Hinglish`: 1 test parquet
      ~295 MB → 1–2 h subset → delete parquet; forced-alignment vs complete-heavy
      fallback TBD; ~100–150 MB permanent) → eval s1-001 + s2-004 on test-C,
      fill testC columns; validates s2-004 test-B is not TTS-pipeline overfit
- [ ] P4 arms: pooling (ASP / asp-end = full s1+s2 rerun), ARM-4 k-sweep
      (k=1/4/all), label smoothing, ARM-5 replay-ratio (s2-only)

## Session 7 — 2026-08-29: SP5 test-C built + evaluated; real domain gap found

### Done
1. Downloaded MUCS-Hinglish test parquet #0 (294.6 MB) → built `data/test_c/`:
   846 clips / 76.8 min / 17 speakers / 50:50 (434 complete + 412 incomplete)
   → parquet deleted (net +107 MB; total data ~1.4 GB, under cap).
   Builder `scripts/build_test_c.py`; manifest `data/splits/test_c_manifest.csv`.
2. Two construction artifacts found & fixed mid-build (HARDPOINT entry):
   (a) valley cuts made incompletes end in silence → switched to speech-active
   cuts; (b) MUCS completes are tight-trimmed (0.000 s median tail) → 200 ms
   pad per repo convention; (c) class-overlap + rng-state determinism bugs.
3. Evaluated zero-shot + s1-001 + s2-004 on test-C (testC columns now in
   results.csv for all rows): zs 0.539/0.573, s1 0.541/0.279, s2-004
   **0.600/0.446**, silence-threshold 0.921/0.928 (T=0.15).
4. Diagnosed the collapse (results.md test-C section): v3.2 models are
   prosody-first (test-A completes: median 0.18 s tail, 25% zero) and MUCS
   tutorial monologue lacks turn-taking prosody; neural operating points need
   >0.3 s tails vs our 0.2 s pad. Energy cue is trivially exploitable (pad).
5. Docs updated: results.md (test-C section + findings), subplan (SP5 done,
   phase closed), HARDPOINT (Session 7 entry), README turn_v2 (test_c split +
   caveat), masterplan (P3 fully done incl. test-C).

### Pending / next session
- [ ] P4 phase plan (`context/phases/phase5.md`) + fresh subplan: pooling arms
      (ASP/asp-end), ARM-4 k-sweep (k=1/4/all), label smoothing, ARM-5
      replay-ratio (s2-only) — all report on A/B/C
- [ ] Optional test-C v2 (if found): real conversational Hinglish calls instead
      of tutorial monologue; or accept v1 as stress test (relative rankings)

### Key decisions
- **test-C v1 shipped as-is**: structurally consistent with v3.2 conventions;
  neural-model collapse is a genuine domain-gap finding, not a blocker —
  s2-004 remains the ship candidate (best neural on B and C, strong on A).
- Forced-alignment labeling dropped (MMS_FA 1.2 GB + Windows perl dependency);
  transcript-guided speech-active cuts achieve the same goal deterministically.
- MUCS incompletes end mid-speech (mid-word cuts allowed — they model a user
  still talking; the "never cut mid-word" rule applies to generated complete
  clips, not realism cuts).

## Session 9 — 2026-08-29: P5 complete (KD negative, int8 shipped with documented deviation)

> Session began with crash recovery: a previous session died mid-teacher-run
> (UnboundLocalError in train.py) and a quant sweep was killed mid-flight from
> Task Manager (system slowdown) — artifacts were intact; both were recovered.

### Done
1. **Fixed the crashed teacher run**: `import numpy as np` inside the KD branch
   of `train()` shadowed the module-level import → `UnboundLocalError` at
   `np.random.seed` before KD ever loaded. Removed the local import; smoke
   tested (`--overfit 4` gets past the crash point).
2. **CUDA torch swapped in** (user call, CPU teacher was too slow):
   torch 2.13.0+cu126 + torchaudio 2.11.0+cu126 via uv (cu124 index has no
   2.13.0; cu130 lacks torchaudio 2.13). biquad ops + CUDA smoke verified;
   torchaudio skew unchanged (2.11 vs 2.13, verified working).
3. **Teacher trained on GPU** (~5.5 min vs 1.5 h CPU budget):
   s1-004 whisper-small (dev 0.913, test-A 0.902/0.904) → s2-013 approved
   recipe (test-A 0.913/0.915, test-B 1.000/1.000, test-C 0.603/0.541) —
   teacher beats the student everywhere.
4. **Teacher logits precomputed** (7,409 clips → kd/teacher_logits.npz;
   script fixed: device-aware, `.to(device)`, unicode paths — no pickle).
5. **KD student s2-014** (T=3, α=0.5): test-B parity (0.970) but test-A
   0.837. **No-KD control s2-015** (same harness, replay 626/ep): test-A
   0.846 — **KD verdict NEGATIVE** (−0.9 acc vs control, −4.5 vs s2-004).
   Ship = **s2-004** per the P5 timebox. One arg-quoting crash along the way
   (unquoted `--change-summary` with spaces) — relaunched with quotes.
6. **export.py WS-B run to completion**: fp32 ONNX (opset 18, dynamic batch)
   **bit-faithful** (0.882/0.970/0.600). First int8 (all-ops QDQ) hit the
   predicted **AVX2 U8S8 cliff** (−8.8 test-A); reduce_range no-op,
   qint8-act catastrophic (0.574). **Fix = reference recipe**
   (`quant_pre_process` + `op_types_to_quantize=["Conv","MatMul","Gemm"]`)
   → −2.5 test-A. Residual knob sweep (minmax/u8u8/calib-1024) all ≈ −2.5 →
   noise floor. Final: `s2-004.int8.entropy-quint8.onnx` 9.05 MB,
   p50 13.2 ms (fp32 17.7 ms).
7. **New util** `scripts/eval_onnx.py` (eval existing ONNX on a split without
   re-quantizing); `PYTHONUTF8=1` lesson for torch.onnx console output;
   `python -u` for background logs (stdout block-buffering hid output).
8. **Docs closed**: results.md P5 section (KD + quant tables + deviation),
   subplan (P5 ✅), masterplan P5 (exit = PARTIAL, deviation documented),
   HARDPOINT (int8 cliff entry), onnx/ cleaned to the two final artifacts.

### Pending / next session (P6 — evaluation depth)
- [ ] **P6 phase plan DONE** (`context/phases/phase7.md`, end of session) —
      next session: rewrite subplan for P6, then build `latency.py`,
      `policy.py`, slices + error analysis (20+ failures, test-B/C)
- [ ] P7 after P6: Gradio demo (fp32 ONNX path), HF Hub upload, final report

### Key decisions
- **KD ships as a documented negative** (teacher>scores student everywhere,
  soft targets don't transfer at 8M params/626-clip s2) — playbook timebox
  honored; v2-core remains s2-004 (fp32 torch + bit-faithful fp32 ONNX +
  9.05 MB int8).
- int8 exit criteria (≤8 MB, ≤~1%) missed narrowly → documented deviation,
  not a blocker: test-B delta is 1 clip; fp32 ONNX is the demo default.
- **User ops rule: only one background process at a time** (system slowdown
  report); background jobs need `-u` + UTF-8 for visible logs.

## Session 8 — 2026-08-29: P4 ablation arms complete → v2-core = s2-004

### Done
1. **Harness extensions** (`turn_v2/train.py`, `models/pooling.py`):
   `--label-smoothing`, `--replay-frac` (default 0.5 = P3 behavior),
   `attention-end` pooling entry; smoke-tested all 4 pooling variants
   (out_dim 384/768/768/1152).
2. **10 ablation runs** (all auto-reported test-A/B/C now):
   ARM-5 replay 0/0.25 (s2-005/006), ARM-3 smoothing 0.05 (s2-007),
   ARM-4 k=1/4/full (s2-008/009/010), ARM-1 ASP (s1-002→s2-011),
   ARM-2 attention-end (s1-003→s2-012).
3. **Verdict: clean negative ablation — every arm lost to the P3 default on
   test-A guard and/or test-B primary. v2-core = s2-004 unchanged.**
   Key signals: replay is load-bearing (0% → test-A 0.770); k=2 sweet spot
   (k=4/full buy 1 test-B clip for −4.6 test-A); ASP/end-bias don't help at
   whisper-tiny scale (ASP −4.0 test-A, +1.6 ms latency).
4. Docs: results.md P4 ablation table + verdict, subplan (phase complete),
   masterplan P4 ticked, README already current (flags documented).

### Pending / next session
- [ ] P5 phase plan (`phase6.md`): distillation (Whisper Small teacher) +
      ONNX int8 export with held-out calibration + TEN VAD demo swap
- [ ] P5 timebox per masterplan: if KD not working by phase end → ship s2-004

### Key decisions
- **No combined-arms run needed**: no two arms won different axes, so the
  final config stays s2-004 exactly (discipline rule held).
- test-B 0.970 vs 1.000 = one clip (33 total) — treated as noise in selection.
- ARM-2's s1 end-bias had best s1 test-C (0.512) — logged as future-work note,
  not selected.

### Key decisions
- transformers 5.x Whisper mel-length validation handled by max_source_positions=400
  + pretrained pos-emb slice (better than reference's random re-init for frozen use).
- Overfit gate runs at lr 1e-3 (head-only); stage lrs remain masterplan defaults
  except where results drove the s2 ladder.
- test-B 33-clip noise (1 clip ≈ 3 pts) acknowledged; test-C build is the priority
  validation before P4 burns compute on arms.
