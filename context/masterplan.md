# Masterplan — Hinglish Turn Detection: Train & Fine-Tune

> Operational plan for building `turn_v2/` (our model) on top of the verified data pool,
> following `context/references/Hinglish-Turn-Detection-Playbook.md` (strategy) and
> `context/references/smart_turn_training.md` (reference implementation details).
> Companion log: `context/progress.md` (session-by-session state).

---

## 1. Goal & constraints

Build a **tiny, fast, audio-native turn-detection model for Hinglish speech** —
binary: *complete* (agent should respond) vs *incomplete* (user still talking).

| Constraint | Value |
|---|---|
| Hardware | local RTX 4060, 8 GB VRAM |
| Timeline | ~1 week effective (phase-map in §5) |
| Data cap | ≤ 2 GB on disk (currently at ~1.5 GB incl. example sets) |
| Deliverables | weights (HF Hub) + runnable demo + self-written report |

**Non-negotiable input contract** (identical to Smart Turn, everywhere):
- 16 kHz mono float32 in [-1, 1]
- ≤ 8 s: shorter → **zero-pad front**; longer → **keep last 8 s**
  (`smart_turn_reference/audio_utils.py:truncate_audio_to_last_n_seconds` semantics)
- Generated clips end with ~200 ms real silence; never cut mid-word
- Labels: `endpoint_bool` (1 = complete); keep `midfiller`/`endfiller` as eval metadata

---

## 2. Current state (verified)

### 2.1 On disk

| Asset | Status | Numbers |
|---|---|---|
| `data/train_pool/` | ✅ complete | 8,800 FLAC / 1.15 GB — hin 6,000 · eng 2,000 · mar 400 · ben 400, each exactly 50:50 complete/incomplete |
| `data/test_a/` | ✅ complete | 1,200 FLAC / 175 MB — hin 600 · eng 600 (300:300 each) from `pipecat-ai/smart-turn-data-v3.2-test` |
| `data/example_hinglish_fixed/` | ✅ complete | 165 TTS Hinglish FLAC, Smart Turn layout, 80:20 → `train_pool/` 132 + `test_b/` 33 (seed 42) |
| `data/data_result_example/` | ✅ source template | original CSV+MP3s, compatibility report included |
| `data/tts_hinglish/` | ✅ pilot only | 100 pilot FLAC (50:50 labels) + `pilot_report.md`; scale-up CANCELLED (pivot) |
| `data/splits/dev_v1.csv` | ✅ complete | 881 clips, 10% stratified lang × label-folder, seed 42; doubles as int8 calibration set |
| `models_ref/` | ✅ complete | `smart-turn-v3.2-cpu.onnx` (int8, 8.68 MB); hardlink at root as `smart-turn-v3.1.onnx` |
| `scripts/download_data.py` | ✅ working | resumable streaming downloader; `--only train_pool|test_a` |
| `scripts/structure_example_hinglish.py` | ✅ working | MP3→FLAC + Smart Turn folder layout |
| `scripts/split_example_hinglish.py` | ✅ working | 80:20 stratified split |
| `turn_v2/` | 🟡 started | `results.csv`/`results.md` seeded (P1 rows); model code to be built |

### 2.2 Missing before training can start

1. ~~Heavy deps~~ ✅ done (P0): transformers 5.16.1, torchaudio 2.11.0, onnx, onnxruntime, edge-tts
2. ~~Smart Turn v3 int8 ONNX weights~~ ✅ done — `models_ref/smart-turn-v3.2-cpu.onnx`
3. ~~Full TTS Hinglish train set (3–4k) + test-B (~600)~~ ❌ **CANCELLED (pivot 2026-08-29)**:
   Stage 2 uses the existing **232 TTS clips only** (132 example-train + 100 pilot); test-B stays at 33
4. test-C (MUCS 2021 real speech) — pending build (rolls into P3 phase scope)
5. ~~dev split~~ ✅ done — `data/splits/dev_v1.csv` (881 clips, doubles as int8 calibration)
6. TTS dev carve (~23 clips, 10% of TTS train, seed 42) — small, for Stage-2 early stopping

---

## 3. Target layout: `turn_v2/`

```
turn_v2/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── dataset.py        # TurnDataset: on-demand loading from folder layout,
│   │                     #   8s contract, label parsing, dev-split carving
│   └── augment.py        # telephony bandpass / noise 10–20dB SNR / speed 0.9–1.1×,
│                         #   applied on-the-fly in __getitem__ (p≈0.5, zero disk)
├── models/
│   ├── __init__.py
│   ├── model.py          # SmartTurnV2Model: Whisper Tiny encoder + configurable head
│   └── pooling.py        # attention-mean (baseline) | ASP (mean+std) | +end-biased concat
├── train.py              # plain PyTorch loop (see §4)
├── evaluate.py           # dev + test-A/B/C; accuracy/F1 + per-language, per-filler slices
├── export.py             # ONNX fp32 → int8 static QDQ with HELD-OUT calibration (dev)
├── policy.py             # threshold sweep → false-interrupt vs latency Pareto frontier
├── latency.py            # accuracy vs trailing-context (0.5/1/2/4/8 s) curve
├── results.csv           # machine log: one row per run
├── results.md            # curated ablation table (report artifact)
└── app.py                # Gradio demo (last phase)
```

**Rules**
- `smart_turn_reference/` stays read-only — import from it (`audio_utils`), never modify
- Data-generation scripts live in root `scripts/` (they are data ops, not model code)
- Every experiment = one change per row; all rows report test-A/B/C

---

## 4. Training recipe — two-stage transfer (`turn_v2/train.py`)

> **STRATEGY PIVOT (2026-08-29, user decision):** two-stage transfer replaces the
> original single-stage mixed-pool recipe. Stage 1 pretrains on original Smart
> Turn `hin`+`eng` only; Stage 2 finetunes on the small existing TTS Hinglish
> set + replay. mar/ben **dropped from training**. TTS scale-up **cancelled**.

### Stage 1 — pretrain (`stage=s1`)

| Hyperparameter | Value |
|---|---|
| Data | train_pool `hin`+`eng` minus dev_v1 → **~7,200 clips** (mar/ben excluded) |
| Base | `openai/whisper-tiny` **encoder only** (frozen by default) |
| Features | `WhisperFeatureExtractor(chunk_length=8)` → (80, 800) log-mel |
| Optimizer | AdamW, lr 5e-5, weight decay 0.01 |
| Schedule | cosine, warmup ratio 0.2 |
| Epochs | 4 (early stop on dev_v1 hin+eng F1, patience ~2 evals) |
| Batch | 32–64 train (fp16, 8 GB VRAM), 128 eval |
| Model selection | best dev_v1 **hin+eng subset** F1 → `ckpt_s1` |

### Stage 2 — finetune (`stage=s2`)

| Hyperparameter | Value |
|---|---|
| Init | `ckpt_s1` |
| Data | **232 TTS Hinglish clips** (132 example-train + 100 pilot; stats-QC only, listening gate waived) + replay |
| Replay mix | **50:50 per epoch**: TTS upsampled ×2 (→464) + equal hin/eng replay drawn fresh each epoch (seeded). Ratio is a cheap P4 arm. |
| Optimizer | AdamW, lr **1e-5**, weight decay 0.01 |
| Schedule | cosine, warmup ratio 0.1 |
| Epochs | **2** (early stop on combined dev: dev_v1 hin+eng + TTS dev ~23 clips) |
| Freeze policy | same as Stage 1 (frozen encoder, head + pooling trainable) |
| Output | `ckpt_s2` — candidate v2 model |

Shared (both stages): plain PyTorch loop — no HF Trainer, no W&B requirement;
label smoothing 0.0 baseline (0.05–0.1 = P4 arm); seeds 42
everywhere; sanity mode `--overfit 100` must pass before any full run; every run
appends to `turn_v2/results.csv`, curated rows into `results.md`.

```python
# loss (mirrors Smart Turn math)
pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
```

**results.csv row schema (v2)**
`run_id, timestamp, stage, init_ckpt, config_tag, change_summary, seed, train_n,
dev_f1, dev_acc, testA_f1, testA_acc, testB_f1, testB_acc, testC_f1, testC_acc,
params_M, latency_ms_p50_cpu, notes`
(`stage` ∈ baseline|s1|s2; pre-pivot rows backfilled as `baseline`, `init_ckpt=none`)

---

## 5. Phased execution

Each phase has explicit exit criteria — do not advance until met.

### P0 — Environment & baseline weights
- [x] `uv add transformers torchaudio onnx onnxruntime scikit-learn edge-tts`
- [x] Download Smart Turn v3 int8 ONNX from HF Hub into `models_ref/` (gitignored)
      — `smart-turn-v3.2-cpu.onnx` 8.68 MB, hardlinked at root as `smart-turn-v3.1.onnx`
      (filename hardcoded in `smart_turn_reference/inference.py:7`)
- [x] Run baseline inference on a few local wavs (16 kHz mono) — 6-clip smoke test,
      5/6 correct; see `context/progress.md` Session 2
- [x] Listening log: 50+ clips from train_pool (hin first) → `context/listening_log.md`
      (60 clips sampled seed 42; human annotation of notes column pending)
- **Exit:** zero-shot prediction works locally ✅; log exists ✅

### P1 — Baselines (the thesis table v1)
- [x] Silence-threshold baseline: sweep threshold T on test-A (dev split pending
      P2) — near-chance (best acc 0.578); trailing silence uninformative alone
- [x] Smart Turn zero-shot on test-A (full): acc 0.927 / F1 0.930;
      test-B (33-clip pilot): acc 0.788 / F1 0.759; test-C pending P2 (MUCS)
- **Exit:** table exists in `turn_v2/results.md`: model × test-set → acc/F1 ✅
      (strong A, weaker B as expected; C pending)

### P2 — Data completion
- [x] `scripts/generate_tts_hinglish.py`: edge-tts, 4 voices
      (hi-IN-SwaraNeural, hi-IN-MadhurNeural, en-IN-NeerjaNeural, en-IN-PrabhatNeural),
      logistics/customer-care scripts, complete/incomplete **pairs** from shared templates
      (marker-based: `|` cut, `^` midfiller; 25 scripts, slots for combinatorial scale-up)
- [x] Conventions: FLAC, 16 kHz mono, +250 ms tail silence, filler sub-labels,
      **split scripts train vs test-B before rendering** (enforced in code)
- [x] **Pilot 100 clips generated** (0 failures, 96/100 ≤ 8 s, 50:50 labels) →
      `data/tts_hinglish/pilot_report.md` — listening gate WAIVED (pivot:
      stats-QC admission only); ~~full 3–4k train + ~600 test-B generation~~
      CANCELLED — Stage 2 uses the existing 232 TTS clips, test-B stays at 33
- [x] test-C scout: `dianavdavidson/MUCS-Hinglish` — 1 test parquet (~295 MB)
      → 1–2 h subset; incomplete clips via forced alignment or complete-heavy
      fallback (build = next work block)
- [x] Carve dev split: `data/splits/dev_v1.csv` — 10% of train_pool (881 clips),
      stratified lang × label-folder, seed 42; doubles as int8 calibration set
- [x] Verify counts/durations/balance after every step; disk 1.30 GB / 2 GB cap
- **Exit:** dataset card matches playbook targets — ✅ **complete per pivot**:
  TTS scale-up/test-B-600/listening-gate cancelled by user decision (2026-08-29);
  only **test-C build remains → rolls into P3 phase scope** as parallel workstream

### P3 — Two-stage reproduction
- [ ] `turn_v2/data/dataset.py` + `augment.py`; `models/model.py` (attention-mean baseline)
- [ ] `--overfit 100` sanity gate — must pass before any full run
- [ ] On-the-fly augmentation wired (telephony / noise / speed)
- [ ] Pilot stats-QC admission (labels/durations/16 kHz) + TTS dev carve (~23 clips, seed 42)
- [ ] test-C build: 1 MUCS test parquet → 1–2 h subset → delete parquet
      (parallel workstream; forced alignment or complete-heavy fallback)
- [ ] **Stage 1 run** (hin+eng ~7,200, lr 5e-5, 4 ep) → `ckpt_s1`; eval dev + A/B
      (expect test-A ≈ Smart-Turn-level, test-B weak)
- [ ] **Stage 2 run** (232 TTS ×2 + 50:50 hin/eng replay, lr 1e-5, 2 ep) from
      `ckpt_s1` → `ckpt_s2`; eval dev + A/B/C; thesis table v2
- **Exit:** `ckpt_s2` beats Smart Turn zero-shot on **test-B** (0.788 acc / 0.759 F1)
      AND test-A regression vs `ckpt_s1` ≤ ~1–2%

### P4 — Ablation arms (one change per row, reusing P3 harness)
> Stage dimension: architecture arms (ARM-1/2) change the head input dim →
> **full s1+s2 rerun** required (ckpt_s1 incompatible). Training-only arms
> (ARM-3/4/5) = cheap **s2-only reruns** from `ckpt_s1`.
- [ ] ARM-1: ASP pooling — weighted mean + std → classifier in 768-dim (playbook §7.1) — full s1+s2
- [ ] ARM-2: end-biased — concat mean of last ~50 encoder frames (playbook §7.2) — full s1+s2
- [ ] ARM-3: label smoothing 0.05–0.1 — s2-only
- [ ] ARM-4: unfreeze last 1–2 encoder blocks (if head-only plateaus) — s2-only
- [ ] ARM-5: replay ratio sweep (0 / 25% / 50%) — s2-only, cheapest arm
- **Exit:** ablation table complete; best arm selected as "v2-core"

### P5 — Distillation + deployment
- [ ] Teacher: Whisper Small encoder trained with the **same two-stage recipe**
      (s1 hin+eng → s2 TTS+replay); **precompute logits** once
- [ ] KD into v2-core student, applied at the student's **s2 run**: T=2–5, α≈0.5 (playbook §7.3)
- [ ] `export.py`: ONNX fp32 → int8 static QDQ (QUInt8/QInt8, per-channel, Entropy)
      with **held-out calibration from dev** (fixes the reference flaw)
- [ ] Report FP32 vs int8 delta; watch for AVX2 U8S8 saturation cliff
- [ ] TEN VAD swap-in for the demo's VAD gate (drop-in, 16 kHz frames)
- **Timebox:** if KD not working by end of P5 → ship best P4 arm. No exceptions.
- **Exit:** best model ≤ 8 MB int8, quantization delta ≤ ~1%

### P6 — Evaluation depth
- [ ] `latency.py`: accuracy/F1 vs trailing context 0.5/1/2/4/8 s
- [ ] `policy.py`: threshold sweep → false-interrupt rate vs added-latency Pareto frontier
- [ ] Per-category: language × filler × test-set; code-switch density slices where possible
- [ ] Error analysis: 20+ failures from test-B/C pulled, categorized, written up
- **Exit:** curves + frontier + error-analysis section drafted for the report

### P7 — Ship
- [ ] `turn_v2/app.py`: Gradio demo (mic/file → verdict + probability + inference ms)
- [ ] Weights + feature extractor → HF Hub; README results table + reproduce commands
- [ ] Final report per playbook §8 (framing → audit → strategy → ablations → errors → limits)
- **Exit:** deliverables checklist green

---

## 6. Experiment discipline

1. **One change per ablation row.** No combined arms except the final "v2-core = best arms" run.
2. Every row reports **all three test sets** + dev — never a single number.
3. Fixed seeds; identical data ordering across arms (same seed → same batches).
4. The 8-second contract is shared code (import from `smart_turn_reference/audio_utils.py`) — never re-implement it locally.
5. Verify audio stats (counts × label, durations, disk) after every data step.
6. Prefer tables/curves over prose in `results.md`.
7. Never bulk-download the 41 GB train set; stream-filter only (existing script does this).

---

## 7. Risks & fallbacks

| Risk | Mitigation |
|---|---|
| Tiny Stage-2 set (232 clips) → overfit / no adaptation signal | 50:50 replay mix + lr 1e-5 + 2 epochs + combined-dev early stop; the pivot is an approach **probe** — invest in TTS scale-up only if s2 shows test-B gains |
| TTS quality issues in pilot clips (listening gate waived) | stats-QC admission (labels/durations/16 kHz); per-clip anomalies surface in error analysis (P6) |
| MUCS segmentation effort blows up | cap at 1–2 h subset; if timestamps unusable, fall back to utterance-boundary clips only (complete-heavy test-C) |
| 8 GB VRAM OOM on Whisper Small teacher | batch 16 + grad accumulation; logits precompute makes KD cheap |
| int8 accuracy cliff on AVX2 | `reduce_range` or U8U8 path (playbook §4.7) |
| KD stalls | timebox rule (§5 P5) — ship best ablation arm |
| Everything slips | minimum shippable = P1 thesis table + P3 two-stage reproduction + P4 ASP arm + P6 policy sweep |

---

## 8. Deliverables checklist

- [ ] Repo: `turn_v2/` code, pinned deps, seeds, README with results table + run commands
- [ ] `turn_v2/results.md`: ablation table, latency curve, Pareto frontier, FP32-vs-int8
- [ ] Weights on HF Hub (fp32 + int8 ONNX + feature extractor)
- [ ] Gradio demo (mic/file, verdict + probability + latency)
- [ ] Self-written report: problem framing, dataset audit (83%-TTS finding, listening log),
      data strategy, approach, experiments, error analysis, limitations, next steps
      (text conditioning, 3-class hold phrases, streaming encoder, QAT, human Hinglish data)
