# Hinglish Turn Detection — Final Report

**Project:** tiny, fast, audio-native turn detection for Hinglish (Hindi–English
code-mixed) telephony speech · **Ship:** [`Shrey160/hinglish-turn-v2`](https://huggingface.co/Shrey160/hinglish-turn-v2)
(fp32 + int8 ONNX) · **Demo:** `turn_v2/app.py` (Gradio, local) ·
**Reference baseline:** Pipecat Smart Turn v3.2 (read-only copy in `smart_turn_reference/`)

The model answers one binary question per ≤8 s speech segment: **is the user
done speaking (complete → agent should respond) or just pausing (incomplete →
keep listening)?**

---

## 1. Problem framing

Turn detection is the piece that decides *when* a voice agent may speak. Getting
it wrong in either direction is user-visible: respond during a pause = **false
interrupt** (the agent talks over the user); wait too long = **dead air**. Silence
thresholds — still the production default — fail precisely on the cases that
matter in Indian customer-support calls: contemplative pauses, mid-list pauses
(dictating an order ID, an address), and filler/hold phrases ("matlab…", "umm…
ek minute").

**Why audio-native for Hinglish.** The text-based alternative (ASR → LLM
judgment) has three structural problems here: (a) code-switching inflates
telephony ASR WER (14–16% is typical for Hinglish), so the text the judge sees is
corrupted exactly on this market's speech; (b) ASR strips fillers — which are
among the strongest "not done yet" cues; (c) text discards prosody, the primary
carrier of "I'm still going" vs "I'm done". The baseline (Smart Turn v3.2) is
already audio-native; our contribution is adapting it to a language pair it
skipped, with the evaluation depth the field actually uses (policies, not single
accuracies).

## 2. Dataset audit (the finding)

Auditing the v3.2 training pool (270k rows, 41.4 GB — audited via HF
datasets-server metadata + streaming samples, never bulk-downloaded):

- **~83% of v3.2 is TTS-generated; only ~1.6% is human speech** (human_5: 469,
  human_convcollector_1: 96 clips in the sampled subset). The remaining ~15% is
  scripted/translated content.
- **Its Hindi is not code-mixed Hinglish** — Hindi clips are largely pure Hindi.
  Fillers are marked (`midfiller`/`endfiller`) but Hinglish-specific code-switch
  prosody is absent.
- Our own listening log (`context/listening_log.md`, 60 stratified clips seed 42)
  supported this: hin clips were predominantly pure Hindi; TTS artifacts audible
  on the synthetic share.

**Implication:** a strong test-A score is a statement about TTS-era Hindi/English,
not about real Hinglish calls. This is why we built domain-specific test sets
before training anything.

## 3. Data strategy (≤2 GB cap)

| Split | Size | Purpose |
|---|---|---|
| `train_pool` (v3.2-train subset) | 8,800 FLAC / 1.15 GB — hin 6,000, eng 2,000, mar 400, ben 400, each 50:50 complete/incomplete | supervised signal |
| `dev_v1` | 881 (10%, stratified lang × label, seed 42) | selection + int8 calibration |
| `test-A` | 1,200 (v3.2-test hin 600 / eng 600) | in-distribution eval |
| TTS Hinglish | 232 (132 example set + 100 edge-tts pilot) | s2 finetune domain |
| `test-B` | 33 held-out TTS Hinglish | Hinglish-domain eval |
| `test-C` | 846 / 76.8 min / 17 speakers (MUCS-Hinglish) | real-speech stress test |

Acquisition: streaming row-filter of the 41.4 GB HF dataset (resumable
downloader, `scripts/download_data.py`); per-language 50:50 label balance;
midfiller/endfiller diversity preserved; total on disk **~1.4 GB / 2 GB cap**.

**TTS Hinglish generation** (`scripts/generate_tts_hinglish.py`): 25 handwritten
script *pairs* (logistics/customer-care domain) with marker-based complete vs
incomplete derivation (`|` = final-clause cut, `^` = midfiller position), 4
Azure voices (hi-IN Swara/Madhur, en-IN Neerja/Prabhat), rate +8% to fit ≤8 s,
FLAC 16 kHz mono + 250 ms tail, 0 failures on the 100-clip pilot. This mirrors
the very method Pipecat used for 83% of their own data — applied to the language
pair they skipped.

**test-C construction** (`scripts/build_test_c.py`): MUCS test parquet → 846
clips; completes = full utterances + 200 ms trailing pad (repo convention);
incompletes = transcript-guided speech-active mid-word cuts (35–70% of
transcript, ≥3 words remaining). Two construction artifacts were caught and
fixed (valley cuts, tight-trimmed completes — see HARDPOINT).

**Input contract everywhere** (identical to Smart Turn, shared imported code):
16 kHz mono float32 [-1,1]; ≤8 s; shorter → zero-pad FRONT; longer → keep LAST 8 s.

## 4. Approach

**Architecture (`turn_v2/models/`):** whisper-tiny encoder (frozen in s1; last 2
blocks unfrozen in s2) → attention-mean pooling (384-d) → MLP head
(Linear→LN 256→GELU→Dropout 0.1→Linear 64→GELU→Linear 1). ~8M params,
~20 ms p50 e2e on CPU. Chosen for: audio-native pretrained prosody, CPU-scale
inference, and drop-in comparability with the reference's pooling/head.

**Two-stage transfer recipe** (user-directed pivot from single-stage mixed
training):

1. **Stage 1 — pretrain:** hin+eng minus dev (7,200 clips), frozen encoder,
   lr 5e-5, 4 ep → s1-001 (dev 0.851).
2. **Stage 2 — finetune:** 232 TTS Hinglish ×2 upsample + 50:50 hin/eng replay
   per epoch (836 clips/ep), **lr 1e-4, 3 ep, unfreeze last 2 encoder blocks**
   → **s2-004**.

Two locked-hyper failures shaped this recipe (both root-caused in HARDPOINT):
the original overfit gate stalled at lr 5e-5 (calibrated for full-encoder
training, not head-only) and s2-001 was flat at the locked lr 1e-5/2 ep. The
lesson: **head-only regimes need head-scaled learning rates**. Replay proved
load-bearing (0% replay costs −11.2 test-A acc — the s1 representation is not
retained without it).

**Discipline:** one change per row; every row reports all three test sets; seeds
fixed; the 8 s contract is shared imported code; `--overfit 100` gate must pass
before any full run; every run appended to `turn_v2/results.csv` (19-col schema).

## 5. Experiments

### 5.1 Headline (thesis) table — the crossover *is* the story

| Model | test-A acc/F1 | test-B acc/F1 | test-C acc/F1 |
|---|---|---|---|
| Smart Turn v3.2 zero-shot (int8) | **0.927 / 0.930** | 0.788 / 0.759 | 0.539 / 0.573 |
| s1-001 (stage 1 only) | 0.833 / 0.829 | 0.727 / 0.571 | 0.541 / 0.279 |
| **s2-004 (v2-core, ships)** | 0.882 / 0.886 | **0.970 / 0.968** | **0.600 / 0.446** |

The baseline wins on its own training distribution (test-A); our Hinglish
fine-tune wins on both Hinglish domains (+18.2 acc on test-B, +6.1 on test-C
where every neural model struggles). In-distribution benchmarks overstate
production readiness — this table is the quantified version of that claim.

Also notable: s2-004 rebalances the languages (hin 0.880 vs eng 0.883 — the
zero-shot gap was 5.5 pts) and fixes the baseline's FIR-heavy behavior on TTS
audio (FIR 0.167 → 0.056 on test-B).

### 5.2 Ablations (P4, one change per row; full table in `turn_v2/results.md`)

Every arm lost to the P3 default on the test-A guard and/or test-B primary:
ASP pooling (test-A 0.842 vs 0.882), attention-end (0.836), label smoothing 0.05
(0.857), unfreeze k=1/4/full (0.835/0.836/0.830), replay 0/0.25 (0.770/0.816).
**Verdict: clean negative ablation; s2-004 = v2-core unchanged.** ASP's expected
prosodic-variance gain does not materialize at whisper-tiny scale on this task.

### 5.3 Knowledge distillation (P5) — documented negative

whisper-small teacher (same recipe) beats the student everywhere (test-A 0.913,
test-B 1.000) but KD into the student (T=3, α=0.5) **hurt**: 0.837 vs the 0.846
no-KD control and 0.882 v2-core. Soft targets did not transfer at 8M params /
626-clip s2. Timebox honored; ship = s2-004.

### 5.4 Trailing-context curve (P6, `turn_v2/latency.py`)

Keep only the last b s of every clip (front-pad to 8 s):

| b | 0.5 s | 1 s | 2 s | 4 s | 8 s |
|---|---|---|---|---|---|
| v2-core test-A | 0.545 | 0.738 | 0.844 | **0.882** | 0.882 |
| v2-core test-B | 0.636 | 0.939 | **1.000** | 0.970 | 0.970 |

v2-core saturates at **~4 s (test-A) / 2 s (test-B)** — a streaming policy can
commit on 2–4 s windows. Under truncation the two models fail in *opposite*
directions: v2-core toward HOLD (0.905 hold at 0.5 s — goes silent, never
interrupts), zero-shot toward FIR (0.797 — responds on anything). The s2
fine-tune flipped the failure mode to the operationally safer side.

### 5.5 Policy sweep (P6, `turn_v2/policy.py`)

τ sweep on v2 probabilities: FIR = P(pred complete | incomplete) vs HOLD =
P(pred incomplete | complete) (added-latency proxy):

| Split | τ | FIR | HOLD | acc |
|---|---|---|---|---|
| test-A | 0.45 (knee) | 0.168 | 0.067 | 0.882 |
| test-A | **0.50 (default)** | 0.160 | 0.077 | 0.882 |
| test-A | 0.70 (FIR-averse) | 0.128 | 0.120 | 0.876 |
| test-B | 0.25–0.85 (plateau) | 0.056 | 0.000 | 0.970 |
| test-C | 0.15 (acc peak) | 0.228 | 0.486 | 0.639 |
| — ref @ 0.5 | A / B / C | 0.115 / 0.167 / 0.534 | 0.032 / 0.267 / 0.396 | 0.927 / 0.788 / 0.537 |

The frontier is monotone on test-A (each −0.01 FIR costs ~+0.015 HOLD); test-B
has a wide free plateau. **Shipped: τ=0.5** — single global threshold, matches
all stored rows; per-domain thresholds deferred until a conversational test-C
exists. This is the policy-layer maturity the field evaluates (probabilities →
thresholds → Pareto), not one accuracy number.

### 5.6 FP32 vs int8 ONNX (P5, `turn_v2/export.py`)

| Split | fp32 (30.9 MB) | int8 (9.05 MB) | Δacc | p50 ms |
|---|---|---|---|---|
| test-A | 0.882 / 0.886 | 0.857 / 0.858 | −2.5 | 17.7 → 13.2 |
| test-B | 0.970 / 0.968 | 0.939 / 0.933 | −3.0 (1 clip) | 17.3 → 13.3 |
| test-C | 0.600 / 0.446 | 0.604 / 0.455 | +0.4 | 16.9 → 12.9 |

fp32 ONNX is bit-faithful to PyTorch. int8 = static QDQ with **held-out**
calibration (200 dev clips — fixes the reference's calibrate-on-test flaw) +
the reference recipe (quant_pre_process, Conv/MatMul/Gemm only). The naive
all-ops route hit the AVX2 U8S8 cliff at −8.8; the residual −2.5 is the
exhaustive noise floor (minmax/reduce_range/u8u8/calib-1024 all identical).
Documented deviation from the ≤8 MB / ≤1% target.

## 6. Error analysis (P6, `results/error_analysis_failures.csv`)

83 categorized failures (all test-B fails; test-C balanced samples per failing
model, seed 42) with per-clip duration, trailing-silence (TEN VAD / energy),
and counterpart-model probabilities. Full write-up in `turn_v2/results.md` §P6;
headline findings:

1. **v2-core's test-C failures are conservative** (147/338 are completes read as
   incomplete; FIR 0.097 vs HOLD 0.687) — confidently wrong (median p 0.03–0.13),
   not borderline.
2. **Tail-cue reliance is real but bounded:** the biggest bucket is completes
   ending on the 0.2 s pad (the neural operating point needs >0.3 s tails), yet
   some v2-only completes with ≥0.59 s tails are still held → monologue prosody,
   not just the silence cue. Conversely, incompletes with 0.44 s median tails
   fooled both models.
3. **Error overlap quantifies the domain gap:** 190/338 v2 test-C failures are
   also zero-shot failures (56%) — the shared MUCS monologue gap; the 148
   v2-only corrections (44%) are the TTS-finetune transfer. Zero-shot fails in
   the *opposite* direction (eager completes).
4. **test-B:** v2's single failure is an endfiller filler-confusion; the
   zero-shot's 7 failures are mostly short-tail completes — the s2 fine-tune
   fixed exactly the FIR-heavy behavior.
5. **Slices:** midfiller=True remains the hard core (test-A acc 0.824 vs 0.919)
   with FIR 0.285 there — pause-vs-end disambiguation is where future gains live.

## 7. Limitations

1. **test-C is tutorial monologue**, not conversation: all neural models collapse
   there; the silence-threshold baseline trivially wins (0.921). We ship it as a
   stress test (relative rankings), not an absolute benchmark.
2. **test-B is 33 clips sharing the TTS pipeline** with s2 training data — one
   clip ≈ 3 acc pts. Directional, not definitive.
3. **Feature convention:** transformers 5 silently dropped `do_normalize`, so v2
   trains/infers on *unnormalized* mels — fully self-consistent, but the
   pretrained encoder's native input is normalized; retraining on normalized
   mels is an untested potential gain (HARDPOINT 2026-08-29).
4. **232 TTS clips is a probe, not a domain solution**; listening-gate QC was
   waived at the pivot (stats-QC admission only).
5. Non-streaming architecture: the 8 s contract re-featurizes per decision; the
   latency curve shows 2–4 s windows are viable but a streaming/cached encoder
   would be the production shape.
6. HF Space hosting is now PRO-gated for Gradio SDK (402) — the demo ships as a
   local app + public weights; Space upload is one command away
   (`scripts/upload_hf.py --space`).

## 8. Next steps

- **Human Hinglish data** (top lever): the corpus is 83% TTS / 1.6% human; even
  a few hundred human conversational Hinglish calls for s2 would de-risk the
  TTS-pipeline overfit and likely move test-C.
- **Conversational test-C v2** (real calls) to replace the monologue stress test
  and enable per-domain threshold shipping.
- **Streaming encoder** (cached blocks + incremental mel) for real-time VAD-loop
  integration; TEN VAD gate is already wired in the demo.
- **3-class output** (end-turn / hold-phrase / mid-turn) using the existing
  `midfiller`/`endfiller` metadata — the hard core is pause-vs-end, not speech
  vs silence.
- **Text conditioning** (code-switch density, transcript fillers) as a side
  channel where ASR is available.
- **QAT / distillation retry** at higher capacity or on normalized features.

## 9. Reproduce

```powershell
uv sync
uv run python scripts/download_data.py                # subset of v3.2 (streaming)
uv run python scripts/generate_tts_hinglish.py        # TTS Hinglish pilot
uv run python scripts/build_test_c.py                 # MUCS test-C subset
uv run python -m turn_v2.train --stage s1 --overfit 100
uv run python -m turn_v2.train --stage s1 --workers 4
uv run python -m turn_v2.train --stage s2 --init-ckpt turn_v2/ckpt/s1-001/best.pt --lr 1e-4 --epochs 3 --unfreeze-last-k 2 --workers 4   # s2-004
uv run python -m turn_v2.evaluate --ckpt turn_v2/ckpt/s2-004/best.pt --splits test_a test_b test_c
uv run python turn_v2/latency.py                      # P6 context curve
uv run python turn_v2/policy.py                       # P6 policy frontier
uv run python scripts/error_analysis.py               # P6 slices + failures
uv run python turn_v2/export.py --ckpt turn_v2/ckpt/s2-004/best.pt --eval --quant-config "entropy,quint8"
uv run python scripts/upload_hf.py                    # HF Hub upload
uv run python turn_v2/app.py                          # demo UI
```

Exact per-run provenance: `turn_v2/results.csv`; curated tables:
`turn_v2/results.md`; planning/decision history: `context/`.
