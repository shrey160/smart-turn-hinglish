# Results — curated (report artifact)

> Machine log: `results.csv` (one row per run). Curated tables below.
> Test sets: **A** = v3.2-test hin+eng subset (1,200) · **B** = TTS Hinglish
> (33-clip held-out) · **C** = MUCS real speech (build = P3 parallel workstream).
> Dev = dev_v1 hin+eng (800) + dev_tts_v1 (23) for s2 selection.

## P6 — Evaluation depth (2026-08-29, Session 10)

No retraining; all tables evaluate frozen v2-core (`s2-004`) vs zero-shot
reference (`smart-turn-v3.2-cpu.onnx`). Zero-shot evals feed the reference its
native normalized features (`TurnDataset(normalize=True)` — see HARDPOINT
2026-08-29 feature-convention entry); v2 keeps its training convention.

### WS-A — Trailing-context curve (`turn_v2/latency.py`)

Keep only the LAST b s of every clip (front zero-pad to 8 s per the shared
contract) → acc/F1 vs how much end-context the model actually sees.
CSV: `results/latency_curve.csv`.

**acc (F1)** by split × budget:

| Model | split | b=0.5 s | b=1 s | b=2 s | b=4 s | b=8 s |
|---|---|---|---|---|---|---|
| v2-core s2-004 | A | 0.545 (0.173) | 0.738 (0.681) | 0.844 (0.847) | **0.882 (0.885)** | **0.882 (0.886)** |
| v2-core s2-004 | B | 0.636 (0.333) | 0.939 (0.929) | **1.000 (1.000)** | 0.970 (0.968) | **0.970 (0.968)** |
| v2-core s2-004 | C | 0.487 (0.005) | 0.506 (0.150) | 0.543 (0.295) | 0.552 (0.334) | **0.600 (0.446)** |
| zero-shot ref | A | 0.598 (0.712) | 0.732 (0.786) | 0.855 (0.870) | 0.921 (0.924) | **0.927 (0.930)** |
| zero-shot ref | B | 0.606 (0.667) | 0.788 (0.774) | 0.727 (0.710) | 0.727 (0.667) | **0.788 (0.759)** |
| zero-shot ref | C | 0.600 (0.714) | **0.615 (0.705)** | 0.595 (0.661) | 0.543 (0.580) | 0.537 (0.572) |

(b=8 rows reproduce the stored thesis numbers exactly — anchor check ✓.)

**Findings:**
1. **v2-core saturates at ~4 s trailing context on test-A** (b=4 = b=8) and at
   **2 s on test-B** (1.000 — its single failure at b=8 disappears with less
   context). A demo/VAD policy can commit on ~2–4 s windows.
2. **Error direction under truncation is opposite:** v2-core fails toward HOLD
   (at b=0.5 s: HOLD 0.905, FIR 0.005 — goes silent, never interrupts);
   zero-shot fails toward FIR (FIR 0.797 at b=0.5 s — responds on anything).
   The TTS finetune flipped the failure mode to the operationally safer side.
3. test-C: v2 accuracy grows monotonically with context (0.487→0.600) while
   zero-shot peaks at b=1 and decays — more context does not fix the monologue
   domain gap.

### WS-B — Policy sweep (`turn_v2/policy.py`)

τ sweep on v2-core probabilities. FIR = P(pred complete | incomplete) = the
costly false-interrupt; HOLD = P(pred incomplete | complete) = added-latency
proxy (each held turn ≈ one extra VAD re-check). CSVs:
`results/policy_sweep_<split>.csv`, `results/policy_frontier_<split>.csv`.

| Split | τ | FIR | HOLD | acc | note |
|---|---|---|---|---|---|
| test-A | 0.45 (knee) | 0.168 | 0.067 | 0.882 | best balanced acc 0.883 |
| test-A | **0.50 (default)** | 0.160 | 0.077 | **0.882** | shipped operating point |
| test-A | 0.70 | 0.128 | 0.120 | 0.876 | −20% FIR for −0.6 acc |
| test-A | 0.90 | 0.068 | 0.273 | 0.829 | FIR floor, HOLD ×3.5 |
| test-B | 0.25–0.85 (plateau) | 0.056 | 0.000 | 0.970 | flat 1-clip FIR |
| test-B | 0.90 | 0.000 | 0.067 | 0.970 | zero FIR (1 hold) |
| test-C | 0.15 | 0.228 | 0.486 | **0.639** | acc peak (monologue domain) |
| test-C | 0.50 (default) | 0.097 | 0.687 | 0.600 | conservative direction |
| — ref @ τ=0.5 | A / B / C | 0.115 / 0.167 / 0.534 | 0.032 / 0.267 / 0.396 | 0.927 / 0.788 / 0.537 | reference operating point |

**Findings:**
- test-A's frontier is monotone (every τ non-dominated): each −0.01 FIR costs
  ~+0.015 HOLD. The reference dominates v2-core in-distribution (its home
  data); v2-core dominates it on test-B at every τ.
- test-B has a wide flat plateau (τ 0.25–0.85 identical) — threshold is a free
  parameter on the TTS domain.
- **Recommendation: keep τ=0.5 for the demo** (all stored numbers correspond to
  it; test-A knee is 0.45 — within noise). If false interrupts dominate UX
  cost, τ=0.7 is the defensible aggressive variant. test-C wants τ≈0.15–0.2,
  but per-domain thresholds need conversational test-C v2 before shipping.

### WS-C — Slices + error analysis (`scripts/error_analysis.py`)

Confusion slices (acc, F1, FIR/HOLD, TP/FP/FN/TN) for v2-core + zero-shot per
lang × midfiller × endfiller × test-set → `results/slices_v2core.csv` (32 rows).

v2-core highlights (full CSV has both models): test-A midfiller=True remains
the hard core (0.824 vs 0.919 midfiller=False) with FIR 0.285 on that slice;
hin/eng now balanced (0.880/0.883); test-B endfiller=True 16/17 correct.

Failures pulled → `results/error_analysis_failures.csv` (83 rows: all 7
test-B fails; test-C sampled 30 v2-only + 30 both-fail + 30 zs-only,
balanced per (label, pred) stratum, seed 42) with per-clip duration /
trailing-silence / counterpart-model p. Category summary:

| Failing model | Category | n | med p(pred) | med tail | med dur |
|---|---|---|---|---|---|
| v2 | tail-cue:complete-short-tail | 12 | 0.134 | 0.18 s | 4.75 s |
| v2 | domain-gap:monologue | 4 | 0.440 | 0.59 s | 4.41 s |
| v2+zs | tail-cue:complete-short-tail | 14 | 0.027 | 0.18 s | 5.25 s |
| v2+zs | tail-cue:incomplete-long-tail | 7 | 0.737 | 0.44 s | 2.66 s |
| v2+zs | domain-gap:monologue | 9 | 0.890 | 0.10 s | 2.97 s |
| v2+zs | filler-confusion | 1 | 0.866 | 0.08 s | 4.07 s |
| zs | domain-gap:monologue | 14 | 0.739 | 0.00 s | 2.93 s |
| zs | tail-cue:complete-short-tail | 17 | 0.296 | 0.18 s | 6.75 s |
| zs | tail-cue:incomplete-long-tail | 3 | 0.981 | 1.50 s | 1.84 s |
| zs | filler-confusion / other | 2 | 0.559–0.607 | — | ~4 s |

**Write-up:**
1. **v2-core's dominant test-C failure is "complete read as incomplete"**
   (147/338 v2 fails are completes-held; FIR 0.097 vs HOLD 0.687) — the
   conservative direction (fewer agent interruptions). Median p on those fails
   is 0.03–0.13: confidently wrong, not borderline.
2. **Tail-cue reliance is real but bounded:** completes ending with the 0.2 s
   pad (med tail 0.18 s) are the biggest bucket — the neural operating point
   needs >0.3 s tails (Session 7). But 4 v2-only completes with **≥0.59 s
   median tail** are still held → prosody/domain remains, silence alone would
   not fix them. Conversely 7 incompletes with 0.44 s median tails fooled both
   models into "complete".
3. **Error overlap quantifies the domain gap:** 190/338 v2 test-C fails are
   also zero-shot fails (56%). The shared errors are the MUCS monologue gap;
   the 148 v2-only corrections (44%) are the TTS-Hinglish finetune transfer.
4. **Zero-shot fails in the opposite direction on test-C** (FIR 0.534, med p
   0.74–0.98 on its FP rows — eagerly predicts complete). v2 and ref fail
   toward different sides; only 7/338 clips fail in both directions.
5. test-B: v2's single failure is an endfiller filler-confusion; zero-shot's 7
   failures are mostly completes with short tails (tail-cue) — i.e., the s2
   finetune fixed exactly the FIR-heavy behavior zero-shot shows on TTS audio.


## test-C (MUCS real speech, v1 — 2026-08-29, Session 7)

**Construction** (`scripts/build_test_c.py`, from `dianavdavidson/MUCS-Hinglish`
test parquet #0, deleted after extraction): 846 clips / 76.8 min / 17 speakers /
all 16 kHz. complete = 434 full utterances + 200 ms trailing-silence pad (repo
convention; MUCS trims segments to 0.000 s median tail, unbuildable without the
pad). incomplete = 412 mid-speech cuts: cut word index in [35%, 70%] of the
transcript (≥3 words remain), snapped to a speech-active 20 ms frame ±0.35 s.
eng_ratio filter 0.02–0.70; clips 1.5–8 s; speaker cap 40; seed 42.
Manifest: `data/splits/test_c_manifest.csv`.

| Model | test-C acc | test-C F1 |
|---|---|---|
| Silence-threshold (T=0.15 s) | **0.921** | **0.928** |
| **s2-004** | **0.600** | 0.446 |
| s1-001 | 0.541 | 0.279 |
| Smart Turn v3.2 int8 (zero-shot) | 0.539 | **0.573** |

**Findings (all three neural models collapse; two verified causes):**
1. Trailing-silence cue structure differs: on test-A completes have median
   0.18 s tail (25% zero) so models learned prosody-first detection; MUCS
   completes are tight-trimmed (median 0.000 s) and only the 200 ms pad gives
   the cue — which the energy threshold trivially exploits (0.921) but the
   neural models' operating point needs >0.3 s tails (zs: 0.956 acc on ≥0.3 s
   bin vs 0.497 on 0.1–0.3 s bin).
2. Domain gap is real: zero-shot 0.927 (test-A) → 0.539 (test-C). Tutorial
   monologue prosody (no conversational turn-taking) reads as non-final to
   every v3.2-trained model.
3. **s2-004 is the best neural model on test-C (0.600, +6.1 acc over
   zero-shot)** — the TTS Hinglish finetune transferred out of domain. F1 is
   low because it over-predicts incomplete (conservative: fewer false
   agent interruptions).
4. Caveat: test-C v1 over-weights the trailing-silence cue relative to test-A
   (energy baseline beats all neural models) — treat absolute numbers as
   pessimistic for neural models; relative model ranking is the usable signal.

## P5 — Distillation + deployment (2026-08-29, Session 9)

### KD verdict: NEGATIVE — ship s2-004 (timebox rule applied)

Teacher = whisper-small encoder (87.7M params), same two-stage recipe.
GPU note: torch 2.13.0+cu126 swapped in mid-phase — teacher s1 went from a
budgeted ~1.5 h (CPU) to ~5.5 min.

| Run | Config | dev F1 | test-A acc | test-B acc | test-C acc |
|---|---|---|---|---|---|
| s1-004 | teacher s1 (whisper-small, frozen) | 0.913 | 0.902 | 0.697 | 0.508 |
| s2-013 | teacher s2 (approved recipe) | 0.930 | 0.913 | 1.000 | 0.603 |
| **s2-004** | **v2-core (no KD)** | **0.884** | **0.882** | 0.970 | **0.600** |
| s2-014 | KD from s2-013 logits (T=3, α=0.5) | 0.862 | 0.837 | 0.970 | 0.590 |
| s2-015 | no-KD control, same harness | 0.875 | 0.846 | 0.970 | 0.580 |

**Verdict: KD fails the phase bar** (test-B parity yes, but −0.9 test-A vs the
no-KD control and −4.5 vs v2-core). Teacher beats the student everywhere, yet
the soft targets did not help at whisper-tiny capacity with 626 clips/epoch —
logged as a clean negative result. `s2-004` remains the ship candidate.

Harness note: s2-014/s2-015 use the Session-8 harness (replay 626/epoch) vs
s2-004's P3 run (836/epoch) — the control isolates KD from that difference;
neither harness value reaches s2-004, reinforcing the P3 default.

### ONNX int8 export (WS-B)

fp32 ONNX (opset 18, dynamic batch) is **bit-faithful** to PyTorch
(0.882/0.970/0.600 on A/B/C). int8 static QDQ, per-channel weights, held-out
calibration (200 dev_v1 clips — fixes the reference's calibrate-on-test flaw).

| int8 recipe (test-A) | acc | Δacc vs fp32 | notes |
|---|---|---|---|
| naive: all-ops QDQ, entropy | 0.793 | −8.8 | AVX2 U8S8 cliff (LayerNorm/act quantized) |
| + reduce_range | 0.793 | −2.5→0 change | no effect on this graph |
| activations QInt8 (symmetric) | 0.574 | −30.8 | catastrophic |
| **ref recipe: quant_pre_process + Conv/MatMul/Gemm only, Entropy, calib 200** | **0.857** | **−2.5** | ✅ shipped |
| ref recipe, calib 1024 | 0.848 | −3.4 | more calibration ≠ better |
| ref recipe + MinMax / + reduce_range / U8U8 | 0.855–0.857 | −2.5…−2.7 | converged — noise floor |

Final artifacts (`turn_v2/onnx/`): `s2-004.fp32.onnx` (30.9 MiB) +
`s2-004.int8.entropy-quint8.onnx` (9.05 MB).

| Split | fp32 acc/F1 | int8 acc/F1 | Δacc | p50 fp32 → int8 |
|---|---|---|---|---|
| test-A | 0.882/0.886 | 0.857/0.858 | −2.5 | 17.7 → 13.2 ms |
| test-B | 0.970/0.968 | 0.939/0.933 | −3.0 (=1 clip/33) | 17.3 → 13.3 ms |
| test-C | 0.600/0.446 | 0.604/0.455 | +0.4 | 16.9 → 12.9 ms |

**Exit criterion deviation, documented:** masterplan P5 target was int8 ≤ 8 MB
with delta ≤ ~1%. Shipped: 9.05 MB, test-A −2.5 / test-B −3.0 (1 clip) /
test-C +0.4. The −8.8 cliff was fixed by replicating the reference recipe
(quant_pre_process + quantize Conv/MatMul/Gemm only); the residual −2.5 is
the noise floor across every remaining knob (rr/minmax/u8u8/calib-size all
identical). QAT listed as future work. For the demo, fp32 ONNX is the
zero-degradation option (31 MB); int8 (8.6 MiB) is the tiny-footprint option.

## P4 ablation table (2026-08-29, Session 8)

All s2-only arms init `ckpt/s1-001/best.pt`, approved recipe (lr 1e-4, 3 ep)
unless noted; arch arms = full s1+s2 rerun. One change per row vs **s2-004**
(attention-mean, k=2, replay 0.5, no smoothing) = the P3 default.

| Run | Arm (single change) | dev F1 | test-A acc | test-B acc | test-C acc |
|---|---|---|---|---|---|
| **s2-004** | **(default — v2-core)** | **0.884** | **0.882** | 0.970 | **0.600** |
| s2-005 | ARM-5: replay-frac 0 | 0.815 | 0.770 | 0.970 | 0.553 |
| s2-006 | ARM-5: replay-frac 0.25 | 0.840 | 0.816 | 0.970 | 0.599 |
| s2-007 | ARM-3: label smoothing 0.05 | 0.876 | 0.857 | 0.970 | 0.597 |
| s2-008 | ARM-4: k=1 | 0.853 | 0.835 | 0.970 | 0.600 |
| s2-009 | ARM-4: k=4 | 0.850 | 0.836 | **1.000** | 0.582 |
| s2-010 | ARM-4: full unfreeze | 0.847 | 0.830 | **1.000** | 0.579 |
| s1-002 / s2-011 | ARM-1: ASP pooling (full rerun) | 0.868 | 0.842 | 0.909 | 0.455 |
| s1-003 / s2-012 | ARM-2: attention-end (full rerun) | 0.866 | 0.836 | 0.970 | 0.539 |

**Verdict: no arm beats the P3 default — s2-004 is v2-core.**
- Replay is load-bearing (0% replay costs −11.2 test-A acc; the s1 pretrain
  representation is not retained without it).
- k=2 is the unfreeze sweet spot (k=1 underfits, k=4/full trade test-A for a
  1-clip test-B gain, 32/33 → 33/33).
- ASP/attention-end do not help here (−4.0/−4.6 test-A) and ASP adds ~1.6 ms
  latency (21.8 vs 20.2 ms p50); playbook's expected prosodic-variance gain
  does not materialize at whisper-tiny scale on this task.
- test-B is 33 clips: 0.970 vs 1.000 = one clip — treat as noise.
- ARM-2's s1 stage (s1-003) had the best test-C of any s1 (0.512) — end-bias
  may help domain robustness pre-finetune; noted for future work, not selected.

## Thesis table v2 (P3 two-stage reproduction, 2026-08-29)

| Model | dev F1 | test-A acc | test-A F1 | test-B acc | test-B F1 | p50 ms e2e (CPU) |
|---|---|---|---|---|---|---|
| Silence-threshold (T=0.05 s*) | — | 0.537 | 0.560 | 0.485 | 0.622 | ~0 |
| Smart Turn v3.2 int8 (zero-shot) | — | **0.927** | **0.930** | 0.788 | 0.759 | 15 |
| s1-001: whisper-tiny frozen, attn-mean head (hin+eng 7,200) | 0.851 | 0.833 | 0.829 | 0.727 | 0.571 | 20.1 |
| s2-001: + TTS finetune @ locked lr 1e-5 | 0.849 | 0.842 | 0.846 | 0.727 | 0.571 | 20.4 |
| s2-002: s2 lr 1e-4 | 0.845 | 0.834 | 0.843 | 0.758 | 0.692 | 19.5 |
| s2-003: s2 lr 1e-4, 3 ep | 0.845 | 0.835 | 0.844 | 0.788 | 0.741 | 19.4 |
| **s2-004: s2 lr 1e-4, 3 ep, unfreeze last 2 enc blocks** | **0.884** | 0.882 | 0.886 | **0.970** | **0.968** | 20.2 |

\* best F1 on the T sweep; acc-best T=0.20 s → test-A acc 0.578 / F1 0.534. Near-chance either way.

**P3 exit: MET by s2-004** — beats zero-shot on test-B (0.970/0.968 vs 0.788/0.759)
AND test-A improves over ckpt_s1 (+4.9 acc, no regression). All s2 rows init from
`s1-001`; s2 data = 209 TTS ×2 + 50:50 hin/eng replay (836/epoch).

**Caveats:** test-B is 33 clips (1 clip ≈ 3 acc pts) and shares the TTS generation
pipeline with s2 training data — real-speech validation waits on test-C (MUCS).
Head-only rows (s1/s2-001..003) show the frozen-encoder lr handicap: locked lrs
(5e-5 s1 / 1e-5 s2) assume full-encoder training; see HARDPOINT 2026-08-29.

## s2-004 slices (test-A, n=1,200)

| Slice | n | acc | F1 |
|---|---|---|---|
| eng | 600 | 0.883 | 0.883 |
| hin | 600 | 0.880 | 0.889 |
| midfiller=False | 729 | 0.919 | 0.920 |
| midfiller=True | 471 | 0.824 | 0.839 |

Zero-shot reference slices: eng 0.955 / hin 0.900 / midfiller=True 0.858 —
s2-004 trades some eng for hin balance (hin now ±0.003 of eng vs −5.5 before),
midfiller hard core still the weak slice.

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
uv run python scripts/eval_baselines.py --splits test_a test_b   # P1 baselines -> results/
uv run python -m turn_v2.train --stage s1 --overfit 100          # sanity gate (must PASS)
uv run python -m turn_v2.train --stage s1 --workers 4            # s1-001 (CPU) / s1-004 teacher: --base openai/whisper-small
uv run python -m turn_v2.train --stage s2 --init-ckpt turn_v2/ckpt/s1-001/best.pt --lr 1e-4 --epochs 3 --unfreeze-last-k 2 --workers 4   # s2-004
uv run python -m turn_v2.evaluate --ckpt turn_v2/ckpt/s2-004/best.pt --splits test_a test_b
# P5:
uv run python -m turn_v2.train --stage s2 --init-ckpt turn_v2/ckpt/s1-001/best.pt --unfreeze-last-k 2 --lr 1e-4 --epochs 3 --kd-teacher turn_v2/kd/teacher_logits.npz --kd-temp 3 --kd-alpha 0.5 --workers 4   # s2-014 (negative)
uv run python scripts/precompute_teacher_logits.py --ckpt turn_v2/ckpt/s2-013/best.pt
uv run python turn_v2/export.py --ckpt turn_v2/ckpt/s2-004/best.pt --eval --calib-n 200 --quant-config "entropy,quint8"   # final fp32+int8
uv run python scripts/eval_onnx.py --split test_a --paths turn_v2/onnx/s2-004.fp32.onnx turn_v2/onnx/s2-004.int8.entropy-quint8.onnx
# P6:
uv run python turn_v2/latency.py                                                  # trailing-context curve -> results/latency_curve.csv
uv run python turn_v2/policy.py                                                   # tau sweep + Pareto frontier -> results/policy_*.csv
uv run python scripts/error_analysis.py                                           # slices + categorized failures -> results/slices_v2core.csv, error_analysis_failures.csv
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
