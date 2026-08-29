# Results — curated (report artifact)

> Machine log: `results.csv` (one row per run). Curated tables below.
> Test sets: **A** = v3.2-test hin+eng subset (1,200) · **B** = TTS Hinglish
> (33-clip held-out) · **C** = MUCS real speech (build = P3 parallel workstream).
> Dev = dev_v1 hin+eng (800) + dev_tts_v1 (23) for s2 selection.

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
