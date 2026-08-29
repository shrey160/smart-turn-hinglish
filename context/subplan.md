# Subplan — Phase 6 (P5: Distillation + deployment)

> Previous phases: P0 ✅ · P1 ✅ · P2 ✅ · P3 ✅ (two-stage repro + test-C) ·
> P4 ✅ (ablations; v2-core = s2-004 unchanged).
> **P5 ✅ (Session 9) — KD negative (ship s2-004 per timebox); int8 exported
> with documented −2.5/−3.0/+0.4 delta; TEN VAD deferred to P7.**

## SP1 — WS-A teacher track
- [x] `--base` + KD flags in train.py (`--kd-teacher/--kd-alpha/--kd-temp`,
      soft-BCE KD loss, logits npz loader, base threaded to ckpt)
- [x] `scripts/precompute_teacher_logits.py` (s2-pool clean-audio logits → npz)
- [x] Teacher s1: whisper-small, hin+eng 7,200, lr 5e-5, 4 ep → `s1-004`
      (dev 0.913 / test-A 0.902/0.904 — CUDA torch made this ~5.5 min)
- [x] Teacher s2: approved recipe from `s1-004` → `s2-013`
      (test-A 0.913 / test-B 1.000 / test-C 0.603)
- [x] Precompute teacher logits → `turn_v2/kd/teacher_logits.npz` (7,409 clips)
- [x] Student KD s2: init ckpt_s1-001 + teacher logits (T=3, α=0.5) → `s2-014`
- [x] KD verdict vs s2-004 on test-A/B/C: **NEGATIVE** (control s2-015 isolates
      the harness confound; KD −0.9 test-A, test-B tie) → ship s2-004

## SP2 — WS-B export + int8
- [x] `turn_v2/export.py`: ONNX fp32 export (opset 18, dynamic batch; bit-faithful)
- [x] int8 static QDQ with held-out dev calibration (200 dev_v1 clips) +
      `quant_pre_process` + Conv/MatMul/Gemm-only (reference recipe; fixed the
      −8.8 AVX2 cliff)
- [x] fp32 vs int8 eval on test-A/B/C: −2.5 / −3.0 (1 clip) / +0.4 acc;
      9.05 MB int8; p50 17.7→13.2 ms
- [x] AVX2 U8S8 cliff: found and fixed (rr no-op here; qint8-act catastrophic;
      minmax/u8u8/calib-1024 = no gain — noise floor, documented)

## SP3 — Docs + phase close
- [x] results.md P5 section (KD verdict table + quantization table + deviation note)
- [x] results.csv rows s1-004/s2-013/s2-014/s2-015; progress.md Session 9
- [x] masterplan P5 checkboxes + exit-criteria deviation; HARDPOINT int8 entry

## Exit
- KD verdict recorded: **negative → ship s2-004** (timebox rule).
- int8 ≤ 8 MB **missed narrowly** (9.05 MB) with delta ≤ ~1% **missed**
  (−2.5 test-A) — deviation documented; artifacts + docs current.
