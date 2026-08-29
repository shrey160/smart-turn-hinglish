# Phase 1 — P0: Environment & Baseline Weights

> First phase plan (maps to masterplan §5 P0). Follows playbook Day 1.
> Progress trace: completed phases stay in `context/phases/`.

## Objective

Get the reference baseline running end-to-end locally and build the human
listening log — the audit foundation every later phase depends on.

## Tasks (from masterplan P0)

1. Install heavy ML deps: `transformers torchaudio onnx onnxruntime edge-tts`
   (scikit-learn + torch already present).
2. Download Smart Turn v3 int8 ONNX from `huggingface.co/pipecat-ai/smart-turn-v3`
   into `models_ref/` (gitignored; hardlink at repo root so
   `smart-turn/inference.py`'s CWD-relative `ONNX_MODEL_PATH` resolves).
3. Run zero-shot baseline inference on a few local clips (complete + incomplete,
   hin / eng / TTS example) via `smart-turn/predict.py` semantics.
4. Listening log: sample 50+ clips from train_pool (hin first) with audio stats
   (duration, RMS, tail silence) → `context/listening_log.md` template for
   human annotation (naturalness, code-mix, filler quality notes).

## Exit criteria (masterplan P0)

- [ ] Zero-shot prediction works locally on real clips
- [ ] `context/listening_log.md` exists with 50+ sampled clips ready to annotate

## Notes / constraints

- `smart-turn/` stays read-only; import from it, never modify.
- Never bulk-download the 41 GB train set (already respected — pool is on disk).
- If the HF repo filename differs from `smart-turn-v3.1.onnx` (expected by
  `smart-turn/inference.py:7`), adapt via hardlink name, not code edits.
