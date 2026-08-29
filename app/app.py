"""Gradio demo — Hinglish turn detection (v2-core s2-004, ONNX).

Mic or file upload -> 16 kHz mono float32 -> shared 8 s contract (keep LAST
context seconds, zero-pad front) -> whisper log-mel (80, 800) WITHOUT per-bin
normalization (v2 training convention — see REPORT.md §limitations) -> ONNX
-> verdict + p(complete) + inference ms.

Extras:
- Model selector: fp32 ONNX (bit-faithful) or int8 ONNX (9 MB, -2.5 test-A).
- Decision threshold tau (default 0.5 per the P6 policy sweep; higher = fewer
  false interrupts at the cost of longer holds).
- Trailing-context slider (0.5–8 s): simulated streaming — how early would the
  detector fire with only the last b seconds of audio (P6 latency curve)?
- TEN VAD trailing-silence readout (the production VAD gate; falls back to an
  energy floor if ten_vad is unavailable).

Self-contained on purpose: runs from the repo OR as an HF Space — model files
resolve locally first, else download from the HF Hub repo.

Local run:
  uv run python turn_v2/app.py                # UI on :7860
  uv run python turn_v2/app.py --selftest     # headless check on test-B clips
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "Shrey160/hinglish-turn-v2"
MODEL_FILES = {
    "fp32 ONNX (bit-faithful)": "onnx/s2-004.fp32.onnx",
    "int8 ONNX (9 MB)": "onnx/s2-004.int8.entropy-quint8.onnx",
}
LOCAL_DIRS = [ROOT / "turn_v2" / "onnx", ROOT / "models_ref"]
SAMPLE_RATE = 16000
ENERGY_FLOOR = 0.01  # fallback tail-silence floor (same as eval_baselines.py)

_SESSIONS = {}
_FE = None
_TEN = None


def feature_extractor():
    global _FE
    if _FE is None:
        from transformers import WhisperFeatureExtractor

        _FE = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", chunk_length=8)
    return _FE


def resolve_model(which):
    """Local file if present, else download from the Hub (cached)."""
    fname = MODEL_FILES[which]
    for d in LOCAL_DIRS:
        p = d / Path(fname).name
        if p.exists():
            return str(p)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=REPO_ID, filename=fname)


def get_session(which):
    if which not in _SESSIONS:
        path = resolve_model(which)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _SESSIONS[which] = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    return _SESSIONS[which]


def to_16k_mono(audio, sr):
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) if np.issubdtype(audio.dtype, np.floating) else audio.astype(np.float32) / 32768.0
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def ten_vad_tail_silence(audio_f32):
    """Trailing silence via TEN VAD (production VAD gate); None if unavailable."""
    global _TEN
    try:
        if _TEN is None:
            from ten_vad import TenVad

            _TEN = TenVad(hop_size=256, threshold=0.5)
        x16 = (np.clip(audio_f32, -1, 1) * 32767).astype(np.int16)
        n = (len(x16) // 256) * 256
        tail = 0
        for i in range(n - 256, -1, -256):
            prob, _flag = _TEN.process(x16[i : i + 256])
            if prob > 0.5:
                break
            tail += 1
        return round(tail * 256 / SAMPLE_RATE, 3)
    except Exception:
        return None


def energy_tail_silence(audio_f32):
    """Fallback trailing-silence via 30 ms frame RMS floor (same as eval_baselines)."""
    win, hop = 480, 320
    if len(audio_f32) < win:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(audio_f32, win)[::hop]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    tail = 0
    for r in rms[::-1]:
        if r < ENERGY_FLOOR:
            tail += 1
        else:
            break
    return round(tail * hop / SAMPLE_RATE, 3)


def predict(audio, model_choice, tau, context_s):
    if audio is None:
        return "—", 0.0, 0.0, "—", "upload or record audio"
    sr, data = audio
    audio_f32 = to_16k_mono(np.asarray(data), int(sr))
    if len(audio_f32) < SAMPLE_RATE // 10:
        return "—", 0.0, 0.0, "—", "clip too short (<0.1 s)"

    b = float(context_s)
    n = int(round(b * SAMPLE_RATE))
    ctx_audio = audio_f32[-n:] if n > 0 else audio_f32  # keep LAST b s (simulated streaming)

    fe = feature_extractor()
    feats = fe(
        ctx_audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=8 * SAMPLE_RATE,
        truncation=True,
        do_normalize=False,  # v2 convention (unnormalized mels) — do not change
    )["input_features"].astype(np.float32)

    sess = get_session(model_choice)
    t0 = time.perf_counter()
    out = sess.run(None, {sess.get_inputs()[0].name: feats})[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    logit = float(np.asarray(out).reshape(-1)[0])
    p = 1.0 / (1.0 + np.exp(-logit))  # our export outputs logits (unlike the reference's sigmoid)
    verdict = "COMPLETE — agent can respond" if p >= tau else "INCOMPLETE — keep listening"
    tail_ten = ten_vad_tail_silence(audio_f32)
    tail = energy_tail_silence(audio_f32) if tail_ten is None else tail_ten
    vad_label = "TEN VAD" if tail_ten is not None else "energy floor"
    note = (
        f"p(complete)={p:.3f} vs τ={tau:.2f} · trailing silence {tail:.2f} s ({vad_label}) · "
        f"last {b:.1f} s of audio shown to the model"
    )
    return verdict, round(p, 4), round(infer_ms, 2), f"{tail:.2f} s", note


def run_selftest():
    clips = sorted((ROOT / "data" / "example_hinglish_fixed" / "test_b" / "hinglish").glob("*/*.flac")) if (ROOT / "data").exists() else []
    if not clips:
        print("no local test_b clips — selftest needs the repo data folder")
        return
    complete = [c for c in clips if c.parent.name.startswith("complete")]
    incomplete = [c for c in clips if c.parent.name.startswith("incomplete")]
    import soundfile as sf

    # s2-004 has exactly one documented test-B false interrupt (FIR=1/18, P6
    # policy sweep) — the selftest tolerates <=1 miss among sampled incompletes.
    group_results = []
    for group, want, allow in ((complete[:4], 1, 0), (incomplete[:4], 0, 1)):
        misses = 0
        for c in group:
            audio, sr = sf.read(c, dtype="float32")
            verdict, p, ms, tail, _ = predict((sr, audio), list(MODEL_FILES)[0], 0.5, 8.0)
            pred = 1 if p >= 0.5 else 0
            miss = pred != want
            misses += miss
            print(f"  {c.parent.name:22s} p={p:.4f} want={want} -> {'PASS' if not miss else 'MISS'} ({ms:.1f} ms)")
        group_results.append(misses <= allow)
    ok = all(group_results)
    print("SELFTEST", "PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--server-port", type=int, default=7860)
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    import gradio as gr

    examples = []
    ex_dir = ROOT / "data" / "example_hinglish_fixed" / "test_b" / "hinglish"
    if ex_dir.exists():
        for sub in ("complete-False-False", "incomplete-False-False", "incomplete-True-False"):
            files = sorted((ex_dir / sub).glob("*.flac"))[:1]
            examples.extend([str(f) for f in files])

    with gr.Blocks(title="Hinglish Turn Detection") as demo:
        gr.Markdown(
            f"## Hinglish Turn Detection — v2 (`s2-004`)\n"
            f"Whisper-tiny encoder + attention head, two-stage trained for Hinglish "
            f"(Hindi–English code-mixed) telephony audio. Binary verdict: is the user "
            f"done speaking (agent can respond) or just pausing (keep listening)?\n\n"
            f"Model files: [`{REPO_ID}`](https://huggingface.co/{REPO_ID}) — fp32 (bit-faithful) "
            f"or int8 (9 MB). Input contract: ≤8 s, shorter zero-pad FRONT, longer keep LAST 8 s."
        )
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(label="Audio (mic or upload)", type="numpy")
                model_dd = gr.Dropdown(list(MODEL_FILES), value=list(MODEL_FILES)[0], label="Model")
                tau_sl = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Decision threshold τ (higher = fewer false interrupts)")
                ctx_sl = gr.Slider(0.5, 8.0, value=8.0, step=0.5, label="Trailing context (s) — simulated streaming: keep only the last b s")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                verdict_out = gr.Textbox(label="Verdict")
                prob_out = gr.Number(label="p(complete)", precision=4)
                ms_out = gr.Number(label="Inference ms (ONNX fwd)", precision=2)
                tail_out = gr.Textbox(label="Trailing silence")
                note_out = gr.Textbox(label="Detail")
        btn.click(predict, [audio_in, model_dd, tau_sl, ctx_sl], [verdict_out, prob_out, ms_out, tail_out, note_out])
        audio_in.change(predict, [audio_in, model_dd, tau_sl, ctx_sl], [verdict_out, prob_out, ms_out, tail_out, note_out])
        if examples:
            gr.Examples(examples, inputs=[audio_in])

    demo.launch(server_port=args.server_port)


if __name__ == "__main__":
    main()
