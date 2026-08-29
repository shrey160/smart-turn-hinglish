"""P7 WS-B: upload v2-core artifacts to the HF Hub (model repo + optional Space).

Uploads to model repo ``Shrey160/hinglish-turn-v2``:
  pytorch_model.pt                       (s2-004 torch ckpt, fp32)
  onnx/s2-004.fp32.onnx                  (bit-faithful, opset 18, dynamic batch)
  onnx/s2-004.int8.entropy-quint8.onnx   (9.05 MB QDQ, reference recipe)
  preprocessor_config.json               (whisper-tiny FE, chunk_length=8)
  README.md                              (model card)

Optionally creates the gradio Space ``Shrey160/hinglish-turn-demo`` with the
self-contained app.py. The token is loaded from .env in-process and never
printed.

Usage:
  uv run python scripts/upload_hf.py --dry-run
  uv run python scripts/upload_hf.py
  uv run python scripts/upload_hf.py --space
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO_ID = "Shrey160/hinglish-turn-v2"
SPACE_ID = "Shrey160/hinglish-turn-demo"

ARTIFACTS = [
    (ROOT / "turn_v2" / "ckpt" / "s2-004" / "best.pt", "pytorch_model.pt"),
    (ROOT / "turn_v2" / "onnx" / "s2-004.fp32.onnx", "onnx/s2-004.fp32.onnx"),
    (ROOT / "turn_v2" / "onnx" / "s2-004.int8.entropy-quint8.onnx", "onnx/s2-004.int8.entropy-quint8.onnx"),
]

MODEL_CARD = """---
license: mit
language:
- hi
- en
tags:
- audio
- turn-detection
- endpointing
- hinglish
- whisper
- onnx
library_name: onnx
pipeline_tag: audio-classification
---

# Hinglish Turn Detection — v2 (s2-004)

Tiny audio-native turn detector for **Hinglish (Hindi–English code-mixed)
telephony speech**: given up to 8 s of audio, is the user done speaking
(**complete** → agent can respond) or just pausing (**incomplete** → keep
listening)? Two-stage fine-tune of the [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3.2) approach for
the language pair it skipped.

## Results

| Model | test-A (v3.2-test hin+eng, 1,200) | test-B (TTS Hinglish, 33) | test-C (MUCS real speech, 846) |
|---|---|---|---|
| Smart Turn v3.2 zero-shot | **0.927** / **0.930** | 0.788 / 0.759 | 0.539 / 0.573 |
| **v2 s2-004 (this model)** | 0.882 / 0.886 | **0.970** / **0.968** | **0.600** / 0.446 |

(acc / F1). The intended story: the baseline wins on its own training
distribution; the Hinglish fine-tune wins on Hinglish domains.
int8 variant: −2.5 / −3.0 (1 clip) / +0.4 acc, 9.05 MB, p50 ~13 ms CPU.
Policy (P6 sweep): τ=0.5 default; knee τ≈0.45; test-B flat τ 0.25–0.85.
Trailing context: accuracy saturates at ~4 s (test-A) / 2 s (test-B).

## Input contract (identical to Smart Turn)

- 16 kHz mono float32 in [-1, 1]
- ≤ 8 s: shorter → zero-pad FRONT; longer → keep the LAST 8 s
- **Feature convention:** whisper log-mel (80, 800) via `chunk_length=8`
  **WITHOUT per-bin normalization** (`do_normalize=False`). The v2 models were
  trained and exported self-consistently on unnormalized mels (transformers 5
  silently dropped the FE default). Do NOT enable normalization at inference.
- Output: 1 logit — apply sigmoid; ≥ 0.5 → complete (at the shipped τ=0.5)

```python
import numpy as np, onnxruntime as ort, soundfile as sf
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", chunk_length=8)
sess = ort.InferenceSession(hf_hub_download("Shrey160/hinglish-turn-v2", "onnx/s2-004.fp32.onnx"))

audio, sr = sf.read("clip.wav", dtype="float32")   # resample to 16 kHz mono first
feats = fe(audio, sampling_rate=16000, return_tensors="np", padding="max_length",
           max_length=8 * 16000, truncation=True, do_normalize=False)["input_features"]
logit = sess.run(None, {sess.get_inputs()[0].name: feats.astype(np.float32)})[0].reshape(-1)[0]
p_complete = 1 / (1 + np.exp(-logit))
```

## Training (two-stage transfer, whisper-tiny encoder)

1. **Stage 1** — frozen encoder, head trained on original Smart Turn
   hin+eng (7,200 clips), lr 5e-5, 4 epochs.
2. **Stage 2** — 232 TTS Hinglish clips (×2 upsample) + 50:50 hin/eng replay,
   lr 1e-4, 3 epochs, last 2 encoder blocks unfrozen (approved amendment).
   Ablations (ASP pooling, attention-end, label smoothing, unfreeze k, replay
   ratio) and KD (whisper-small teacher) all lost to this default or shipped
   as documented negatives.

## Limitations

- test-C is MUCS *tutorial monologue* — all neural models collapse there
  (prosody domain gap); treat as stress test, relative rankings only.
- test-B is 33 clips sharing the TTS pipeline with s2 training data.
- mels are unnormalized (see above); normalized retraining = future work.
- 83% of the v3.2 training pool is TTS-generated; human Hinglish data is the
  top scaling lever.
"""

SPACE_CARD = """---
title: Hinglish Turn Detection
emoji: 🎧
colorFrom: indigo
colorTo: pink
sdk: gradio
app_file: app.py
pinned: true
license: mit
---

Hinglish (Hindi–English code-mixed) turn detection: is the user done speaking
or just pausing? Two-stage whisper-tiny model (v2 s2-004). Upload or record
audio, pick fp32/int8, tune τ and the trailing-context budget to simulate
streaming. Weights: https://huggingface.co/{REPO_ID}
"""

SPACE_REQUIREMENTS = """onnxruntime
transformers
huggingface_hub
numpy
librosa
soundfile
ten-vad
"""


def get_api():
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("no HF token in environment (.env: HUGGINGFACE_API_KEY)")
    from huggingface_hub import HfApi

    return HfApi(token=token)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--space", action="store_true", help="also create/update the gradio Space")
    args = ap.parse_args()

    api = get_api()
    user = api.whoami().get("name")
    print(f"uploading as: {user}")

    for local, remote in ARTIFACTS:
        assert local.exists(), f"missing artifact: {local}"
        size = local.stat().st_size / 1e6
        print(f"  {remote:42s} <- {local.name} ({size:.2f} MB)")

    with tempfile.TemporaryDirectory() as td:
        from transformers import WhisperFeatureExtractor

        fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", chunk_length=8)
        fe.save_pretrained(td)
        fe_file = Path(td) / "preprocessor_config.json"
        print(f"  {'preprocessor_config.json':42s} <- whisper-tiny FE (chunk_length=8)")

        if args.dry_run:
            print("dry run — nothing uploaded")
            return

        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)
        for local, remote in ARTIFACTS:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=REPO_ID, repo_type="model")
            print(f"  uploaded {remote}")
        api.upload_file(path_or_fileobj=str(fe_file), path_in_repo="preprocessor_config.json", repo_id=REPO_ID, repo_type="model")
        print("  uploaded preprocessor_config.json")
        api.upload_file(path_or_fileobj=MODEL_CARD.encode("utf-8"), path_in_repo="README.md", repo_id=REPO_ID, repo_type="model")
        print("  uploaded README.md (model card)")

        # readback verification
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(repo_id=REPO_ID, filename="onnx/s2-004.int8.entropy-quint8.onnx")
        print(f"  readback OK: {p}")
        print(f"model repo: https://huggingface.co/{REPO_ID}")

    if args.space:
        api.create_repo(repo_id=SPACE_ID, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
        api.upload_file(
            path_or_fileobj=SPACE_CARD.replace("{REPO_ID}", REPO_ID).encode("utf-8"),
            path_in_repo="README.md", repo_id=SPACE_ID, repo_type="space",
        )
        api.upload_file(path_or_fileobj=str(ROOT / "app" / "app.py"), path_in_repo="app.py", repo_id=SPACE_ID, repo_type="space")
        api.upload_file(path_or_fileobj=SPACE_REQUIREMENTS.encode("utf-8"), path_in_repo="requirements.txt", repo_id=SPACE_ID, repo_type="space")
        print(f"space: https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
