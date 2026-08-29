# Hinglish Turn Detection (v2)

Tiny, fast, **audio-native** turn detection for Hinglish (Hindi–English
code-mixed) telephony speech. Given up to 8 s of audio, decide: is the user
**done speaking** (agent can respond) or **just pausing** (keep listening)?

Built on top of the [Pipecat Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn)
approach (read-only reference in `smart_turn_reference/`), adapted to the
language pair it skipped — with the policy-layer evaluation the field actually
uses.

- **Weights:** [Shrey160/hinglish-turn-v2](https://huggingface.co/Shrey160/hinglish-turn-v2) — fp32 (bit-faithful) + int8 (9 MB) ONNX, torch ckpt, feature extractor config
- **Report:** [REPORT.md](REPORT.md) — framing, dataset audit, strategy, experiments, errors, limitations
- **Demo:** `turn_v2/app.py` (Gradio; mic/file → verdict + p + inference ms + simulated streaming)
- **Curated numbers:** [`turn_v2/results.md`](turn_v2/results.md) · machine log: `turn_v2/results.csv`

## Results

| Model | test-A (v3.2-test hin+eng) | test-B (TTS Hinglish) | test-C (MUCS real speech) |
|---|---|---|---|
| Smart Turn v3.2 zero-shot | **0.927 / 0.930** | 0.788 / 0.759 | 0.539 / 0.573 |
| **v2 s2-004 (ours)** | 0.882 / 0.886 | **0.970 / 0.968** | **0.600 / 0.446** |

acc / F1. The crossover is the point: the baseline wins on its own training
distribution; the Hinglish fine-tune wins on Hinglish domains. Latency ~20 ms
p50 e2e on CPU (int8 ~13 ms, 9 MB). Policy: τ=0.5 shipped; trailing-context
curve shows the model can commit on 2–4 s windows.

```
audio (≤8 s, 16 kHz mono)  ──  contract: pad FRONT / keep LAST 8 s
        │
        ▼
whisper-tiny log-mel (80 × 800)          [unnormalized — v2 convention, see REPORT §7]
        │
        ▼
whisper-tiny encoder (s1 frozen / s2 last-2 blocks unfrozen)
        │
        ▼
attention-mean pooling (384-d) → MLP head → logit → sigmoid → p(complete)
        │
        ▼
p ≥ τ (0.5)  →  COMPLETE: agent responds     |     p < τ  →  INCOMPLETE: keep listening
```

## Training recipe (two-stage transfer)

1. **Stage 1** — frozen whisper-tiny encoder, head trained on Smart Turn
   hin+eng (7,200 clips), lr 5e-5, 4 epochs.
2. **Stage 2** — 232 TTS Hinglish clips (×2) + 50:50 hin/eng replay per epoch,
   lr 1e-4, 3 epochs, last 2 encoder blocks unfrozen.

Data: ~1.4 GB on disk (2 GB cap) — 8,800-clip v3.2 subset, TTS Hinglish
(edge-tts, marker-based complete/incomplete script pairs), and a real-speech
MUCS stress set. Dataset audit finding: **83% of v3.2 is TTS, ~1.6% human** —
the report treats this as a first-class result.

## How to run

```powershell
uv sync                                              # env (Python 3.11)
uv run python turn_v2/app.py --selftest              # demo selftest
uv run python turn_v2/app.py                         # Gradio demo UI

uv run python -m turn_v2.train --stage s1 --overfit 100        # sanity gate
uv run python -m turn_v2.train --stage s1 --workers 4          # stage 1
uv run python -m turn_v2.train --stage s2 --init-ckpt turn_v2/ckpt/s1-001/best.pt --lr 1e-4 --epochs 3 --unfreeze-last-k 2 --workers 4
uv run python -m turn_v2.evaluate --ckpt turn_v2/ckpt/s2-004/best.pt --splits test_a test_b test_c

uv run python turn_v2/latency.py                     # trailing-context curve
uv run python turn_v2/policy.py                      # FIR vs HOLD Pareto frontier
uv run python scripts/error_analysis.py              # slices + categorized failures
uv run python scripts/upload_hf.py                   # HF Hub upload (--space for a Space)
```

Full reproduce commands: [REPORT.md §9](REPORT.md); module/CLI docs:
[`turn_v2/README.md`](turn_v2/README.md); planning & decision history: `context/`.

## Repo layout

```
├── smart_turn_reference/   # Pipecat baseline (READ-ONLY; imported, never modified)
├── turn_v2/                # our model: data/, models/, train, evaluate, export,
│                           #   latency, policy, eval_common, app (demo), results.{md,csv}
├── scripts/                # data ops: download, TTS generation, test-C build, HF upload
├── data/                   # datasets (gitignored)
├── models_ref/             # baseline ONNX (gitignored)
├── context/                # masterplan → phases → subplan → HARDPOINT → progress
└── REPORT.md               # final self-written report
```

## Limitations (short version)

test-C is tutorial monologue (neural models collapse there — stress test only);
test-B is 33 TTS clips; v2 mels are unnormalized (self-consistent, see report);
232 TTS clips = probe, not domain solution; HF Spaces now PRO-gates Gradio
hosting (demo runs locally; Space upload is one command). Details + next steps:
[REPORT.md §7–8](REPORT.md).
