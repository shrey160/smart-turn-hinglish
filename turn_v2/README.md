# turn_v2 — Two-Stage Hinglish Turn Detection

P3 artifact of the masterplan: a whisper-tiny encoder + lightweight head that
answers one binary question per ≤8 s speech segment — **is the user done
speaking (endpoint) or just pausing?** — trained for Hinglish (Hindi-English
code-mixed) telephony audio.

Two-stage recipe (masterplan §4, as amended 2026-08-29):

1. **Stage 1 (s1)** — frozen whisper-tiny encoder, train the head on original
   Smart Turn data (`hin` + `eng`, 7,200 clips).
2. **Stage 2 (s2)** — finetune on TTS Hinglish (209 clips ×2) + 50:50 hin/eng
   replay, **lr 1e-4, 3 epochs, unfreeze last 2 encoder blocks** (approved
   amendment; the originally locked lr 1e-5/2 ep/frozen recipe failed — see
   `context/HARDPOINT.md`).

Current best: `ckpt/s2-004/best.pt` — test-A 0.882/0.886, test-B 0.970/0.968
(acc/F1), p50 e2e latency ~20 ms/clip on CPU. Curated numbers:
`results.md`; machine log: `results.csv` (19-col schema, one row per run).

## Layout

```
turn_v2/
├── data/
│   ├── dataset.py      # clip discovery, manifests, 8 s contract, log-mel (80×800)
│   └── augment.py      # telephony bandpass / noise 10–20 dB / speed 0.9–1.1× (p≈0.5)
├── models/
│   ├── model.py        # SmartTurnV2Model: whisper-tiny + pooling + MLP head
│   └── pooling.py      # attention-mean (384) / asp (768) / asp-end (1152)
├── train.py            # two-stage trainer (s1/s2), overfit gate, KD flags, results.csv logger
├── evaluate.py         # acc/F1 + per-language/filler slices + latency
├── export.py           # ONNX fp32 + int8 static QDQ (held-out dev calibration)
├── ckpt/<run_id>/best.pt
├── kd/                 # teacher logits (npz) + teacher run logs
├── onnx/               # s2-004.fp32.onnx (30.9 MiB) + s2-004.int8.entropy-quint8.onnx (9.05 MB)
├── results.md          # thesis tables (P1 + v2 two-stage + P4 ablations + P5 KD/int8)
└── results.csv         # append-only experiment log
```

## Input contract (must match Smart Turn exactly)

- 16 kHz mono float32 in [-1, 1]
- ≤8 s; shorter → **zero-pad FRONT**; longer → keep the **LAST 8 s**
  (semantics of `smart_turn_reference/audio_utils.py`, imported — never
  re-implemented)
- Labels: `endpoint_bool` (1 = complete = agent should respond);
  `midfiller`/`endfiller` preserved from folder names
- Features: whisper log-mel, `(80, 800)` via `chunk_length=8`

Label folders parse as `complete|incomplete[-midfiller][-endfiller]`
(Python-bool casing, e.g. `incomplete-False-True`).

## Quickstart

```powershell
# 1) sanity gate — MUST pass before any full run (lr 1e-3 head-only)
uv run python -m turn_v2.train --stage s1 --overfit 100

# 2) Stage 1 (frozen encoder, ~12 min on CPU)
uv run python -m turn_v2.train --stage s1 --workers 4

# 3) Stage 2 (approved recipe)
uv run python -m turn_v2.train --stage s2 --init-ckpt turn_v2/ckpt/s1-001/best.pt \
    --lr 1e-4 --epochs 3 --unfreeze-last-k 2 --workers 4

# 4) evaluate any checkpoint
uv run python -m turn_v2.evaluate --ckpt turn_v2/ckpt/s2-004/best.pt --splits test_a test_b

# 5) ONNX export + int8 (reference recipe; see results.md §P5)
$env:PYTHONUTF8 = "1"
uv run python turn_v2/export.py --ckpt turn_v2/ckpt/s2-004/best.pt --eval --calib-n 200 --quant-config "entropy,quint8"
```

Every training run appends a row to `results.csv` and writes
`ckpt/<run_id>/best.pt` (best on weighted dev F1; combined dev = dev_v1 +
dev_tts_v1 for s2). One change per row — tag it with `--tag`, `--change-summary`.

## CLI

### train.py

| Arg | Default | Notes |
|---|---|---|
| `--stage` | required | `s1` or `s2` |
| `--init-ckpt` | `""` | s2 loads `ckpt/s1-*/best.pt` here |
| `--pooling` | `attention-mean` | `asp`, `asp-end` available (P4 arms) |
| `--lr` | 0.0 | required; overfit mode defaults 1e-3, else pass explicitly |
| `--epochs` | 0 | required |
| `--unfreeze-last-k` | 0 | unfreeze last k encoder blocks (s2-004 used 2) |
| `--no-freeze-encoder` | off | full-encoder finetune |
| `--overfit N` | 0 | sanity gate: N clips, no augment, lr 1e-3; prints PASS/FAIL (f1 ≥ 0.95, loss ≤ 0.1) |
| `--batch-size` / `--eval-batch-size` | 32 / 128 | |
| `--warmup-ratio`, `--eval-steps`, `--patience` | None/0/2 | warmup+cosine; per-epoch eval; early stop on 2 evals w/o best |
| `--seed` | 42 | |
| `--workers` | 0 | use 4 on this machine (CPU) |
| `--no-augment` | off | |
| `--kd-teacher` | `""` | npz of precomputed teacher logits (`scripts/precompute_teacher_logits.py`) |
| `--kd-alpha` / `--kd-temp` | 0.5 / 3.0 | soft-BCE KD at the s2 run (P5: negative result — s2-014 vs control s2-015) |
| `--tag`, `--change-summary`, `--notes` | `""` | results.csv provenance |

Loss: BCE with batch-level pos_weight (KD adds temperature-scaled soft term ×T²).
Optimizer: AdamW, wd 0.01.

### evaluate.py

| Arg | Default | Notes |
|---|---|---|
| `--ckpt` | required | path to `best.pt` |
| `--splits` | `test_a test_b` | also `dev`, `dev_v1`, `tts_dev`, `tts_all`, `test_c` |
| `--max`, `--batch-size`, `--workers` | 0/128/0 | |

Prints n / acc / F1, per-language and per-filler slices, p50 latency (e2e incl.
feature extraction + fwd-only).

### export.py (P5)

| Arg | Default | Notes |
|---|---|---|
| `--ckpt` | required | torch ckpt → fp32 ONNX (opset 18, dynamic batch) |
| `--splits` | `test_a test_b test_c` | eval splits |
| `--quant-config` | `entropy,quint8` | `calib,act[:rr]`; combine via `;` — shipped recipe: `entropy,quint8` |
| `--calib-n` | 200 | held-out dev_v1 calibration clips |
| `--sweep` | off | quantize all configs, eval on first split only (needs `--eval`) |
| `--eval` | off | fp32 vs int8 acc/F1/p50 on splits |

int8 recipe = reference-faithful: `quant_pre_process` + QDQ, per-channel,
QUInt8/QInt8, Entropy, **Conv/MatMul/Gemm only** — replicating it fixed the
−8.8 AVX2 U8S8 cliff (HARDPOINT 2026-08-29). Result tables in `results.md`.

### scripts/eval_onnx.py

| Arg | Notes |
|---|---|
| `--paths` | one or more `.onnx` files |
| `--split` | `test_a`/`test_b`/`test_c` — evals existing files without re-quantizing |

## Modules

- **dataset.py** — split builders return clip dicts
  (`{path, lang, label, midfiller, endfiller}`): `train_pool_clips()`,
  `test_a_clips()`, `test_b_clips()`, `tts_clips()`, `dev_v1_clips()`
  (manifest `data/splits/dev_v1.csv`), `test_c_clips()`. `carve_tts_dev()`
  made `data/splits/dev_tts_v1.csv` (23 clips, seed 42) — s2 selection uses
  combined dev. `TurnDataset(cache=)` caches log-mel tensors per clip.
- **augment.py** — `WaveAugment(p≈0.5, seed)`: telephony bandpass 300–3400 Hz,
  additive noise 10–20 dB SNR, speed 0.9–1.1× (applied pre-contract).
- **model.py** — `SmartTurnV2Model`: frozen whisper-tiny, `max_source_positions=400`
  (transformers 5 fix) with pretrained positional rows 0–399, pooling → head
  (Linear→LN 256→GELU→Dropout 0.1→Linear 64→GELU→Linear 1; init N(0, 0.1)
  mirrored from reference). `freeze_encoder()` / `unfreeze_last_k()`.
- **pooling.py** — `POOLING_REGISTRY`: `attention-mean` (out 384, default),
  `asp` (768), `asp-end` (1152).

## Data expectations

Layouts live under `data/` (gitignored): `train_pool/<lang>/<label-folders>/`,
`data/example_hinglish_fixed/test_b/`, `data/tts_hinglish/...`,
`data/test_c/mucs-hinglish/`, manifests in `data/splits/`. Verify
counts/durations after any data step — every run prints per-epoch `n`;
test-A must be 1,200 (hin 600 / eng 600); test-C is 846 (434:412).

**test-C caveat** (built by `scripts/build_test_c.py` from MUCS-Hinglish,
parquet deleted after extraction): real human Hinglish *tutorial monologue* —
all neural models score near chance on it (prosody-first v3.2 training does
not transfer to non-conversational speech; see `results.md` test-C section).
Use it for relative model ranking, not absolute thresholds.

## Status / open items

- P0–P5 complete. **v2-core = s2-004** (attention-mean, k=2, replay 0.5):
  test-A 0.882/0.886, test-B 0.970/0.968, test-C 0.600/0.446.
- P5 closed: KD negative (teacher whisper-small s2-013 beats student but KD
  itself hurt — see results.md); ONNX fp32 bit-faithful + int8 9.05 MB at
  −2.5/−3.0/+0.4 (documented deviation from the ≤8 MB/≤1% target).
- Next: P6 (`latency.py` trailing-context curve, `policy.py` Pareto frontier,
  error analysis) → P7 (Gradio demo on fp32/int8 ONNX, HF Hub, report).
- torch is now **2.13.0+cu126 (CUDA)** — GPU runs take minutes; CPU still
  works. Keep `--workers 4` for dataloaders; one background job at a time.
- Re-running experiments: `uv run python -m turn_v2.train --help`; exact run
  provenance per row in `results.csv`.
