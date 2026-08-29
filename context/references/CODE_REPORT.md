# Smart Turn v3.2 — Codebase Report

> Generated from a knowledge-graph analysis of the full repository (`graphify-out/graph.json`, 203 nodes / 320 edges / 18 communities) combined with a line-by-line read of every source file.

---

## 1. What this project is

**Smart Turn** is an open-source, audio-native **turn detection model for conversational voice AI**. It answers a single binary question about a segment of human speech:

> **Is this the end of the user's turn (should the voice agent respond now), or is the user still talking?**

This is the "should-I-speak-now" decision that every voice agent has to make. The key design bet of the project (documented in `README.md:9`):

- Legacy voice agents use **VAD only** (voice activity detection) — pure speech/silence segmentation, which ignores *linguistic and acoustic content*.
- Smart Turn instead classifies each segment using **speech understanding**, so it can match human turn-taking behavior (finished thought vs. continuing thought).

The model runs **only during periods of silence** (triggered conjointly with a lightweight VAD like Silero). It does not work on whispered/quiet utterances — it needs the full turn with context, hence the "run during silence" design.

---

## 2. Repository layout — file map

```
smart-turn/
├── README.md                        # project overview, usage, model notes
├── requirements*.txt                # dependency sets (training / aarch64 / inference-only)
├── audio_utils.py                   # shared audio normalization (truncate/pad to 8 s)
├── inference.py                     # ONNX inference wrapper — the ONLY inference entry point
├── predict.py                        # CLI: run inference on a single audio file
├── record_and_predict.py             # CLI: live mic streaming + Silero VAD + Smart Turn
├── train.py                          # everything: model, datasets, training, export, quantization
├── benchmark.py                      # accuracy + latency benchmarking for ONNX models
├── logger.py                         # structured console/W&B logging + training progress callback
├── train_local.py                    # local orchestration wrapper around train.py
├── train_modal.py                    # Modal (cloud GPU) orchestration wrapper around train.py
├── datasets/
│   └── scripts/
│       ├── raw_to_hf_dataset.py      # raw labeled FLAC folders → HuggingFace dataset
│       └── upload-to-hub.py          # local dataset → HuggingFace Hub
└── docs/
    ├── static/confusion_matrix_*.png # eval artifact (1,360-sample test confusion matrix)
    └── data_generation_contribution_guide.md  # labeling/convention spec for contributed data
```

---

## 3. Graph — the 18 communities at a glance

The clustering step found these cohesive modules, which map 1:1 to the real architecture:

| Community | Graph label | Real code |
|---|---|---|
| 0 | Training Run & Evaluation | `train.py` orchestrator + eval callbacks + `logger.py` helpers |
| 1 | Benchmark & Confusion Metrics | `benchmark.py` |
| 2 | Inference & VAD Recording | `inference.py`, `predict.py`, `record_and_predict.py` |
| 3 | ONNX Quantization Calibration | `train.py` → `CalibrationDataset` / `ONNXCalibrationDataReader` |
| 4 | Progress Logging Callback | `logger.py` → `ProgressLoggerCallback` |
| 5 | Dataset Conversion Pipeline | `datasets/scripts/raw_to_hf_dataset.py` |
| 6 | VAD & Data Labeling Guide | `docs/data_generation_contribution_guide.md` + `record_and_predict.py` |
| 7 | Model Architecture & Pipecat | `SmartTurnV3Model` + README architecture notes |
| 8 | Local Training Orchestration | `train_local.py` |
| 9 | Markdown Report Table | `benchmark.py` → `MarkdownTable` |
| 10 | Modal Training Orchestration | `train_modal.py` |
| 11 | Upload Dataset to Hub | `datasets/scripts/upload-to-hub.py` |
| 12 | Confusion Matrix Analysis | `docs/static/*.png` |
| 13 | Dataset Format & Export | FLAC format rules + dataset creation |

The **god nodes** (highest-degree, the true core of this codebase):

1. `do_training_run()` (train.py:665) — 17 edges
2. `benchmark()` (benchmark.py:708) — 15 edges
3. `ProgressLoggerCallback` (logger.py:293) — 15 edges
4. `log_progress()` — 10 edges
5. `run_accuracy()` / `predict_endpoint()` / `run_e2e_perf()` — 9 / 9 / 8 edges

---

## 4. The model — architecture deep dive

### 4.1 Inputs and expected format

- **16 kHz mono PCM** (`float32`, in `[-1, 1]`).
- **Up to 8 seconds** of audio per segment; shorter segments are **zero-padded at the front** so the audio sits at the *end* of the vector.
- Longer segments are truncated to the **last 8 seconds** (keep the ending, drop the beginning).
- Graph artifact `audio_utils.py:4`: `truncate_audio_to_last_n_seconds()` implements both behaviors in one function — and this single function is the **only shared leaf node** among the training, quantization, and inference communities.

### 4.2 Architecture

Defined in `train.py:63` (`SmartTurnV3Model`):

```
input: log-mel spectrogram, shape (batch, 80 mel bands, 800 frames ≈ 8 s @ 16 kHz)
    │
    ▼
Whisper Tiny Encoder (only)          ← frozen-ish backbone, 400 max positions
    ▼
last_hidden_state                     shape (batch, ~1500 tokens, 384)
    ▼
Attention Pooling (pool_attention)  ← Linear(384→256) → Tanh → Linear(256→1) → softmax over tokens
    → weighted-sum pooled vector      shape (batch, 384)
    ▼
Classifier                           ← Linear(384→256) → LayerNorm → GELU → Dropout(0.1)
                                       → Linear(256→64) → GELU → Linear(64→1)
    ▼
logit → sigmoid → probability in [0,1]   (1 = "complete" = agent should respond)
```

Key details:
- Only the **Whisper encoder** is used (not the full seq2seq decoder) — it embeds speech into a representation, then two small heads convert that into a single turn-completeness score.
- Attention pooling: instead of a mean-pool over the T=8s worth of encoder frames, the model **learns frame weights** and produces a weighted sum — this lets it focus on prosody-bearing regions (the trailing intonation/filler, for example).
- The classifier has a **`Dropout(0.1)`** and GELU activations; weights are initialized `N(0, 0.1)` with zero biases (`train.py:63-101`).
- **Loss:** binary cross-entropy with `pos_weight` set from the **actual batch class ratio** (clamped 0.1–10), so it self-balances against dataset skew (`train.py:122-127`).

### 4.3 Deployment variants

- **FP32 ONNX** (~32 MB) — for GPU, slightly more accurate (~1 % on accuracy vs int8).
- **int8 quantized ONNX** (~8 MB) — for CPU, produced by *static* quantization with a 1,024-sample calibration set (`train.py:266-313`).

---

## 5. Data pipeline

### 5.1 Data collection conventions (`docs/...guide.md`)

- **FLAC** files preferred (lossy formats discouraged), mono 16-bit, ≤ 16 s per file, ideally **ending in ~200 ms silence** (annotates the silence window Smart Turn will actually run in).
- Each file = **one turn from one speaker**.
- Labels: `complete` vs `incomplete` (see below), kept at a **50:50 balance**.
- **Prosody is treated as equally important as words** — the guide explicitly says a grammatically complete sentence with "thinking" prosody should be marked *incomplete* (`guide:48-52`).
- Incomplete samples end with a **filler** ("um", "er"), a **connective** ("and", "but"), or **hanging prosody** — but must never be cut off mid-word.

### 5.2 Raw data → HuggingFace dataset (`datasets/scripts/raw_to_hf_dataset.py`)

Input directory layout (language → label suffixes):

```
raw/
├── eng/
│   ├── complete-midfiller/<uuid>.flac
│   ├── incomplete-endfiller/<uuid>.flac
│   └── complete-midfiller-endfiller/<uuid>.flac
├── rus/
│   └── incomplete-nofiller/<uuid>.flac
…
```

The converter (`create_audio_dataset`, line 127):
1. Walks language dirs → subdir suffixes.
2. `parse_directory_suffix()` (line 43) decodes `complete-*` / `incomplete-*` + optional `-midfiller`/`-endfiller`/`-nofiller` into `endpoint_bool`, `midfiller`, `endfiller`.
3. Validates UUID filenames.
4. Copies a renamed FLAC (`{lang}_{subdir}_{uuid}.flac`) into `audio/`, plus a `metadata.jsonl`.
5. Loads with HuggingFace `audiofolder`, saves to disk.

`upload-to-hub.py` pushes that saved dataset to `huggingface.co/datasets`.

### 5.3 On-demand feature generation (`OnDemandSmartTurnDataset`, train.py:323)

Features are **not pre-computed** — the dataset object extracts Whisper log-mel features *on demand* in `__getitem__`, keeping only 8 s (via the shared `truncate_audio_to_last_8s` helper), plus slices `endpoint_bool`, `language`, `dataset`, `midfiller`, `endfiller`.

`SmartTurnDataCollator` batches them; language/midfiller/endfiller are passed along as **lists** so the eval callbacks can slice per-category metrics later.

---

## 6. Training (`train.py`)

### 6.1 Config (top-of-file, structural)

```python
CONFIG = {
    base: "openai/whisper-tiny",
    train/test: pipecat-ai/smart-turn-data-v3.2-(train|test),
    lr 5e-5, epochs 4, batch 384/128,
    warmup 0.2, weight_decay 0.01, cosine schedule,
    eval_steps 500, save_steps 500, logging 100,
    onnx_opset 18, calibration_size 1024,
}
```

### 6.2 The training loop — `do_training_run()` (line:665)

1. **Sanity/device log**: `log_dependencies()`, `log_device_info()`.
2. Requires `WANDB_API_KEY`; initializes a **W&B run** (`endpointing-training` project) with config attached.
3. Loads `SmartTurnV3Model` from `openai/whisper-tiny` (num_labels=1, ignore mismatched sizes).
4. `prepare_datasets_ondemand()` — loads both train/test sources, splits *train → 90% train / 10% eval* (seed 42), concatenates, shuffles, wraps in on-demand datasets.
5. Builds `TrainingArguments` (cosine LR, wandb reporter, pinned memory, 6 dataloader workers, `disable_tqdm=True`).
6. Logs dataset statistics per split.
7. Builds a **Trainer** with:
   - `compute_metrics=compute_metrics` (accuracy/precision/recall/F1 + confusion counts).
   - `data_collator=SmartTurnDataCollator()`.
   - **`ProgressLoggerCallback`** (replaces tqdm with structured log lines).
   - **`ExternalEvalCallback`** — see below.
8. Trains, saves the final model + feature extractor, and **exports ONNX FP32**.

### 6.3 The external evaluation callback (line:468) — a key architectural surprise

`ExternalEvalCallback.on_evaluate` runs, at every HF eval step, the **full test set** (not just the eval split): for each test dataset it produces per-example soft probs via `predict()`, then for every category in the test set:
- **per-`language` metrics**
- **per-`midfiller` / per-`endfiller` metrics**
…then logs W&B histograms of the probability distribution and a global "lowest-accuracy test set" report.

The graph flagged this: it is **semantically twin to `compute_per_category_metrics()` in benchmark.py** — same grouping-by-category logic, same metric functions, one in the training harness and one in the benchmark tool. Nothing structural (no shared symbol) — they drifted apart over time.

### 6.4 Metric functions

- `compute_metrics()` (line:608) — sparse metrics via sklearn, plus confusion matrix components as dict keys.
- `compute_per_category_metrics` (benchmark, line:176) — group probs/labels by language or dataset, compute full metric dict per group.
- Both **never produce a NaN** — `process_predictions()` guards against NaN/infinite values.

---

## 6.5 ONNX export & quantization

### `export_to_onnx_fp32()` (train.py:194)
- Wraps model to output `(batch, 1)` logits (a single head).
- Validates batch=1 **and** batch=2 output consistency (shape `(1,1)`/`(2,1)`).
- Exports opset 18, dynamic batch dim, ONNX-verified with `onnxruntime`.

### `quantize_onnx_model()` (train.py:266)
- `quant_pre_process` (fold constants) → build `CalibrationDataset` → `ONNXCalibrationDataReader` → `quantize_static` with **QuantFormat.QDQ** (Quantize-Dequantize), `QUInt8` activations / `QInt8` weights, **per-channel**, `Entropy` calibration, quantizing `Conv/MatMul/Gemm` layers.

### The full pipeline (hyperedge from graph):
```
train → ONNX export → int8 quantization → benchmark
```
exactly matches the three sub-jobs in `do_training_run` → `do_quantization_run` → `do_benchmark_run`.

---

## 7. Benchmarking (`benchmark.py`)

Two independent benchmarks:

1. **Accuracy (`run_accuracy`, line:609)** — runs the ONNX model over the full test set, computes:
   - overall metrics + **per-language** and **per-dataset** matrices
   - `false_positive_rate` / `false_negative_rate` (how often the model wrongly says "complete"/"incomplete")
   - prints progress with ETA.
2. **Performance (`benchmark` entry, line:708)** — 3 latency measurements:
   - *direct* — pre-extracted zero features → pure ONNX run
   - *feature-extract* — Whisper mel feature extraction alone
   - *end-to-end* — full 8 s audio → features → inference
   Each = p50/p90/mean latency + throughput, with warmup loops, CPU + GPU (CUDA) providers detected.

The `MarkdownTable` class produces aligned, monospace-friendly markdown tables, and `format_markdown_report` renders one .md per model per timestamp.

**Running it:** `train.py` — `do_benchmark_run(model_paths)` → passes on each ONNX to `benchmark()` with a merged test dataset and `batch_size=256`.

---

## 8. The inference stack (the part that actually runs in production)

### 8.1 `inference.py` — the engine

- `build_session()` — one-time ONNX `InferenceSession` with sequential execution, 1 inter-op thread, full graph optimization.
- `predict_e_endpoint(audio_array)` (line 20):
  1. `truncate_audio_to_last_seconds()` — normalize to ≤8 s / pad front
  2. `WhisperFeatureExtractor(chunk_length=8)` on 16 kHz, `padding=“max”` — builds log-mel
  3. squeeze → add batch dim
  4. `session.run(None, {"input_features": ...})` — raw sigmoid probability out
  5. returns `{"prediction": 0|1, "probability": float}`

**This is the *only* function every other consumer calls** (`predict.py` and `record_and_predict.py` both route through it). It's also the hinge in the graph either connecting "Inference & VAD Recording" to "ONNX Quantization Calibration" — because the ONNX-exported artifact produced by `train.py` is exactly what this session runs.

### 8.2 `predict.py` — single-file batch

Loads a file (`librosa.load`), resamples to 16 kHz if needed, normalizes to `[-1, 1]`, and prints the endpoint verdict.

### 8.3 `record_and_predict.py` — live microphone + VAD

The real-world demo:
- Pumps 512-sample buffers from `pyaudio` at 16 kHz (`CHUNK=512`, Silero's expected window).
- Wraps **Silero VAD ONNX** in `SileroVAD` — component with 64-sample context, resetting every 5 s (`MODEL_RESET_STATES_TIME`).
- Keeps a ring buffer of pre-speech (200 ms of silence-before-speech is kept).
- State machine:
  - trigger on speech prob >  0.5
  - accumulate segment (speech + trailing silence)
  - end when trailing silence ≥ 1000ms, OR hard cap ≥ 0 seconds
- On segment end → `_process_segment()` → calls `predict_endpoint()` on the whole segment, prints verdict + inference time.

This is exactly aligned with the README design: *only run Smart Turn during silence* (`README.md:20`) and *re-run on the full turn, not the new piece* (`README.md:94`).

---

## 9. Logging (`logger.py`)

- Module-level `logging` instance with one console handler.
- `log_dependencies()` / `log_device_info()` — dump `pip list`, CUDA/MPS status, key env vars.
- `log_model_structure()` — layers, embed dims, parameter counts (mostly used for debugging earlier Whisper snapshots).
- `log_dataset_statistics()` — positive/negative ratios, language distribution, feature shape/predicted audio length.
- `ProgressLoggerCallback` — `TrainerCallback` with step/epoch/EASET logging (no tqdm), used to replace the default bar.

Functions are exported as an aggregate module in `logger.py:11-21` so callers just `from logger import log as log`.

---

## 10. Orchestration wrappers

### `train_local.py` (CLI)
- `--training-run-name`, `--quantize <onnx>`, `--benchmark <dir>` — thin arg wrappers that just call the `train.*` run functions.
- Also remembers to set `OMP_NUM_THREADS=1` / `PASSIVE` to avoid MKL thread over-subscription on CPU-bound hosts.

### `train_modal.py` (Modal cloud GPU)
- Defines a `modal.App`, a `modal.Volume` ("endpointing") at `/data`.
- `training_run` / `quant_ / bench` as modal functions with:
  - training: GPU `L4`, 32 GB, 8 CPU
  - quantization: no GPU, 128 GB RAM, 16 CPU
  - benchmark: GPU `T4`, 32 GB, 8 CPU
- Runs `wandb-secret` secret for `TrainingArguments(report_to=["wandb"])`.
- Main entry also turns `training_run_name`, `quantize`, `benchmark` flags into **remote** invocations.

---

## 11. Graph-derived observations (worth writing down)

The clustering uncovered several things you likely didn't know the code does:

1. **`truncate_audio_to_last_n_seconds()` is your most load-bearing utility.**
   - Betweenness 0.163 — the highest in the graph.
   - Because `predict_endpoint` (inference), `CalibrationDataset`, and `OnDemandSmartTurnDataset` (training) all share the "keep last 8 s" contract, this one function is what keeps train ↔ inference ↔ quantize in sync. If it drifts in any asymmetry (padding style, integer sample types), behavior diverges per-case precisely because no such symmetry across those three pipelines outside this contract.

2. **`train.py` and `benchmark.py` have drifted.** The graph finds *five* `semantically_similar_to` edges where both modules define near-identical logic (build/ONNX, load_dataset, process_predictions, compute_metrics-ish, per-category). The types are the same, the metric math is the same, but the two never import each other — a candidate for extracting a shared `metrics.py` / `features.py` module.

3. **Inference is intentionally asymmetric.** The eval harness (`benchmark.py`) uses fully-precomputed `input_features` + resized zero audio for perf runs, while real inference (`inference.py`) uses the feature extractor live. This is correct (perf needs a fixed-cost baseline) but it means *the benchmark's "e2e" numbers are for the pipeline that inference actually uses* — exactly what you want to trust when shipping.

4. **Mutable `feature_extractor` is built at module import time.** `inference.py:16` instantiates the extractor and the ONNX session at import — so importing `inference` (as `predict.py`, `record_and_predict.py` do) immediately fires a model load. This is fine for scripts but will surprise any long-running server/CGI worker that imports the module lazily; worth noting if you ever serve this in memory.

5. **Dataset labels live in directory names, not in files.** The `raw_to_hf_dataset.py` pipeline decodes semantic state from *suffix strings*; the label is re-derived later in `__getitem__`, so any future label convention change must stay in sync with `parse_directory_suffix` + the dataset schema (endpoint, midfiller, endfiller) or training breaks silently.

6. **Per-category analysis is a duplicate.** `benchmark.compute_per_category_metrics()` and `ExternalEvaluationCallback._log_category_metrics()` solve the exact same problem (group → metric table → W&B histogram) on both sides of the train/benchmark boundary. The most object consistency-checked pair in the graph.

7. **Quantization is calibration-loaded but not used from train.** The `CalibrationDataset` uses a seed 42 subset of the *training* split, not a held-out calibration set. In many production int8 pipelines the calibration set is dedicated/opaque; here the same OnDemand preprocessing path feeds both training and calibration, so any feature-parity bug hits both.

---

## 12. Walkthroughs: three end-to-end journeys

### A. "Record from microphone, get a prediction" (the demo)
```
record_and_predict.py
  ├─ SileroVAD (ONNX, silero_vad.onnx, auto-download)     → VAD speech probability per 512-sample chunk
  ├─ ring buffer (200 ms pre-speech)
  ├─ state machine (trigger; accumulate; stop: 1000 ms silence OR 8 s cap)
  └─ _process_segment()
       └─ prediction → predict_endpoint()                     → inference_external.py
               ├─ truncate_audio(8 s, keep end, pad front)
               ├─ WhisperFeatureExtractor(8) → log-mel (80, 800)
               ├─ ONNX session.run(input_features)           → sigmoid probability
               └─ { "prediction": 0|1, "probability": p }
```
> The model only ever sees an 8 s window aligned to the *end* of the turn, zero-padded in front.

### B. Train a new model
```
train_local.py --training-run-name "my-run"
  → train.do_training_run()
      ├─ wandb init, log deps/device/model
      ├─ prepare_datasets_ondemand()   (HF datasets, 90/10, shuffle 42)
      ├─ Trainer (whisper-tiny + heads, BCE pos_weight, cosine LR)
      │   ├─ ProgressLoggerCallback (log)
      │   ├─ ExternalEvalCallback (per-language, per-filler, W&B)
      │   └─ compute_metrics (accuracy/F1/confusion)
      ├─ save final_model/
      └─ export_to_onnx_fp32() → model_fp32.onnx
train_local.py --quantize <fp32 path>
  → do_quantization_run() → quantize_static(QDQ, int8, 1024 calib) → *_int8_static.onnx
train_local.py --benchmark <model root>
  → do_benchmark_run() → benchmark(md + JSON metrics)
      ├─ perf (direct / e2e, CPU+GPU)
      └─ accuracy (overall + per-language + per-dataset) → .md report
```

### C. Data contribution flow
```
raw/flac folders (lang/suffix convention)
  → raw_to_hf_dataset.py (parse suffix → endpoint_bool/midfiller/endfiller, rename, metadata.jsonl)
  → audiofolder → save_to_disk()
  → upload-to-hub.py --upload --hub-id pipecat-ai/smart-turn-data-v3.2-train
  → future train run picks it up
```

---

## 12. Cross-cutting concerns & edge notes

- **Torchcodec requirement**: `ensure_torchcodec_available()` fail-fasts before dataset loading (`train.py:52`) — HF audio decoding needs it on 3.10+ transformers builds.
- **Num workers**: 6, prefetch 4, pinned memory, `tf32=False` (Ampere-safe).
- **Determinism**: shuffle seeds are set to 42 in both dataset prep and on-demand calibration.
- **NaN-** iences: `process_predictions` raises on non-finite logits rather than warn .
- **Bandwidth**: full W&B logging off by defaults under `report_to=["wandb"]` toggle — but `do_training_run` hard-requires the env key.

---

## 13. Dependencies at a shot

| Runtime | Dexes |
|---|---|
| Model | torch 2.9, transformers 4.48.2 (whisper), torchaudio, torchcodec |
| ONNX | onnx 1.19, onnxruntime-gpu 1.23, onnxscript |
| Data | datasets 4.4, scikit-learn, numpy 2.3 |
| Audio | soundfile / librosa (inference samples), pyaudio (mic) |
| Observability | wandb, logging |
| Edge | Modal (`train_modal.py`) |

---

## 14. Project goals & extension ideas visible in the code

The graph + README agree there are clean seams to grow:
- **Text conditioning** — two usages of a "mode" context (credit card, phone number) are sketched in `README.md:112-114`. The model already takes slices per-category, so a text-embedding fused into the encoder is a natural next step.
- **More base models** mapped out — wav2vec2-BERT, wav2vec2, LSTM, deeper transformer head were all tried (README:120).
- **More data** as `smart-turn-data-vX-server` — the directory-back designer has a per-filler knob already.
- **The twin logic in `train.py` vs `benchmark.py`** is the biggest refactoring opportunity surfaced (unlike the rest of the codebase, it's actual duplication).

---

*Generated from graphify knowledge graph + source. Line numbers point at the file as of this writing.*