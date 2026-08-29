# Smart Turn v3.2 — Training Pipeline Reference

> Distilled from `smart_turn_reference/train.py`, `train_local.py`, `train_modal.py`, `audio_utils.py`, `inference.py`, `docs/data_generation_contribution_guide.md`, `datasets/scripts/raw_to_hf_dataset.py`, `README.md`, and `context/references/CODE_REPORT.md`.

---

## 1. Task definition

Smart Turn answers one binary question about a speech segment:

- **1 = `complete`** — the speaker has finished their turn; the agent should respond.
- **0 = `incomplete`** — the speaker is likely still talking (filler, connective, hanging prosody); wait.

The model is **audio-native**: it consumes raw 16 kHz mono PCM rather than text, so it can use prosody and intonation cues.

---

## 2. Training data format & conventions

### 2.1 Raw source format (for contributors)

- **Container:** FLAC (preferred; avoid lossy MP3/Opus).
- **Audio:** mono, 16-bit; 16 kHz is required for training, but higher sample rates are accepted and resampled.
- **Length:** ≤ 16 s per clip; one speaker per clip; one turn per clip.
- **Trailing silence:** aim for ~200 ms of silence at the end (Smart Turn runs only after VAD detects silence).
- **Labels encoded in directory names:**

```
raw/
├── eng/
│   ├── complete-midfiller/<uuid>.flac
│   ├── incomplete-endfiller/<uuid>.flac
│   └── complete-midfiller-endfiller/<uuid>.flac
└── hin/
    └── incomplete-nofiller/<uuid>.flac
```

Directory suffixes are parsed by `raw_to_hf_dataset.py:parse_directory_suffix()`:

| Prefix | `endpoint_bool` |
|---|---|
| `complete-*` | `True` |
| `incomplete-*` | `False` |

| Suffix | `midfiller` | `endfiller` |
|---|---|---|
| `-nofiller` | False | False |
| `-midfiller` | True | False |
| `-endfiller` | False | True |
| `-midfiller-endfiller` | True | True |

- **Balance:** target a 50:50 split of `complete` vs `incomplete`.
- **Incomplete rules:** must end with a filler ("um", "er"), a connective ("and", "but", "because"), or hanging prosody — **never cut off mid-word**.
- **Prosody matters:** a grammatically complete sentence with "thinking" intonation should be labeled `incomplete`.

### 2.2 HuggingFace dataset schema

`raw_to_hf_dataset.py` converts the folder tree into an `audiofolder` dataset with this metadata (`metadata.jsonl`):

```json
{
  "file_name": "audio/{language}_{subdir}_{uuid}.flac",
  "id": "<uuid>",
  "language": "eng",
  "endpoint_bool": true,
  "midfiller": false,
  "endfiller": true
}
```

At runtime, each row exposes:

- `audio["array"]` — raw waveform (numpy float32, already loaded by `datasets`).
- `endpoint_bool` — boolean label.
- `language`, `midfiller`, `endfiller`, `dataset`, `id`.

The canonical training/eval repos are:

- `pipecat-ai/smart-turn-data-v3.2-train`
- `pipecat-ai/smart-turn-data-v3.2-test`

---

## 3. Audio preprocessing

Every audio clip is normalized to exactly **8 seconds** by `audio_utils.truncate_audio_to_last_n_seconds()`:

```python
def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    max_samples = n_seconds * sample_rate  # 128,000 samples
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]                # keep the END
    elif len(audio_array) < max_samples:
        padding = max_samples - len(audio_array)
        return np.pad(audio_array, (padding, 0),         # pad FRONT
                      mode='constant', constant_values=0)
    return audio_array
```

This "keep last 8 s, zero-pad front" rule is the **single shared contract** between training, quantization calibration, and inference.

---

## 4. Feature extraction

Features are **not pre-computed**. The `OnDemandSmartTurnDataset` extracts them on `__getitem__`:

```python
audio_array = sample["audio"]["array"]
audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

inputs = feature_extractor(
    audio_array,
    sampling_rate=16000,
    return_tensors="pt",
    padding="max_length",
    max_length=8 * 16000,
    truncation=True,
    do_normalize=True,
)
```

- `feature_extractor = WhisperFeatureExtractor(chunk_length=8)`.
- Output tensor shape per example: `(80 mel bins, 800 frames)`.
- Batched shape: `(batch_size, 80, 800)`.

`SmartTurnDataCollator` simply stacks `input_features` and `labels`; it also passes through `language`, `midfiller`, `endfiller` as lists so the evaluation callback can compute per-category metrics.

---

## 5. Model architecture

Defined in `train.py:SmartTurnV3Model`:

```
input: log-mel spectrogram (batch, 80, 800)
        │
        ▼
Whisper Tiny Encoder only
(config.max_source_positions = 400)
        │
        ▼
last_hidden_state (batch, ~1500 tokens, 384)
        │
        ▼
Attention Pooling (pool_attention)
  Linear(384 → 256) → Tanh → Linear(256 → 1) → softmax over time
  → weighted-sum pooled vector (batch, 384)
        │
        ▼
Classifier
  Linear(384 → 256) → LayerNorm → GELU → Dropout(0.1)
  Linear(256 → 64) → GELU → Linear(64 → 1)
        │
        ▼
sigmoid → probability of completion
```

- Base: `openai/whisper-tiny` encoder only (not the decoder).
- Final output: a single logit → sigmoid probability.
- Weight init: `N(0, 0.1)` for linear layers, zero biases.
- The attention pooling lets the model focus on prosody-bearing frames (e.g., trailing fillers).

---

## 6. Loss & training config

Top-level `CONFIG` in `train.py`:

| Hyperparameter | Value |
|---|---|
| Base model | `openai/whisper-tiny` |
| Learning rate | `5e-5` |
| Epochs | 4 |
| Train batch size | 384 |
| Eval batch size | 128 |
| Warmup ratio | 0.2 |
| Weight decay | 0.01 |
| LR schedule | cosine |
| Eval/save/logging steps | 500 / 500 / 100 |
| ONNX opset | 18 |
| Calibration dataset size | 1024 |

Loss function (`BCEWithLogitsLoss`) with **dynamic positive weight** per batch:

```python
pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

This self-balances against class imbalance without fixing a global weight.

Training uses the HuggingFace `Trainer`:

- `TrainingArguments` with `report_to=["wandb"]`.
- `dataloader_num_workers=6`, `prefetch_factor=4`, `pin_memory=True`.
- `tf32=False`.
- `disable_tqdm=True` — progress is logged via a custom `ProgressLoggerCallback`.

---

## 7. Training flow (`do_training_run`)

1. Log dependencies, device info, model structure.
2. Require `WANDB_API_KEY`; initialize a Weights & Biases run (`project="speech-endpointing"`).
3. Load `SmartTurnV3Model.from_pretrained("openai/whisper-tiny", num_labels=1, ignore_mismatched_sizes=True)`.
4. `prepare_datasets_ondemand()`:
   - Load each training repo, `train_test_split(test_size=0.1, seed=42)`.
   - Concatenate all training splits; concatenate all eval splits.
   - Load each test repo separately.
   - Wrap everything in `OnDemandSmartTurnDataset`.
5. Log dataset statistics.
6. Build `Trainer` with:
   - `compute_metrics` (accuracy, precision, recall, F1, confusion counts).
   - `SmartTurnDataCollator`.
   - `ProgressLoggerCallback`.
   - `ExternalEvaluationCallback` — see below.
7. Train.
8. Save final model + feature extractor to `final_model/`.
9. Export FP32 ONNX to `final_model/exports/model_fp32.onnx`.

### 7.1 External evaluation callback

At every eval step, the full **test set** is run (not just the eval split). The callback computes and logs to W&B:

- Overall accuracy/F1/precision/recall per test dataset.
- Per-language metrics.
- Per-filler (`midfiller`, `endfiller`) metrics.
- Probability-distribution histograms.
- Lowest-accuracy test set, mean accuracy, variance.

---

## 8. ONNX export & quantization

### 8.1 FP32 export (`export_to_onnx_fp32`)

- Wraps the model so it returns a 2D tensor of shape `(batch_size, 1)`.
- Validates output shapes for batch sizes 1 and 2.
- Exports with opset 18 and **dynamic batch dimension**.
- Verifies the exported model with `onnxruntime`.

### 8.2 INT8 quantization (`quantize_onnx_model`)

- `quant_pre_process` on the FP32 model (constant folding enabled).
- Build a 1,024-sample calibration set from the **training split** (seed 42 shuffle).
- Run `onnxruntime.quantization.quantize_static`:
  - `QuantFormat.QDQ`
  - activations: `QUInt8`
  - weights: `QInt8`
  - `per_channel=True`
  - `calibrate_method=Entropy`
  - quantize only `Conv`, `MatMul`, `Gemm`
- Output: `model_int8_static_calib1024.onnx` (~8 MB vs ~32 MB FP32).

---

## 9. Inference (production path)

`inference.py:predict_endpoint(audio_array)`:

1. `truncate_audio_to_last_n_seconds(audio_array, 8)` — keep last 8 s, zero-pad front.
2. `WhisperFeatureExtractor(chunk_length=8)` → log-mel `(80, 800)`.
3. Add batch dimension → `(1, 80, 800)`.
4. ONNX Runtime session run.
5. Sigmoid probability output.
6. Threshold at `0.5` to return `{prediction: 0|1, probability: float}`.

ONNX session options used:

```python
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

---

## 10. Local vs cloud orchestration

| Script | Purpose |
|---|---|
| `train_local.py` | CLI wrapper: `--training-run-name`, `--quantize <path>`, `--benchmark <dir>` |
| `train_modal.py` | Modal.com cloud wrapper (L4 for training, no GPU for quantization, T4 for benchmarking) |

Both set:

```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
```

to avoid MKL thread oversubscription.

---

## 11. Key takeaways for our Hinglish model

1. **Keep the 8-second contract everywhere** — truncate to last 8 s, pad zeros at the front. Any mismatch between train/quantize/inference silently breaks the model.
2. **Labels are not just complete/incomplete** — the `midfiller` and `endfiller` flags are tracked per sample and used for fine-grained evaluation, even though the model only sees a single binary target.
3. **Prosody is a first-class signal** — the dataset guide explicitly labels hanging intonation as `incomplete`; our Hinglish data generation should preserve this.
4. **Dynamic batch positive weight** — useful for handling class imbalance if our Hinglish subsets are not perfectly balanced.
5. **Attention pooling over Whisper encoder frames** — the model learns where in the 8-second window to look (e.g., the trailing filler).
6. **Features are extracted on-demand** — no offline feature cache is required, but it means training I/O is audio-decoding heavy; `torchcodec` is required.
7. **Export path is fixed** — train → FP32 ONNX → INT8 static quantization → ONNX Runtime inference.

---

## 12. File references

| File | Role |
|---|---|
| `smart_turn_reference/train.py` | Model, dataset wrapper, training loop, ONNX export, quantization |
| `smart_turn_reference/train_local.py` | Local CLI orchestration |
| `smart_turn_reference/train_modal.py` | Modal.com cloud orchestration |
| `smart_turn_reference/audio_utils.py` | Shared truncate/pad to 8 s |
| `smart_turn_reference/inference.py` | ONNX Runtime production inference |
| `smart_turn_reference/predict.py` | Single-file inference example |
| `smart_turn_reference/docs/data_generation_contribution_guide.md` | Raw data collection & labeling rules |
| `smart_turn_reference/datasets/scripts/raw_to_hf_dataset.py` | Raw FLAC folders → HF dataset |
| `smart_turn_reference/benchmark.py` | Accuracy + latency benchmarking |
