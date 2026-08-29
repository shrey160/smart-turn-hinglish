"""P1 baselines: Smart Turn v3.2 (int8) zero-shot + silence-threshold baseline.

Builds the thesis table (baseline x test-set -> acc/F1) per masterplan P1.
Reuses the read-only smart_turn_reference/ inference path; the 8 s audio contract is
imported from smart_turn_reference/audio_utils (never re-implemented locally).

Run (from anywhere; script pins CWD to repo root for the CWD-relative ONNX path):
  uv run python scripts/eval_baselines.py --splits test_a test_b
  uv run python scripts/eval_baselines.py --splits test_a --max 50   # quick pilot

Outputs: results/preds_<split>.csv (regenerable) + printed summary table.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "smart_turn_reference"))

from audio_utils import truncate_audio_to_last_n_seconds  # noqa: E402  (shared contract)

FRAME_S, HOP_S = 0.03, 0.02
ENERGY_FLOOR = 0.01  # absolute frame-RMS floor for "silence"
T_GRID = np.round(np.arange(0.05, 1.55, 0.05), 2)
RESULTS_DIR = ROOT / "results"


def iter_clips(split):
    if split == "test_a":
        base = ROOT / "data" / "test_a"
        for lang_dir in sorted(base.iterdir()):
            for label_dir in sorted(lang_dir.iterdir()):
                for f in sorted(label_dir.glob("*.flac")):
                    yield f, lang_dir.name, label_dir.name
    elif split == "test_b":
        base = ROOT / "data" / "example_hinglish_fixed" / "test_b" / "hinglish"
        for label_dir in sorted(base.iterdir()):
            for f in sorted(label_dir.glob("*.flac")):
                yield f, "tts-hinglish", label_dir.name
    else:
        raise ValueError(f"unknown split {split}")


def load_16k(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio.astype(np.float32)


def trailing_silence(audio):
    """Seconds of trailing silence under the shared 8 s contract."""
    audio = truncate_audio_to_last_n_seconds(audio, n_seconds=8)
    win, hop = int(FRAME_S * 16000), int(HOP_S * 16000)
    if len(audio) < win:
        return 0.0
    n = 1 + (len(audio) - win) // hop
    rms = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(audio, win)[::hop] ** 2, axis=1))
    tail = 0
    for r in rms[::-1]:
        if r < ENERGY_FLOOR:
            tail += 1
        else:
            break
    return tail * HOP_S


def prf(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, zero_division=0)


def filler_of(folder):
    parts = folder.split("-")
    return (parts[1], parts[2]) if len(parts) == 3 else ("?", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test_a", "test_b"])
    ap.add_argument("--max", type=int, default=0, help="cap clips per split (0 = all)")
    ap.add_argument("--skip-zero-shot", action="store_true")
    ap.add_argument("--skip-silence", action="store_true")
    args = ap.parse_args()

    predict_endpoint = None
    if not args.skip_zero_shot:
        from inference import predict_endpoint  # builds ONNX session once (CWD = ROOT)

    RESULTS_DIR.mkdir(exist_ok=True)
    summaries, sweep_rows = [], []

    for split in args.splits:
        rows = []
        t_start = time.perf_counter()
        for i, (path, lang, folder) in enumerate(iter_clips(split)):
            if args.max and i >= args.max:
                break
            row = {
                "path": str(path.relative_to(ROOT)),
                "lang": lang,
                "folder": folder,
                "label": 1 if folder.startswith("complete") else 0,
                "midfiller": filler_of(folder)[0],
                "endfiller": filler_of(folder)[1],
            }
            audio = load_16k(path)
            if not args.skip_silence:
                row["tail_sil"] = trailing_silence(audio)
            if predict_endpoint is not None:
                t0 = time.perf_counter()
                out = predict_endpoint(audio)
                row["prob"] = out["probability"]
                row["pred"] = out["prediction"]
                row["infer_ms"] = (time.perf_counter() - t0) * 1000
            rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"  [{split}] {i + 1} clips...", flush=True)

        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / f"preds_{split}.csv", index=False)
        print(f"[{split}] {len(df)} clips in {time.perf_counter() - t_start:.0f}s -> results/preds_{split}.csv")

        y = df["label"].to_numpy()
        if predict_endpoint is not None:
            acc, f1 = prf(y, df["pred"].to_numpy())
            lat = df["infer_ms"].median()
            summaries.append(("zero-shot smart-turn-v3.2-int8", split, len(df), acc, f1, lat))
            for col in ("lang", "midfiller", "endfiller"):
                for val, g in df.groupby(col):
                    a, f = prf(g["label"], g["pred"])
                    print(f"    zs {col}={val}: n={len(g)} acc={a:.3f} f1={f:.3f}")

        if not args.skip_silence:
            sweep = [(T, *prf(y, (df["tail_sil"] >= T).astype(int).to_numpy())) for T in T_GRID]
            best_T, best_f1, best_acc = max(sweep, key=lambda r: (r[2], r[1]))
            sweep_rows.extend((split, T, a, f) for T, a, f in sweep)
            acc, f1 = prf(y, (df["tail_sil"] >= best_T).astype(int).to_numpy())
            summaries.append((f"silence-threshold T={best_T:.2f}s", split, len(df), acc, f1, np.nan))
            print(f"    silence best T={best_T:.2f}s: acc={acc:.3f} f1={f1:.3f}")

    print("\n=== Summary (baseline x split) ===")
    for name, split, n, acc, f1, lat in summaries:
        lat_s = f" p50={lat:.0f}ms" if not np.isnan(lat) else ""
        print(f"{name:38s} {split:7s} n={n:5d} acc={acc:.3f} f1={f1:.3f}{lat_s}")

    pd.DataFrame(sweep_rows, columns=["split", "T_s", "acc", "f1"]).to_csv(
        RESULTS_DIR / "silence_sweep.csv", index=False
    )


if __name__ == "__main__":
    main()
