"""Evaluation: dev + test-A/B/C accuracy/F1 with per-language and per-filler slices + latency."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turn_v2.data.dataset import (  # noqa: E402
    TurnDataset,
    collate_turn,
    dev_v1_clips,
    test_a_clips,
    test_b_clips,
    test_c_clips,
    tts_clips,
)
from turn_v2.models.model import SmartTurnV2Model  # noqa: E402


def split_clips(name):
    if name == "dev":
        return dev_v1_clips(("hin", "eng"))
    if name == "test_a":
        return test_a_clips()
    if name == "test_b":
        return test_b_clips()
    if name == "test_c":
        return test_c_clips()
    if name == "tts_all":
        return tts_clips()
    raise ValueError(f"unknown split '{name}'")


@torch.no_grad()
def evaluate_model(model, clips, device, batch_size=128, workers=0, max_clips=0, seed=42):
    if not clips:
        return None
    if max_clips:
        clips = clips[:max_clips]
    ds = TurnDataset(clips, cache=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_turn, num_workers=workers)
    model.eval().to(device)
    probs, labels, langs, mids, ends, paths = [], [], [], [], [], []
    e2e_ms, fwd_ms = [], []
    for batch in loader:
        t0 = time.perf_counter()
        x = batch["input_features"].to(device)
        t1 = time.perf_counter()
        logits = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        n = x.size(0)
        e2e_ms.append((t2 - t0) * 1000 / n)
        fwd_ms.append((t2 - t1) * 1000 / n)
        probs.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(int(v) for v in batch["label"])
        langs.extend(batch["lang"])
        mids.extend(batch["midfiller"])
        ends.extend(batch["endfiller"])
        paths.extend(batch["path"])
    y = np.array(labels)
    p = np.array(probs)
    pred = (p >= 0.5).astype(int)
    result = {
        "n": len(y),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "latency_ms_p50_e2e": float(np.median(e2e_ms)),
        "latency_ms_p50_fwd": float(np.median(fwd_ms)),
        "probs": p,
        "labels": y,
        "preds": pred,
        "langs": langs,
        "mids": mids,
        "ends": ends,
        "paths": paths,
    }
    result["slices"] = {}
    for col, vals in (("lang", langs), ("midfiller", mids), ("endfiller", ends)):
        for v in sorted(set(vals)):
            mask = np.array([x == v for x in vals])
            result["slices"][f"{col}={v}"] = (
                int(mask.sum()),
                float(accuracy_score(y[mask], pred[mask])),
                float(f1_score(y[mask], pred[mask], zero_division=0)),
            )
    return result


def print_eval(name, r, show_slices=True):
    if r is None:
        print(f"[{name}] no clips — skipped")
        return
    print(
        f"[{name}] n={r['n']} acc={r['acc']:.3f} f1={r['f1']:.3f} "
        f"lat p50 e2e={r['latency_ms_p50_e2e']:.1f}ms fwd={r['latency_ms_p50_fwd']:.1f}ms"
    )
    if show_slices:
        for k, (n, a, f) in r["slices"].items():
            print(f"    {k:22s} n={n:5d} acc={a:.3f} f1={f:.3f}")


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = SmartTurnV2Model(
        base=ckpt.get("base", "openai/whisper-tiny"),
        pooling=ckpt.get("pooling", "attention-mean"),
        freeze_encoder=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["test_a", "test_b"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} (stage={ckpt.get('stage')} pooling={ckpt.get('pooling')}) on {device}")
    for split in args.splits:
        print_eval(split, evaluate_model(model, split_clips(split), device, args.batch_size, args.workers, args.max))


if __name__ == "__main__":
    main()
