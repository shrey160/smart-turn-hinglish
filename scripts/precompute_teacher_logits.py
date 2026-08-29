"""Precompute teacher logits for KD (P5).

Loads a teacher checkpoint (any base), runs inference over the full s2 pool
(tts_train + hin_pool + eng_pool, same construction as build_stage_data s2),
and saves {paths: [...], logits: [...]} as npz keyed by clip rel path.
Teacher sees CLEAN audio (no augmentation) — stable soft targets.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turn_v2.data.dataset import TurnDataset, collate_turn, tts_clips, carve_tts_dev, train_pool_clips  # noqa: E402
from turn_v2.evaluate import load_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ROOT / "turn_v2" / "kd" / "teacher_logits.npz"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tts_train, _ = carve_tts_dev(tts_clips())
    hin = train_pool_clips(("hin",), exclude_dev=True)
    eng = train_pool_clips(("eng",), exclude_dev=True)
    clips = tts_train + hin + eng
    print(f"teacher pool: {len(tts_train)} tts + {len(hin)} hin + {len(eng)} eng = {len(clips)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.ckpt, device=device)
    print(f"loaded {args.ckpt} (base={ckpt.get('base')}, dev_f1={ckpt.get('dev_f1')}, device={device})")

    ds = TurnDataset(clips, augment=None, cache=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_turn, num_workers=args.workers)
    paths, logits = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["input_features"].to(device)
            out = model(x)
            logits.extend(out.cpu().numpy().tolist())
            paths.extend(batch["path"])
            if (i + 1) % 20 == 0:
                print(f"  {len(paths)}/{len(clips)}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, paths=np.array(paths), logits=np.array(logits, dtype=np.float32))
    print(f"saved {len(paths)} logits -> {out_path}")


if __name__ == "__main__":
    main()
