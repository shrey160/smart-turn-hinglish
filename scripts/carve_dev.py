"""Carve the dev split from train_pool — 10% per lang x label-folder, seed 42.

Manifest-based: files stay in place (download resumability preserved); the
dev manifest doubles as the held-out int8 quantization calibration set
(masterplan P2/P5). Training set = train_pool minus dev manifest.

Run:  uv run python scripts/carve_dev.py
Out:  data/splits/dev_v1.csv  (rel_path, lang, label_folder, midfiller, endfiller)
"""
import random
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_POOL = ROOT / "data" / "train_pool"
OUT = ROOT / "data" / "splits" / "dev_v1.csv"
SEED, FRAC = 42, 0.10

rows = []
for lang_dir in sorted(TRAIN_POOL.iterdir()):
    for label_dir in sorted(lang_dir.iterdir()):
        clips = sorted(label_dir.glob("*.flac"))
        k = round(len(clips) * FRAC)
        for f in random.Random(SEED).sample(clips, k):
            parts = label_dir.name.split("-")
            rows.append({
                "rel_path": str(f.relative_to(ROOT)),
                "lang": lang_dir.name,
                "label_folder": label_dir.name,
                "label": 1 if label_dir.name.startswith("complete") else 0,
                "midfiller": parts[1] if len(parts) == 3 else "?",
                "endfiller": parts[2] if len(parts) == 3 else "?",
            })

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
c = Counter((r["lang"], r["label_folder"]) for r in rows)
print(f"dev_v1: {len(rows)} clips (10% of 8,800)")
for k in sorted(c):
    print(" ", k, c[k])
