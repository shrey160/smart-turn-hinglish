import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

rows = list(csv.DictReader(open(ROOT / "results" / "preds_test_c.csv", encoding="utf-8")))
by_label = {0: [], 1: []}
for r in rows:
    by_label[int(r["label"])].append(r)

for label in (1, 0):
    ts = np.array([float(r["tail_sil"]) for r in by_label[label]])
    print(
        f"label={label} n={len(ts)} | tail_sil s: mean={ts.mean():.3f} "
        f"p25={np.percentile(ts,25):.3f} p50={np.percentile(ts,50):.3f} p75={np.percentile(ts,75):.3f} "
        f"| frac <0.10s: {(ts<0.10).mean():.3f} | frac <0.20s: {(ts<0.20).mean():.3f}"
    )
