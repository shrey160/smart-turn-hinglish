import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(open(ROOT / "results" / "preds_test_c.csv", encoding="utf-8")))

bins = [(0.0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 1.01)]
print("label tail_sil_bin   n  pred=1(complete) rate   acc(pred==label)")
for label in (1, 0):
    for lo, hi in bins:
        sel = [r for r in rows if int(r["label"]) == label and lo <= float(r["tail_sil"]) < hi]
        if not sel:
            continue
        pred1 = np.mean([int(r["pred"]) == 1 for r in sel])
        acc = np.mean([int(r["pred"]) == int(r["label"]) for r in sel])
        print(
            f"  {label}   [{lo:.2f},{hi:.2f})  {len(sel):4d}        {pred1:.3f}                {acc:.3f}"
        )

conf = np.zeros((2, 2), dtype=int)
for r in rows:
    conf[int(r["label"]), int(r["pred"])] += 1
print("\nconfusion (rows=label 1=complete, cols=pred):")
print(conf)
