import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "test_c" / "mucs-hinglish"

preds = {}
with open(ROOT / "results" / "preds_test_c.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        preds[r["path"]] = r  # columns: path, label, pred (check actual)

if not preds:
    print(open(ROOT / "results" / "preds_test_c.csv", encoding="utf-8").readline())
    sys.exit(1)

man = {}
with open(ROOT / "data" / "splits" / "test_c_manifest.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        man[r["segment_id"]] = r

key = list(preds.values())[0]
print("pred csv cols:", list(key.keys()))

stats = {0: {}, 1: {}}
for p, r in preds.items():
    path = ROOT / p if not Path(p).is_absolute() else Path(p)
    sid = path.stem
    if sid not in man:
        continue
    label = int(r["label"])
    wav, _ = sf.read(path, dtype="float32")
    tail = wav[-int(0.2 * 16000):]
    tail_rms = float(np.sqrt((tail**2).mean())) if len(tail) else 0.0
    pred = int(float(r.get("pred", r.get("pred_label", 0))))
    b = stats[label]
    b.setdefault("rms", []).append(tail_rms)
    b.setdefault("pred", []).append(pred)

for label in (0, 1):
    b = stats[label]
    rms = np.array(b["rms"])
    pred = np.array(b["pred"])
    med = np.median(rms)
    hi = rms > med
    print(
        f"label={label} n={len(rms)} | tail_rms median={med:.4f} | "
        f"pred=1 rate: hi-tail {pred[hi].mean():.3f} vs lo-tail {pred[~hi].mean():.3f}"
    )
