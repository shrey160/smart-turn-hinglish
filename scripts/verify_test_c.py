from collections import Counter
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turn_v2.data.dataset import test_c_clips, load_audio

clips = test_c_clips()
print("loader n =", len(clips), Counter((c["label"], c["midfiller"], c["endfiller"]) for c in clips))
print("lang set:", {c["lang"] for c in clips})

inc = [c for c in clips if c["label"] == 0][:3]
man = {r["segment_id"]: r for r in csv.DictReader(open("data/splits/test_c_manifest.csv", encoding="utf-8"))}
for c in inc:
    sid = c["path"].replace("\\", "/").split("/")[-1].replace(".wav", "")
    m = man[sid]
    wav = load_audio(c["path"])
    print(
        f"{sid}: dur={len(wav)/16000:.2f}s orig={float(m['orig_dur_s']):.2f} "
        f"cut_word_idx={m['cut_word_idx']} rms={np.sqrt((wav**2).mean()):.4f} "
        f"last10ms_rms={np.sqrt((wav[-160:]**2).mean()):.5f}"
    )
    print("   words kept:", " ".join(m["transcript"].split()[: int(m["cut_word_idx"])])[:100])

comp = [c for c in clips if c["label"] == 1][:2]
for c in comp:
    wav = load_audio(c["path"])
    print(f"{c['path'].split('/')[-1]}: dur={len(wav)/16000:.2f}s rms={np.sqrt((wav**2).mean()):.4f}")
