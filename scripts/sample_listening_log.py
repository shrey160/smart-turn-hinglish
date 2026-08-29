"""Sample clips for the human listening log (playbook Day 1, masterplan P0).

Stratified, seed-42 sampling from train_pool (hin/eng/mar/ben) + TTS example set.
Writes context/listening_log.md with per-clip audio stats and an empty notes column.
Run:  uv run python scripts/sample_listening_log.py
"""
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "context" / "listening_log.md"
SEED = 42
QUOTA = {  # lang-dir -> n_clips (label-balanced below)
    "hin": 30, "eng": 15, "mar": 5, "ben": 5,
}
TTS_QUOTA = 5

rng = random.Random(SEED)


def clip_stats(path: Path, sr_expect: int = 16000):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    # trailing silence: 20ms frames from the end below 5% of clip RMS (min floor)
    frame = int(0.02 * sr)
    thresh = max(0.005, 0.05 * rms)
    tail = 0.0
    i = len(audio)
    while i - frame >= 0 and float(np.sqrt(np.mean(audio[i - frame : i] ** 2))) < thresh:
        tail += 0.02
        i -= frame
    return sr, dur, rms, peak, tail


def sample_label_balanced(group_dirs: list[Path], n: int) -> list[Path]:
    """Split clips into complete* / incomplete* and take n/2 from each."""
    comp, inc = [], []
    for d in group_dirs:
        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            bucket = comp if sub.name.startswith("complete") else inc
            bucket.extend(sub.glob("*.flac"))
    take = [(comp, n - n // 2), (inc, n // 2)]
    out = []
    for pool, k in take:
        out.extend(rng.sample(pool, min(k, len(pool))))
    return out


rows = []
for lang, n in QUOTA.items():
    lang_dir = ROOT / "data" / "train_pool" / lang
    clips = sample_label_balanced([lang_dir], n)
    rng.shuffle(clips)
    rows.extend((lang, p) for p in clips)

tts_dirs = [ROOT / "data" / "example_hinglish_fixed" / "train_pool" / "hinglish"]
clips = sample_label_balanced(tts_dirs, TTS_QUOTA)
rows.extend(("tts-hinglish", p) for p in clips)

lines = [
    "# Listening Log — Day-1 audit (masterplan P0)",
    "",
    "Sampled with seed 42 by `scripts/sample_listening_log.py`. Annotate the",
    "`notes` column after listening: naturalness, pure-Hindi vs code-mixed,",
    "filler realism (midfiller/endfiller), background noise, truncation artifacts.",
    "",
    "| # | clip | lang | label | midfiller | endfiller | dur_s | rms | peak | tail_sil_s | notes |",
    "|---|------|------|-------|-----------|-----------|-------|-----|------|------------|-------|",
]
for idx, (lang, p) in enumerate(rows, 1):
    parts = p.parent.name.split("-")  # e.g. incomplete-False-True
    label, midf, endf = parts[0], parts[1], parts[2] if len(parts) == 3 else "?"
    sr, dur, rms, peak, tail = clip_stats(p)
    rel = p.relative_to(ROOT)
    assert sr == 16000, f"unexpected sample rate {sr} in {rel}"
    lines.append(
        f"| {idx} | `{rel}` | {lang} | {label} | {midf} | {endf} "
        f"| {dur:.2f} | {rms:.4f} | {peak:.2f} | {tail:.2f} |  |"
    )

LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {LOG_PATH} with {len(rows)} clips")
