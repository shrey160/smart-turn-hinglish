"""Write data/tts_hinglish/pilot_report.md from the manifest (listening gate).

Run:  $env:PYTHONIOENCODING="utf-8"; uv run python scripts/tts_report.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
mdf = pd.read_csv(ROOT / "data" / "tts_hinglish" / "manifest.csv")
p = mdf[mdf["pilot"] == True]  # noqa: E712

lines = [
    "# TTS Hinglish — Pilot listening report (100 clips)",
    "",
    "**Listening gate before scaling to 3-4k + test-B.** For each clip judge:",
    "(1) naturalness of code-mixing, (2) filler realism (midfiller/endfiller),",
    "(3) completeness contrast (does a complete clip actually sound finished?),",
    "(4) pronunciation of English tokens in hi voices / Hindi words in en voices,",
    "(5) truncation artifacts (>8s clips; loader keeps last 8s).",
    "Verdict per clip: keep / re-voice / re-script / drop.",
    "",
    "## Counts per label-folder x voice",
    "",
    p.groupby(["label_folder", "voice"]).size().unstack(fill_value=0).to_markdown(),
    "",
    "## Durations (s)",
    "",
    p["duration_s"].describe().round(2).to_markdown(),
    "",
    "## Clips > 8s (status=too_long; loader applies last-8s contract)",
    "",
    (p[p["status"] == "too_long"][["file", "duration_s"]].to_markdown(index=False)
     if (p["status"] == "too_long").any() else "none"),
    "",
    "## Full clip list",
    "",
    "| file | voice | variant | label_folder | dur_s | text |",
    "|---|---|---|---|---|---|",
]
for _, r in p.iterrows():
    lines.append(
        f"| {r['file']} | {r['voice']} | {r['variant']} | {r['label_folder']} "
        f"| {r['duration_s']} | {str(r['text'])[:60]} |"
    )

out = ROOT / "data" / "tts_hinglish" / "pilot_report.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"report written: {out} ({len(p)} clips)")
