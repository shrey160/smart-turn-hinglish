"""
Download the Smart Turn v3.2 data subsets per playbook Part 5.2.

Targets (~1.9 GB total):
  train pool:  hin 6000 | eng 2000 | mar 400 | ben 400   (50:50 complete/incomplete each)
  test-A eval: v3.2-test hin+eng subset, ~1200 total

Strategy: HF streaming (Option A from the playbook). Rows stream through;
only matching rows are written to disk. Network traffic > disk usage because
non-matching row-groups still download - that is the accepted cost of not
pulling 41 GB.

Usage:
    uv run python scripts/download_data.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from dotenv import load_dotenv
from datasets import Audio, load_dataset

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

TRAIN_TARGETS = {  # language -> {endpoint_bool: quota}
    "hin": {True: 3000, False: 3000},
    "eng": {True: 1000, False: 1000},
    "mar": {True: 200, False: 200},
    "ben": {True: 200, False: 200},
}
TEST_TARGETS = {
    "hin": {True: 300, False: 300},
    "eng": {True: 300, False: 300},
}

TRAIN_REPO = "pipecat-ai/smart-turn-data-v3.2-train"
TEST_REPO = "pipecat-ai/smart-turn-data-v3.2-test"


def out_dir(split: str, lang: str, endpoint: bool, midfiller: bool, endfiller: bool) -> Path:
    label = f"{'complete' if endpoint else 'incomplete'}-{midfiller}-{endfiller}"
    return DATA_DIR / split / lang / label


def scan_existing(split: str, targets: dict) -> dict:
    """Count already-downloaded files so re-runs resume where they left off."""
    counts = {lang: {True: 0, False: 0} for lang in targets}
    base = DATA_DIR / split
    if not base.exists():
        return counts
    for f in base.rglob("*.flac"):
        lang = f.relative_to(base).parts[0]
        if lang in counts:
            # startswith, not substring: "incomplete-..." also contains "complete-"
            ep = f.parent.name.startswith("complete-")
            counts[lang][ep] += 1
    return counts


def remaining(targets: dict, counts: dict) -> dict:
    return {
        lang: {ep: max(0, targets[lang][ep] - counts[lang][ep]) for ep in (True, False)}
        for lang in targets
    }


def download_split(repo: str, split_name: str, targets: dict):
    counts = scan_existing(split_name, targets)
    print(f"\n=== {repo} -> data/{split_name}/ ===")
    for lang in sorted(targets):
        c = counts[lang]
        print(f"  existing {lang}: complete={c[True]} incomplete={c[False]} "
              f"(targets {targets[lang][True]}/{targets[lang][False]})")

    rem = remaining(targets, counts)
    if all(v == 0 for cell in rem.values() for v in cell.values()):
        print("  all quotas met, skipping.")
        return

    # both repos expose their single split as "train" (verified via datasets error)
    ds = load_dataset(repo, split="train", streaming=True, token=HF_TOKEN)
    # raw FLAC bytes, no torchcodec/FFmpeg decoding
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"  streaming (total rows upstream unknown; Ctrl+C safe to interrupt & re-run)")

    filler_stats = defaultdict(lambda: [0, 0])  # (lang,ep) -> [n_midfiller, n_endfiller]
    seen = scanned = 0

    try:
        for ex in ds:
            scanned += 1
            lang = ex["language"]
            if lang not in targets:
                continue
            ep = bool(ex["endpoint_bool"])
            if rem[lang][ep] <= 0:
                continue

            mf, ef = bool(ex["midfiller"]), bool(ex["endfiller"])
            audio = ex["audio"]
            path = out_dir(split_name, lang, ep, mf, ef)
            path.mkdir(parents=True, exist_ok=True)
            fname = path / f"{ex['id']}.flac"

            # audio bytes arrive as dict with either 'path' (cached file) or raw bytes
            if "bytes" in audio and audio["bytes"] is not None:
                fname.write_bytes(audio["bytes"])
            else:
                import shutil
                shutil.copyfile(audio["path"], fname)

            rem[lang][ep] -= 1
            counts[lang][ep] += 1
            filler_stats[(lang, ep)][0] += int(mf)
            filler_stats[(lang, ep)][1] += int(ef)

            seen += 1
            if seen % 250 == 0:
                tot_left = sum(v for cell in rem.values() for v in cell.values())
                print(f"  saved {seen} new (scanned {scanned}) | rows left needed: {tot_left}")

            if all(v == 0 for cell in rem.values() for v in cell.values()):
                print("  all quotas met.")
                break
    except KeyboardInterrupt:
        print("\ninterrupted - progress saved, re-run the same command to resume.")

    print(f"\n--- final counts for {split_name} (scanned {scanned} rows) ---")
    for lang in sorted(targets):
        c = counts[lang]
        m = filler_stats[(lang, True)]
        n = filler_stats[(lang, False)]
        print(f"  {lang}: complete={c[True]}/{targets[lang][True]} incomplete={c[False]}/{targets[lang][False]}"
              f" | midfiller c/i={m[0]}/{n[0]} endfiller c/i={m[1]}/{n[1]}")


def main():
    # usage: download_data.py [--only train_pool|test_a]
    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            only = sys.argv[idx + 1]
    if only in ("train_pool", "test_a"):
        splits = [only]
    else:
        splits = ["train_pool", "test_a"]

    if "train_pool" in splits:
        download_split(TRAIN_REPO, "train_pool", TRAIN_TARGETS)
    if "test_a" in splits:
        download_split(TEST_REPO, "test_a", TEST_TARGETS)

    # summary table
    print("\n================ DATA SUMMARY ================")
    rows = []
    for split_name in splits:
        base = DATA_DIR / split_name
        size = sum(f.stat().st_size for f in base.rglob("*") if f.is_file()) / 1e6
        n = len(list(base.rglob("*.flac")))
        rows.append((split_name, n, f"{size:.0f} MB"))
    print(f"{'split':<12} {'clips':>8} {'size':>10}")
    for r in rows:
        print(f"{r[0]:<12} {r[1]:>8} {r[2]:>10}")


if __name__ == "__main__":
    sys.exit(main())
