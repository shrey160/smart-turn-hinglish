"""
Restructure the example TTS Hinglish dataset into the Smart Turn training layout.

Reads   data/data_result_example/   (dataset_tts.csv + audio/*.mp3)
Writes  data/example_hinglish_fixed/
         train_pool/hinglish/{complete|incomplete}-{midfiller}-{endfiller}/{uuid}.flac
         dataset_fixed.csv        (original columns + Smart Turn schema fields)
         conversion_map.csv       (uuid <-> original file mapping)

Only structural changes (per user request):
  - MP3 -> FLAC (lossless decode + re-encode, 16 kHz mono float32 preserved)
  - label directories parsed from completeness/ending columns
  - UUID filenames (uuid5, deterministic from original name)
Audio content is NOT altered (no silence padding, no trimming, no resampling).

Usage:
    uv run python scripts/structure_example_hinglish.py
"""

import csv
import uuid
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "data_result_example"
DST = ROOT / "data" / "example_hinglish_fixed"
LANG = "hinglish"

# deterministic namespace so uuids are stable across re-runs
UUID_NS = uuid.uuid5(uuid.NAMESPACE_URL, "hinglish-tts-example")


def label_parts(completeness: str, ending) -> tuple[bool, bool, bool]:
    """Map (completeness, ending) -> (endpoint_bool, midfiller, endfiller)."""
    endpoint = completeness == "complete"
    # midfiller not annotated in source -> conservative False
    # filler/connective endings = endfiller; trailoff = nofiller incomplete
    endfiller = (not endpoint) and ending in ("filler", "connective")
    return endpoint, False, endfiller


def out_dir(endpoint: bool, midfiller: bool, endfiller: bool) -> Path:
    label = f"{'complete' if endpoint else 'incomplete'}-{midfiller}-{endfiller}"
    return DST / "train_pool" / LANG / label


def main():
    src_csv = SRC / "dataset_tts.csv"
    with open(src_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"source rows: {len(rows)}")

    DST.mkdir(parents=True, exist_ok=True)

    fixed_rows, map_rows, counts = [], [], {}
    for i, r in enumerate(rows):
        src_path = SRC / r["audio"].replace("\\", "/")
        endpoint, midfiller, endfiller = label_parts(r["completeness"], r["ending"] or "")

        uid = str(uuid.uuid5(UUID_NS, src_path.stem))
        dst_dir = out_dir(endpoint, midfiller, endfiller)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / f"{uid}.flac"

        audio, sr = sf.read(str(src_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sf.write(str(dst_path), audio, sr, format="FLAC")

        fixed_rows.append({
            **r,
            "id": uid,
            "endpoint_bool": endpoint,
            "midfiller": midfiller,
            "endfiller": endfiller,
            "audio_flac": str(dst_path.relative_to(DST)).replace("\\", "/"),
        })
        map_rows.append({
            "id": uid,
            "original_audio": r["audio"],
            "flac_audio": str(dst_path.relative_to(DST)).replace("\\", "/"),
        })

        key = dst_dir.name
        counts[key] = counts.get(key, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  converted {i + 1}/{len(rows)}")

    # fixed-schema CSV (same column order as source + derived fields)
    with open(DST / "dataset_fixed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fixed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fixed_rows)

    with open(DST / "conversion_map.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(map_rows[0].keys()))
        writer.writeheader()
        writer.writerows(map_rows)

    print("\n--- conversion summary ---")
    total = 0
    for label in sorted(counts):
        n = counts[label]
        total += n
        print(f"  {label}: {n}")
    print(f"  total: {total}")

    # verify audio stats (working rule: verify after every data step)
    print("\n--- verification ---")
    flacs = list((DST / "train_pool").rglob("*.flac"))
    assert len(flacs) == len(rows), f"file count mismatch: {len(flacs)} != {len(rows)}"
    bad = 0
    for fp in flacs:
        info = sf.info(str(fp))
        if info.samplerate != 16000 or info.channels != 1 or info.format != "FLAC":
            bad += 1
            print(f"  BAD: {fp.name} sr={info.samplerate} ch={info.channels} fmt={info.format}")
    print(f"  flac files: {len(flacs)}, non-conforming: {bad}")
    size_mb = sum(f.stat().st_size for f in flacs) / 1e6
    print(f"  total size: {size_mb:.1f} MB (source mp3: "
          f"{sum(f.stat().st_size for f in (SRC / 'audio').glob('*.mp3')) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
