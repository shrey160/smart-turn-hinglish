"""
Split the restructured example Hinglish dataset 80:20 into train/test.

Moves 20% of clips per label folder from train_pool/ to test_b/ (stratified by
label, deterministic seed 42), rewrites dataset_fixed.csv with a `split` column
and updated `audio_flac` paths, refreshes conversion_map.csv paths, then
verifies counts on disk.

Note: all 165 scripts are unique, so no script-overlap risk between splits
(test-B discipline from the playbook is satisfied by construction here).

Usage:
    uv run python scripts/split_example_hinglish.py
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data" / "example_hinglish_fixed"
SEED = 42
TEST_FRACTION = 0.2


def label_name(row) -> str:
    endpoint = row["endpoint_bool"] == "True"
    midfiller = row["midfiller"] == "True"
    endfiller = row["endfiller"] == "True"
    return f"{'complete' if endpoint else 'incomplete'}-{midfiller}-{endfiller}"


def main():
    with open(DST / "dataset_fixed.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"rows in dataset_fixed.csv: {len(rows)}")

    # group ids per label folder, sorted for stable ordering
    by_label = defaultdict(list)
    for r in rows:
        by_label[label_name(r)].append(r["id"])
    for ids in by_label.values():
        ids.sort()

    rng = random.Random(SEED)
    test_ids = set()
    for label in sorted(by_label):
        ids = by_label[label]
        n_test = round(len(ids) * TEST_FRACTION)
        rng.shuffle(ids)
        chosen = ids[:n_test]
        test_ids.update(chosen)
        print(f"  {label}: total={len(ids)} -> test={n_test}")

    # move files + update rows
    moved = 0
    for r in rows:
        label = label_name(r)
        new_split = "test_b" if r["id"] in test_ids else "train_pool"
        src = DST / r["audio_flac"]
        dst_dir = DST / new_split / "hinglish" / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{r['id']}.flac"
        if src != dst:
            shutil.move(str(src), str(dst))
            moved += 1
        r["split"] = new_split
        r["audio_flac"] = str(dst.relative_to(DST)).replace("\\", "/")

    print(f"files moved: {moved}")

    with open(DST / "dataset_fixed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # refresh conversion_map paths
    flac_by_id = {r["id"]: r["audio_flac"] for r in rows}
    with open(DST / "conversion_map.csv", newline="", encoding="utf-8") as f:
        map_rows = list(csv.DictReader(f))
    for m in map_rows:
        m["flac_audio"] = flac_by_id[m["id"]]
    with open(DST / "conversion_map.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(map_rows[0].keys()))
        writer.writeheader()
        writer.writerows(map_rows)

    # verification: disk counts vs csv, per split x label
    print("\n--- verification ---")
    csv_counts = defaultdict(int)
    for r in rows:
        csv_counts[(r["split"], label_name(r))] += 1
    ok = True
    for split in ("train_pool", "test_b"):
        base = DST / split / "hinglish"
        disk = {d.name: len(list(d.glob("*.flac"))) for d in sorted(base.iterdir()) if d.is_dir()}
        for label, n_disk in disk.items():
            n_csv = csv_counts[(split, label)]
            status = "OK" if n_disk == n_csv else "MISMATCH"
            if n_disk != n_csv:
                ok = False
            print(f"  {split}/{label}: disk={n_disk} csv={n_csv} {status}")
    total_disk = len(list(DST.rglob("*.flac")))
    print(f"  total flac on disk: {total_disk} (expected {len(rows)})")
    if not ok or total_disk != len(rows):
        raise SystemExit("verification failed")
    print("  all checks passed")


if __name__ == "__main__":
    main()
