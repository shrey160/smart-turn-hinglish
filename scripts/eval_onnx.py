"""Evaluate existing ONNX files on a split without re-quantizing.

Usage:
  uv run python scripts/eval_onnx.py --split test_a --paths a.fp32.onnx a.int8.onnx
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turn_v2.data.dataset import test_a_clips, test_b_clips, test_c_clips  # noqa: E402
from turn_v2.export import ort_eval, ort_session  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_a", choices=["test_a", "test_b", "test_c"])
    ap.add_argument("--paths", nargs="+", required=True)
    args = ap.parse_args()

    clips = {"test_a": test_a_clips, "test_b": test_b_clips, "test_c": test_c_clips}[args.split]()
    for p in args.paths:
        r = ort_eval(ort_session(p), clips)
        print(f"[{args.split}] {Path(p).name:40s} acc={r['acc']:.3f} f1={r['f1']:.3f} p50={r['lat_ms']:.1f}ms")


if __name__ == "__main__":
    main()