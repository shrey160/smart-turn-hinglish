"""P6 WS-A: accuracy/F1 vs trailing-context budget curve.

For each budget b in {0.5, 1, 2, 4, 8} s, keep only the LAST b seconds of every
clip (front zero-pad to 8 s per the shared contract — same feature path), then
evaluate v2-core (torch ckpt) and the zero-shot reference ONNX on test-A/B/C.

Usage:
  uv run python turn_v2/latency.py
  uv run python turn_v2/latency.py --splits test_b --max 16        # smoke
  uv run python turn_v2/latency.py --budgets 1 2 8                  # subset

Output: results/latency_curve.csv + printed tables (one per model).
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turn_v2.data.dataset import test_a_clips, test_b_clips, test_c_clips  # noqa: E402
from turn_v2.eval_common import confusion, predict_both, ref_session  # noqa: E402

SPLIT_CLIPS = {"test_a": test_a_clips, "test_b": test_b_clips, "test_c": test_c_clips}
MODELS = ("v2-core-s2-004", "zero-shot-ref")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(ROOT / "turn_v2" / "ckpt" / "s2-004" / "best.pt"))
    ap.add_argument("--splits", nargs="+", default=["test_a", "test_b", "test_c"])
    ap.add_argument("--budgets", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max", type=int, default=0, help="cap clips per split (0 = all)")
    ap.add_argument("--out", default=str(ROOT / "results" / "latency_curve.csv"))
    args = ap.parse_args()

    import torch

    from turn_v2.evaluate import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} on {device}")
    sess = ref_session()
    print(f"reference ONNX input '{sess.get_inputs()[0].name}' ready")

    rows = []
    for split in args.splits:
        clips = SPLIT_CLIPS[split]()
        if args.max:
            clips = clips[: args.max]
        print(f"[{split}] n={len(clips)}")
        for b in args.budgets:
            r = predict_both(
                model, sess, clips, device, args.batch_size, args.workers,
                truncate_s=None if b >= 8 else b,
            )
            preds = {
                MODELS[0]: (r["probs"] >= 0.5).astype(int),
                MODELS[1]: (r["ref_probs"] >= 0.5).astype(int),
            }
            for name, pred in preds.items():
                c = confusion(r["labels"], pred)
                rows.append({"split": split, "model": name, "budget_s": b, **{k: c[k] for k in ("n", "acc", "f1", "fir", "hold")}})
                print(
                    f"  b={b:>3.1f}s {name:16s} acc={c['acc']:.3f} f1={c['f1']:.3f} "
                    f"fir={c['fir']:.3f} hold={c['hold']:.3f}"
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")

    for name in MODELS:
        print(f"\n=== acc by split x budget — {name} ===")
        print(f"{'split':8s}" + "".join(f"  b={b:>3.1f}s" for b in args.budgets))
        for split in args.splits:
            cells = []
            for b in args.budgets:
                row = next((r for r in rows if r["split"] == split and r["model"] == name and r["budget_s"] == b), None)
                cells.append(f"  {row['acc']:>6.3f}" if row else "     n/a")
            print(f"{split:8s}" + "".join(cells))


if __name__ == "__main__":
    main()
