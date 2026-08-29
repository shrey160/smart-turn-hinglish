"""P6 WS-B: decision-threshold sweep -> false-interrupt vs added-latency Pareto frontier.

Sweeps tau over v2-core (s2-004) probabilities per split; reports per tau:
  FIR   = P(pred=complete | incomplete)  (costly error: agent interrupts user)
  HOLD  = P(pred=incomplete | complete)  (added-latency proxy: each held turn
          costs one extra VAD re-check before the agent responds)
plus acc/F1/balanced-acc and FIR slices (lang, midfiller). The reference's
operating point (zero-shot ONNX at tau=0.5) is reported on the same features.

Usage:
  uv run python turn_v2/policy.py
  uv run python turn_v2/policy.py --splits test_b --max 16          # smoke

Outputs: results/policy_sweep_<split>.csv, results/policy_frontier_<split>.csv.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turn_v2.data.dataset import test_a_clips, test_b_clips, test_c_clips  # noqa: E402
from turn_v2.eval_common import confusion, predict_both, ref_session  # noqa: E402

SPLIT_CLIPS = {"test_a": test_a_clips, "test_b": test_b_clips, "test_c": test_c_clips}
RESULTS = ROOT / "results"


def fir_slice(y, pred, mask):
    if mask.sum() == 0 or (y[mask] == 0).sum() == 0:
        return float("nan")
    return float(np.mean(pred[mask][y[mask] == 0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(ROOT / "turn_v2" / "ckpt" / "s2-004" / "best.pt"))
    ap.add_argument("--splits", nargs="+", default=["test_a", "test_b", "test_c"])
    ap.add_argument("--taus", nargs="+", type=float, default=[round(t, 2) for t in np.arange(0.05, 0.96, 0.05)])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    import torch

    from turn_v2.evaluate import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.ckpt, device)
    sess = ref_session()
    print(f"loaded {args.ckpt} on {device}")

    RESULTS.mkdir(exist_ok=True)
    for split in args.splits:
        clips = SPLIT_CLIPS[split]()
        if args.max:
            clips = clips[: args.max]
        r = predict_both(model, sess, clips, device, args.batch_size, args.workers)
        y = r["labels"]
        langs = np.array(r["langs"])
        mids = np.array(r["mids"])
        print(f"\n[{split}] n={len(y)} (incomplete={int((y == 0).sum())}, complete={int((y == 1).sum())})")

        sweep = []
        for tau in args.taus:
            pred = (r["probs"] >= tau).astype(int)
            c = confusion(y, pred)
            row = {
                "tau": tau,
                "fir": c["fir"],
                "hold": c["hold"],
                "acc": c["acc"],
                "f1": c["f1"],
                "bacc": 1 - (c["fir"] + c["hold"]) / 2,
                "fir_midfiller": fir_slice(y, pred, mids == True),  # noqa: E712
                **{f"fir_lang_{lg}": fir_slice(y, pred, langs == lg) for lg in sorted(set(r["langs"]))},
            }
            sweep.append(row)

        # Pareto frontier: non-dominated (minimize FIR and HOLD)
        frontier = []
        for i, row in enumerate(sweep):
            dominated = any(
                (o["fir"] <= row["fir"] and o["hold"] <= row["hold"])
                and (o["fir"] < row["fir"] or o["hold"] < row["hold"])
                for j, o in enumerate(sweep) if j != i
            )
            if not dominated:
                frontier.append(row)

        zs_pred = (r["ref_probs"] >= 0.5).astype(int)
        zc = confusion(y, zs_pred)
        print(
            f"  reference operating point (tau=0.5): acc={zc['acc']:.3f} f1={zc['f1']:.3f} "
            f"fir={zc['fir']:.3f} hold={zc['hold']:.3f}"
        )

        fields = list(sweep[0].keys())
        with open(RESULTS / f"policy_sweep_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sweep)
        with open(RESULTS / f"policy_frontier_{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(sorted(frontier, key=lambda x: x["fir"]))

        print(f"  {'tau':>5s} {'FIR':>6s} {'HOLD':>6s} {'acc':>6s} {'bacc':>6s}  frontier")
        for row in sweep:
            mark = " *" if row in frontier else ""
            print(
                f"  {row['tau']:>5.2f} {row['fir']:>6.3f} {row['hold']:>6.3f} "
                f"{row['acc']:>6.3f} {row['bacc']:>6.3f}{mark}"
            )
        knee = min(frontier, key=lambda x: x["fir"] + x["hold"])
        print(f"  frontier taus: {[row['tau'] for row in sorted(frontier, key=lambda x: x['fir'])]}")
        print(f"  knee (min FIR+HOLD): tau={knee['tau']:.2f} FIR={knee['fir']:.3f} HOLD={knee['hold']:.3f}")
        print(f"  wrote results/policy_sweep_{split}.csv + policy_frontier_{split}.csv")


if __name__ == "__main__":
    main()
