"""P6 WS-C: confusion slices + categorized failure analysis for v2-core and zero-shot.

Step 1 (slices): lang x midfiller x endfiller confusion slices per test set for
s2-004 (torch) and the zero-shot reference ONNX -> results/slices_v2core.csv.
Step 2 (failures): all v2-core failures + all zero-shot failures on test-B;
capped balanced sample (seed 42) of v2-core failures on test-C. Per clip:
y/pred, p(complete), duration, trailing-silence s (same RMS-floor logic as
eval_baselines.py), counterpart model's p; categorized:
  tail-cue:complete-short-tail   FN with tail < 0.3 s (model waits for longer silence)
  tail-cue:incomplete-long-tail  FP with tail >= 0.3 s (long silence fooled it)
  filler-confusion               midfiller/endfiller = True
  domain-gap:monologue           test-C residual (no turn-taking prosody)
  other
-> results/error_analysis_failures.csv.

Usage:
  uv run python scripts/error_analysis.py
  uv run python scripts/error_analysis.py --skip-slices --max-c-fails 8   # smoke
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
TAIL_CUE_S = 0.3  # neural operating point needs >0.3 s tails (Session 7 diagnosis)


def categorize(y, v2_pred, v2_p, tail_sil, split, mid, end):
    if y == 1 and v2_pred == 0 and tail_sil < TAIL_CUE_S:
        return "tail-cue:complete-short-tail"
    if y == 0 and v2_pred == 1 and tail_sil >= TAIL_CUE_S:
        return "tail-cue:incomplete-long-tail"
    if mid or end:
        return "filler-confusion"
    if split == "test_c":
        return "domain-gap:monologue"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(ROOT / "turn_v2" / "ckpt" / "s2-004" / "best.pt"))
    ap.add_argument("--slices-splits", nargs="+", default=["test_a", "test_b", "test_c"])
    ap.add_argument("--fail-splits", nargs="+", default=["test_b", "test_c"])
    ap.add_argument("--skip-slices", action="store_true")
    ap.add_argument("--max-c-fails", type=int, default=30, help="cap on sampled test-C v2 failures")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    import torch

    from turn_v2.evaluate import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.ckpt, device)
    sess = ref_session()
    print(f"loaded {args.ckpt} on {device}")
    RESULTS.mkdir(exist_ok=True)

    # ---- Step 1: confusion slices ----------------------------------------
    if not args.skip_slices:
        slice_rows = []
        for split in args.slices_splits:
            clips = SPLIT_CLIPS[split]()
            r = predict_both(model, sess, clips, device, args.batch_size, args.workers)
            y = r["labels"]
            langs = np.array(r["langs"])
            mids = np.array(r["mids"])
            ends = np.array(r["ends"])
            for mname, p in (("v2-core-s2-004", r["probs"]), ("zero-shot-ref", r["ref_probs"])):
                pred = (p >= 0.5).astype(int)
                blocks = [("all", np.ones(len(y), dtype=bool))]
                for lg in sorted(set(r["langs"])):
                    blocks.append((f"lang={lg}", langs == lg))
                for mv in (False, True):
                    blocks.append((f"midfiller={mv}", mids == mv))
                for ev in (False, True):
                    blocks.append((f"endfiller={ev}", ends == ev))
                for sname, mask in blocks:
                    if mask.sum() == 0:
                        continue
                    c = confusion(y[mask], pred[mask])
                    slice_rows.append(
                        {
                            "split": split,
                            "model": mname,
                            "slice": sname,
                            **{k: c[k] for k in ("n", "acc", "f1", "fir", "hold", "tp", "fp", "fn", "tn")},
                        }
                    )
            print(f"[{split}] slices computed")
            for row in slice_rows:
                if row["split"] == split and row["model"] == "v2-core-s2-004" and row["n"] > 0:
                    print(
                        f"    v2 {row['slice']:16s} n={row['n']:5d} acc={row['acc']:.3f} "
                        f"f1={row['f1']:.3f} fir={row['fir']:.3f} hold={row['hold']:.3f}"
                    )
        with open(RESULTS / "slices_v2core.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(slice_rows[0].keys()))
            w.writeheader()
            w.writerows(slice_rows)
        print(f"wrote {RESULTS / 'slices_v2core.csv'} ({len(slice_rows)} rows)")

    # ---- Step 2: failures --------------------------------------------------
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_baselines import load_16k, trailing_silence  # noqa: E402

    fail_rows = []
    for split in args.fail_splits:
        clips = SPLIT_CLIPS[split]()
        r = predict_both(model, sess, clips, device, args.batch_size, args.workers)
        y = r["labels"]
        v2_pred = (r["probs"] >= 0.5).astype(int)
        zs_pred = (r["ref_probs"] >= 0.5).astype(int)

        fail_v2 = np.where(y != v2_pred)[0]
        fail_zs = np.where(y != zs_pred)[0]
        rng = np.random.default_rng(42)
        if split == "test_c":
            v2_only = sorted(set(fail_v2.tolist()) - set(fail_zs.tolist()))
            both = sorted(set(fail_v2.tolist()) & set(fail_zs.tolist()))
            zs_only = sorted(set(fail_zs.tolist()) - set(fail_v2.tolist()))
            print(
                f"[{split}] v2 fails={len(fail_v2)} (v2-only={len(v2_only)}, both={len(both)}), "
                f"zs-only fails={len(zs_only)}"
            )
            sel = {"v2+zs": both, "v2": v2_only, "zs": zs_only}
        else:
            both = sorted(set(fail_v2.tolist()) & set(fail_zs.tolist()))
            v2_only = sorted(set(fail_v2.tolist()) - set(fail_zs.tolist()))
            zs_only = sorted(set(fail_zs.tolist()) - set(fail_v2.tolist()))
            print(f"[{split}] v2 fails={len(fail_v2)}, zs fails={len(fail_zs)} (all included)")
            sel = {"v2+zs": both, "v2": v2_only, "zs": zs_only}

        for fmodel, idxs in sel.items():
            if not idxs:
                continue
            if split == "test_c":
                pred_arr = zs_pred if fmodel == "zs" else v2_pred
                strata = {}
                for i in idxs:
                    strata.setdefault((int(y[i]), int(pred_arr[i])), []).append(int(i))
                cap = max(1, args.max_c_fails // max(1, len(strata)))
                keep = set()
                for key in sorted(strata):
                    s = strata[key]
                    if len(s) > cap:
                        s = rng.choice(s, size=cap, replace=False).tolist()
                    keep.update(s)
                idxs = sorted(keep)
            for i in idxs:
                path = ROOT / r["paths"][i]
                audio = load_16k(path)
                tail = trailing_silence(audio)
                pred_i, p_i = (
                    (int(v2_pred[i]), float(r["probs"][i]))
                    if fmodel != "zs"
                    else (int(zs_pred[i]), float(r["ref_probs"][i]))
                )
                cat = categorize(int(y[i]), pred_i, p_i, tail, split, bool(r["mids"][i]), bool(r["ends"][i]))
                fail_rows.append(
                    {
                        "split": split,
                        "failed_model": fmodel,
                        "path": r["paths"][i],
                        "lang": r["langs"][i],
                        "label": int(y[i]),
                        "v2_pred": int(v2_pred[i]),
                        "v2_p": round(float(r["probs"][i]), 4),
                        "zs_pred": int(zs_pred[i]),
                        "zs_p": round(float(r["ref_probs"][i]), 4),
                        "midfiller": bool(r["mids"][i]),
                        "endfiller": bool(r["ends"][i]),
                        "dur_s": round(len(audio) / 16000, 2),
                        "tail_sil_s": round(tail, 3),
                        "category": cat,
                    }
                )

    out = RESULTS / "error_analysis_failures.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fail_rows[0].keys()))
        w.writeheader()
        w.writerows(fail_rows)
    print(f"wrote {out} ({len(fail_rows)} failure rows)")

    print("\n=== failure categories (per failing model) ===")
    cats = sorted({(row["failed_model"], row["category"]) for row in fail_rows})
    for fmodel, cat in cats:
        rows = [r for r in fail_rows if r["failed_model"] == fmodel and r["category"] == cat]
        pcol = "v2_p" if fmodel != "zs" else "zs_p"
        med_p = np.median([r[pcol] for r in rows])
        med_tail = np.median([r["tail_sil_s"] for r in rows])
        med_dur = np.median([r["dur_s"] for r in rows])
        print(
            f"  {fmodel:6s} {cat:32s} n={len(rows):3d}  med p={med_p:.3f}  med tail={med_tail:.2f}s  med dur={med_dur:.2f}s"
        )


if __name__ == "__main__":
    main()
