"""Shared evaluation helpers for the P6 scripts (latency.py / policy.py / error analysis).

- ref_session(): zero-shot Smart Turn v3.2 int8 ONNX session (CWD-independent).
- predict_both(): run the v2 torch model and the reference ONNX over the same
  clip list with identical features (batched), returning aligned per-clip arrays.
- confusion(): acc/F1/FIR/HOLD + confusion counts for a thresholded prediction.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REF_ONNX = ROOT / "models_ref" / "smart-turn-v3.2-cpu.onnx"


def ref_session():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(REF_ONNX), sess_options=so)


@torch.no_grad()
def predict_both(model, sess, clips, device="cpu", batch_size=128, workers=0, truncate_s=None, progress_every=0):
    """Evaluate torch model + reference ONNX on the same clips.

    v2 features use the turn_v2 convention (do_normalize=False, as trained);
    reference features use its native path (do_normalize=True) so zero-shot
    numbers match the P1 baseline (0.927 test-A anchor, not the degraded 0.915).

    Returns dict of aligned arrays: probs (v2), ref_probs (zero-shot, sigmoid),
    labels, langs, mids, ends, paths.
    """
    from torch.utils.data import DataLoader

    from turn_v2.data.dataset import TurnDataset, collate_turn

    ds_v2 = TurnDataset(clips, cache=True, truncate_s=truncate_s)
    ds_ref = TurnDataset(clips, cache=True, truncate_s=truncate_s, normalize=True)
    loader_v2 = DataLoader(ds_v2, batch_size=batch_size, shuffle=False, collate_fn=collate_turn, num_workers=workers)
    loader_ref = DataLoader(ds_ref, batch_size=batch_size, shuffle=False, collate_fn=collate_turn, num_workers=workers)
    model.eval().to(device)
    out = {k: [] for k in ("probs", "ref_probs", "labels", "langs", "mids", "ends", "paths")}
    input_name = sess.get_inputs()[0].name
    seen = 0
    for b2, br in zip(loader_v2, loader_ref):
        x = b2["input_features"].to(device)
        logits = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        out["probs"].extend(torch.sigmoid(logits).cpu().tolist())
        xn = br["input_features"].numpy().astype(np.float32)
        rp = sess.run(None, {input_name: xn})[0]
        out["ref_probs"].extend(np.asarray(rp).reshape(-1).tolist())
        out["labels"].extend(int(v) for v in b2["label"])
        out["langs"].extend(b2["lang"])
        out["mids"].extend(b2["midfiller"])
        out["ends"].extend(b2["endfiller"])
        out["paths"].extend(b2["path"])
        seen += int(x.size(0))
        if progress_every and seen % progress_every < batch_size:
            print(f"    {seen} clips...", flush=True)
    return {
        "probs": np.array(out["probs"], dtype=np.float64),
        "ref_probs": np.array(out["ref_probs"], dtype=np.float64),
        "labels": np.array(out["labels"], dtype=int),
        "langs": out["langs"],
        "mids": out["mids"],
        "ends": out["ends"],
        "paths": out["paths"],
    }


def confusion(y, pred):
    """acc/F1/FIR/HOLD + counts. FIR = P(pred=complete | incomplete) (false interrupt),
    HOLD = P(pred=incomplete | complete) (added-latency proxy)."""
    from sklearn.metrics import accuracy_score, f1_score

    y = np.asarray(y)
    pred = np.asarray(pred, dtype=int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {
        "n": len(y),
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fir": fp / n0 if n0 else float("nan"),
        "hold": fn / n1 if n1 else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
