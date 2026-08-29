"""ONNX export + int8 static quantization (P5 WS-B).

fp32:  torch.onnx.export, opset 17, fixed input (1, 80, 800), 8 s contract.
int8:  static QDQ (QUInt8/QInt8, per-channel weights, Entropy activations)
       calibrated on a held-out dev_v1 subset — fixes the reference's flaw of
       calibrating on test data.

Eval:  onnxruntime CPU, accuracy/F1 + p50 latency on any split; reports
fp32-vs-int8 deltas. AVX2 U8S8 saturation cliff: re-quantize with
reduce_range=True if int8 accuracy craters.

Usage:
  uv run python turn_v2/export.py --ckpt turn_v2/ckpt/s2-004/best.pt
  uv run python turn_v2/export.py --ckpt ... --splits test_a test_b test_c --eval
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ONNX_DIR = ROOT / "turn_v2" / "onnx"


def build_wrapped(ckpt_path):
    from turn_v2.evaluate import load_model

    model, ckpt = load_model(ckpt_path, device="cpu")
    return model, ckpt


def export_fp32(model, out_path):
    import onnx

    model.eval()
    dummy = torch.randn(1, 80, 800, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input_features"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes={"input_features": {0: "batch"}, "logits": {0: "batch"}},
    )
    m = onnx.load(str(out_path), load_external_data=True)
    for t in m.graph.initializer:
        t.ClearField("data_location")
        t.ClearField("external_data")
    onnx.save(m, str(out_path))  # inline weights into a single file
    data_file = Path(str(out_path) + ".data")
    if data_file.exists():
        data_file.unlink()
    return out_path


def quantize_int8(fp32_path, out_path, calib_features, reduce_range=False,
                  calib_method="entropy", act_type="quint8", pre_process=True):
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static, quant_pre_process
    from onnxruntime.quantization import QuantFormat, CalibrationMethod

    class Reader(CalibrationDataReader):
        def __init__(self, feats):
            self.feats = feats
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.feats):
                return None
            f = self.feats[self.idx]
            self.idx += 1
            return {"input_features": f[None].astype(np.float32)}

    src = str(fp32_path)
    if pre_process:
        pre = str(fp32_path) + ".pre.onnx"
        quant_pre_process(src, pre, skip_optimization=False, skip_symbolic_shape=True, verbose=1)
        src = pre

    quantize_static(
        model_input=src,
        model_output=str(out_path),
        calibration_data_reader=Reader(calib_features),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8 if act_type == "qint8" else QuantType.QUInt8,
        weight_type=QuantType.QUInt8 if act_type == "u8u8" else QuantType.QInt8,
        per_channel=True,
        reduce_range=reduce_range,
        calibrate_method=CalibrationMethod.MinMax if calib_method == "minmax" else CalibrationMethod.Entropy,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        extra_options={"ActivationSymmetric": act_type == "qint8"},
    )
    return out_path


def ort_session(path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def ort_eval(sess, clips, batch_size=128):
    from sklearn.metrics import accuracy_score, f1_score

    from turn_v2.data.dataset import TurnDataset, collate_turn
    from torch.utils.data import DataLoader

    ds = TurnDataset(clips, augment=None, cache=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_turn)
    ys, ps = [], []
    lat = []
    for batch in loader:
        x = batch["input_features"].numpy().astype(np.float32)
        t0 = time.perf_counter()
        logits = sess.run(["logits"], {"input_features": x})[0]
        lat.append((time.perf_counter() - t0) * 1000 / x.shape[0])
        ps.extend((logits > 0).astype(int).tolist())
        ys.extend(batch["label"].tolist())
    return {
        "n": len(ys),
        "acc": accuracy_score(ys, ps),
        "f1": f1_score(ys, ps, zero_division=0),
        "lat_ms": float(np.median(lat)),
    }


def calibration_features(n=200, seed=42):
    from turn_v2.data.dataset import TurnDataset, dev_v1_clips, collate_turn
    from torch.utils.data import DataLoader

    clips = dev_v1_clips(("hin", "eng"))
    ds = TurnDataset(clips, augment=None, cache=True)
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_turn)
    take = set(idx.tolist())
    feats, pos = [], 0
    for batch in loader:
        for j in range(batch["input_features"].shape[0]):
            if pos in take:
                feats.append(batch["input_features"][j].numpy())
            pos += 1
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["test_a", "test_b", "test_c"])
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--quant-config", default="entropy,quint8",
                    help="calib,act[:rr]; repeatable via ';' e.g. 'entropy,quint8;minmax,quint8:rr'")
    ap.add_argument("--sweep", action="store_true",
                    help="quantize+eval several configs on the first split only")
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--eval", action="store_true", help="eval fp32 and int8 on splits")
    args = ap.parse_args()

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(args.ckpt).parent.name
    fp32_path = ONNX_DIR / f"{stem}.fp32.onnx"
    int8_path = ONNX_DIR / f"{stem}.int8.onnx"

    print(f"loading {args.ckpt} ...")
    model, ckpt = build_wrapped(args.ckpt)
    print(f"base={ckpt.get('base')} pooling={ckpt.get('pooling')} dev_f1={ckpt.get('dev_f1')}")

    print(f"exporting fp32 onnx -> {fp32_path}")
    export_fp32(model, fp32_path)
    print(f"fp32 size: {fp32_path.stat().st_size / 1e6:.2f} MB")

    if not args.no_quant:
        print(f"collecting {args.calib_n} calibration features from dev_v1 (held-out) ...")
        feats = calibration_features(args.calib_n)
        configs = []
        for spec in args.quant_config.split(";"):
            cfg = [c.strip() for c in spec.split(",")]
            calib_method, act_type = cfg[0], cfg[1]
            reduce_range = "rr" in cfg[2:]
            tag = f"{calib_method}-{act_type}{'-rr' if reduce_range else ''}"
            out = ONNX_DIR / f"{stem}.int8.{tag}.onnx"
            print(f"quantizing int8 (QDQ, per-channel, {tag}) -> {out}")
            quantize_int8(fp32_path, out, feats, reduce_range=reduce_range,
                          calib_method=calib_method, act_type=act_type)
            print(f"int8 size: {out.stat().st_size / 1e6:.2f} MB")
            configs.append((tag, out))
        if not args.sweep:
            int8_path = configs[-1][1]

    if args.eval:
        from turn_v2.data.dataset import test_a_clips, test_b_clips, test_c_clips

        split_map = {"test_a": test_a_clips, "test_b": test_b_clips, "test_c": test_c_clips}
        sess_fp32 = ort_session(fp32_path)
        if args.sweep:
            split = args.splits[0]
            clips = split_map[split]()
            r32 = ort_eval(sess_fp32, clips)
            print(f"[{split}] fp32: acc={r32['acc']:.3f} f1={r32['f1']:.3f} p50={r32['lat_ms']:.1f}ms")
            for tag, out in configs:
                r8 = ort_eval(ort_session(out), clips)
                print(
                    f"[{split}] {tag:18s} int8: acc={r8['acc']:.3f} "
                    f"f1={r8['f1']:.3f} p50={r8['lat_ms']:.1f}ms | dacc={r8['acc'] - r32['acc']:+.3f}"
                )
            return
        sess_int8 = ort_session(int8_path) if not args.no_quant else None
        for split in args.splits:
            clips = split_map[split]()
            r32 = ort_eval(sess_fp32, clips)
            line = f"[{split}] fp32: n={r32['n']} acc={r32['acc']:.3f} f1={r32['f1']:.3f} p50={r32['lat_ms']:.1f}ms"
            if sess_int8 is not None:
                r8 = ort_eval(sess_int8, clips)
                line += (
                    f" | int8: acc={r8['acc']:.3f} f1={r8['f1']:.3f} p50={r8['lat_ms']:.1f}ms"
                    f" | delta acc={r8['acc'] - r32['acc']:+.3f} f1={r8['f1'] - r32['f1']:+.3f}"
                )
            print(line)


if __name__ == "__main__":
    main()
