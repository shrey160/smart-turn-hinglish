"""Two-stage training (masterplan §4): plain PyTorch loop, no HF Trainer.

Stage 1 (s1): pretrain on train_pool hin+eng minus dev_v1 (~7,200) — lr 5e-5, 4 ep.
Stage 2 (s2): finetune ckpt_s1 on TTS Hinglish x2 + 50:50 hin/eng replay per epoch — lr 1e-5, 2 ep.

Sanity gate: --overfit N trains on N clips with no augmentation and must reach
~zero loss before any full run. Every non-overfit run appends a results.csv row
(schema v2: stage + init_ckpt columns).

Run:
  uv run python -m turn_v2.train --stage s1
  uv run python -m turn_v2.train --stage s2 --init-ckpt turn_v2/ckpt/s1-001/best.pt
  uv run python -m turn_v2.train --stage s1 --overfit 100
"""
import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turn_v2.data.augment import WaveAugment  # noqa: E402
from turn_v2.data.dataset import (  # noqa: E402
    DEV_TTS_V1_CSV,
    TurnDataset,
    carve_tts_dev,
    collate_turn,
    dev_v1_clips,
    test_a_clips,
    test_b_clips,
    test_c_clips,
    train_pool_clips,
    tts_clips,
    write_manifest_csv,
)
from turn_v2.evaluate import evaluate_model, print_eval  # noqa: E402
from turn_v2.models.model import SmartTurnV2Model  # noqa: E402

RESULTS_CSV = ROOT / "turn_v2" / "results.csv"
CKPT_ROOT = ROOT / "turn_v2" / "ckpt"
RESULT_COLUMNS = [
    "run_id", "timestamp", "stage", "init_ckpt", "config_tag", "change_summary",
    "seed", "train_n", "dev_f1", "dev_acc", "testA_f1", "testA_acc",
    "testB_f1", "testB_acc", "testC_f1", "testC_acc",
    "params_M", "latency_ms_p50_cpu", "notes",
]

STAGE_DEFAULTS = {
    "s1": {"epochs": 4, "lr": 5e-5, "warmup_ratio": 0.2},
    "s2": {"epochs": 2, "lr": 1e-5, "warmup_ratio": 0.1},
}


def append_results_row(row):
    rows = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    if not rows:
        rows = [RESULT_COLUMNS]
    rows.append([str(row.get(c, "")) for c in RESULT_COLUMNS])
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def next_run_id(stage):
    n = 0
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["run_id"].startswith(f"{stage}-"):
                    n += 1
    return f"{stage}-{n + 1:03d}"


def batch_loss(logits, labels, smoothing=0.0, teacher_logits=None, kd_alpha=0.5, kd_temp=3.0):
    pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
    targets = labels.float()
    if smoothing > 0:
        targets = targets * (1.0 - smoothing) + smoothing / 2.0
    hard = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    if teacher_logits is None:
        return hard
    soft_t = torch.sigmoid(teacher_logits / kd_temp)
    soft = F.binary_cross_entropy_with_logits(logits / kd_temp, soft_t) * kd_temp**2
    return (1.0 - kd_alpha) * hard + kd_alpha * soft


def build_stage_data(args):
    if args.stage == "s1":
        train_clips = train_pool_clips(("hin", "eng"), exclude_dev=True)
        dev_v1 = dev_v1_clips(("hin", "eng"))
        return {"train": train_clips, "dev_parts": {"dev_v1": dev_v1}, "n_tts": 0}
    tts_all = tts_clips()
    tts_train, tts_dev = carve_tts_dev(tts_all)
    if not DEV_TTS_V1_CSV.exists():
        write_manifest_csv(DEV_TTS_V1_CSV, tts_dev)
        print(f"wrote {DEV_TTS_V1_CSV} ({len(tts_dev)} clips)")
    dev_v1 = dev_v1_clips(("hin", "eng"))
    return {
        "train": tts_train,
        "dev_parts": {"dev_v1": dev_v1, "tts_dev": tts_dev},
        "n_tts": len(tts_train),
        "hin_pool": train_pool_clips(("hin",), exclude_dev=True),
        "eng_pool": train_pool_clips(("eng",), exclude_dev=True),
    }


def epoch_indices(data, stage, epoch, seed, replay_frac=0.5):
    n_tts = data.get("n_tts", 0)
    if stage == "s1" or n_tts == 0:
        return list(range(len(data["train"])))
    rng = random.Random(seed * 1000 + epoch)
    k = round(replay_frac * n_tts)
    idx = list(range(n_tts)) * 2
    if k > 0:
        hin = rng.sample(range(n_tts, n_tts + len(data["hin_pool"])), k)
        eng = rng.sample(range(n_tts + len(data["hin_pool"]), len(data["train"]) + len(data["hin_pool"]) + len(data["eng_pool"])), k)
        idx += hin + eng
    return idx


def run_eval_and_checkpoint(model, data, device, args, run_id, out_dir, best, evals_since_best, global_step):
    model.eval()
    parts = {name: evaluate_model(model, clips, device, args.eval_batch_size, workers=0)
             for name, clips in data["dev_parts"].items()}
    all_n = sum(r["n"] for r in parts.values())
    all_correct = sum(r["acc"] * r["n"] for r in parts.values())
    dev_f1 = sum(r["f1"] * r["n"] for r in parts.values()) / all_n if all_n else 0.0
    dev_acc = all_correct / all_n if all_n else 0.0
    for name, r in parts.items():
        print_eval(f"step {global_step} | {name}", r, show_slices=False)
    improved = dev_f1 > best["f1"] + 1e-4
    if improved:
        best.update({"f1": dev_f1, "acc": dev_acc, "step": global_step})
        evals_since_best = 0
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": state,
                "pooling": args.pooling,
                "base": args.base,
                "stage": args.stage,
                "seed": args.seed,
                "dev_f1": dev_f1,
                "dev_acc": dev_acc,
                "run_id": run_id,
                "step": global_step,
            },
            out_dir / "best.pt",
        )
        print(f"  new best dev_f1={dev_f1:.4f} -> {out_dir / 'best.pt'}")
    else:
        evals_since_best += 1
    model.train()
    return best, evals_since_best


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    defs = STAGE_DEFAULTS[args.stage]
    lr = args.lr if args.lr else (1e-3 if args.overfit else defs["lr"])
    warmup_ratio = args.warmup_ratio if args.warmup_ratio is not None else defs["warmup_ratio"]

    data = build_stage_data(args)
    train_clips = data["train"]
    if args.overfit:
        rng = random.Random(args.seed)
        rng.shuffle(train_clips)
        train_clips = train_clips[: args.overfit]
        data["train"] = train_clips
        data["dev_parts"] = {"overfit_train": train_clips}
        epochs = args.epochs or 30
        batch_size = min(args.batch_size, 16)
    else:
        epochs = args.epochs if args.epochs else defs["epochs"]
        batch_size = args.batch_size

    print(f"stage={args.stage} device={device} train_n={len(train_clips)} epochs={epochs} lr={lr} bs={batch_size}")
    for name, clips in data["dev_parts"].items():
        print(f"  dev[{name}] n={len(clips)}")

    augment = WaveAugment(p=0.5, seed=args.seed) if (not args.no_augment and not args.overfit) else None
    teacher_map = None
    if args.kd_teacher:
        z = np.load(args.kd_teacher, allow_pickle=False)
        teacher_map = dict(zip(z["paths"].tolist(), z["logits"].tolist()))
        print(f"kd: {len(teacher_map)} teacher logits from {args.kd_teacher} "
              f"(alpha={args.kd_alpha}, T={args.kd_temp})")
    full_ds_train = TurnDataset(train_clips + data.get("hin_pool", []) + data.get("eng_pool", []), augment, cache=bool(args.overfit))

    model = SmartTurnV2Model(base=args.base, pooling=args.pooling, freeze_encoder=not args.no_freeze_encoder)
    if args.unfreeze_last_k:
        model.unfreeze_last_k(args.unfreeze_last_k)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"init from {args.init_ckpt} (stage={ckpt.get('stage')} dev_f1={ckpt.get('dev_f1')})")
    model.to(device)
    total_params, trainable_params = model.count_params()
    print(f"params total={total_params / 1e6:.2f}M trainable={trainable_params / 1e6:.2f}M")
    encoder_frozen = all(not p.requires_grad for p in model.encoder.parameters())

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(
        (len(epoch_indices(data, args.stage, 0, args.seed, args.replay_frac))
         if args.stage == "s2" else len(train_clips)) / batch_size
    )
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_ratio * total_steps))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    run_id = "overfit" if args.overfit else next_run_id(args.stage)
    out_dir = CKPT_ROOT / run_id
    best = {"f1": -1.0, "acc": 0.0, "step": 0}
    evals_since_best = 0
    global_step = 0
    t_start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        if encoder_frozen:
            model.encoder.eval()
        if augment is not None:
            augment.set_seed(args.seed + epoch)
        idx = epoch_indices(data, args.stage, epoch, args.seed, args.replay_frac)
        loader = DataLoader(
            full_ds_train, batch_size=batch_size, sampler=SubsetRandomSampler(idx),
            collate_fn=collate_turn, num_workers=args.workers,
        )
        running, seen = 0.0, 0
        for batch in loader:
            x = batch["input_features"].to(device)
            y = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                t_logits = None
                if teacher_map is not None:
                    t_logits = torch.tensor(
                        [teacher_map.get(p, 0.0) for p in batch["path"]],
                        dtype=torch.float32, device=device,
                    )
                loss = batch_loss(
                    model(x), y, args.label_smoothing,
                    teacher_logits=t_logits, kd_alpha=args.kd_alpha, kd_temp=args.kd_temp,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            running += loss.item() * x.size(0)
            seen += x.size(0)
        print(f"epoch {epoch + 1}/{epochs} train_loss={running / max(seen, 1):.4f} lr={scheduler.get_last_lr()[0]:.2e} "
              f"elapsed={time.perf_counter() - t_start:.0f}s")
        if args.overfit:
            continue
        if args.eval_steps and global_step % args.eval_steps == 0:
            best, evals_since_best = run_eval_and_checkpoint(
                model, data, device, args, run_id, out_dir, best, evals_since_best, global_step
            )
            if evals_since_best >= args.patience:
                print(f"early stop at step {global_step} (no dev improvement for {args.patience} evals)")
                break
        else:
            best, evals_since_best = run_eval_and_checkpoint(
                model, data, device, args, run_id, out_dir, best, evals_since_best, global_step
            )
            if evals_since_best >= args.patience:
                print(f"early stop at step {global_step} (no dev improvement for {args.patience} evals)")
                break

    if not args.overfit:
        print(f"loading best ckpt (dev_f1={best['f1']:.4f} @ step {best['step']})")
        ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])

    if args.overfit:
        model.eval()
        gate = evaluate_model(model, train_clips, device, batch_size=64, workers=0)
        print_eval("overfit-train", gate, show_slices=False)
        passed = gate["f1"] >= 0.95 and running / max(seen, 1) <= 0.1
        print(f"OVERFIT GATE ({args.overfit} clips): {'PASS' if passed else 'FAIL'} "
              f"(train_f1={gate['f1']:.3f} final_loss={running / max(seen, 1):.4f})")
        return

    dev_results = {name: evaluate_model(model, clips, device, args.eval_batch_size, workers=0)
                   for name, clips in data["dev_parts"].items()}
    all_n = sum(r["n"] for r in dev_results.values())
    dev_f1 = sum(r["f1"] * r["n"] for r in dev_results.values()) / all_n
    dev_acc = sum(r["acc"] * r["n"] for r in dev_results.values()) / all_n
    test_a = evaluate_model(model, test_a_clips(), device, args.eval_batch_size, workers=0)
    test_b = evaluate_model(model, test_b_clips(), device, args.eval_batch_size, workers=0)
    test_c = evaluate_model(model, test_c_clips(), device, args.eval_batch_size, workers=0)
    for name, r in dev_results.items():
        print_eval(f"final {name}", r, show_slices=False)
    print_eval("final test_a", test_a)
    print_eval("final test_b", test_b, show_slices=False)
    print_eval("final test_c", test_c, show_slices=False)

    row = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "stage": args.stage,
        "init_ckpt": args.init_ckpt or "none",
        "config_tag": args.tag or f"{args.stage}-{args.pooling}",
        "change_summary": args.change_summary,
        "seed": args.seed,
        "train_n": len(train_clips) if args.stage == "s1"
        else len(epoch_indices(data, args.stage, 0, args.seed, args.replay_frac)),
        "dev_f1": round(dev_f1, 3),
        "dev_acc": round(dev_acc, 3),
        "testA_f1": round(test_a["f1"], 3),
        "testA_acc": round(test_a["acc"], 3),
        "testB_f1": round(test_b["f1"], 3),
        "testB_acc": round(test_b["acc"], 3),
        "testC_f1": round(test_c["f1"], 3) if test_c else "",
        "testC_acc": round(test_c["acc"], 3) if test_c else "",
        "params_M": round(total_params / 1e6, 2),
        "latency_ms_p50_cpu": round(test_a["latency_ms_p50_e2e"], 1),
        "notes": args.notes or (
            f"{len(train_clips)} base clips; encoder_frozen={encoder_frozen}; "
            f"unfreeze_last_k={args.unfreeze_last_k}; label_smoothing={args.label_smoothing}; "
            f"replay_frac={args.replay_frac}; kd={Path(args.kd_teacher).name if args.kd_teacher else 'off'}; "
            f"per-epoch n={steps_per_epoch * batch_size}"
        ),
    }
    append_results_row(row)
    print(f"results row appended: {run_id}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["s1", "s2"])
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--pooling", default="attention-mean", choices=["attention-mean", "attention-end", "asp", "asp-end"])
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--replay-frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--eval-steps", type=int, default=0, help="eval every N steps (0 = per epoch)")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--overfit", type=int, default=0, help="sanity gate: train on N clips, no augment")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-freeze-encoder", action="store_true")
    ap.add_argument("--unfreeze-last-k", type=int, default=0)
    ap.add_argument("--base", default="openai/whisper-tiny")
    ap.add_argument("--kd-teacher", default="", help="npz with teacher logits {paths, logits} keyed by clip rel path")
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=3.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--change-summary", default="")
    ap.add_argument("--notes", default="")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
