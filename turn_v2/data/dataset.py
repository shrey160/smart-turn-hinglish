"""TurnDataset: on-demand folder-layout loading, 8 s contract, label parsing, split builders.

Folder layout (Smart Turn convention, see AGENTS.md / masterplan):
    data/{split}/{lang}/{label_folder}/{id}.flac
    label_folder = complete|incomplete-{midfiller}-{endfiller}  (Python bool casing)

The 8 s audio contract (keep last 8 s, zero-pad front) is imported from the
read-only smart_turn_reference/audio_utils - never re-implemented locally.
"""
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "smart_turn_reference"))

from audio_utils import truncate_audio_to_last_n_seconds  # noqa: E402  (shared contract)

BASE_MODEL = "openai/whisper-tiny"
SAMPLE_RATE = 16000
FE = None

DEV_V1_CSV = ROOT / "data" / "splits" / "dev_v1.csv"
DEV_TTS_V1_CSV = ROOT / "data" / "splits" / "dev_tts_v1.csv"


def get_feature_extractor():
    global FE
    if FE is None:
        from transformers import WhisperFeatureExtractor

        FE = WhisperFeatureExtractor.from_pretrained(BASE_MODEL, chunk_length=8)
    return FE


def parse_label_folder(folder):
    parts = folder.split("-")
    if len(parts) != 3:
        raise ValueError(f"bad label folder: {folder}")
    label = 1 if parts[0] == "complete" else 0
    return label, parts[1] == "True", parts[2] == "True"


def _norm(p):
    return str(p).replace("\\", "/")


def _clip_from_path(path, lang, folder):
    label, mid, end = parse_label_folder(folder)
    return {
        "path": str(path),
        "rel": _norm(path.relative_to(ROOT)),
        "lang": lang,
        "label": label,
        "midfiller": mid,
        "endfiller": end,
    }


def _is_label_dir(name):
    try:
        parse_label_folder(name)
        return True
    except ValueError:
        return False


def _enumerate_dir(base, lang=None):
    """Walk `{base}/{lang}/{label_folder}/*.(flac|wav)` (auto-detects flat `{base}/{label_folder}/*`)."""
    base = Path(base)
    if not base.exists():
        return []

    def _clips_from(label_dir, use_lang):
        files = sorted(label_dir.glob("*.flac")) + sorted(label_dir.glob("*.wav"))
        return [_clip_from_path(f, use_lang, label_dir.name) for f in files]

    children = sorted(d for d in base.iterdir() if d.is_dir())
    if children and _is_label_dir(children[0].name):
        return [c for ld in children for c in _clips_from(ld, lang or "unknown")]
    clips = []
    for lang_dir in children:
        use_lang = lang if lang is not None else lang_dir.name
        for label_dir in sorted(lang_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            clips.extend(_clips_from(label_dir, use_lang))
    return clips


def train_pool_clips(langs=("hin", "eng"), exclude_dev=True):
    clips = _enumerate_dir(ROOT / "data" / "train_pool")
    if langs is not None:
        clips = [c for c in clips if c["lang"] in langs]
    if exclude_dev:
        dev_rels = {c["rel"] for c in dev_v1_clips(langs=None)}
        clips = [c for c in clips if c["rel"] not in dev_rels]
    return clips


def test_a_clips():
    return _enumerate_dir(ROOT / "data" / "test_a")


def test_b_clips():
    return _enumerate_dir(ROOT / "data" / "example_hinglish_fixed" / "test_b", lang="tts-hinglish")


def test_c_clips():
    return _enumerate_dir(ROOT / "data" / "test_c")


def tts_clips():
    clips = _enumerate_dir(ROOT / "data" / "example_hinglish_fixed" / "train_pool", lang="tts-hinglish")
    clips += _enumerate_dir(ROOT / "data" / "tts_hinglish" / "train", lang="tts-hinglish")
    return clips


def _manifest_clips(csv_path):
    clips = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = ROOT / Path(row["rel_path"])
            clips.append(
                {
                    "path": str(path),
                    "rel": _norm(row["rel_path"]),
                    "lang": row["lang"],
                    "label": int(row["label"]),
                    "midfiller": row["midfiller"] == "True",
                    "endfiller": row["endfiller"] == "True",
                }
            )
    return clips


def dev_v1_clips(langs=("hin", "eng")):
    clips = _manifest_clips(DEV_V1_CSV)
    if langs is not None:
        clips = [c for c in clips if c["lang"] in langs]
    return clips


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32)


def to_features(audio):
    audio = truncate_audio_to_last_n_seconds(audio, n_seconds=8, sample_rate=SAMPLE_RATE)
    fe = get_feature_extractor()
    out = fe(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=8 * SAMPLE_RATE,
        truncation=True,
    )["input_features"]
    feat = np.asarray(out, dtype=np.float32)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    return torch.from_numpy(feat)


class TurnDataset(Dataset):
    """Clips -> (80, 800) log-mel on demand; optional on-the-fly waveform augmentation."""

    def __init__(self, clips, augment=None, cache=False):
        self.clips = list(clips)
        self.augment = augment
        self.cache = cache
        self._cache = {}

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx]
        if self.cache and idx in self._cache:
            feat = self._cache[idx]
            return {
                "input_features": feat,
                "label": c["label"],
                "lang": c["lang"],
                "midfiller": c["midfiller"],
                "endfiller": c["endfiller"],
                "path": c["rel"],
            }
        audio = load_audio(c["path"])
        if self.augment is not None:
            audio = self.augment(audio)
        feat = to_features(audio)
        if self.cache:
            self._cache[idx] = feat
        return {
            "input_features": to_features(audio),
            "label": c["label"],
            "lang": c["lang"],
            "midfiller": c["midfiller"],
            "endfiller": c["endfiller"],
            "path": c["rel"],
        }


def collate_turn(batch):
    return {
        "input_features": torch.stack([b["input_features"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "lang": [b["lang"] for b in batch],
        "midfiller": [b["midfiller"] for b in batch],
        "endfiller": [b["endfiller"] for b in batch],
        "path": [b["path"] for b in batch],
    }


def carve_tts_dev(clips, frac=0.1, seed=42):
    """Deterministic per-label-folder carve; returns (train, dev)."""
    rng = random.Random(seed)
    by_folder = {}
    for c in sorted(clips, key=lambda c: c["rel"]):
        folder = Path(c["path"]).parent.name
        by_folder.setdefault(folder, []).append(c)
    train, dev = [], []
    for folder in sorted(by_folder):
        group = by_folder[folder]
        k = round(frac * len(group))
        dev_idx = set(rng.sample(range(len(group)), k)) if k else set()
        for i, c in enumerate(group):
            (dev if i in dev_idx else train).append(c)
    return train, dev


def write_manifest_csv(path, clips):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "lang", "label_folder", "label", "midfiller", "endfiller"])
        for c in clips:
            w.writerow(
                [c["rel"], c["lang"], Path(c["path"]).parent.name, c["label"], c["midfiller"], c["endfiller"]]
            )
