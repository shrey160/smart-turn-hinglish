"""On-the-fly waveform augmentation: telephony band-limit / additive noise / speed perturbation.

Applied in TurnDataset.__getitem__ (p≈0.5, one transform per hit) BEFORE the 8 s
contract so length changes re-enter the pad/truncate rule. Zero disk footprint.
"""
import random

import numpy as np

SR = 16000


def _telephony(wav):
    import torch
    import torchaudio.functional as AF

    x = torch.from_numpy(wav).unsqueeze(0)
    x = AF.resample(x, SR, 8000)
    x = AF.highpass_biquad(x, 8000, 300)
    x = AF.lowpass_biquad(x, 8000, 3400)
    x = AF.resample(x, 8000, SR)
    return x.squeeze(0).numpy().astype(np.float32)


def _add_noise(wav, rng):
    rms = float(np.sqrt(np.mean(wav**2)))
    if rms < 1e-6:
        return wav
    snr_db = rng.uniform(10.0, 20.0)
    gen = np.random.default_rng(rng.getrandbits(32))
    noise = gen.normal(0.0, 1.0, size=wav.shape).astype(np.float32)
    noise *= rms * (10 ** (-snr_db / 20.0)) / max(float(np.sqrt(np.mean(noise**2))), 1e-9)
    return (wav + noise).astype(np.float32)


def _speed(wav, factor):
    n_out = int(round(len(wav) / factor))
    return np.interp(
        np.linspace(0.0, len(wav) - 1, n_out), np.arange(len(wav)), wav
    ).astype(np.float32)


class WaveAugment:
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.rng = random.Random(seed)

    def set_seed(self, seed):
        self.rng = random.Random(seed)

    def __call__(self, wav):
        if self.rng.random() >= self.p:
            return wav
        choice = self.rng.choice(("telephony", "noise", "speed"))
        if choice == "telephony":
            return _telephony(wav)
        if choice == "noise":
            return _add_noise(wav, self.rng)
        return _speed(wav, self.rng.uniform(0.9, 1.1))
