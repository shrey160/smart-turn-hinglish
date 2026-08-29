"""Pooling heads over Whisper encoder frames (B, T, D).

- attention-mean (baseline, mirrors Smart Turn): Linear-Tanh-Linear softmax weighted sum -> (B, D)
- ASP (playbook §7.1): weighted mean + weighted std -> (B, 2D)
- asp-end (playbook §7.2): ASP + mean of last-k frames concat -> (B, 3D)

Init mirrors Smart Turn train.py:63-101: Linear weights N(0, 0.1), zero biases.
"""
import torch
import torch.nn as nn


def _init_linear(module, std=0.1):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=std)
            if m.bias is not None:
                m.bias.data.zero_()


class AttentionMeanPooling(nn.Module):
    out_dim = None

    def __init__(self, dim, attn_dim=256, last_k=0):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
        self.last_k = last_k
        self.out_dim = dim * (2 if last_k > 0 else 1)
        _init_linear(self)

    def _weights(self, hidden):
        return torch.softmax(self.attn(hidden), dim=1)

    def forward(self, hidden):
        w = self._weights(hidden)
        mean = (hidden * w).sum(dim=1)
        if self.last_k > 0:
            return torch.cat([mean, hidden[:, -self.last_k :, :].mean(dim=1)], dim=1)
        return mean


class ASPPooling(AttentionMeanPooling):
    def __init__(self, dim, attn_dim=256, last_k=0):
        super().__init__(dim, attn_dim, last_k)
        self.out_dim = dim * (3 if last_k > 0 else 2)

    def forward(self, hidden):
        w = self._weights(hidden)
        mean = (hidden * w).sum(dim=1)
        var = (w * (hidden - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = torch.sqrt(var.clamp(min=1e-5))
        parts = [mean, std]
        if self.last_k > 0:
            parts.append(hidden[:, -self.last_k :, :].mean(dim=1))
        return torch.cat(parts, dim=1)


POOLING_REGISTRY = {
    "attention-mean": AttentionMeanPooling,
    "attention-end": lambda dim, attn_dim=256: AttentionMeanPooling(dim, attn_dim, last_k=50),
    "asp": ASPPooling,
    "asp-end": lambda dim, attn_dim=256: ASPPooling(dim, attn_dim, last_k=50),
}


def build_pooling(name, dim):
    if name not in POOLING_REGISTRY:
        raise ValueError(f"unknown pooling '{name}'; choose from {sorted(POOLING_REGISTRY)}")
    return POOLING_REGISTRY[name](dim)
