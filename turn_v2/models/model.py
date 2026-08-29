"""SmartTurnV2Model: Whisper Tiny frozen encoder + configurable pooling + Smart Turn classifier head."""
import torch
import torch.nn as nn
from transformers import WhisperModel

from turn_v2.models.pooling import build_pooling

BASE_MODEL = "openai/whisper-tiny"


class SmartTurnV2Model(nn.Module):
    def __init__(self, base=BASE_MODEL, pooling="attention-mean", freeze_encoder=True, dropout=0.1):
        super().__init__()
        self.base = base
        self.pooling_name = pooling
        whisper = WhisperModel.from_pretrained(base)
        self.encoder = whisper.encoder
        self.hidden_size = self.encoder.config.d_model
        # 8 s contract: 800 mel frames -> 400 encoder positions (conv1 stride 1, conv2 stride 2).
        # Keep the pretrained embedding's first 400 rows (mirrors reference max_source_positions=400).
        n_pos = self.encoder.embed_positions.weight.shape[0]
        if n_pos != 400:
            new_pos = nn.Embedding(400, self.hidden_size)
            new_pos.weight.data = self.encoder.embed_positions.weight.data[:400].clone()
            self.encoder.embed_positions = new_pos
        self.encoder.config.max_source_positions = 400
        self.pool = build_pooling(pooling, self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.pool.out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        for module in list(self.classifier) + list(self.pool.modules()):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()
        if freeze_encoder:
            self.freeze_encoder()
        else:
            self.freeze_encoder(freeze=False)

    def freeze_encoder(self, freeze=True):
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def unfreeze_last_k(self, k):
        if k <= 0:
            return
        for layer in self.encoder.layers[-k:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.encoder.layer_norm.parameters():
            p.requires_grad = True

    def forward(self, input_features):
        hidden = self.encoder(input_features=input_features).last_hidden_state
        pooled = self.pool(hidden)
        return self.classifier(pooled).squeeze(-1)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
