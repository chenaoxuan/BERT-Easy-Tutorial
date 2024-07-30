import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, hidden, eps=1e-6):
        super(LayerNorm, self).__init__()
        self._gamma = nn.Parameter(torch.ones(hidden))
        self._beta = nn.Parameter(torch.zeros(hidden))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self._gamma * (x - mean) / (std + self.eps) + self._beta
