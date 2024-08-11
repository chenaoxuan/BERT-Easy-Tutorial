import math

import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        # The use of log + exp is to avoid intermediate numerical calculations exceeding the range of float
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class RotaryEmbedding:
    def __init__(self, hidden, max_len=10):
        # Method 1 and Method 2 are completely equivalent
        # Method 1: Generate pos, then pos.unsqueeze (1), and finally calculate pos * inv_term
        # Method 2: Generate pos, then calculate torch.outer(pos, inv_term)

        # Method 1 is completely equivalent to Method 2. Recommend Method 1 for more accurate numerical calculations.
        # Method 1: inv_term = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden))
        # Method 2: 1.0 / (10000.0 ** (torch.arange(0, hidden, 2)[: (hidden // 2)].float() / hidden))

        pos = torch.arange(max_len, dtype=torch.float32)
        inv_term = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden))
        freqs = torch.outer(pos, inv_term)
        self.freqs = torch.polar(torch.ones_like(freqs), freqs)
        # Expand num_head dimension
        # [max_len, hidden//2] -> [max_len, 1, hidden//2]
        self.freqs = self.freqs[:, None, :]

    def apply_rotary_emb(self, q, k):
        '''

        :param q: q with shape [batch_size, q_len, num_head, hidden]
        :param k: k with shape [batch_size, k_len, num_head, hidden]
        :return:
        '''
        # [batch_size, len, num_head, hidden] -> [batch_size, len, num_head, hidden//2, 2] -> [batch_size, len, num_head, hidden//2]
        q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        q_out = torch.view_as_real(q_ * self.freqs[:q_.size(1)]).flatten(-2)
        k_out = torch.view_as_real(k_ * self.freqs[:k_.size(1)]).flatten(-2)
        return q_out.type_as(q), k_out.type_as(k)
