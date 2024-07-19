import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int = None, num_head: int = 8, head_dim: int = 32):
        super().__init__()

        # Universal code: suitable not only for self-transformer but also for cross-transformer.
        # Where q is the query and k\v is the context
        if context_dim is None:
            context_dim = query_dim

        inner_dim = num_head * head_dim
        self.num_head = num_head
        self.head_dim = head_dim

        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(context_dim, inner_dim)
        self.to_v = nn.Linear(context_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor = None, mask: torch.Tensor = None):
        if context is None:
            context = query

        B, q_len, c_len = query.shape[0], query.shape[1], context.shape[1]
        q = self.to_q(query).reshape(B, q_len, self.num_head, self.head_dim)
        k = self.to_k(context).reshape(B, c_len, self.num_head, self.head_dim)
        v = self.to_v(context).reshape(B, c_len, self.num_head, self.head_dim)

        energy = torch.einsum('bqhd,bkhd->bhqk', [q, k]) / self.head_dim ** 0.5

        del q, k

        if mask is not None:
            assert mask.shape[-2] == q_len and mask.shape[-1] == c_len, 'Mask must have shape B * q_len * c_len'
            mask = mask.unsqueeze(1)
            energy.masked_fill_(~mask, -1e9)

        attention = energy.softmax(dim=-1)
        out = torch.einsum('bhqc,bchd->bqhd', [attention, v]).reshape(B, q_len, -1)
        out = self.to_out(out)
        return out
