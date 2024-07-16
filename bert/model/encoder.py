import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import PositionwiseFeedForward, LayerNorm


class Encoder(nn.Module):
    """
    Transformer Encoder = MultiHead-Self-Attention + Feed-Forward with residual connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of encoder
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.norm1 = LayerNorm(features=hidden)
        self.drop1 = nn.Dropout(p=dropout)
        self.norm2 = LayerNorm(features=hidden)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = x + self.drop1(self.attention(x, x, x, mask=mask))
        x = self.norm1(x)
        x = x + self.drop2(self.feed_forward(x))
        x = self.norm2(x)
        return x
