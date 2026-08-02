"""Reference implementation for step 4. Don't read this until you've
attempted the TODOs in models/transformer.py yourself.
"""

import torch.nn as nn

# This imports the solutions copy of multi-head attention, not yours, so
# the file works even if your steps 2 and 3 aren't finished.
from solutions.multi_head import MultiHeadAttention


class FeedForward(nn.Module):
    """Same as the one in models/transformer.py, copied here so this file
    stands on its own."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention half, with residual
        x = x + self.ffwd(self.ln2(x))   # feed-forward half, with residual
        return x
