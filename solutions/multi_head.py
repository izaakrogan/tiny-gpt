"""Reference implementation for step 3. Don't read this until you've
attempted the TODOs in models/multi_head.py yourself.
"""

import torch
import torch.nn as nn

# This imports the solutions copy of the head, not yours, so the file
# works even if your step 2 isn't finished.
from solutions.single_attention_head import CausalSelfAttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [CausalSelfAttentionHead(n_embd, head_size, block_size)
             for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        out = self.proj(out)
        return out
