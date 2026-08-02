"""Reference implementation for step 2. Don't read this until you've
attempted the TODOs in models/single_attention_head.py yourself.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionHead(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x):
        B, T, _ = x.shape
        q = self.query(x)                                    # (B, T, head_size)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) / self.head_size ** 0.5   # (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                  # rows sum to 1
        out = weights @ v                                    # (B, T, head_size)
        return out
