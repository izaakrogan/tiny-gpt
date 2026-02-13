"""Step 1: the bigram baseline. Complete and working.

A lookup table: for each current character, a row of logits over what comes
next. No attention, no masking, no positions. It exists so the training loop
holds no surprises before attention arrives, and so every later model has a
baseline to beat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int, **_):
        super().__init__()
        # vocab_size x vocab_size table: row = current char, columns = logits
        # for the next char. That table IS the whole model.
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) integer character ids
        logits = self.table(idx)  # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            # cross_entropy is the -log(p) loss from module 282
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature  # only the last position matters
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
