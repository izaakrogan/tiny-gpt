"""Step 1: the bigram baseline. Complete and working.

The whole model is one table, vocab_size by vocab_size. You give it the
character you're on, it looks up that character's row, and the row holds a
score for every character that could come next. Training does nothing more
than adjust the rows. It can't see anything earlier in the text than the
character you gave it, so what it writes is bad, and that's the point: it
gets the training loop working and it sets a baseline score that we want
the attention model to beat.
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
            # cross_entropy is the -log(p) loss from the Transformers II module
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
