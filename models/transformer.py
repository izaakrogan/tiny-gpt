"""Step 4: the full transformer. This is the outline, you are going to
write the Block.

A block is: layer norm, multi-head attention, add the input back, layer
norm, feed-forward, add the input back. Stack the block n_layer times and
you have the model. The attention inside it is your step 2 head via your
step 3 multi-head; the feed-forward below is written for you.

Fill in the TODOs in Block.forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multi_head import MultiHeadAttention


class FeedForward(nn.Module):
    """The per-position processing step. Two linear layers with a ReLU
    between them, applied to every position independently. Attention is
    the only place information crosses between positions; this layer
    never does. The 4x expansion is the standard transformer choice."""

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
        # TODO 1: the attention half.
        #   Norm first, then attention, then add x back on:
        #   x = x + self.attn(self.ln1(x))
        #   The add is the residual connection. Without it the block
        #   replaces its input; with it the block refines its input.
        #   Write it, then reread it until you can say which part is
        #   the residual.
        x = ...

        # TODO 2: the feed-forward half. Same shape: norm, ffwd, add.
        x = ...

        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 64, block_size: int = 64,
                 n_head: int = 4, n_layer: int = 4, **_):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final norm before the head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
