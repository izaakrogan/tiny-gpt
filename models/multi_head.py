"""Step 3: multi-head attention. This is the outline, you are going to
write the multi-head bits.

One head produces one set of attention weights per position, so each
position gets to ask one question of the context. Several heads let it ask
several at once. The head class itself is the one you wrote in step 2,
imported unchanged: this step is about running copies of it side by side.

The harness is written for you. Fill in the TODOs in MultiHeadAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.single_attention_head import CausalSelfAttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        # Each head works in a smaller space. Four heads in a 64-dim model
        # get 16 dims each, so the total work stays the same as one
        # full-width head.
        head_size = n_embd // n_head

        # TODO 1: create the heads.
        #   You want n_head copies of CausalSelfAttentionHead, each built
        #   the same way as in step 2: (n_embd, head_size, block_size),
        #   with head_size as above. Plain Python lists don't register
        #   their contents as trainable, so use nn.ModuleList.
        self.heads = ...

        # The output projection: mixes the concatenated head outputs back
        # into one vector. Written for you.
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # x is (B, T, n_embd).

        # TODO 2: run every head on x and concatenate the results.
        #   Each head returns (B, T, head_size). Concatenated along the
        #   last dimension they give (B, T, n_embd) again.
        #   torch.cat takes a list of tensors and a dim.
        out = ...

        # TODO 3: pass the concatenated output through the projection.
        #   Each head has written its answer into its own 16 dims of the
        #   vector, and nothing has compared them yet. The projection is
        #   one learned matrix that reads across all the heads at once.
        out = ...

        return out


class MultiHeadModel(nn.Module):
    """Same harness as SingleHeadModel, with the head swapped for many."""

    def __init__(self, vocab_size: int, n_embd: int = 64, block_size: int = 64,
                 n_head: int = 4, **_):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.attn(x)
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
