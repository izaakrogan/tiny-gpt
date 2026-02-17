"""Step 2: a single attention head. Guided skeleton, you write the attention.

What you implement here is the week 1-4 material: QKV projections,
scaled scores, the causal mask, softmax, weighted sum.

The harness is written for you: embeddings, positions, the language-model
head, generate. Fill in the TODOs in CausalSelfAttentionHead.forward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionHead(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        # The three projections from module 279. bias=False because nothing
        # is being shifted here, the vector is only being re-expressed.
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular matrix of ones. Cell (i, j) is 1 when position i
        # is allowed to look at position j. That is the shape of the mask.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x):
        # x is (B, T, n_embd): B sequences, T positions, n_embd dims each.
        B, T, _ = x.shape

        # TODO 1: project x into Query, Key, Value.         (module 279)
        #   q, k and v are each (B, T, head_size).
        q = ...
        k = ...
        v = ...

        # TODO 2: raw relevance scores, scaled.             (module 279)
        #   Dot every Query with every Key, so scores is (B, T, T).
        #   q @ k.transpose(-2, -1) gives you the dots. Then divide by
        #   self.head_size ** 0.5, per the "why divide by root d" exercise.
        scores = ...

        # TODO 3: the causal mask.                          (this week)
        #   Future positions need to come out of softmax with zero weight.
        #   Which raw score does that? Use masked_fill:
        #   scores.masked_fill(self.tril[:T, :T] == 0, <that score>)
        #   Hint: float("-inf"), since e^(-inf) is 0.
        scores = ...

        # TODO 4: softmax the scores into attention weights. (module 282)
        #   weights is (B, T, T), each row positive and summing to 1.
        weights = ...

        # TODO 5: weighted sum of the Values.               (module 279)
        #   out is (B, T, head_size).
        out = ...

        return out


class SingleHeadModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 64, block_size: int = 64, **_):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # One learned vector per position, added to the token embedding
        # below. Without these, attention sees an unordered bag of vectors,
        # so "man bites dog" and "dog bites man" look identical to it.
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.head = CausalSelfAttentionHead(n_embd, head_size=n_embd, block_size=block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                                   # (B, T, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))      # (T, n_embd)
        x = tok + pos            # note that positions are added, not concatenated
        x = self.head(x)
        logits = self.lm_head(x)                                    # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
