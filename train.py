"""The training harness: loads the data, builds whichever model you ask for,
runs the training loop, then samples some text from the trained model.

    python train.py --model bigram
    python train.py --model head
    python train.py --model multihead
    python train.py --model transformer

Every run writes losses_<model>.csv and ends by generating text. There is
a lot going on in this file. You do not need to understand all of it to
build your model, but if you want to, paste it into your favourite LLM
and ask it to walk you through the moving parts.
"""

import argparse
import time

import torch

from data import CharTokenizer, load_text
from models import MODELS


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=50):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(MODELS), default="bigram")
    p.add_argument("--data", default=None, help="path to any plain-text file (default: tiny shakespeare)")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--block-size", type=int, default=64, help="context length")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--sample-chars", type=int, default=200)
    p.add_argument("--seed", type=int, default=42, help="change for a different run")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    text = load_text(args.data) if args.data else load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"vocab {tok.vocab_size}, train {n:,} chars, val {len(data) - n:,} chars")

    model = MODELS[args.model](
        vocab_size=tok.vocab_size, block_size=args.block_size
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model} ({n_params:,} parameters)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    rows = ["iter,train_loss,val_loss"]
    for it in range(args.iters + 1):
        if it % 300 == 0:
            losses = estimate_loss(model, splits, args.block_size, args.batch_size, device)
            rows.append(f"{it},{losses['train']:.4f},{losses['val']:.4f}")
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {time.time() - t0:5.1f}s")
        x, y = get_batch(splits["train"], args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    csv_name = f"losses_{args.model}.csv"
    with open(csv_name, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\nwrote {csv_name}")

    print(f"\n--- {args.sample_chars} generated characters (temperature {args.temperature}) ---")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=args.sample_chars, temperature=args.temperature)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
