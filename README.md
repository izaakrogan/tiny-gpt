# Tiny GPT

Build a transformer in stages.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data.py        # downloads tiny shakespeare and reports the vocab
```

It runs on a laptop CPU.

## How to use this repo

Start by looking at the tokenizer. It turns the text into integers so the models have something they can read, and it tells you the vocab size.

Then look at the bigram baseline. It works on two characters at a time: you give it the character you're on, it gives you back a distribution over what comes next. Nothing before that character reaches it. What it generates is bad, and it's meant to be. It's here to get the training loop running and to give us some bad output to compare against.

The single attention head is the thing you're writing today. It pulls together everything we've covered on attention so far, and it can look back over the whole context rather than one character. See how much better you can make it than the bigram.

## Step 0: tokenizer

`data.py` builds a character-level tokenizer. Every distinct character in the dataset gets an integer id. Run it, then read it. You should be able to say what the vocab size is and why character-level tokenisation is a reasonable choice for a model this small.

## Step 1: bigram baseline

```
python train.py --model bigram
```

This one is finished. It is a lookup table from the current character to a distribution over next characters. No attention, and no notion of position. It is here so that the training loop is working before you start on attention: cross-entropy loss, gradients, AdamW.

Loss should fall from roughly 4.8 to roughly 2.7. Read the 200 characters it generates at the end and keep them in mind, because that is what your attention model has to improve on.

## Step 2: single attention head

```
python train.py --model head
```

`models/single_attention_head.py` gives you the shapes and the wiring. You write the attention and I've given you five TODOs:

1. Project the embedding into Query, Key and Value
2. Compute scores as Q Kᵀ, divided by √d
3. Set future positions to −∞ before the softmax
4. Softmax the scores into weights
5. Take the weighted sum of the Values

Position embeddings are already wired in. Find where they are added and work out why the model needs them.

One bug to expect: if you leave out the mask, the training loss falls further than it should while the generated text stays bad. Every position has seen the answer during training. If your loss looks too good, check the mask first.

Every run writes `losses.csv` and generates about 200 characters at the end. `python plot.py` turns the CSV into a PNG. It needs matplotlib.
