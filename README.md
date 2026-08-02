# Tiny GPT

Run a bigram baseline, build a single attention head and continue from there.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data.py        # downloads tiny shakespeare and reports the vocab
```

It runs on a laptop CPU.

## How to use this repo

Start by looking at the tokenizer. It does two things: it builds the vocabulary (every distinct character in the dataset gets an integer id, 65 in all), and it converts text to those ids and back. Nothing is learned here and the ids are not embeddings, they are just labels. The models turn them into vectors later with an embedding layer.

Then look at the bigram baseline. It works on two characters at a time: you give it the character you're on, it gives you back a distribution over what comes next. What it generates is bad, and it's meant to be. It's here to get the training loop running and to give us some bad output to compare against, and hopefully improve on.

The single attention head is the first model you write yourself. It pulls together the attention material from the course, and it can look back over the whole context rather than one character. See how much better you can make it than the bigram.

Steps 3 and 4 reuse the same head. Step 3 runs several copies of it side by side, step 4 stacks that into a full transformer. Your step 2 code is imported unchanged.

Reference implementations are in `solutions/`. Have a proper go at each step before you look.

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

Every run writes `losses_<model>.csv` and generates about 200 characters at the end. `python plot.py` plots every model you've trained so far onto one chart. It needs matplotlib.

## Step 3: multi-head attention

```
python train.py --model multihead
```

Your single head gives each position one set of attention weights: one question about what came before. This step runs several smaller heads side by side, so each position can ask several.

`models/multi_head.py` gives you the shapes and the wiring, and the head class is your own from step 2, imported unchanged. Three TODOs:

1. Create the heads: n_head copies of your head, each working in n_embd / n_head dimensions
2. Run them all and concatenate the results
3. Mix the concatenated output through the projection

Compare the loss with your single head. The attention weights are the same size (four heads of 16 dims costs the same as one head of 64). The only new parameters are the output projection, about 4k on a 25k model. So most of the improvement is the heads specialising, not a bigger model.

## Step 4: the full transformer

```
python train.py --model transformer
```

`models/transformer.py` has the last two pieces. The feed-forward layer is written for you: two linear layers with a ReLU, applied to every position independently. You write the Block: layer norm, your multi-head attention, a residual add, layer norm, feed-forward, a residual add. The model stacks the block four times.

There are only two TODOs and each is one line. Make sure you know which part of each line is the residual before you move on.

It is also worth seeing what happens without the residual connections. Train the full model, then delete the two `x +` from your Block so each half replaces its input instead of adding to it, and train again. The loss drops to about 3.3 in the first 300 iterations and then stops improving for the rest of the run. That is worse than the bigram, even though the model has 50 times as many parameters.

Rough val losses with the default settings: bigram 2.7, single head 2.4, multi-head 2.2, transformer 1.8. Also read the four generations side by side. The difference is easier to see in the text than in the numbers.

## Resources

- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/): a real GPT-2 running in your browser, with every intermediate step on show. Type a sentence and watch the attention weights move.
- [LLM Visualization](https://bbycroft.net/llm): a 3D walkthrough of a small GPT, layer by layer. Good for seeing how the single head you're writing fits into the whole machine.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/): Jay Alammar's classic walkthrough of the full architecture, diagram by diagram.
- [Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc): 3Blue1Brown on attention, the same mechanism you're implementing, animated.
