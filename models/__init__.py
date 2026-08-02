"""The model registry. Each import happens inside its function, so a
syntax error in the model you're working on doesn't stop the finished
ones from running.
"""


def bigram(**kwargs):
    from .bigram import BigramModel
    return BigramModel(**kwargs)


def head(**kwargs):
    from .single_attention_head import SingleHeadModel
    return SingleHeadModel(**kwargs)


def multihead(**kwargs):
    from .multi_head import MultiHeadModel
    return MultiHeadModel(**kwargs)


MODELS = {
    "bigram": bigram,
    "head": head,
    "multihead": multihead,
}
