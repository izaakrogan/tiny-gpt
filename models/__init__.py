from .bigram import BigramModel
from .single_attention_head import SingleHeadModel

MODELS = {
    "bigram": BigramModel,
    "head": SingleHeadModel,
}
