"""Step 0: the tokenizer.

Character-level: every distinct character in the dataset gets an integer id.
Small vocab (~65).
"""

import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_FILE = os.path.join(os.path.dirname(__file__), "input.txt")


def load_text(path: str = DATA_FILE) -> str:
    if not os.path.exists(path):
        # Only the default dataset gets downloaded. If you passed your own
        # path and it doesn't exist, that's an error, not a download.
        if path != DATA_FILE:
            raise FileNotFoundError(f"no such file: {path}")
        print(f"downloading dataset to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self._to_id = {ch: i for i, ch in enumerate(chars)}
        self._to_ch = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self._to_id[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._to_ch[i] for i in ids)


if __name__ == "__main__":
    text = load_text()
    tok = CharTokenizer(text)
    print(f"dataset: {len(text):,} characters")
    print(f"vocab size: {tok.vocab_size}")
    demo = "The bank of the river was"
    print(f"encode({demo!r}) -> {tok.encode(demo)}")
    print(f"round trip ok: {tok.decode(tok.encode(demo)) == demo}")
