from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

Context = tuple[str, ...]
Counts = dict[Context, Counter[str]]


def build_ngram_model(tokens: Iterable[str], n: int) -> Counts:
    if n < 1:
        raise ValueError("n must be at least 1")
    token_list = list(tokens)
    if len(token_list) < n:
        raise ValueError(f"Need at least {n} tokens to build a {n}-gram model")
    model: defaultdict[Context, Counter[str]] = defaultdict(Counter)
    history_size = n - 1
    for index in range(history_size, len(token_list)):
        context = tuple(token_list[index - history_size:index]) if history_size else ()
        model[context][token_list[index]] += 1
    return dict(model)


@dataclass(frozen=True)
class NGramModel:
    n: int
    counts: Mapping[Context, Counter[str]]

    @classmethod
    def train(cls, tokens: Iterable[str], n: int) -> "NGramModel":
        return cls(n=n, counts=build_ngram_model(tokens, n))

    @property
    def context_count(self) -> int:
        return len(self.counts)
