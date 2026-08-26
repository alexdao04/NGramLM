from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

Context = tuple[str, ...]
Counts = dict[Context, Counter[str]]
CountsByContextLength = dict[int, Counts]


def build_ngram_model(tokens: Iterable[str], n: int) -> Counts:
    if n < 1:
        raise ValueError("n must be at least 1")
    token_list = list(tokens)
    if len(token_list) < n:
        raise ValueError(f"Need at least {n} tokens to build a {n}-gram model")
    model: defaultdict[Context, Counter[str]] = defaultdict(Counter)
    # an n-gram uses the previous n - 1 tokens as its context
    history_size = n - 1
    for index in range(history_size, len(token_list)):
        # map each context to counts of the tokens that followed it
        context = tuple(token_list[index - history_size:index]) if history_size else ()
        model[context][token_list[index]] += 1
    return dict(model)


@dataclass(frozen=True)
class NGramModel:
    n: int
    counts: Mapping[Context, Counter[str]]
    counts_by_context_length: Mapping[int, Mapping[Context, Counter[str]]]

    @classmethod
    def train(cls, tokens: Iterable[str], n: int) -> "NGramModel":
        token_list = list(tokens)
        # keep every lower order so generation can back off when needed
        counts_by_context_length = {
            context_length: build_ngram_model(token_list, context_length + 1)
            for context_length in range(n)
        }
        return cls(
            n=n,
            counts=counts_by_context_length[n - 1],
            counts_by_context_length=counts_by_context_length,
        )

    @property
    def context_count(self) -> int:
        return len(self.counts)
