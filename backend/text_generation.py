import random
import re
from typing import Sequence

from .backoff import backoff
from .model import NGramModel


def generate_tokens(
    model: NGramModel,
    number_of_tokens: int,
    *,
    start: Sequence[str] = (),
    seed: int | None = None,
) -> list[str]:
    if number_of_tokens < 0:
        raise ValueError("number_of_tokens cannot be negative")
    if model.n < 1 or not model.counts:
        raise ValueError("model must be a non-empty n-gram model")

    rng = random.Random(seed)
    history_size = model.n - 1
    # starting words are normalized to match the lowercase training tokens
    supplied = tuple(word.lower() for word in start)
    if history_size == 0 or len(supplied) >= history_size:
        generated = list(supplied)
    else:
        # complete a short start by finding contexts with the same prefix
        possible = [
            context
            for context in model.counts
            if context[:len(supplied)] == supplied
        ]
        if not possible:
            raise ValueError("Starting words do not occur as a model context")
        generated = list(rng.choice(possible))

    while len(generated) < number_of_tokens:
        # use the most recent n - 1 tokens to find possible next tokens
        context = tuple(generated[-history_size:]) if history_size else ()
        choices = backoff(model.counts_by_context_length, context)
        if not choices:
            raise ValueError("model has no available continuation")

        # corpus counts act as weights, so common continuations are more likely
        generated.append(
            rng.choices(list(choices), weights=choices.values(), k=1)[0]
        )
    return generated[:number_of_tokens]


def untokenize(tokens: Sequence[str]) -> str:
    """Join tokens and correct punctuation spacing."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%\)\]’])", r"\1", text)
    text = re.sub(r"([\(\[‘])\s+", r"\1", text)
    text = re.sub(r"\s+(['’])\s+", r"\1", text)
    return text
