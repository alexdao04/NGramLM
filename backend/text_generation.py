import random
import re
from typing import Sequence

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
    supplied = tuple(word.lower() for word in start)
    if history_size == 0 or len(supplied) >= history_size:
        generated = list(supplied)
    else:
        possible = [
            context
            for context in model.counts
            if context[:len(supplied)] == supplied
        ]
        if not possible:
            raise ValueError("Starting words do not occur as a model context")
        generated = list(rng.choice(possible))

    while len(generated) < number_of_tokens:
        context = tuple(generated[-history_size:]) if history_size else ()
        choices = model.counts.get(context)
        if choices:
            generated.append(
                rng.choices(list(choices), weights=choices.values(), k=1)[0]
            )
        else:
            generated.extend(rng.choice(list(model.counts)))
    return generated[:number_of_tokens]


def untokenize(tokens: Sequence[str]) -> str:
    """Join tokens and correct punctuation spacing."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%\)\]’])", r"\1", text)
    text = re.sub(r"([\(\[‘])\s+", r"\1", text)
    text = re.sub(r"\s+(['’])\s+", r"\1", text)
    return text
