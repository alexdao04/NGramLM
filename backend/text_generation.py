from collections import Counter, defaultdict
import random
import re
import requests
import spacy

def weighted_choice(counter):
    choices = list(counter.keys())
    weights = list(counter.values())
    return random.choices(choices, weights=weights, k=1)[0]

def generate_tokens(model, n, number_of_tokens, start=()):
    start = tuple(word.lower() for word in start)

    if len(start) >= n - 1:
        generated = list(start)
    else:
        possible_contexts = [context for context in model if context[:len(start)] == start]
        if not possible_contexts:
            raise ValueError("Those starting words do not occur as a context in this book.")
        generated = list(random.choice(possible_contexts))

    while len(generated) < number_of_tokens:
        context = tuple(generated[-(n - 1):])
        choices = model.get(context)

        # A context at the very end of the book may have no continuation.
        # In that rare case, begin again from another context of the same model.
        if not choices:
            restart = random.choice(list(model.keys()))
            generated.extend(restart)
        else:
            generated.append(weighted_choice(choices))

    return generated[:number_of_tokens]

def untokenize(tokens):
    # Join tokens, then remove spaces before common punctuation.
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%\)\]’])", r"\1", text)
    text = re.sub(r"([\(\[‘])\s+", r"\1", text)
    text = re.sub(r"\s+(['’])\s+", r"\1", text)
    return text

