from collections import Counter
from typing import Mapping


Context = tuple[str, ...]
OrderCounts = Mapping[int, Mapping[Context, Counter[str]]]


def backoff(
    counts_by_context_length: OrderCounts,
    context: Context,
) -> Counter[str]:
    """Return choices for the longest available suffix of context."""
    maximum_length = min(len(context), max(counts_by_context_length))

    # try the full context first, then remove one word from the left each time
    for context_length in range(maximum_length, -1, -1):
        current_context = context[-context_length:] if context_length else ()
        choices = counts_by_context_length[context_length].get(current_context)
        if choices:
            return choices

    return Counter()
