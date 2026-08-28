from collections import Counter
from typing import Mapping


Context = tuple[str, ...]
OrderCounts = Mapping[int, Mapping[Context, Counter[str]]]
 
def stupid_backoff(
    counts_by_context_length: OrderCounts,
    context: Context,
    alpha: float = 0.4,
) -> Counter[str]:
    # we return counts for longest context first then shorter if none found.
    maximum_length = min(len(context), max(counts_by_context_length))
    vocabulary = counts_by_context_length[0].get((), Counter())

    scores: Counter[str] = Counter()

    # try the full context first, then remove one word from the left each time
    for word in vocabulary: 
        penalty = 1.0 # backoff penalty starts at 1.0 (multiplied by alpha for each backoff step)

        for length in range(maximum_length, -1, -1): # start stop step
            sub_context = context[-length:] if length > 0 else () # get the last 'length' words of the context, or empty tuple for length 0
            counts = counts_by_context_length[length].get(sub_context, Counter()) # get the counts for this sub-context, or empty Counter if not found

            if counts: # if we found counts for this context length
                probability = counts[word] / sum(counts.values()) # calculate probability for this word in the current context
                scores[word] += counts[word] * penalty # add the weighted counts to the scores
                break  # found counts for this context, no need to back off further

            penalty *= alpha  # apply backoff penalty

    return scores # return the final scores for each word based on the backoff model
