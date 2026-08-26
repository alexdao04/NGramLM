# import ...

def build_ngram_model(tokens, n):
    model = defaultdict(Counter)

    for i in range(0, len(tokens)-1):
      history = tuple(tokens[i-(n-1):i])
      next = tokens[i]
      model[history][next] += 1
    return model

bigram_model = build_ngram_model(tokens, 2)
trigram_model = build_ngram_model(tokens, 3)
quadrigram_model = build_ngram_model(tokens, 4)

print(f"Bigram contexts: {len(bigram_model):,}")
print(f"Trigram contexts: {len(trigram_model):,}")
print(f"Quadrigram contexts: {len(quadrigram_model):,}")
