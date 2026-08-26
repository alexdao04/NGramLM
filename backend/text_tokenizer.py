import spacy
from spacy.language import Language
# corpus tokenizer

def make_tokenizer(*, max_length: int = 5_000_000) -> Language:
    nlp = spacy.blank("en")
    nlp.max_length = max_length
    return nlp


def tokenize(text: str, *, nlp: Language | None = None) -> list[str]:
    """Return lowercase alphabetic tokens from text."""
    tokenizer = nlp or make_tokenizer(max_length=max(5_000_000, len(text) + 1))
    return [token.text.lower() for token in tokenizer(text) if token.is_alpha]
