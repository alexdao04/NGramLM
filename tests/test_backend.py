"""Tests for model training, generation, and tokenization."""

import unittest
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.model import NGramModel, build_ngram_model
from backend.text_generation import generate_tokens
from backend.text_tokenizer import tokenize


class NGramModelTests(unittest.TestCase):
    def setUp(self):
        self.tokens = "one fish two fish red fish blue fish".split()

    def test_unigram_uses_empty_context(self):
        counts = build_ngram_model(self.tokens, 1)[()]
        print(f"\n[unigram] counts={dict(counts)}")
        self.assertEqual(counts, Counter(self.tokens))

    def test_bigram_counts_all_continuations(self):
        model = build_ngram_model(self.tokens, 2)
        print(f"\n[bigram] after 'fish'={dict(model[('fish',)])}")
        self.assertEqual(model[("fish",)], Counter({"two": 1, "red": 1, "blue": 1}))

    def test_arbitrary_window(self):
        model = NGramModel.train(self.tokens, 5)
        print(f"\n[5-gram] contexts={model.context_count}")
        self.assertEqual(model.n, 5)
        self.assertEqual(model.context_count, 4)

    def test_seed_makes_generation_repeatable(self):
        model = NGramModel.train(self.tokens, 3)
        first = generate_tokens(model, 20, seed=7)
        print(f"\n[generation] seed=7 tokens={' '.join(first)}")
        self.assertEqual(first, generate_tokens(model, 20, seed=7))
        self.assertEqual(len(first), 20)

    def test_invalid_window_is_rejected(self):
        with self.assertRaises(ValueError) as error:
            build_ngram_model(self.tokens, 0)
        print(f"\n[validation] error={error.exception}")


class TokenizerTests(unittest.TestCase):
    def test_lowercases_and_removes_punctuation(self):
        tokens = tokenize("Hello, N-Grams!")
        print(f"\n[tokenizer] tokens={tokens}")
        self.assertEqual(tokens, ["hello", "n", "grams"])


if __name__ == "__main__":
    unittest.main()
