import argparse
from pathlib import Path
from .corpus import fetch_wikipedia_article, read_text
from .model import NGramModel
from .text_generation import generate_tokens, untokenize
from .text_tokenizer import tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--article", default="Google", help="Wikipedia article title")
    source.add_argument("--file", type=Path, help="local UTF-8 corpus")
    parser.add_argument("--n", type=int, nargs="+", default=[2, 3, 4], help="window sizes")
    parser.add_argument("--tokens", type=int, default=100, help="generated token count")
    parser.add_argument("--start", nargs="*", default=(), help="optional starting words")
    parser.add_argument("--seed", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(n < 1 for n in args.n):
        raise SystemExit("Every --n value must be at least 1")
    text = read_text(args.file) if args.file else fetch_wikipedia_article(args.article)
    corpus_tokens = tokenize(text)
    print(f"Corpus: {len(corpus_tokens):,} tokens")
    for n in args.n:
        model = NGramModel.train(corpus_tokens, n)
        generated = generate_tokens(model, args.tokens, start=args.start, seed=args.seed)
        print(f"\n{n}-gram ({model.context_count:,} contexts)\n{untokenize(generated)}")


if __name__ == "__main__":
    main()
