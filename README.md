# N-Gram Language Model Project
*Author: Alexander Dao*<br>

Generates text using n-gram models trained on a Wikipedia article or text file.<br>
The original Jupyter notebook for which this is based on is in the "references" folder.<br>

## Setup

```bash
python -m venv dependencies
source dependencies/bin/activate
pip install -r requirements.txt
```

## Usage

Generate text from a Wikipedia article:

```bash
python -m backend.experiment --article Google --n <int>
```

Generate text from a local file:

```bash
python -m backend.experiment --file corpus.txt --n <int>
```

Useful options:

- `--n 2 3 4`: n-gram window sizes to compare
- `--tokens 100`: number of tokens to generate
- `--start the search`: starting words
- `--seed 6`: random seed for repeatable output

## Tests

Run the tests in verbose mode to verify that model training, text generation,
validation, and tokenization work:

```bash
python tests/test_backend.py -v
```

The tests print debugging details such as learned counts, generated tokens, and
tokenizer output. Everything is working when all tests end with:

```text
Ran 7 tests

OK
```
