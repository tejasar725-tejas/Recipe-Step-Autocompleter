# Recipe Step Autocompleter

An offline command-line tool that predicts likely continuations for incomplete cooking instructions using an n-gram language model.

## Features

- Built-in cooking-instruction corpus, with support for a custom text corpus
- Text normalization and tokenization
- Unigram, bigram, and trigram counts
- Maximum-likelihood estimates with configurable Laplace smoothing
- Greedy or weighted-random next-token generation
- Multiple suggestions and configurable continuation length
- Interactive CLI and one-shot command mode
- Unit and smoke tests

## Requirements

- Python 3.9+
- No third-party runtime dependencies

## Setup

```bash
git clone <your-repository-url>
cd recipe-step-autocompleter
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Interactive mode:

```bash
python -m app.cli
```

One-shot mode:

```bash
python -m app.cli --prompt "Add the chopped onions" --num-suggestions 3 --length 8
```

Use your own corpus (one instruction per line):

```bash
python -m app.cli --corpus data/recipe_instructions.txt --prompt "Preheat the oven"
```

Useful options:

```text
--order 3             n-gram order (default: 3)
--alpha 1.0           Laplace smoothing strength (default: 1.0)
--length 10           maximum generated tokens (default: 10)
--num-suggestions 3   number of continuations (default: 3)
--strategy greedy     greedy or sample
--corpus PATH         optional UTF-8 text file, one instruction per line
```

In interactive mode, type a partial instruction and press Enter. Type `:quit` to exit.

## How it works

1. Each instruction is lowercased, punctuation is separated, and text is tokenized.
2. The model counts observed n-grams up to the selected order.
3. For a context `c` and candidate token `w`, the smoothed estimate is:

   `P(w | c) = (count(c, w) + alpha) / (count(c) + alpha * vocabulary_size)`

4. The generator repeatedly predicts a token and appends it until it reaches the requested length or emits an end-of-sequence token.

This is a small statistical language model, not an LLM. Predictions depend on the examples in the corpus; add varied, high-quality instructions to improve results.

## Project structure

```text
recipe-step-autocompleter/
  .env.example
  .github/workflows/ci.yml
  .gitignore
  README.md
  app/
    __init__.py
    cli.py
    core.py
    llm.py
    storage.py
  pytest.ini
  requirements.txt
  tests/
    test_core.py
    test_smoke.py
```

## Tests

```bash
pytest
```

## Future improvements

- Import recipe datasets from CSV/JSON
- Compare Laplace smoothing with backoff or interpolation
- Evaluate predictions using held-out data and perplexity
- Add a curses or web interface
