"""Command-line interface for recipe step autocomplete."""

from __future__ import annotations

import argparse

from .core import NGramModel
from .storage import load_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autocomplete incomplete cooking instructions with an n-gram model."
    )
    parser.add_argument("--prompt", "-p", help="Partial recipe instruction. Omit for interactive mode.")
    parser.add_argument("--corpus", help="UTF-8 text file containing one instruction per line.")
    parser.add_argument("--order", type=int, default=3, help="N-gram order (default: 3).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing strength (default: 1.0).")
    parser.add_argument("--length", type=int, default=10, help="Maximum number of generated tokens.")
    parser.add_argument("--num-suggestions", type=int, default=3, help="Number of suggestions to display.")
    parser.add_argument("--strategy", choices=("greedy", "sample"), default="sample", help="Prediction strategy.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable sampling.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.order < 1 or args.alpha < 0 or args.length < 1 or args.num_suggestions < 1:
        parser.error("--order and --length and --num-suggestions must be positive; --alpha cannot be negative")

    try:
        corpus = load_corpus(args.corpus)
    except (OSError, UnicodeError) as exc:
        parser.error(str(exc))
    model = NGramModel(order=args.order, alpha=args.alpha).fit(corpus)
    print(f"Loaded {len(corpus)} recipe instructions. N-gram order: {args.order}.")

    if args.prompt:
        prompts = [args.prompt]
    else:
        print("Enter an incomplete recipe step. Type :quit to exit.")
        prompts = None

    def show(prompt: str) -> None:
        suggestions = model.suggest(
            prompt, num_suggestions=args.num_suggestions,
            max_tokens=args.length, strategy=args.strategy, seed=args.seed,
        )
        for index, suggestion in enumerate(suggestions, start=1):
            print(f"{index}. {suggestion}")

    if prompts is not None:
        show(prompts[0])
        return 0

    while True:
        try:
            prompt = input("\nRecipe step> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if prompt.lower() in {":quit", ":q", "exit"}:
            print("Goodbye.")
            break
        if not prompt:
            print("Please enter part of a recipe instruction.")
            continue
        show(prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
