"""Loading and saving recipe instruction corpora."""

from pathlib import Path
from typing import Iterable

from .core import DEFAULT_CORPUS


def load_corpus(path: str | Path | None = None) -> list[str]:
    """Load one instruction per line, or return the built-in starter corpus."""
    if path is None:
        return list(DEFAULT_CORPUS)
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    lines = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def save_corpus(instructions: Iterable[str], path: str | Path) -> None:
    """Save instructions as UTF-8 text, one instruction per line."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(item.strip() for item in instructions if item.strip()) + "\n", encoding="utf-8")
