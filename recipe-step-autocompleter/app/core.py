"""N-gram language model for recipe instruction autocomplete."""

from __future__ import annotations

from collections import Counter, defaultdict
import random
import re
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\w\s]", re.IGNORECASE)


def preprocess(text: str) -> str:
    """Normalize whitespace and lowercase text while preserving punctuation."""
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word, number, and punctuation tokens."""
    return _TOKEN_RE.findall(preprocess(text))


DEFAULT_CORPUS = [
    "Preheat the oven to 180 degrees Celsius.",
    "Preheat the oven to 200 degrees Celsius and line a baking tray with parchment paper.",
    "Heat the oil in a large pan over medium heat.",
    "Add the chopped onion and cook until soft and translucent.",
    "Add the minced garlic and cook for one minute until fragrant.",
    "Stir in the chopped tomatoes and simmer for ten minutes.",
    "Season with salt, black pepper, and a pinch of chilli flakes.",
    "Bring a large pot of salted water to a boil.",
    "Add the pasta and cook until al dente, following the package instructions.",
    "Drain the pasta and reserve a cup of the cooking water.",
    "Mix the flour, baking powder, and salt in a bowl.",
    "Whisk the eggs, milk, and melted butter in a separate bowl.",
    "Pour the wet ingredients into the dry ingredients and mix until just combined.",
    "Fold in the grated cheese and chopped herbs.",
    "Transfer the mixture to the prepared baking dish.",
    "Bake for twenty to twenty five minutes until golden brown.",
    "Allow the cake to cool completely before serving.",
    "Wash and finely chop the fresh coriander.",
    "Add the vegetables and stir-fry for three to four minutes.",
    "Pour in the coconut milk and bring the mixture to a gentle simmer.",
    "Cover the pan and cook over low heat until the lentils are tender.",
    "Taste and adjust the seasoning with more salt or lemon juice.",
    "Garnish with fresh coriander and serve hot with rice.",
    "Place the chicken in a bowl and add the yogurt and spices.",
    "Cover and marinate in the refrigerator for at least thirty minutes.",
    "Heat a little oil in a pan and brown the chicken on all sides.",
    "Reduce the heat, cover, and cook until the chicken is cooked through.",
    "Squeeze fresh lemon juice over the dish before serving.",
    "Roll the dough into small balls and place them on a lined tray.",
    "Let the dough rest for ten minutes before shaping.",
]


class NGramModel:
    """Counts n-grams and predicts next tokens using Laplace smoothing."""

    def __init__(self, order: int = 3, alpha: float = 1.0) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.order = order
        self.alpha = float(alpha)
        self.vocabulary: set[str] = set()
        self.ngram_counts: dict[int, Counter[tuple[str, ...]]] = {
            n: Counter() for n in range(1, order + 1)
        }
        self.context_counts: dict[int, Counter[tuple[str, ...]]] = {
            n: Counter() for n in range(1, order)
        }
        self._trained = False

    def fit(self, instructions: Iterable[str]) -> "NGramModel":
        """Build n-gram and context counts from one instruction per string."""
        self.vocabulary.clear()
        for counter in self.ngram_counts.values():
            counter.clear()
        for counter in self.context_counts.values():
            counter.clear()

        for instruction in instructions:
            tokens = tokenize(instruction)
            if not tokens:
                continue
            sequence = ["<s>"] * (self.order - 1) + tokens + ["</s>"]
            self.vocabulary.update(tokens)
            self.vocabulary.add("</s>")
            for n in range(1, self.order + 1):
                for index in range(len(sequence) - n + 1):
                    gram = tuple(sequence[index:index + n])
                    self.ngram_counts[n][gram] += 1
                    if n > 1:
                        self.context_counts[n - 1][gram[:-1]] += 1

        # A non-empty vocabulary is needed for probability estimates.
        if not self.vocabulary:
            self.vocabulary = {"</s>"}
        self._trained = True
        return self

    def _context_for(self, tokens: Sequence[str]) -> tuple[str, ...]:
        if self.order == 1:
            return ()
        return tuple(tokens[-(self.order - 1):])

    def probabilities(self, context: Sequence[str] = ()) -> dict[str, float]:
        """Return next-token probabilities for a context, with backoff if unseen."""
        if not self._trained:
            raise RuntimeError("Call fit() before requesting probabilities.")
        vocab = sorted(self.vocabulary)
        context_tuple = tuple(context[-(self.order - 1):]) if self.order > 1 else ()

        # Back off to shorter contexts until a context with observations is found.
        while context_tuple:
            total = self.context_counts[len(context_tuple)].get(context_tuple, 0)
            if total:
                break
            context_tuple = context_tuple[1:]
        if not context_tuple and self.order > 1:
            counts = Counter({(word,): count for (word,), count in self.ngram_counts[1].items()})
            total = sum(counts.values())
            numerator = {word: counts.get((word,), 0) for word in vocab}
        else:
            n = len(context_tuple) + 1
            total = self.context_counts[len(context_tuple)].get(context_tuple, 0) if context_tuple else sum(self.ngram_counts[1].values())
            numerator = {
                word: self.ngram_counts[n].get(context_tuple + (word,), 0)
                for word in vocab
            }

        denominator = total + self.alpha * len(vocab)
        if denominator == 0:
            return {word: 1.0 / len(vocab) for word in vocab}
        return {word: (count + self.alpha) / denominator for word, count in numerator.items()}

    def predict_next(self, context: Sequence[str] = (), strategy: str = "greedy", rng: random.Random | None = None) -> str:
        """Predict one token using greedy maximum probability or weighted sampling."""
        probs = self.probabilities(context)
        if strategy == "greedy":
            # Stable tie-breaking makes output reproducible.
            return max(sorted(probs), key=lambda token: probs[token])
        if strategy == "sample":
            generator = rng or random
            words, weights = zip(*probs.items())
            return generator.choices(words, weights=weights, k=1)[0]
        raise ValueError("strategy must be 'greedy' or 'sample'")

    def generate(self, prompt: str, max_tokens: int = 10, strategy: str = "greedy", seed: int | None = None) -> str:
        """Generate a continuation; the returned string includes the original prompt."""
        if max_tokens < 1:
            return prompt.strip()
        if not self._trained:
            raise RuntimeError("Call fit() before generating text.")
        prompt_tokens = tokenize(prompt)
        context = (["<s>"] * max(0, self.order - 1 - len(prompt_tokens))) + prompt_tokens
        rng = random.Random(seed) if seed is not None else None
        generated: list[str] = []
        for _ in range(max_tokens):
            next_token = self.predict_next(context, strategy=strategy, rng=rng)
            if next_token == "</s>":
                break
            generated.append(next_token)
            context.append(next_token)
        continuation = self._detokenize(generated)
        base = prompt.rstrip()
        if not continuation:
            return base
        separator = "" if continuation[0] in ".,!?;:%)]}" or not base else " "
        return base + separator + continuation

    @staticmethod
    def _detokenize(tokens: Sequence[str]) -> str:
        text = ""
        no_space_before = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}"}
        no_space_after = {"(", "[", "{"}
        for token in tokens:
            if not text or token in no_space_before or text[-1:] in no_space_after:
                text += token
            else:
                text += " " + token
        return text

    def suggest(self, prompt: str, num_suggestions: int = 3, max_tokens: int = 10, strategy: str = "sample", seed: int | None = 42) -> list[str]:
        """Return several continuations, using distinct seeds for sampled results."""
        if num_suggestions < 1:
            raise ValueError("num_suggestions must be at least 1")
        return [
            self.generate(prompt, max_tokens=max_tokens, strategy=strategy,
                          seed=None if seed is None else seed + i)
            for i in range(num_suggestions)
        ]
