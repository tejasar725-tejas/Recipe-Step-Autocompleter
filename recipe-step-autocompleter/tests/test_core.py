import pytest

from app.core import NGramModel, preprocess, tokenize


def test_preprocess_and_tokenize():
    assert preprocess("  ADD   GARLIC ") == "add garlic"
    assert tokenize("Add 2 cloves of garlic.") == ["add", "2", "cloves", "of", "garlic", "."]


def test_ngram_counts_and_smoothed_probabilities():
    model = NGramModel(order=2, alpha=1).fit(["add salt", "add pepper"])
    assert model.ngram_counts[2][("add", "salt")] == 1
    probabilities = model.probabilities(["add"])
    assert set(probabilities) == model.vocabulary
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["salt"] > 0


def test_generation_includes_prompt_and_continuation():
    model = NGramModel(order=2).fit(["add salt and stir", "add pepper and stir"])
    result = model.generate("add", max_tokens=3, strategy="greedy")
    assert result.startswith("add ")
    assert len(result.split()) >= 2


def test_invalid_arguments():
    with pytest.raises(ValueError):
        NGramModel(order=0)
    with pytest.raises(ValueError):
        NGramModel(alpha=-0.1)
    with pytest.raises(ValueError):
        NGramModel().fit(["add salt"]).suggest("add", num_suggestions=0)
