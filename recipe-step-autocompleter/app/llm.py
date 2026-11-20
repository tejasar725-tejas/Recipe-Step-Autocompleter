"""Optional extension point for future LLM-backed suggestions.

The project intentionally uses its local n-gram model by default and does not
require an API key or network access.
"""


def llm_available() -> bool:
    """Return False until an external LLM provider is implemented."""
    return False
