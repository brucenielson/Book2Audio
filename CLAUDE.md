# Claude Instructions for Book2Audio

## Test-Driven Development

This project uses test-driven development. **Always write failing unit tests before making any code change.**

- Write the test(s) first and confirm they fail for the right reason
- Then prompt the user before implementing the fix or feature to make them pass
- Then run the full fast suite to confirm nothing else broke

This applies to bug fixes, new features, and refactors. Do not skip ahead to the implementation even if the fix seems obvious.

## Running Tests

When verifying a code change, run only the fast unit tests:

```
pytest -m "not integration and not canonical"
```

Do **not** run the canonical or integration tests yourself. Instead, ask the user to run them when needed:

- **Canonical tests** (`-m canonical`) parse full PDF and EPUB documents against saved canonical output. They are slow but token-free.
- **Integration tests** (`-m integration`) call a local LLM via Ollama. They are slow and burn tokens.
