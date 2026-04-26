# Claude Instructions for Book2Audio

## Running Tests

When verifying a code change, run only the fast unit tests:

```
pytest -m "not integration and not canonical"
```

Do **not** run the canonical or integration tests yourself. Instead, ask the user to run them when needed:

- **Canonical tests** (`-m canonical`) parse full PDF and EPUB documents against saved canonical output. They are slow but token-free.
- **Integration tests** (`-m integration`) call a local LLM via Ollama. They are slow and burn tokens.
