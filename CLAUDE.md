# CLAUDE.md — RAG Generator

## Mission
Build a RAG Generator that accepts documents at runtime, indexes them into a named
collection, and answers questions grounded ONLY in that collection, with citations.
Any document set must work with zero code changes (collection name + files in, answers out).

## Non-negotiable rules
1. **TDD, always.** For every unit: write the failing test → run `pytest` and SHOW the red
   output → write the minimum code → SHOW green → refactor. No production code without a
   failing test first. Never edit a test just to make it pass.
2. **Function ≤ 10 lines.** Counted from the `def` line to the last line, excluding blank
   lines and the docstring. If a function would exceed 10, extract well-named helpers.
3. **File ≤ 200 lines.** If a file would exceed 200, split it by responsibility.
4. **Limits are enforced by code.** `python scripts/check_limits.py` must pass after every
   step; `tests/test_code_limits.py` runs it inside `pytest`.
5. **Comments:** one-line comments only, no block comments. Names must explain what a
   function does without opening it (`split_text_into_overlapping_chunks`, not `process`).
6. **Types:** full type hints; `ruff check .` must be clean.
7. **No network in unit tests.** Use `tests/fakes.py` (FakeEmbedder, FakeLLM).
8. **Depend on ports, not adapters.** Only `pipeline/container.py` constructs concrete classes.
9. **No new dependencies** beyond `pyproject.toml` without asking first.
10. **Don't invent scope.** If a requirement is ambiguous, stop and ask.

## Architecture
```
            ┌──────────── interfaces ────────────┐
            │   cli.py (Typer)    api.py (FastAPI)│
            └───────────────┬────────────────────┘
                            ▼
                 RagService (Facade)  ◄── container.py (composition root)
                 ├── IngestService                 
                 │     LoaderRegistry → Chunker → Embedder → VectorStore.add
                 └── AnswerService
                       Embedder → VectorStore.search → score filter → LLM → Answer
```
Ports (Protocols in `ports.py`): `DocumentLoader`, `Chunker`, `Embedder`, `VectorStore`, `LLM`.

## Design patterns (use these, don't add others without reason)
- **Strategy / Ports & Adapters:** every external concern is a Protocol; adapters are swappable.
- **Registry (Factory):** `LoaderRegistry` maps file extension → loader. New format = one new
  class + one `register()` call. Unsupported extension raises `UnsupportedFileTypeError`.
- **Repository:** `VectorStore` is scoped per collection and persists to `data/<collection>/`.
- **Facade:** `RagService` exposes exactly `ingest(collection, paths)`, `ask(collection, question)`,
  `list_collections()`.
- **Composition root:** `container.build_rag_service(settings)` wires all adapters.

## Domain models (`domain.py`, frozen dataclasses)
- `Document(source: str, text: str)`
- `Chunk(source: str, index: int, text: str)`
- `ScoredChunk(chunk: Chunk, score: float)`
- `Citation(source: str, chunk_index: int, score: float)`
- `Answer(text: str, citations: list[Citation], grounded: bool)`

## Grounding contract
1. Embed question, retrieve `top_k` (default 5) by cosine similarity.
2. Drop chunks with score < `min_score` (default 0.3).
3. If nothing remains → return `Answer(NOT_FOUND_MESSAGE, [], grounded=False)` WITHOUT calling the LLM.
4. Prompt tells the LLM: answer only from numbered context, cite as `[n]`, otherwise reply
   exactly with `NOT_FOUND_MESSAGE`.
5. Asking a collection that doesn't exist raises `CollectionNotFoundError`.

## Adapters (defaults)
| Port | Adapter | Notes |
|---|---|---|
| DocumentLoader | text (.txt/.md), pypdf (.pdf), python-docx (.docx) | registry by extension |
| Chunker | `RecursiveCharacterChunker` | size 800, overlap 100, splits on paragraph → sentence → char |
| Embedder | `FastEmbedEmbedder` | `BAAI/bge-small-en-v1.5`, local ONNX, no API key |
| VectorStore | `NumpyVectorStore` | cosine, persisted as `.npy` + `.json` per collection |
| LLM | `AnthropicLLM` | model from `RAG_LLM_MODEL` env |

## Folder layout
```
src/rag_generator/
  config.py  domain.py  ports.py  errors.py
  loaders/     registry.py text_loader.py pdf_loader.py docx_loader.py
  chunking/    recursive_chunker.py
  embedding/   fastembed_embedder.py
  store/       numpy_store.py
  llm/         anthropic_llm.py prompts.py
  pipeline/    ingest.py answer.py rag_service.py container.py
  interfaces/  cli.py api.py
tests/
  fakes.py  conftest.py  test_code_limits.py
  unit/        test_<module>.py (mirror src)
  integration/ test_end_to_end.py
scripts/check_limits.py
sample_docs/  hr/  product/     # two different doc sets for the e2e proof
```

## Test conventions
- Name: `test_<unit>_<behavior>_<condition>` (e.g. `test_chunker_keeps_overlap_between_chunks`).
- Arrange / Act / Assert, one behavior per test.
- Use `tmp_path` for any file or persistence test.
- Integration test MUST ingest two different doc sets into two collections and prove
  answers come only from the queried collection.

## Commands
```
pip install -e ".[dev]"
pytest -q
python scripts/check_limits.py
ruff check .
rag ingest --collection hr sample_docs/hr
rag ask --collection hr "How many leave days do employees get?"
uvicorn rag_generator.interfaces.api:app --reload
```

## Definition of done (every step)
- [ ] Tests written first and shown failing
- [ ] `pytest -q` green
- [ ] `python scripts/check_limits.py` passes
- [ ] `ruff check .` clean
- [ ] Two commits: `test(<module>): ...` then `feat(<module>): ...` (TDD visible in git history)

## Out of scope
Auth, web UI, OCR for scanned PDFs, streaming, multi-user tenancy, re-ranking models.
