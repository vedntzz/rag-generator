# RAG Generator

Drop in any set of documents at runtime, get a question-answering system grounded in them.
No code changes between document sets: each set lives in its own named **collection**.

## What it does
- **Ingests at runtime:** `.txt`, `.md`, `.pdf`, `.docx` via CLI or HTTP upload
- **Builds a per-collection index:** chunk → embed → persist to `data/<collection>/`
- **Answers with grounding:** every answer cites the source file and chunk it came from
- **Refuses to guess:** if nothing relevant is retrieved, it says so and never calls the LLM

## Quick start
```bash
git clone <repo-url> && cd rag-generator
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add ANTHROPIC_API_KEY

rag ingest --collection hr sample_docs/hr
rag ask --collection hr "How many leave days do employees get?"
```

Swap in a completely different document set, with no code changes:
```bash
rag ingest --collection product sample_docs/product
rag ask --collection product "What does the Pro plan include?"
```

## HTTP API
```bash
uvicorn rag_generator.interfaces.api:app --reload
```
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/collections/{name}/documents` | multipart `files` | `{ "chunks_indexed": int }` |
| POST | `/collections/{name}/ask` | `{ "question": str }` | `{ "answer", "grounded", "citations": [...] }` |
| GET | `/collections` | none | `["hr", "product"]` |

Example response:
```json
{
  "answer": "Employees receive 24 days of paid leave per year [1].",
  "grounded": true,
  "citations": [{ "source": "leave_policy.pdf", "chunk_index": 3, "score": 0.82 }]
}
```

## Architecture
```
 CLI / FastAPI
      │
 RagService (facade) ── built by container.py (composition root)
   ├─ IngestService:  LoaderRegistry → Chunker → Embedder → VectorStore
   └─ AnswerService:  Embedder → VectorStore.search → score filter → LLM → Answer
```
All five components (Loader, Chunker, Embedder, VectorStore, LLM) are **Protocols** in
`ports.py`. Adapters can be swapped in `container.py` without touching business logic.

### Design decisions
| Decision | Why | Trade-off |
|---|---|---|
| Ports & adapters (Strategy) | Swap embedder/LLM/store without touching business logic | Slightly more files |
| Loader registry by extension | New format = one class + one line | None meaningful |
| Local embeddings (fastembed, bge-small) | No second API key, fast, deterministic | Weaker than large hosted embedders |
| NumPy store persisted per collection | Zero infrastructure, fast to review | Not built for millions of chunks; swap for pgvector/Qdrant |
| Score threshold before LLM | Cheap, hard guard against hallucinated answers | Threshold needs tuning per corpus |
| Recursive chunking (800 / 100 overlap) | Respects paragraph and sentence boundaries | Not semantic chunking |

## Configuration (`.env`)
| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | LLM access |
| `RAG_LLM_MODEL` | `claude-haiku-4-5-20251001` | Generation model |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `RAG_DATA_DIR` | `./data` | Where collections persist |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `100` | Chunking |
| `RAG_TOP_K` / `RAG_MIN_SCORE` | `5` / `0.3` | Retrieval and grounding threshold |

## Engineering standards
- **Test-driven:** every module was built red → green → refactor. Git history shows
  `test(...)` commits before `feat(...)` commits.
- **Hard limits, enforced by a test:** no function over 10 lines, no file over 200 lines
  (`scripts/check_limits.py`, run inside `pytest`).
- **No network in unit tests:** fakes for the embedder and LLM live in `tests/fakes.py`.

```bash
pytest -q                          # unit + integration + limits
python scripts/check_limits.py     # limits only
ruff check .
```

## Project layout
```
src/rag_generator/
  config.py domain.py ports.py errors.py
  loaders/ chunking/ embedding/ store/ llm/
  pipeline/    ingest.py answer.py rag_service.py container.py
  interfaces/  cli.py api.py
tests/ unit/ integration/ fakes.py test_code_limits.py
scripts/check_limits.py
sample_docs/ hr/ product/
transcripts/  # full AI agent session export
```

## How this was built
Built with Claude Code using agentic TDD. `CLAUDE.md` holds the rules the agent followed.
The full session transcript is in `transcripts/`.

## Limitations and next steps
- No OCR: scanned PDFs yield no text
- Brute-force cosine search is fine to ~100k chunks; beyond that use an ANN store
- Next: hybrid retrieval (BM25 + dense), a re-ranker, streaming answers, an eval set with
  faithfulness scoring
