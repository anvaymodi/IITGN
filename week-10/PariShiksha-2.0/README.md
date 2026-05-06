# PariShiksha Study Assistant v2.0
**Week 10 · Production-Grade RAG · PG Diploma AI-ML · IIT Gandhinagar**

Builds on the Week 9 BM25 baseline. This version adds dense embeddings, persistent Chroma vector store, strict grounding with citation, honest 12-question evaluation on three axes, and one targeted fix with a measured delta.

---

## Architecture

```
PDFs (iesc108, iesc109)
        │
        ▼  PyMuPDF + pdfplumber fallback
   Raw text per page
        │
        ▼  tiktoken cl100k_base · 250 tokens · 40 overlap
   wk10_chunks.json
   content_type: prose | worked_example | question_or_exercise
        │
        ├──── BM25Okapi  (lexical, Stage 1 sanity checks)
        │
        └──── text-embedding-3-small → Chroma (cosine, persisted)
                    │
                    ▼  top-5 dense retrieve · similarity threshold ≥ 0.35
              Retrieved chunks
                    │
                    ▼  Strict grounding prompt + citation requirement
              claude-haiku-4-5  (temperature = 0)
                    │
                    ▼
        {answer, sources, chunk_ids, cited_ids}
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Fill in `.env` with your API keys:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Download NCERT PDFs from https://ncert.nic.in/textbook.php?iesc1=0-11 and place in `data/`:
- `iesc108.pdf` — Chapter 8: Motion
- `iesc109.pdf` — Chapter 9: Force and Laws of Motion

```bash
jupyter notebook notebook.ipynb
```

Run cells top to bottom. Cell 6 embeds chunks into Chroma on first run and skips on subsequent runs.

---

## Deliverables

| File | Stage | Description |
|---|---|---|
| `wk10_chunks.json` | 1 | Token-aware chunks with metadata |
| `chunking_diff.md` | 1 | Wk9 vs Wk10 chunk strategy |
| `retrieval_log.json` | 2 | Dense retrieval on 10 queries |
| `retrieval_misses.md` | 2 | 3 miss diagnoses |
| `prompt_diff.md` | 3 | Permissive vs strict prompt |
| `eval_raw.csv` | 4 | Raw LLM output for 12 questions |
| `eval_scored.csv` | 4 | Scored on correctness / grounded / refused_oos |
| `eval_v2_scored.csv` | 5 | After similarity threshold fix |
| `fix_memo.md` | 5 | Fix rationale and delta |
| `reflection.md` | — | Reflection questionnaire |

---

## Evaluation Summary

| Version | Correct | Grounded | OOS Refused |
|---|---|---|---|
| v1 (Stage 4) | 8 / 12 | 10 / 12 | 2 / 3 |
| v2 (Stage 5, fix applied) | 9 / 12 | 11 / 12 | 3 / 3 |

Worst failure diagnosed: E12 — plausibly-answerable OOS (Moon weight calculation). Failure mode: mixed structure. Fix: similarity threshold guard at 0.35 cosine similarity.

---

## Loom

[Add Loom link here before final submission]

---

## Key Design Decisions

**250 tokens, not 400** — smaller chunks give sharper cosine similarity signal. Worked examples emitted as single atomic chunks so problem and solution never split.

**tiktoken cl100k_base** — matches text-embedding-3-small tokenisation. Week 9 used word count; token count is more accurate.

**Same embedding model at index and query time** — locked to one constant. No drift between runs.

**Similarity threshold 0.35** — OOS queries that retrieve vaguely-related chunks score 0.30–0.40. In-scope queries score ≥ 0.45. Threshold forces refusal before the LLM call.

**temperature = 0** — evaluation is reproducible. v1 and v2 scores are comparable.
