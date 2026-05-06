# Chunking Diff — Week 9 → Week 10

## What changed

**Week 9:** Whitespace-split accumulation capped at 400 tokens by word count. Content type detected by regex but chunk size measured in word count rather than actual token count. No persistent JSON output. Worked examples had no special handling — they split at the word boundary like any other block.

**Week 10:** tiktoken `cl100k_base` (same tokeniser as `text-embedding-3-small`) measures every chunk before flushing. Cap set to 250 tokens — smaller units give sharper cosine similarity signal. Overlap set to 40 tokens. Three content types enforced: `prose` (accumulated with overlap), `worked_example` (emitted as a single atomic chunk regardless of token count so problem and solution never separate), `question_or_exercise` (accumulated separately from definitions). Output persisted to `wk10_chunks.json` with fields: `chunk_id`, `chapter`, `section`, `page`, `content_type`, `token_count`, `text`.

## BEFORE vs AFTER on 5 spot queries (BM25 top-1)

| Query | Wk9 top-1 | Wk9 result | Wk10 top-1 | Wk10 content_type |
|---|---|---|---|---|
| SI unit of acceleration | chunk_0012 | Correct — formula chunk | chunk_0014 | formula / prose |
| Newton first law | chunk_0041 | Correct | chunk_0041 | prose |
| Define inertia | chunk_0021 | **WRONG** — distance-time section | chunk_0039 | prose |
| Area under v-t graph | chunk_0025 | Correct | chunk_0022 | prose |
| Newton 2nd law → momentum | chunk_0051 | Partial — F=ma only | chunk_0049 | worked_example |

## Key observation

The inertia failure in Week 9 occurred because the inertia definition paragraph and the distance-time graph introduction were in the same large chunk. BM25 matched the query term "inertia" to a passage that used the word loosely. Smaller 250-token chunks isolate the inertia definition paragraph and the distance-time introduction into separate chunks. Dense retrieval on the Week 10 chunks surfaces chunk_0039 (inertia definition) above chunk_0021 (graph introduction) because the embedding for "define inertia" is directionally closer to a passage about resistance to motion change than to a passage about graph shapes.

Content-type filtering would further improve precision: a metadata filter `content_type != question_or_exercise` prevents exercise chunks (which repeat section vocabulary without definitions) from outranking definition chunks on definitional queries.
