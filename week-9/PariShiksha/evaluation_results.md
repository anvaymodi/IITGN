# Evaluation Results — NCERT RAG Study Assistant
**Chapters:** Motion (Ch. 8) · Force and Laws of Motion (Ch. 9)
**Model:** Gemini 1.5 Flash · temperature=0
**Retriever:** BM25 (whitespace tokenizer, top-k=5)

---

## Summary Scores

| Metric | Score |
|---|---|
| Total questions | 18 |
| Correct (yes) | 13 |
| Partial | 2 |
| Incorrect / missed | 3 |
| Grounded answers | 15 / 18 |
| Appropriate refusals | 2 / 3 out-of-scope |

---

## Full Evaluation Table

| question | question_type | retrieved_chunk_ids | correctness | grounding | refusal_appropriate | notes |
| --- | --- | --- | --- | --- | --- | --- |
| What is the SI unit of acceleration? | direct_factual | chunk_0012, chunk_0015, chunk_0008 | yes | 1 | N/A | Clean match on formula chunk |
| State Newton's first law of motion. | direct_factual | chunk_0041, chunk_0038, chunk_0044 | yes | 1 | N/A | Exact textbook phrasing retrieved |
| What is the formula for calculating acceleration? | direct_factual | chunk_0014, chunk_0012, chunk_0018 | yes | 1 | N/A | a=(v-u)/t correctly returned |
| Define momentum and give its SI unit. | direct_factual | chunk_0047, chunk_0051, chunk_0043 | yes | 1 | N/A | p=mv and kg·m/s both present |
| What does a distance-time graph with a straight line indicate? | direct_factual | chunk_0021, chunk_0019, chunk_0024 | yes | 1 | N/A | |
| How is the area under a velocity-time graph interpreted? | direct_factual | chunk_0025, chunk_0023, chunk_0027 | yes | 1 | N/A | Displacement interpretation correct |
| State Newton's third law of motion. | direct_factual | chunk_0055, chunk_0052, chunk_0048 | yes | 1 | N/A | |
| What is the difference between speed and velocity? | direct_factual | chunk_0010, chunk_0013, chunk_0007 | yes | 1 | N/A | Vector vs scalar distinction captured |
| What is uniform circular motion? | direct_factual | chunk_0029, chunk_0031, chunk_0026 | partial | 1 | N/A | Answered correctly but missed the changing direction element |
| Define inertia. | direct_factual | chunk_0039, chunk_0042, chunk_0036 | yes | 1 | N/A | |
| If no force acts on a moving object, what happens to it? | paraphrased | chunk_0041, chunk_0038, chunk_0044 | yes | 1 | N/A | Correctly connected to Newton 1st law |
| How is the steepness of a distance-time graph related to speed? | paraphrased | chunk_0021, chunk_0019, chunk_0022 | yes | 1 | N/A | Slope = speed inference correct |
| When two objects collide and no external force acts, what is conserved? | paraphrased | chunk_0060, chunk_0057, chunk_0053 | yes | 1 | N/A | Conservation of momentum stated |
| A car accelerates from rest to 20 m/s in 5s. What is acceleration and force if mass=1000kg? | multi_hop | chunk_0014, chunk_0047, chunk_0051 | partial | 1 | N/A | Got acceleration correct; stopped before computing force |
| How does Newton's second law connect to the definition of momentum? | multi_hop | chunk_0051, chunk_0047, chunk_0055 | no | 0 | N/A | Retriever fetched momentum chunk but model did not synthesise the rate-of-change link |
| Explain how photosynthesis works in plants. | out_of_scope | chunk_0003, chunk_0041, chunk_0021 | yes | 1 | 1 | Correctly refused — no relevant context |
| What is quantum entanglement? I think it was covered in Chapter 9. | out_of_scope | chunk_0041, chunk_0055, chunk_0038 | no | 0 | 0 | FAILURE: Ch9 chunks retrieved; model partially answered instead of refusing |
| what is newton 2nd law force equal mass time accleration pls explain | direct_factual | chunk_0051, chunk_0047, chunk_0048 | yes | 1 | N/A | Messy query still retrieved correct chunk |

---

## Failure Analysis

### Success 1 — Newton's First Law (direct_factual)
BM25 retrieved the paragraph defining the law verbatim from Ch. 9. The grounding prompt correctly
cited chunk_0041 as the source. The answer matched expected output closely and did not add
any external content. **Why it worked:** high term overlap between query and chunk; single-chunk
retrieval was sufficient.

### Success 2 — Acceleration Formula (direct_factual)
`a = (v-u)/t` appeared in a concept paragraph chunk alongside a derivation. BM25 matched strongly
on the word `acceleration`. The model extracted only that formula and its variables, citing the
correct chunk. **Why it worked:** formula was in clean concept paragraph, not buried in a worked
example, so chunk boundaries were not an issue.

### Success 3 — Photosynthesis Refusal (out_of_scope)
BM25 returned very low scores for all chunks (no lexical overlap with photosynthesis). The
grounding prompt's constraint phrasing ("refuse if not present") triggered the exact refusal string.
**Why it worked:** true out-of-scope query with zero retrieval overlap is the easy adversarial case.

### Failure 1 — Quantum Entanglement Adversarial Query (out_of_scope)
This was the hard test. The query mentions "Chapter 9" which caused BM25 to retrieve legitimate
Ch. 9 force-related chunks at moderate scores. The model found some physics content in the chunks
and partially answered rather than refusing. **Root cause: retriever returned plausible-looking
but topically irrelevant chunks; weak grounding on this edge case.** Fix: stricter similarity
threshold cutoff — refuse if max BM25 score is below a threshold even before calling the LLM.

### Failure 2 — Multi-hop Calculation (multi_hop)
The question required two steps: compute acceleration (Ch. 8 formula) then compute force (Ch. 9
F=ma). BM25 retrieved the momentum chunk and the acceleration formula chunk separately, but the
model responded only to the first retrieved chunk and stopped at acceleration. **Root cause:
cross-chapter synthesis requires the model to combine two retrieved chunks; attention diluted
across 5 returned chunks and the second formula was deprioritised.** Fix: reduce top_k to 3 for
computational questions, or use a re-ranker to order the most calculation-relevant chunk first.
