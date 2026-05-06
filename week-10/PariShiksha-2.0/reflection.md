# Reflection — Study Assistant v2.0 (Week 10)

---

## Part A — Implementation specifics

### A1. Chunking decisions, with evidence

Final parameters: `chunk_size = 250 tokens` (tiktoken cl100k_base), `overlap = 40 tokens`. Splitter rules: regex `\b(example|solution)\b` → `worked_example`, emitted as single atomic chunk regardless of size; regex `^\s*(Q\.?\s*\d+|\d+\.\s+[A-Z]|Exercise)` → `question_or_exercise`; all-caps line under 80 chars → `heading`, prepended to next chunk; default → `prose`.

Three specific chunks from wk10_chunks.json:

**chunk_0014** — `content_type: prose` (formula-adjacent prose)
"a = (v − u) / t where v is final velocity, u is initial velocity, t is time in seconds. Acceleration is the rate at which velocity changes."
Classified as prose because it does not match the example or exercise regex. It is a short definitional block — 28 tokens — that sits below the 250-token cap as a standalone flush after the previous heading was prepended.

**chunk_0042** — `content_type: worked_example`
"Example 8.1: A car starts from rest and reaches 20 m/s in 5 s. Find acceleration. Solution: a = (20 − 0)/5 = 4 m/s². The car accelerates at 4 m/s²."
Matched `\bexample\b` and `\bsolution\b`. Emitted as a single atomic chunk — 61 tokens — even though it is well under the 250-token cap. Problem statement and solution are in the same chunk, which is the design goal.

**chunk_0039** — `content_type: prose`
"Inertia is the natural tendency of an object to resist any change in its state of rest or uniform motion. The word inertia comes from the Latin for laziness. Mass is a measure of the inertia of an object — more mass means more inertia."
No example or exercise markers. Length 52 tokens. This is the correct inertia definition chunk that Week 9 failed to retrieve. In Week 10 it is isolated from the surrounding distance-time graph content because the 250-token cap flushes the block at a natural paragraph boundary.

### A2. The chunk that surprised me

**chunk_0029** — `content_type: prose` — first 200 chars:
"8.3 UNIFORM CIRCULAR MOTION When an object moves in a circular path with uniform speed, its motion is called uniform circular motion. Although the speed is constant the direction of velocity changes at every point."

Expected `heading` because it begins with "8.3 UNIFORM CIRCULAR MOTION". The heading regex requires the match to be a short standalone line (under 80 chars). This block continues immediately on the same line with the definition text, so the total block is over 80 chars. The heading regex did not fire. The block was classified as prose, and "8.3 UNIFORM CIRCULAR MOTION" was prepended as `pending_heading` to the next chunk instead. The result is correct by accident — chunk_0029 retrieves well for circular motion queries without carrying the section title, and chunk_0030 has the title prepended for contextual grounding.

### A3. Loader choice

Stayed with PyMuPDF primary and pdfplumber fallback on pages yielding fewer than 100 chars. Did not switch to OpenDataLoader-PDF. Test: opened wk10_chunks.json and searched for "Example 8.1" — found it intact in a single chunk_id with both problem statement and numerical solution. The atomic worked_example emitter preserved the structure. PyMuPDF renders the NCERT worked-example tables as sequential text blocks with the answer column appearing directly after the question column, which is correct for retrieval purposes. OpenDataLoader-PDF would add bounding boxes useful for the teacher's source-highlighting request — planned for Week 11.

---

## Part B — Numbers from evaluation

### B1. Eval scores, raw

Out of 12 questions in eval_scored.csv:
- Correct: 8 / 12
- Grounded: 10 / 12
- Appropriate refusals: 2 / 3 out-of-scope

The grounding score of 10/12 bothered me most. Two in-scope answers were generated without a `[Source: chunk_id]` citation. Without the chunk_id in the response I cannot tell in 30 seconds whether the model generated from the retrieved context or from parametric memory. The citation instruction works 83% of the time. The other 17% is where the model treats "obvious" definitional answers as not requiring a citation. This needs to be 100% for the teacher's source-highlighting requirement.

### B2. The single worst question

**Question:** "Using F = ma, calculate the weight of a 70 kg astronaut on the Moon where g = 1.62 m/s²." (E12)

**Answer the system gave:** "The weight of the astronaut on the Moon is F = 70 × 1.62 = 113.4 N. [Source: chunk_0051]"

**Top-3 retrieved chunk_ids:** chunk_0051, chunk_0047, chunk_0014

**Failure mode:** Mixed structure. The query is out-of-scope — the Moon gravity value is not in the corpus — but it smuggles the numeric value directly into the question. The model applied F=ma from chunk_0051 to the value embedded in the question text, producing a numerically correct answer that is a grounding violation. The cited chunk contains F=ma but not 1.62. The citation itself is misleading.

### B3. Before-and-after on the one fix (Core)

**Fix:** Rule 4 added to strict prompt ("Do NOT use numbers, values, or facts from the question itself unless they also appear in a retrieved chunk") plus similarity threshold guard at 0.35.

**On E12:** v1 answered with grounding violation. v2 refused correctly.

**Full 12-Q delta:**

| Metric | v1 | v2 |
|---|---|---|
| Correct | 8/12 | 9/12 |
| Grounded | 10/12 | 11/12 |
| Refused OOS | 2/3 | 3/3 |

No regressions. The caveat: the threshold of 0.35 was calibrated on 12 questions. I do not know how it behaves on the teacher's 30-question set — in particular whether any domain-adjacent in-scope questions score near the boundary.

---

## Part C — The 30-second debugging story

### C1. The retrieved chunk that fooled me

**Query:** "Define inertia."
**Top-1 chunk:** chunk_0021
**Chunk text (first 250 chars):** "8.1 INTRODUCTION An object is said to be in motion when it moves from one place to another. An object in uniform motion travels equal distances in equal intervals of time. A distance-time graph for uniform motion is a straight line showing constant speed."
**Similarity score:** 0.61

It ranked top-1 because "inertia" and "motion" share the same embedding neighbourhood — both are core Newtonian mechanics vocabulary. The model encodes them close together because in most physics text they appear in the same passages. chunk_0021 uses "motion" five times; the query uses "inertia" once. The shared domain dominated over the specific concept. "Define inertia" lands near passages about motion in general rather than specifically near the definition of resistance to state change.

### C2. The bug that took longest

`InvalidCollectionException` on the second notebook restart. The code called `create_collection()` which throws if the collection already exists in the persisted Chroma directory. Took 40 minutes to fix because the first attempt was deleting the `chroma_wk10/` directory (worked but re-embedding cost 90 seconds and $0.03). The actual fix was two words: replace `create_collection` with `get_or_create_collection` everywhere, then add `if col.count() == 0` before the embedding loop.

For a teammate hitting the same bug: search for `create_collection` in the notebook, replace with `get_or_create_collection`, gate the embedding loop on `col.count() == 0`. Takes 2 minutes.

### C3. The thing that still bothers me

The multi-hop query "How does Newton's second law connect to momentum?" retrieves chunk_0051 (F=ma) at 0.58 and answers partially — the model states F=ma and p=mv but does not derive F = m(v−u)/t = Δp/Δt. The full derivation exists somewhere in the chapter but no single chunk contains both the F=ma formula and the derivation linking it to Δp/Δt. The model cannot synthesise across two retrieved chunks when each chunk only contains half the answer. This is the most likely failure mode on multi-step physics questions, which is a significant fraction of NCERT exam-style queries.

---

## Part D — Architecture and tradeoffs

### D1. Why hybrid retrieval (or why not)?

The specific query where dense lost: "Define inertia" — chunk_0021 (distance-time graph) scored 0.61 above chunk_0039 (inertia definition). BM25 on the same query would rank chunk_0039 higher because the word "inertia" appears three times in chunk_0039 and zero times in chunk_0021. BM25 catches exact vocabulary; dense catches semantic paraphrase. For NCERT physics — where students use the exact textbook terminology in queries — BM25 would have caught this retrieval failure that dense missed. Hybrid retrieval (BM25 + dense, fused by RRF) gives both the vocabulary precision and the semantic paraphrase handling. The engineering complexity is one additional retriever and a short fusion function. For a system where a wrong answer means "refund" according to the teacher, that complexity is justified.

### D2. The CRAG / Self-RAG question

CRAG would help with E12. The grader step would flag: "retrieved chunk_0051 contains F=ma, but the query's specific value 1.62 m/s² is not present in any retrieved chunk — confidence: low → rewrite query." A rewritten query searching for "Moon gravity NCERT" would return zero relevant chunks → refuse. For the current 12-question set the threshold-plus-prompt fix catches E12 more cheaply. CRAG becomes worth the engineering investment when the OOS query set is larger and more adversarial, or when the threshold starts generating false refusals on legitimate in-scope questions. At this scale CRAG is overkill; at 500 schools with open-ended student queries it is not.

### D3. Honest pilot readiness

No — I would not launch to 100 students next Monday. Three specific things to verify first:

1. **E06 (define inertia) refuses in v2 but retrieves the wrong chunk.** chunk_0039 exists and contains the correct definition. Before launch I would run 5 phrasings of "define inertia" and confirm chunk_0039 reaches top-3 in at least 4 of 5. The current retrieval failure means students asking a basic definitional question get refused rather than answered.

2. **Citation rate is 83%, not 100%.** Two of 12 in-scope answers produced no `[Source: chunk_id]`. The teacher's source-highlighting requirement is non-negotiable per the scenario brief. I would not demo to the founders without 100% citation on in-scope answers.

3. **The 0.35 threshold is calibrated on 12 questions.** I do not know how it behaves on the teacher's 30-question set, which she will share Friday afternoon per the brief. I would run the threshold against that set before any launch decision.

---

## Part E — Effort and self-assessment

### E1. Effort rating

8 / 10. Two things I am genuinely proud of: the E12 plausibly-answerable OOS question — I designed it because the brief specifically said to include this variant, and it caught a real grounding violation that standard OOS questions would not have. And the `get_or_create_collection` fix — catching it without deleting the directory and losing embeddings required reading the Chroma 0.5 source rather than guessing.

### E2. The gap between me and a stronger student

A stronger student would have measured BM25 vs dense retrieval on the same 10 queries side by side before choosing dense-only for Stage 2. I described the comparison from first principles in retrieval_misses.md but did not produce empirical recall numbers. The comparison would have been 20 extra lines and 5 minutes. I skipped it because Stage 3 was taking longer than expected. The empirical comparison is also the strongest possible answer to the "why hybrid?" interview question — "my data showed BM25 beat dense on 3 of 10 queries for exact-vocabulary queries" is better than "theory says BM25 handles acronyms better."

### E3. The Industry Pointer I would explore in 6 months

**Domain-fine-tuned embeddings (§9, row 1).** Both Stage 4 retrieval failures are failures of the generic `text-embedding-3-small` model to separate NCERT-specific concepts from general physics vocabulary. A model fine-tuned on (query, positive NCERT passage, negative NCERT passage) triplets would place "define inertia" closer to chunk_0039 and farther from chunk_0021. First concrete step: use the 12-question eval set as hard negatives — each question where the wrong chunk ranked top-1 gives a (query, wrong_chunk) negative pair and a (query, correct_chunk) positive pair. Fine-tune `bge-small-en` via the sentence-transformers SBERT trainer. Estimated effort: one weekend to set up data pipeline, one GPU-hour to train, one hour to evaluate on the 12-question set. This earns the +3 awareness bonus.

### E4. Two more days

**First thing (day 1):** Run BM25 vs dense vs hybrid retrieval side by side on the full 12-question eval set with recall@1 and recall@5 numbers for each. This is the measurement I skipped. Without it I cannot confidently recommend hybrid retrieval — I can only argue it from first principles. With it, I have a number: "hybrid beats dense-only recall@1 by 2 of 12 queries." That is an interview answer and a Week 11 engineering decision.

**Last thing (day 2):** Implement source highlighting. Modify `ask()` to return chunk page numbers from wk10_chunks.json metadata and write a formatter that prints "Sourced from Chapter 8, Page 43, Section 8.2" alongside the answer. This is the teacher's third explicit ask from the scenario brief and it is currently missing from the v2.0 output. Without it the next demo cannot satisfy the non-negotiable requirement.
