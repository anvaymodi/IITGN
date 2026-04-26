# reflection.md — Week 9 Mini-Project
## Retrieval-Ready Study Assistant for NCERT Science

---

## Part A — Your Implementation Artifacts

### A1. Your chunking parameters

**Parameters settled on:**
- Chunk size: 400 tokens (whitespace split)
- Overlap: 80 tokens
- Special rules: worked_example blocks are never split (kept whole, up to ~600 tokens); heading blocks are prepended to the following paragraph so the chunk carries its section label; formula blocks are emitted as standalone chunks.

I started at 500 tokens. When I printed the first few chunks after running the chunking cell, I noticed that Example 8.1 from Chapter 8 — the car displacement problem — was getting split mid-solution. Chunk N had the problem statement ("A car travels 30 km north…") and chunk N+1 had the solution steps. When I queried "how do I find displacement when directions are at right angles," the retriever returned chunk N with a score of 1.7 but chunk N+1 ranked fourth. The model then produced a confident but incomplete answer that stated the setup without showing the Pythagoras calculation.

That observation was the push. Dropping to 400 tokens kept the example and solution together in a single chunk. The overlap at 80 tokens was chosen to be roughly one sentence worth of context — enough to preserve continuity between adjacent concept paragraphs without duplicating entire examples across chunks.

---

### A2. A retrieved chunk that was wrong for its query

**Query:** "How is the area under a velocity-time graph interpreted?"

**Retrieved chunk (wrong) — chunk_0021, score 1.43:**
"8.1 INTRODUCTION In our daily life, we observe many objects in motion. A bird flying in the air, a fish swimming in water, a car moving on the road and even the movement of the limbs while..."

**Why the retriever returned it:** The query contains "motion" and "time" which are extremely common across every paragraph in Chapter 8. BM25 matched this introduction block on those high-frequency terms even though the block is completely unrelated to velocity-time graphs. The specific phrase "velocity-time graph" existed in a different chunk that ranked second — which is where the useful content was. This is a classic BM25 failure on queries where the key noun ("velocity-time graph") is rare in the corpus but the surrounding words ("area," "time") are common everywhere.

---

### A3. Your grounding prompt, v1 and v(final)

**V1 (first attempt):**
```
Answer the following question using only the provided context.
Context: {context}
Question: {question}
Answer:
```

**What broke with V1:** I tested it on the photosynthesis query and the model returned a two-sentence answer about how plants use energy from sunlight, drawing on its own training knowledge rather than the chunks. The phrase "using only the provided context" is permissive — the model treats it as a preference rather than a hard rule. It interprets "only" to mean "prefer this context" and still reaches into parametric memory when the context feels thin. The failure was clear when I printed the retrieved chunks alongside the answer: none of the chunks mentioned plants, sunlight, or energy in any relevant sense, yet the answer was confident and factually correct (which made it worse, not better, for the grounding requirement).

**V_final (submitted):**
```
You are a study assistant for Class 9 NCERT Science (PariShiksha). You have been given
a set of text chunks retrieved from the NCERT textbook. Your task is to answer the
student's question STRICTLY and ONLY using information present in the provided chunks.

RULES (follow every rule — none are optional):
1. Read all provided chunks carefully before answering.
2. If the answer to the question is NOT present in the chunks, you MUST respond with
   exactly this string and nothing else:
   "I cannot answer this from the provided textbook content."
3. Do NOT use any knowledge outside the provided chunks. If chunks are partially
   relevant but do not fully answer the question, still refuse.
4. At the end of your answer, list the chunk IDs you used, in this format:
   Sources: [chunk_id1, chunk_id2]
5. Keep your answer concise and suitable for a Class 9 student.

RETRIEVED CHUNKS:
{context}

STUDENT QUESTION:
{question}

ANSWER:
```

**What changed and why:** Two changes mattered. First, framing Rule 2 as a MUST with the exact refusal string — the model can't paraphrase or hedge, it has to produce that specific string, which makes automated scoring reliable. Second, Rule 3 explicitly addresses the partial-relevance case ("if chunks are partially relevant but don't fully answer, still refuse") — this is the exact failure mode in the quantum entanglement adversarial query, where the retriever returns Chapter 9 force chunks at moderate scores and a weak prompt would synthesise something from them.

---

## Part B — Numbers from Your Evaluation

### B1. Your evaluation scores

Out of 18 questions:
- Correct: 13 (72%)
- Partial: 2 (11%)
- Incorrect: 3 (17%)
- Grounded: 15 / 18
- Appropriate refusals: 2 / 3 out-of-scope questions correctly refused

The number that bothered me most was the grounding score — 15 out of 18. Three answers were either ungrounded or triggered the wrong path. Two of those three were predictable (the multi-hop calculation and the adversarial entanglement query), but the third was a direct factual question about uniform circular motion where the model added a sentence about centripetal force that didn't appear in any retrieved chunk. That one stung because it was an in-scope question with a good retrieval score, and the model still reached outside the context for one sentence. It's the kind of quiet hallucination that would go unnoticed without careful grounding checks.

---

### B2. Chunk-size experiment (Stretch)

I did not run a formal controlled chunk-size comparison with two scored eval sets. The decision to use 400 tokens came from the observation described in A1 — seeing a worked example split mid-solution during the sanity check — rather than from a scored delta. If I had two more days, this would be the first addition: run the full 18-question eval at chunk_size=250 and chunk_size=500, record correctness and refusal scores, and report the delta. That would turn an observation into a measurement.

---

### B3. Model family comparison (Stretch)

I did not compare model families as a formal Stretch task. The tokenizer comparison in Section A3 of the notebook covers the architectural differences between BPE, WordPiece, and SentencePiece at the tokenization level, but a full side-by-side on the same eval set — extractive QA (RoBERTa-SQuAD) vs encoder-decoder (flan-t5-small) vs Gemini — was not done in this submission.

---

## Part C — Debugging Moments

### C1. The most frustrating bug

The most frustrating issue was the T5 tokenizer download stalling silently the first time. I ran the tokenizer loading cell and it appeared to hang — no progress bar, no error, just nothing for about four minutes. I killed the kernel and tried again, which made it worse because the partially downloaded cache left a corrupted file. When I re-ran, I got a `OSError: model not found` error that had nothing to do with the actual problem.

It took about 45 minutes to fix. I first tried clearing the cell output and re-running, then tried specifying the full model path explicitly (`google/t5-small`), neither of which helped. The actual fix was deleting the Hugging Face cache directory (`~/.cache/huggingface/hub/`) and running the cell again with a stable internet connection. Once the download ran cleanly from scratch, the tokenizer loaded in under a minute.

If someone hits this next week: before assuming the code is wrong, check whether the model cache is corrupted — delete `~/.cache/huggingface/hub/models--t5-small` and re-run.

---

### C2. What still bothers me

The messy student-style query — "what is newton 2nd law force equal mass time accleration pls explain" — retrieved the right chunks and got a correct answer, which was encouraging. What still bothers me is that it worked almost by accident. BM25 matched on "newton," "mass," and "accleration" (misspelled) because the misspelling happens to partially match "acceleration" through a shared prefix in the whitespace split. If the student had typed "nyoton ka doosra niyam" in Devanagari, the system would have returned zero relevant chunks and refused — which is technically correct behaviour but the wrong outcome for PariShiksha's actual students. I don't have a fix for this in the current build. Proper multilingual retrieval requires either a multilingual embedding model or transliteration preprocessing, neither of which I had time for.

---

## Part D — Architecture and Reasoning

### D1. Why not just ChatGPT?

A hiring manager asking "why not just use ChatGPT?" is really asking what a retrieval system buys that a parametric model can't provide on its own.

The most direct answer from this project: when I tested the system without retrieval — just sending the question directly to Gemini with no context — it answered the photosynthesis question correctly and confidently. That's the problem. PariShiksha's contract with parents is that the assistant stays within NCERT content. A parametric model's knowledge has no boundaries and produces no citations. A student who asks about photosynthesis at 10pm should be told the system can't help with that, not given a correct biology answer that has nothing to do with the chapter her exam is on tomorrow.

The second reason is traceable failure. When the quantum entanglement adversarial query failed in my evaluation — the system answered partially instead of refusing — I could trace it to the specific retrieved chunks and the specific gap in the prompt's Rule 3. With a raw API call, the failure would be invisible: the answer would just be wrong, with no way to tell whether the problem was retrieval, prompting, or model knowledge. In a pilot with thin margins and no ML team on call, the ability to isolate failures matters as much as average accuracy.

The third reason is knowledge versioning. NCERT revises chapters. When they do, PariShiksha updates the corpus, re-indexes, and re-evaluates. That's a defined process. Keeping a parametric model current with curriculum changes has no clean equivalent.

---

### D2. The GANs reflection

GANs are the wrong architecture for this problem, and the reason teaches something general about matching generation mechanisms to tasks.

GANs work by training a generator to produce outputs that a discriminator cannot distinguish from real examples. The learning signal is "does this look real?" — not "is this grounded in a specific document?" or "is this factually accurate?" A GAN trained on NCERT text would learn to produce fluent, plausible-looking NCERT-style sentences. It would be very good at sounding like the textbook. It would have no mechanism at all for staying within a specific retrieved chunk, because that concept doesn't exist in the GAN training loop.

The deeper principle is that architectures encode different failure modes as acceptable trade-offs. GANs accept factual incorrectness in exchange for distributional realism — they optimise for the distribution of real examples, not for any individual fact. For a bounded textbook assistant where a single confidently wrong answer can end the pilot, this trade-off is backwards. The intolerable failure for PariShiksha is confident hallucination. GANs are specifically designed to produce confident-sounding outputs regardless of factual grounding.

RAG with a constraint-framing grounding prompt handles this differently: either a chunk supports the answer or the system refuses. The failure mode is under-recall (refusing answerable questions), which is recoverable. GAN-style hallucination is not — a parent who gets a confident wrong answer about Newton's second law doesn't know it's wrong.

---

### D3. Honest pilot readiness

My honest answer to "can we launch next Monday with 100 students?" is: no, but the core retrieval and grounding logic is ready for a smaller internal test.

Three specific things I'd want to verify or fix before any real student touches this:

1. **The adversarial refusal gap.** The quantum entanglement query — out-of-scope but with a topic name that overlaps with Chapter 9 — failed. The system partially answered instead of refusing. The fix is a BM25 score threshold: if the highest score across all retrieved chunks is below a set cutoff, return the refusal string without calling the API at all. I know how to implement this and it would take under an hour, but it needs to be tested on at least five more adversarial queries before I'd trust it.

2. **Hindi-language and Devanagari input.** One messy English query with typos worked. A query in Hindi script would return all-zero BM25 scores. PariShiksha's students in Rajasthan and MP regularly ask in their first language. This is not a minor edge case — it's probably the majority use pattern in Tier-2 centres.

3. **Rate limits under concurrent load.** Every question makes an API call. Free tier Gemini has rate limits in the dozens of requests per minute. One hundred students asking questions simultaneously would hit this immediately. The backend engineer needs to add a request queue and response caching layer before any live deployment.

---

## Part E — Effort and Self-Assessment

### E1. Effort rating

I'd rate my effort at 7 out of 10. Most of my time went into two places: the chunking logic and the grounding prompt iteration, which the project brief's expert hints correctly identified as the highest-leverage areas. I'm genuinely proud of the worked_example boundary handling — I spent time going through the actual NCERT PDF pages to find where example blocks started and ended before writing the classifier, rather than guessing at patterns. The sanity check output confirmed it worked: none of the three worked examples I spot-checked got split across chunk boundaries.

What cost me points on effort was not running the Stretch chunk-size experiment as a formal scored comparison. I made the chunking decision through observation, which is defensible, but a scored delta would have been a stronger justification and better preparation for the interview-style questions in Section 9.

---

### E2. The gap between you and a stronger student

A stronger student would have implemented the BM25 score threshold for the refusal decision — the fix I described in D3 — before submitting rather than after. It's a twenty-line addition to the `answer()` function and directly addresses the one clear failure in the eval set. I identified the fix during the failure analysis but didn't have time to implement and re-run the full evaluation. A stronger student would have planned evaluation time into the schedule from the start, rather than running the eval the same evening as the final notebook cleanup.

There's also the Hindi query gap in the eval set. A stronger student working in an IITGN cohort would know someone from Rajasthan or MP to ask for five informal student-style queries in Hindi, which the brief explicitly suggests. I wrote all 18 eval questions myself, which means they're phrased more like a textbook than a real student would phrase them — the brief warned exactly about this.

---

### E3. What would change with two more days

**First thing:** Add the BM25 score threshold to the `answer()` function and re-run the full 18-question evaluation. This is the highest-leverage change — it directly fixes the one eval failure I can trace to a specific cause (adversarial retrieval), and the implementation is small enough that I'm confident it won't introduce new failures. The new eval numbers would also give me real data for B2 (a genuine chunk-size or threshold experiment) instead of the observation-based reasoning I currently have.

**Last thing:** Run the Stretch model family comparison — extractive QA via RoBERTa-SQuAD alongside flan-t5-small and Gemini on the same eval set. I'd want this last because it's the most time-consuming (model downloads, inference setup, results collation) and the least likely to change any production decision. Gemini is the right choice for this use case regardless of what the comparison shows. The comparison has value for the course rubric and for understanding the architectural trade-offs, but it doesn't change the system.
