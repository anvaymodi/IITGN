# Retrieval Misses — Stage 2

10 queries run on Chroma (text-embedding-3-small, cosine, k=5). Top-1 manually assessed.

**Top-1 correct: 8 / 10. Top-1 wrong: 2 / 10.**

---

## Miss 1 — "Define inertia"

**Wrong chunk retrieved:** chunk_0021
**Similarity score:** 0.61
**Chunk preview:** "8.1 INTRODUCTION An object is said to be in motion when it moves from one place to another. A distance-time graph for uniform motion is a straight line..."

**Why it ranked top-1:** The embedding model places "inertia" near general classical-mechanics vocabulary. chunk_0021 opens the chapter introduction and uses "motion" four times. The correct chunk (chunk_0039, inertia definition) uses "resist change" and "tendency" — phrases that are semantically close to "define inertia" but not as close as the chapter introduction in the generic model's embedding space.

**Diagnosis:** Embedding limitation. The generic `text-embedding-3-small` model has not seen enough NCERT-specific training signal to distinguish "inertia definition" from "motion introduction." Domain-fine-tuned embeddings would fix this (§9 Industry Pointer). Shorter, definition-only chunks also help — chunk_0039 at 250 tokens contains only the inertia paragraph, so its embedding is less diluted.

---

## Miss 2 — "How does Newton second law connect to momentum?"

**Wrong chunk retrieved:** chunk_0051
**Similarity score:** 0.58
**Chunk preview:** "Newton's second law states F = ma. The force F is measured in Newtons, m in kilograms, a in m/s²..."

**Why it ranked top-1:** "Newton's second law" is a strong anchor. chunk_0051 contains that phrase prominently. But the query asks for the *connection to momentum*, which requires the derivation F = m(v−u)/t = Δp/Δt — a prose or worked_example chunk that contains both the formula and the momentum definition together.

**Diagnosis:** Multi-hop failure. No single chunk combines both F=ma and the momentum rate-of-change derivation. BM25 has the same failure. Fix options: (a) query decomposition into two sub-queries, or (b) a worked_example chunk that preserves the full derivation passage intact.

---

## Near-miss — "State Newton's third law"

**Retrieved:** chunk_0055 (action-reaction statement)
**Similarity:** 0.71 — correct chunk but lower than expected.

Several worked_example chunks mentioning "action-reaction" in exercise contexts scored nearly as high (0.68). Structure-aware filtering — prefer `prose` content_type for definitional queries — would improve precision here and is a low-effort Wk11 addition.
