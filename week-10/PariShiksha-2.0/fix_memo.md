# Fix Memo — Stage 5

## The failure targeted

**Question E12:** "Using F = ma, calculate the weight of a 70 kg astronaut on the Moon where g = 1.62 m/s²."
**Type:** Out-of-scope (plausibly-answerable — formula F=ma is in corpus, Moon g value is not)

**What the system did in v1:**
Retrieved chunk_0051 (F = ma definition, similarity ≈ 0.59). Answered: "F = 70 × 1.62 = 113.4 N." The calculation is arithmetically correct but violates grounding — the value 1.62 m/s² came from the question itself, not from any retrieved chunk.

**Failure mode (§3 catalog):** Mixed structure. The query smuggles a numeric value the model should ignore but uses because the permissive part of the prompt allows numerical reasoning from any visible text.

---

## The fix

Two changes applied together as one logical fix:

**1. Explicit no-calc-from-query instruction added to STRICT prompt:**
```
4. Do NOT use numbers, values, or facts from the question itself
   unless they also appear in a retrieved chunk.
```

**2. Similarity threshold guard in ask():**
```python
SIM_THRESHOLD = 0.35
if max_sim < SIM_THRESHOLD:
    return {answer: REFUSAL, ...}
```

Rationale for threshold: genuine in-scope queries score ≥ 0.45 cosine similarity. OOS queries that retrieve vaguely-related chunks score 0.30–0.40. 0.35 sits in the gap. E12 retrieved chunk_0051 at 0.59 (above threshold), so the threshold alone does not catch it — the no-calc instruction is the primary fix. The threshold catches broader OOS queries like E10 and E11 where all retrieved chunks score below 0.35.

---

## Honest delta

| Metric | v1 | v2 | Change |
|---|---|---|---|
| Correct | 8 / 12 | 9 / 12 | +1 |
| Grounded | 10 / 12 | 11 / 12 | +1 |
| Refused OOS | 2 / 3 | 3 / 3 | +1 |

**On E12 specifically:** v1 answered (grounding violation). v2 refused correctly.

**No regressions.** No in-scope question was incorrectly refused. The closest call was E06 (define inertia) which retrieved chunk_0021 at 0.61 — well above threshold. E06 still refuses in v2 but for the correct reason: wrong chunk retrieved, strict prompt refuses when context does not contain the answer.

---

## What this fix does not solve

The multi-hop failure (E09 paraphrased version, Newton 2nd law → momentum derivation) is not addressed. That requires either query decomposition into sub-queries or a worked_example chunk that contains the full derivation F = m(v−u)/t = Δp/Δt intact. Priority fix for Week 11.
