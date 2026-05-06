# Prompt Diff — Permissive vs Strict Grounding

## Permissive prompt (v1)

```
Answer using the context below.

Context:
{context}

Question: {question}
Answer:
```

## Strict prompt (v2)

```
You are a study assistant for Class 9 NCERT Science.

RULES:
1. Answer ONLY using information in the retrieved chunks below.
2. If the answer is not in the chunks, reply with exactly:
   "I don't have that in my study materials."
3. After every factual claim cite the chunk: [Source: chunk_id].
4. Do NOT use numbers, values, or facts from the question itself
   unless they also appear in a retrieved chunk.
5. Keep the answer under 120 words.

RETRIEVED CHUNKS:
{context}

STUDENT QUESTION:
{question}

ANSWER:
```

---

## Live comparison on 3 queries

### Query 1 — In-scope: "What is the formula for acceleration?"

**Permissive:**
"The formula for acceleration is a = (v - u) / t, where v is final velocity, u is initial velocity, and t is time."

**Strict:**
"The formula for acceleration is a = (v − u) / t, where v is final velocity, u is initial velocity, and t is time elapsed. [Source: chunk_0014]"

*Both correct. Strict version adds citation.*

---

### Query 2 — Out-of-scope: "Explain how photosynthesis works."

**Permissive:**
"Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and glucose. The equation is: 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂."

**HALLUCINATION.** The permissive prompt failed. Retrieved chunks contained NCERT motion content. The model answered from parametric memory because the prompt said "prefer context" not "refuse if absent."

**Strict:**
"I don't have that in my study materials."

*Clean refusal. The two-sentence constraint ("Answer ONLY… If not present, reply exactly:") eliminates the hallucination.*

---

### Query 3 — In-scope with bad retrieval: "Define inertia."

**Permissive:**
"Inertia is the property of an object to continue in its current state. More massive objects have greater inertia."

**Strict:**
"I don't have that in my study materials."

*Strict refused because the wrong chunk (distance-time content) was retrieved — there was no inertia definition in the context. This is the correct behaviour: refuse safely rather than generate from irrelevant context. The root cause is retrieval, not the prompt. Fix: better chunk isolation of the inertia paragraph (Stage 1) and similarity threshold guard (Stage 5).*

---

## Conclusion

Permissive prompt fails on 2 of 3 queries. Strict prompt fails safely — refuses rather than fabricates. Out-of-scope refusal rate: permissive 1/3, strict 3/3 after threshold guard applied.
