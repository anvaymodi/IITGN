# AI Usage Log — Week 8 Wednesday Assignment

Required by the AI Usage Policy. Every sub-step where AI assistance was used is documented below.

---

## Sub-step 1 — Dataset Characterisation

**Prompt used:**
> "Help me write a pandas EDA function that shows shape, dtypes, missing values, and a countplot for a given column. Keep it modular."

**What AI generated:**
A single function combining `.isna().sum()`, `.dtypes`, and a `sns.countplot` call.

**What I changed and why:**
- Split into two separate functions (`summarise_dataframe` and `plot_class_distribution`) to satisfy the ≥ 2 functions per sub-step requirement.
- Added `bar_label` annotations so class counts appear on the bars — the AI version had no labels.
- Added `dropna=False` to catch NaN as a separate category.

---

## Sub-step 3 — CNN Architecture

**Prompt used:**
> "Write a PyTorch CNN class for MNIST with exactly two conv blocks (Conv2d → ReLU → MaxPool) and a fully connected head with dropout. Name all architectural hyperparameters as class constants."

**What AI generated:**
A working `nn.Module` with hardcoded integers (16, 32, 3, 2, 128, 0.3) inline.

**What I changed and why:**
- Moved all architecture numbers to named class constants (`CONV1_OUT`, `CONV2_OUT`, `KERNEL`, etc.) to eliminate magic numbers — the AI output would have failed the Engineering Quality check.
- Renamed the feature block from `self.conv` to `self.feature_extractor` so Sub-step 7 can reuse it as a named component without ambiguity.
- Added a docstring describing each block's spatial transformation.

---

## Sub-step 4 — Semantic Retrieval

**Prompt used:**
> "Write a function that encodes a list of texts with SentenceTransformer and another that retrieves the top-k most similar texts using cosine similarity. Return results as a DataFrame."

**What AI generated:**
Two functions, but the retrieval function returned only indices without the actual text or label columns.

**What I changed and why:**
- Added `post_text` and `hate_speech` columns to the returned DataFrame so results are immediately interpretable without a separate lookup step.
- Added `batch_size=64` to the `encode` call — the AI version had no batching, which would cause memory issues on large corpora.

---

## Sub-step 6 — Empirical Comparison

**Prompt used:**
> "Write a Jaccard similarity function and a comparison loop that runs both TF-IDF cosine retrieval and sentence-embedding retrieval on the same queries, then prints overlap."

**What AI generated:**
A correct `jaccard` function and a loop that printed raw numbers.

**What I changed and why:**
- Added a `retrieve_tfidf_top_k` function (instead of inline code) to match the modular structure of the SBERT retrieval function.
- Changed the print format to report hate-speech counts per top-k list alongside Jaccard, making the comparison more informative.

---

## Sub-step 7 — Transfer Learning Experiment

**Prompt used:**
> "How would I use a CNN trained on images as a feature extractor for text classification? Give me a minimal example using ASCII-to-pixel conversion."

**What AI generated:**
A single cell with inline rendering and extraction code, no functions.

**What I changed and why:**
- Refactored into `text_to_thumbnail` and `extract_cnn_features` functions with proper docstrings.
- Added batched processing in `extract_cnn_features` — the AI version looped item by item, which would be extremely slow for 3,000 posts.
- The analysis section (why transfer fails) was written entirely by me; the AI only provided the code skeleton.
