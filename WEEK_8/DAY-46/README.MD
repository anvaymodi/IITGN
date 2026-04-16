# Week 08 — Wednesday Assignment
## CNNs + Embeddings: Content Moderation Pipeline

**Course:** PG Diploma in AI-ML and Agentic AI Engineering — IIT Gandhinagar  
**Topic:** CNNs (Convolution, Pooling, Transfer Learning) + Embeddings (Word2Vec → BERT → Sentence-BERT, Semantic Search)

---

## What This Notebook Does

| Sub-step | Description |
|---|---|
| 1 🟢 | Load & characterise `social_media_posts.csv` — class imbalance analysis |
| 2 🟢 | Load & characterise MNIST via torchvision |
| 3 🟡 | Train a 2-block CNN on MNIST; visualise first-layer filters |
| 4 🟡 | Hate-speech classifier (TF-IDF + LR) + semantic retrieval (Sentence-BERT) |
| 5 🟡 | Two-stage moderation pipeline + recommendation at 100,000 posts/day |
| 6 🔴 | TF-IDF vs embedding similarity — empirical Jaccard comparison |
| 7 🔴 | CNN transfer learning experiment: MNIST → social media (why it fails) |

---

## How to Run

### 1. Clone / navigate to this folder
```bash
git clone <your-repo-url>
cd week-08/wednesday
```

### 2. Python version
```
Python 3.10+
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the dataset
Put `social_media_posts.csv` in the same folder as the notebook.  
If the file is absent, the notebook auto-generates a 3,000-row synthetic stand-in with the same schema so all cells still execute.

### 5. Launch the notebook
```bash
jupyter notebook W8_Wednesday_Band4.ipynb
```
Or open in VS Code / JupyterLab.  
Run **Kernel → Restart & Run All** from a clean environment.

---

## Expected Runtime
| Section | Approx. time (CPU) |
|---|---|
| Sub-steps 1–2 (EDA) | < 1 min |
| Sub-step 3 (CNN training, 5 epochs) | 3–8 min |
| Sub-step 4 (SBERT corpus encoding) | 1–3 min |
| Sub-steps 5–7 | < 2 min |
| **Total** | **~10–15 min** |

GPU (CUDA) reduces CNN training to < 1 min.

---

## Requirements

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
torch>=2.0
torchvision>=0.15
sentence-transformers>=2.2
Pillow>=9.0
jupyter>=1.0
```

See `requirements.txt` for pinned versions.

---

## Folder Structure
```
week-08/
└── wednesday/
    ├── W8_Wednesday_Band4.ipynb   ← main notebook
    ├── social_media_posts.csv     ← dataset (place here from LMS)
    ├── README.md                  ← this file
    ├── requirements.txt           ← pip dependencies
    ├── prompts.md                 ← AI usage log (required by policy)
    └── .gitignore                 ← excludes checkpoints, cache, data
```

---

## Notes
- All file paths are relative — no hardcoded absolute paths.
- Random seed is set globally (`SEED = 42`) for reproducibility.
- All thresholds and constants are named at the top of the notebook — no magic numbers.
- Each sub-step uses at least two functions (modular structure).
- `try/except` guards are present around I/O and model calls.
