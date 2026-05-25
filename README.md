# 📚 Scholarly Label Predictor

> **Data Mining (CO3117 / 252) — Ho Chi Minh University of Technology (HCMUT)**  
> A competition to classify research papers in the field of Answer Set Programming (ASP)
Competed in a two-stage Kaggle competition ([Data Mining 252 — Stage 2](https://www.kaggle.com/competitions/data-mining-252-stage-2/overview)) to classify 3,090 research papers into ordinal label categories, evaluated on Quadratic Weighted Kappa (QWK).

---

## 🎯 Problem Statement

Predict the **category label** (`Label`, integer from 1–5) of research papers based on their metadata and text content. This is a **supervised ordinal classification** task, evaluated using **Quadratic Weighted Kappa (QWK)** — which penalizes larger prediction errors more heavily.

### Dataset

| Split | Samples | Labels |
|---|---|---|
| Train (Stage 2) | 2,494 | ✅ available |
| Public test (Stage 2) | 298 | ❌ not available |
| Private test (Stage 2) | 298 | ❌ not available |

Each paper includes: `id`, `Title`, `Authors`, `Venue`, `Year`, `DOI/Link`, `Label`.

> **Key challenge**: the original dataset has **no abstracts** — abstracts were crawled from external APIs to enrich the data.

🔗 **Competition page**: [Kaggle — Data Mining 252 Stage 2](https://www.kaggle.com/competitions/data-mining-252-stage-2/overview)

### Submission Format
```
id,Label
2491,1
2707,3
...
```

---

## 🗂️ Repository Structure

```
Scholarly-Label-Predictor/
│
├── data/
│   ├── raw/
│   │   ├── Stage1/                        # Stage 1 data (original train + test)
│   │   │   ├── Stage_1_publictrain.csv
│   │   │   ├── test (2).csv
│   │   │   └── sample_submission_DM252.csv
│   │   └── Stage2/                        # Stage 2 data (official competition)
│   │       ├── train.csv
│   │       ├── public_test.csv
│   │       ├── private_test.csv
│   │       └── Test_Submission.csv
│   │
│   ├── external/
│   │   └── crawled_abstracts.csv          # Abstracts crawled from external APIs
│   │
│   ├── interim/                           # Intermediate processed data
│   │   ├── cleaned_train.csv
│   │   ├── cleaned_test.csv
│   │   ├── merged_train.csv               # After merging crawled abstracts into train
│   │   └── merged_test.csv                # After merging crawled abstracts into test
│   │
│   ├── processed/                         # Precomputed features saved to disk
│   │   ├── train_SBERT.npy                # Raw SBERT embeddings (train)
│   │   ├── test_SBERT.npy                 # Raw SBERT embeddings (test)
│   │   ├── train_SBERT_features.npy       # SBERT + metadata features (train)
│   │   ├── test_SBERT_features.npy        # SBERT + metadata features (test)
│   │   ├── train_SVM_features.npz         # TF-IDF sparse matrix (train)
│   │   ├── test_SVM_features.npz          # TF-IDF sparse matrix (test)
│   │   └── train_labels.csv
│   │
│   └── submission/                        # All generated submission files
│       ├── submission_baseline_logreg.csv
│       ├── submission_baseline_nb.csv
│       ├── submission_baseline_svm.csv
│       ├── submission_sbert_xgboost.csv
│       ├── submission_sbert_gbm.csv
│       ├── submission_sbert_knn.csv
│       ├── submission_pure_sbert_logreg.csv
│       ├── submission_bert.csv
│       └── submission_final.csv           # Final submission file
│
├── notebooks/                             # Jupyter notebooks following the pipeline order
│   ├── 01_eda.ipynb                       # Exploratory Data Analysis
│   ├── 02_crawl_abstract.ipynb            # Crawl abstracts from APIs
│   ├── 02.5_proprocess.ipynb              # Preprocessing & data merging
│   ├── 03_feature_engineering.ipynb       # Feature engineering (TF-IDF, SBERT, metadata)
│   ├── 04_baseline_tfidf_svm.ipynb        # Baseline: TF-IDF + SVM / LogReg / Naive Bayes
│   ├── 04.5_sbert_lightgbm.ipynb          # SBERT embeddings + LightGBM
│   ├── 04.5_sbert_test_data.ipynb         # Generate SBERT features for test set
│   ├── 05_sbert_xgboost.ipynb             # SBERT + XGBoost
│   ├── 06_bert_scibert.ipynb              # Fine-tune SciBERT (allenai/scibert_scivocab_uncased)
│   ├── 06_hybrid_model.ipynb              # Hybrid model (SBERT + metadata + classifier)
│   └── 07_error_analysis.ipynb            # Error analysis & confusion matrix
│
├── src/
│   ├── crawler/                           # Multi-source abstract crawling module
│   │   ├── crawler.py                     # BaseCrawler (abstract base class)
│   │   ├── semantic_scholar_api.py        # Semantic Scholar API crawler
│   │   ├── semantic_scholar_paperid_crawler.py
│   │   ├── crossref_api.py                # CrossRef API crawler
│   │   ├── openalex_api.py                # OpenAlex API crawler
│   │   ├── doi_resolver_api.py            # DOI → metadata resolver
│   │   └── fallback_crawler.py            # Fallback when all APIs fail
│   │
│   ├── preprocess/
│   │   └── preprocess.py                  # DataMerger, MissingHandler, AuthorNormalizer, TextCleaner
│   │
│   ├── features/                          # Feature extraction modules
│   │   ├── tfidf_features.py              # TF-IDF for title, abstract, authors (scikit-learn)
│   │   ├── sbert_features.py              # SBERT encoder (all-mpnet-base-v2)
│   │   ├── metadata_features.py           # Metadata features (venue, year, ...)
│   │   ├── keyword_extraction.py          # Keyword extraction
│   │   └── feature_union.py               # Feature combination
│   │
│   ├── models/                            # Model training scripts
│   │   ├── train_svm.py                   # SVM (baseline)
│   │   ├── train_xgboost.py               # XGBoost
│   │   ├── train_lightgbm.py              # LightGBM
│   │   ├── train_bert.py                  # Fine-tune SciBERT with Stratified K-Fold CV
│   │   └── ensemble.py                    # Model ensembling
│   │
│   ├── evaluation/                        # Model evaluation utilities
│   │   ├── cross_validation.py            # Stratified K-Fold CV
│   │   ├── confusion_matrix.py            # Confusion matrix
│   │   ├── macro_f1.py                    # Macro F1 score
│   │   └── error_analysis.py              # Error analysis
│   │
│   ├── inference/
│   │   ├── predict.py                     # Run predictions on test set
│   │   └── make_submission.py             # Generate submission CSV
│   │
│   └── utils/
│       └── utils.py                       # Utility functions
│
├── config.py                              # Paths and hyperparameter configuration
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Overview

```
Raw Data (Stage 2)
       │
       ▼
[02_crawl_abstract]  ──► Crawl abstracts from Semantic Scholar / CrossRef / OpenAlex / DOI
       │
       ▼
[02.5_preprocess]    ──► Merge abstracts, handle missing values, normalize authors, clean text
       │
       ▼
[03_feature_eng]     ──► TF-IDF (title + abstract + authors) | SBERT embeddings | Metadata features
       │
       ├──► [04_baseline]       TF-IDF + SVM / LogReg / Naive Bayes
       ├──► [04.5 / 05]         SBERT + LightGBM / XGBoost / KNN / LogReg
       ├──► [06_bert_scibert]   Fine-tune SciBERT (5-fold CV, early stopping)
       └──► [06_hybrid]         Hybrid: SBERT + metadata features → classifier
                │
                ▼
       [07_error_analysis]  ──► Error analysis, confusion matrix
                │
                ▼
       submission_final.csv
```

---

## 🧩 Component Details

### Abstract Crawling (`src/crawler/`)

Since the original dataset contains no abstracts, a multi-source crawling system was built around a `BaseCrawler` abstract class with five implementations:

- **Semantic Scholar API** — primary source
- **CrossRef API** — first fallback via DOI
- **OpenAlex API** — second fallback
- **DOI Resolver** — direct DOI resolution
- **Fallback Crawler** — handles remaining cases

Normalized output: `{ id, doi, abstract, crawled_authors }`

### Preprocessing (`src/preprocess/preprocess.py`)

Four transformer classes following the sklearn API:

- `DataMerger` — left-joins crawled abstracts into the base dataframe on `id`
- `MissingHandler` — fills NaN values for title, abstract, authors, venue, and year
- `AuthorNormalizer` — normalizes author names (removes special characters, preserves Unicode)
- `TextCleaner` — lowercases text, strips URLs and special characters, normalizes whitespace

### Feature Engineering (`src/features/`)

**TF-IDF** (`tfidf_features.py`):
- Title TF-IDF: `max_features=3000`, ngram range `(1,2)`
- Abstract TF-IDF: `max_features=8000`, ngram range `(1,2)`
- Author TF-IDF: `max_features=1500`, ngram range `(1,1)`, last name only
- Output: sparse matrix (`.npz`), concatenated via `scipy.sparse.hstack`

**SBERT** (`sbert_features.py`):
- Model: `all-mpnet-base-v2`
- Input format: `"Title: {title} [SEP] Abstract: {abstract} Authors: {authors} Venue: {venue} Year: {year}"`
- Output: 768-dimensional dense embedding vectors (`.npy`)

### Models

| Model | Features | Notebook / Script |
|---|---|---|
| SVM (baseline) | TF-IDF (title + abstract + authors) | `train_svm.py` / `04_baseline_tfidf_svm.ipynb` |
| Logistic Regression (baseline) | TF-IDF | `04_baseline_tfidf_svm.ipynb` |
| Naive Bayes (baseline) | TF-IDF | `04_baseline_tfidf_svm.ipynb` |
| XGBoost | SBERT embeddings | `train_xgboost.py` / `05_sbert_xgboost.ipynb` |
| LightGBM | SBERT embeddings | `train_lightgbm.py` / `04.5_sbert_lightgbm.ipynb` |
| KNN | SBERT embeddings | `04.5_sbert_lightgbm.ipynb` |
| **SciBERT** | Raw text (title + abstract + venue + year + authors) | `train_bert.py` / `06_bert_scibert.ipynb` |
| Hybrid | SBERT + metadata | `06_hybrid_model.ipynb` |

### SciBERT (`src/models/train_bert.py`)

The strongest model in the pipeline:

- **Backbone**: `allenai/scibert_scivocab_uncased`
- **Head**: Dropout(0.3) → Linear(768→256) → GELU → Dropout(0.3) → Linear(256→5)
- **Input format**: `"venue: {v} year: {y} authors: {a} {title} ( {v} ) [SEP] {abstract}"`
- **Training**: Stratified 5-Fold CV, `AdamW` lr=2e-5, `OneCycleLR` scheduler (10% warmup)
- **Early stopping**: patience=3, monitored on val Macro F1
- **Inference**: Average softmax probabilities across all 5 fold models (ensemble)
- **Label handling**: 1–5 shifted to 0–4 for CrossEntropyLoss; predictions shifted back +1 at inference

### Evaluation (`src/evaluation/`)

- **Primary metric**: Quadratic Weighted Kappa (QWK) — competition metric
- **Training metric**: Macro F1 (used during cross-validation)
- Confusion matrix and detailed error analysis in `07_error_analysis.ipynb`

---

## ⚙️ Installation

**Requirements**: Python 3.9+

```bash
git clone https://github.com/nnam128/Scholarly-Label-Predictor.git
cd Scholarly-Label-Predictor

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## 🚀 Usage

Run the notebooks in order:

```bash
jupyter notebook
```

1. `02_crawl_abstract.ipynb` — crawl abstracts (skip if `crawled_abstracts.csv` already exists)
2. `02.5_proprocess.ipynb` — merge and clean data
3. `03_feature_engineering.ipynb` — generate and save features
4. `04_baseline_tfidf_svm.ipynb` → `05_sbert_xgboost.ipynb` → `06_bert_scibert.ipynb` — train models
5. `07_error_analysis.ipynb` — analyze results

To run SciBERT with GPU:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📦 Tech Stack

| Category | Libraries |
|---|---|
| Data processing | `pandas`, `numpy` |
| Crawling | `requests`, `beautifulsoup4`, `httpx` |
| Feature extraction | `scikit-learn` (TF-IDF), `sentence-transformers` (SBERT) |
| Gradient boosting | `xgboost`, `lightgbm` |
| Deep learning | `torch`, `transformers` (HuggingFace SciBERT) |
| Visualization | `matplotlib`, `seaborn` |
| Notebooks | `jupyter`, `ipykernel` |

---

## 🏫 Course Information

| | |
|---|---|
| Course | Data Mining — CO3117 / 252 |
| University | Ho Chi Minh University of Technology (HCMUT) |
| Evaluation metric | Quadratic Weighted Kappa (QWK) |
| Competition | 2 stages (Stage 1: exploration, Stage 2: official) |

---

## 👥 Author

**nnam128** — [GitHub](https://github.com/nnam128)