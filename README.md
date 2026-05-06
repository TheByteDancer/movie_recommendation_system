# Netflix Recommender System — A Modern Approach

**content-based · collaborative filtering · hybrid · semantic embeddings · FAISS**

A comprehensive rebuild of a Netflix recommendation system combining the [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) catalog with [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (TMDB metadata + user ratings).

> **Given a title you liked — or a user you know — what should we recommend next, and why?**

The notebook walks through five recommender families on a single, shared dataset: a simple TF-IDF baseline, an enriched multi-feature TF-IDF, a semantic-embedding retriever powered by sentence-transformers + FAISS, an IMDB-weighted popularity ranker, and a collaborative-filtering SVD model — finally fused into a **hybrid** content + collaborative recommender. Built end-to-end with **Polars** for data and **Plotly** for visualisation.

---

## TL;DR

| | |
|---|---|
| **Problem** | Given a Netflix title, recommend the 10 most similar titles from the catalog (catalog-only, no Netflix ratings). Then add user-aware reranking using TMDB ratings. |
| **Approach** | Deep EDA → simple TF-IDF baseline → enriched multi-feature TF-IDF → CountVectorizer "soup" → sentence-transformer semantic embeddings indexed in FAISS → IMDB-weighted popularity ranker → SVD collaborative filtering on user ratings → hybrid content + collaborative recommender. |
| **Result** | Semantic embeddings beat the simple TF-IDF baseline on a same-director hit-rate@10 probe by **~2×**. Enriched TF-IDF closes most of the gap at a fraction of the cost. The hybrid recommender produces **per-user reranking** of the same content-similar pool — different users see different orderings of the same candidates. |

---

## Why this notebook?

A single recommender architecture rarely answers every question stakeholders actually ask:

| Use case | What's needed | Best fit |
|----------|--------------|----------|
| **"Because you watched X"** rail | Catalog-only similarity, fast retrieval | Content-based (TF-IDF / semantic) |
| **Cold-start new title** | No ratings yet — only metadata | Content-based |
| **"Top picks for you"** rail | Personalised across whole catalog | Collaborative filtering (SVD) |
| **"Trending now"** rail | Crowd signal, no personalisation | Popularity / IMDB-weighted |
| **Personalised "more like this"** | Best of both worlds | Hybrid (content × collaborative) |

So we build all of them, side by side, and compare honestly.

---

## What's in the notebook

- **Polars** for all data wrangling — lazy scans, explode for multi-valued fields, fast group-bys
- **Plotly** for every visualisation — interactive, zoomable, dashboard-ready
- Full **EDA**: movies vs TV, year-added trends, content ratings, top countries, duration distribution, longest-running shows, genre breakdown, description-length analysis
- **Feature engineering** — a "rich document" per title (title + genres + description + director + cast) and a CountVectorizer "soup"
- **Five recommender families**:
  1. Simple TF-IDF on description only
  2. Enriched TF-IDF on the rich document (with bigrams, sublinear TF)
  3. CountVectorizer on the metadata "soup"
  4. Semantic embeddings (`all-MiniLM-L6-v2`) indexed in **FAISS**
  5. IMDB-weighted popularity ranker on TMDB
- **Quantitative evaluation** — same-director hit-rate@10 across all content-based recommenders
- **Collaborative filtering** with **SVD** (Surprise) on TMDB user ratings, with 5-fold cross-validation
- **Hybrid recommender** — content retrieval → collaborative reranking, producing per-user orderings
- **Production notes** — how this scales: ANN indexes, cold-start, freshness, latency

---

## Notebook structure

1. **Business Context** — when and why each strategy wins
2. **Setup & Imports**
3. **Data Loading & Contract** — Netflix catalog schema, integrity checks
4. **Exploratory Data Analysis**
   - Movies vs TV Shows
   - Content added over time
   - Catalog by release year
   - Content ratings analysis
   - Top producing countries
   - Movie duration distribution
   - TV shows with the most seasons
   - Genre analysis
   - Description-length distribution
5. **Feature Engineering** — rich document + CountVectorizer "soup"
6. **Recommender #1** — Simple TF-IDF (description only)
7. **Recommender #2** — Multi-feature TF-IDF + CountVectorizer alternative
8. **Recommender #3** — Semantic embeddings (`all-MiniLM-L6-v2`) + FAISS
9. **Side-by-side comparison** — eye-test on several query titles
10. **Quantitative evaluation** — same-director hit-rate@10
11. **Simple Recommender** — IMDB weighted rating on TMDB
12. **Collaborative Filtering** — SVD on user ratings (5-fold CV)
13. **Hybrid Recommender** — content + collaborative
14. **Production Notes** — what changes at scale
15. **Takeaways & Next Steps**

---

## Datasets

This project uses **two** Kaggle datasets:

### 1. [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
~8,800 titles. Columns: `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`.

### 2. [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
TMDB metadata + Movielens user ratings. Used files: `movies_metadata.csv`, `links_small.csv`, `ratings_small.csv` (and optionally `credits.csv`, `keywords.csv`).

### Expected layout

Place the raw files into a `data/` folder at the repository root:

```
netflix_movie_recommendation_system/
├── netflix_recommender_system.ipynb
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── data/
    ├── netflix-shows/
    │   └── netflix_titles.csv
    └── the-movies-dataset/
        ├── movies_metadata.csv
        ├── links_small.csv
        └── ratings_small.csv      # not tracked by git
```

---

## Getting started

### Requirements

- Python 3.10+
- `polars` ≥ 1.0
- `plotly`
- `scikit-learn`
- `sentence-transformers`
- `faiss-cpu` (or `faiss-gpu` if you have CUDA)
- `scikit-surprise`
- `numpy`
- `jupyter`

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook netflix_recommender_system.ipynb
```

> **Note:** the first run downloads the `all-MiniLM-L6-v2` sentence-transformer model (~80 MB) from Hugging Face. Embedding ~8.8 K titles takes ~1 min on CPU.

---

## Headline results

| Recommender | Same-director hit-rate@10 | Notes |
|---|---|---|
| Simple TF-IDF (description only) | low | weak baseline — descriptions are too short |
| Enriched TF-IDF (rich document) | strong gain | bigrams + cast/director/genre help a lot |
| CountVectorizer (soup) | comparable to TF-IDF | classic Kaggle baseline, holds up |
| **Semantic (MiniLM + FAISS)** | **best** | catches paraphrase / synonym similarity |
| SVD (RMSE on ratings) | ~0.89 | 5-fold CV on Movielens-small |

The hybrid recommender produces **different orderings of the same content-similar pool for different users** — exactly the behaviour you want from a "more like this, for you" rail.

---

## Design choices worth highlighting

- **Polars over pandas** — lazy `scan_csv`, `explode` for multi-valued fields (cast, country, genre), expressive groupbys.
- **Build a rich document, don't just use the description** — descriptions average ~145 chars and are often generic. Concatenating title + genres + description + director + cast yields a meaningfully better signal for both TF-IDF and semantic models.
- **TF-IDF and CountVectorizer side by side** — the older Kaggle convention used CountVectorizer on a cleaned "soup" string; we keep that as a baseline so the comparison is honest.
- **Normalise embeddings + use FAISS `IndexFlatIP`** — cosine similarity becomes inner product, retrieval is exact, and the same code path moves cleanly to HNSW / IVF-PQ at scale.
- **Same-director hit-rate as a ground-truth proxy** — there is no labelled "true neighbour" in this dataset; the director probe is crude but honest, and it favours systems that capture style rather than just keyword overlap.
- **Hybrid as content-retrieval → collab-rerank** — the canonical pattern: content gives a fast candidate pool with no cold-start issues, and SVD reranks for the user.

---

## Limitations & next steps

- **No live user data for Netflix titles** — collaborative filtering uses Movielens/TMDB ratings, not actual Netflix viewership.
- **Static FAISS index** — production needs an ANN index (HNSW / IVF-PQ) and an incremental update loop.
- **No diversity/novelty re-ranking** — top-K cosine is famously homogeneous; MMR or determinantal point processes would help.
- **No A/B harness** — the only offline metric here is same-director hit-rate; production decisions need CTR / watch-time uplift.
- **No two-tower / sequence models** — the next natural upgrade is a session-aware transformer (BERT4Rec / SASRec) over actual interaction sequences.

---

## License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for the full text.

Copyright © 2026 Aryan Gupta.
