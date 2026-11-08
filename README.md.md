# CineMatch AI — Large-Scale Personalized Movie Recommendation Engine

CineMatch AI is an educational, end-to-end movie recommendation prototype built with the MovieLens 25M dataset. It uses a **two-stage recommendation pipeline**: a TensorFlow Recommenders two-tower model retrieves candidate movies, and a deep neural network ranks those candidates for a user. FAISS provides efficient nearest-neighbour search over movie embeddings, and Streamlit provides an interactive dashboard.

> **Project status:** Working prototype. The core preprocessing, retrieval, ranking, recommendation, offline evaluation, and dashboard components have been implemented. Production infrastructure such as Spark, Redis, Airflow, Triton Inference Server, and online A/B testing is not currently implemented.

## Features

- MovieLens 25M data preprocessing with DuckDB and chronological per-user train/validation/test splits.
- Two-tower retrieval model trained with TensorFlow Recommenders.
- Movie embedding index built with FAISS (`IndexFlatIP`) using normalized embeddings.
- Deep neural ranking model using user/movie embeddings and movie popularity/rating features.
- Personalized recommendations with watched-movie filtering.
- Popularity-based fallback for users not present in the trained user vocabulary.
- Offline Recall@10 and NDCG@10 evaluation using a sampled candidate set.
- Streamlit dashboard with user selection, recommendation count, genre filtering, and ranking-score display.

## Architecture

```mermaid
flowchart TD
    A[MovieLens 25M Dataset] --> B[Preprocessing with DuckDB]
    B --> C[Train / Validation / Test Data]
    C --> D[Two-Tower Retrieval Model]
    D --> E[User and Movie Embeddings]
    E --> F[FAISS Movie Index]
    F --> G[Candidate Retrieval]
    G --> H[Deep Neural Ranking Model]
    H --> I[Top-N Personalized Recommendations]
    I --> J[Streamlit CineMatch AI Dashboard]
    C --> K[Offline Evaluation]
    K --> L[Recall@10 and NDCG@10]
```

## Technology stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy, DuckDB |
| Retrieval model | TensorFlow, TensorFlow Recommenders |
| Similarity search | FAISS |
| Ranking model | TensorFlow / Keras |
| API / application UI | Streamlit |
| Offline evaluation | Python, NumPy, Pandas |

## Repository structure

The project is organized approximately as follows. Generated model files and processed datasets are created when the corresponding scripts are run.

```text
RecommendationEngine/
├── app.py
├── src/
│   ├── preprocessing.py
│   ├── retrieval.py
│   ├── ranking.py
│   ├── recommend.py
│   └── evaluate.py
├── data/
│   ├── raw/
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── item_popularity.csv
├── models/
│   ├── user_tower.keras
│   ├── movie_tower.keras
│   ├── ranking_model.keras
│   ├── movie_index.faiss
│   └── movie_ids.npy
├── ml-25m/
│   ├── ratings.csv
│   ├── movies.csv
│   └── ...
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This project uses the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/), provided by GroupLens.

Download and extract the dataset so that `ratings.csv` and `movies.csv` are available in the `ml-25m/` directory at the project root. The preprocessing script reads the dataset from that location.

The current prototype uses a deterministic user sample rather than processing every interaction for every training step. The recorded preprocessing run produced:

- Training interactions: **1,763,958**
- Validation interactions: **253,805**
- Test interactions: **512,490**

The retrieval training run used a seeded sample of **500,000** training interactions. These are the recorded counts from the development run; results may differ if the sampling or preprocessing configuration changes.

## Setup on Windows (PowerShell)

Run these commands from the project root:

```powershell
cd D:\RecommendationEngine

python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Dataset preparation

Place the extracted MovieLens files in:

```text
D:\RecommendationEngine\ml-25m\
```

At minimum, the folder should contain `ratings.csv` and `movies.csv`.

### Run the pipeline

Run the scripts from the project root, in this order:

```powershell
python src/preprocessing.py
python src/retrieval.py
python src/ranking.py
python src/evaluate.py
python src/recommend.py
```

**Important:** The training scripts create or overwrite generated files in `data/processed/` and `models/`. If the trained models and processed files are already present and working, you do not need to rerun training just to launch the dashboard.

### Launch the dashboard

```powershell
streamlit run app.py
```

Streamlit will print a local URL in the terminal, normally `http://localhost:8501`. Open that address in your browser.

## Model pipeline

### 1. Candidate retrieval

The two-tower model learns user and movie representations in a shared embedding space. The trained movie tower creates embeddings for indexed movies. FAISS searches the normalized movie vectors to retrieve candidate items for a selected user.

### 2. Candidate ranking

The ranking model scores retrieved candidates using:

- User embedding (32 dimensions)
- Movie embedding (32 dimensions)
- Normalized movie popularity
- Average movie rating feature

The ranker is a feed-forward neural network with dense layers and dropout. Its output is a **ranking score** used to order candidates. It should not be interpreted as a calibrated probability that a user will like a movie.

### 3. Recommendation output

The recommendation function retrieves candidates, removes movies already present in the user's training history, scores the remaining candidates, and returns the top-N movies with their genres and ranking scores. A popularity-based fallback is used for users not represented in the trained user vocabulary.

## Offline evaluation

The recorded evaluation run used 200 eligible users and a sampled candidate set containing held-out positive items and sampled negative items.

| Metric | Recorded result |
|---|---:|
| Recall@10 | 0.7816 |
| NDCG@10 | 0.7588 |
| Users evaluated | 200 |

**How to interpret these results:** Recall@10 measures how many held-out relevant items appear in the top 10. NDCG@10 rewards relevant items appearing higher in the ranking. The evaluation is a prototype offline experiment with sampled negatives and a limited user sample; it is not a full-catalog benchmark, online CTR measurement, or production performance guarantee. The ranker's validation accuracy/AUC are from a constructed positive-versus-negative classification task and should not be presented as recommendation Recall/NDCG.

## Current limitations

- The current implementation is a research/educational prototype, not a production recommendation service.
- The offline evaluation uses a limited sample of users and sampled negatives; evaluation results can change with different sampling choices.
- The ranking score is not calibrated as a probability.
- The current retrieval index uses FAISS `IndexFlatIP`; approximate indexing and large-scale serving optimizations have not been added.
- Spark distributed processing, Redis caching, Airflow orchestration, Triton model serving, online A/B testing, and live CTR tracking are future extensions, not implemented features.
- Recommendations depend on the available interaction history; new-user and new-item cold-start handling is currently basic.

## Future enhancements

1. Evaluate on a larger user sample and compare against popularity and collaborative-filtering baselines.
2. Add richer user/item features and improve cold-start strategies.
3. Explore approximate FAISS indexes for faster large-scale retrieval.
4. Add an API service and Redis caching for low-latency serving.
5. Add Airflow for scheduled data/model pipelines.
6. Deploy models with Triton Inference Server.
7. Run controlled online A/B tests and monitor CTR, coverage, latency, and ranking quality.

## Reproducibility notes

- The development retrieval run used a fixed random seed for interaction sampling.
- Training and evaluation metrics depend on the exact dataset files, sampling configuration, package versions, and random seeds.
- Keep the dataset's original license/terms and cite GroupLens when redistributing project materials.

## Disclaimer

CineMatch AI is an educational project for learning personalized recommendation systems. Recommendations and evaluation metrics are experimental and should not be treated as guarantees of user preference or production performance.
