# 🛍️ E-commerce Recommendation System


## Goal

<p align="center">
  <b>Production-style recommendation system inspired by modern online retail product discovery</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue">
  <img src="https://img.shields.io/badge/FastAPI-0.110-green">
  <img src="https://img.shields.io/badge/scikit--learn-1.4-orange">
  <img src="https://img.shields.io/badge/Docker-Ready-blue">
  <img src="https://img.shields.io/badge/UI-Jinja2-lightgrey">
</p>

---

## Overview

This project is an **e-commerce recommendation system** built with **FastAPI**, **scikit-learn**, and a lightweight web UI.

It simulates a real-world recommendation workflow for an online retail platform by combining multiple recommendation signals:

- content similarity from product metadata
- category and brand affinity
- price similarity
- popularity score
- collaborative-style co-occurrence from user interactions

It uses a realistic **Amazon-style product catalog** and interaction dataset.

---

## Business Goal

The system is designed to support product discovery use cases such as:

- similar product recommendation
- cross-sell opportunities
- product detail page suggestions
- trending product surfacing

This mirrors the kind of recommendation logic used in real e-commerce systems.

---

## Recommendation Approach

The ranking pipeline is intentionally structured to look closer to a production recommendation service.

### 1. Content-based retrieval
A TF-IDF vector is built from:

- title
- category
- brand
- description
- features

### 2. Business-aware scoring
Additional signals improve recommendation quality:

- same category bonus
- same brand bonus
- price similarity
- popularity score

### 3. Interaction-aware signal
A lightweight collaborative-style feature is derived from user interaction history:

- `view`
- `add_to_cart`
- `purchase`

### 4. Final ranking
All signals are combined in a ranking layer:

```text
final_score =
    0.45 * content_score +
    0.15 * category_score +
    0.10 * brand_score +
    0.10 * price_score +
    0.10 * popularity_score +
    0.10 * collaborative_score
```

This keeps the project understandable while still reflecting a more mature system design.

---

## Architecture

```text
User → UI / API
     → FastAPI
     → Recommender Service
         ├── Content Similarity
         ├── Interaction-aware Signal
         ├── Business Features
         └── Ranking Layer
     → Response
```

---

## Project Structure

```text
03_recommendation_system/
├── app/
│   ├── api/                # API routes
│   ├── core/               # Application settings
│   ├── schemas/            # Response models
│   ├── services/           # Recommendation and ranking logic
│   ├── templates/          # Demo UI
│   └── main.py             # FastAPI entrypoint
├── artifacts/
│   └── recommender_bundle.joblib
├── assets/
│   └── RecommendationSystem.png
├── data/
│   ├── products.csv        # E-commerce product catalog
│   └── interactions.csv    # User-product interactions
├── scripts/
│   └── train_recommender.py
├── tests/
│   └── test_health.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Datasets

### `data/products.csv`
Contains product attributes such as:

- `product_id`
- `title`
- `category`
- `brand`
- `price`
- `rating`
- `num_reviews`
- `description`
- `features`

### `data/interactions.csv`
Contains simplified user interaction history:

- `user_id`
- `product_id`
- `action`

Supported actions:
- `view`
- `add_to_cart`
- `purchase`

---

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd 06_recommendation_system_portfolio_ready
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create environment file

```bash
cp .env.example .env
```

### 5. Train the recommender artifact

```bash
python scripts/train_recommender.py
```

This creates:

```text
artifacts/recommender_bundle.joblib
```

### 6. Run the application

```bash
uvicorn app.main:app --reload
```

### 7. Open the app

- UI: http://127.0.0.1:8000/
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

---

## Demo UI

The included UI is intentionally simple, clean, and portfolio-friendly.

It supports:

- browsing sample products
- entering a product ID
- choosing the number of recommendations
- viewing recommendation cards with:
  - category
  - brand
  - price
  - rating
  - final score
  - supporting ranking signals

It also includes a **Trending Products** view for a more product-like experience.

![Demo UI](/assets/RecommendationSystem.png)


---

## API Endpoints

### `GET /health`
Service health check.

### `GET /ready`
Model readiness check.

### `GET /catalog`
Returns the available product catalog.

### `GET /recommend/product/{product_id}?top_k=6`
Returns top-k similar products for a given product.

### `GET /recommend/trending?top_k=6`
Returns top trending products based on popularity.

---

## Training

Run:

```bash
python scripts/train_recommender.py
```

The training script:

- validates both datasets
- builds combined product text
- fits a TF-IDF vectorizer
- computes interaction co-occurrence
- computes popularity signals
- stores everything into a single loadable artifact bundle

---

## Docker

### Build and run

```bash
cp .env.example .env
docker compose up --build
```

### Open

- UI: http://127.0.0.1:8000/
- Docs: http://127.0.0.1:8000/docs

---

## Tests

```bash
pytest
```

---

## Lint

```bash
ruff check .
```

---

## This repository demonstrates:

- recommendation system fundamentals
- multi-signal ranking design
- API serving with FastAPI
- model artifact generation and loading
- product-oriented ML system thinking
- Dockerization
- clean project structure
- demo UI design for explainability


---

## Future Improvements

- user-personalized endpoints
- offline ranking metrics such as Precision@K / Recall@K / MAP@K
- true collaborative filtering
- caching layer
- feature store integration
- cloud deployment

---

# 👨‍💻 Author

**Dr. Tat Dat Tran**

GitHub: https://github.com/tatdattran

---

# ⭐ Star this repo if you find it useful!