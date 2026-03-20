# 🚀 NLP Sentiment Analysis API (FastAPI + Transformers)

<p align="center">
  <b>Production-ready Sentiment Analysis API with Interactive Demo UI</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue">
  <img src="https://img.shields.io/badge/FastAPI-0.110-green">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow">
  <img src="https://img.shields.io/badge/Docker-Ready-blue">
</p>

---

## 📌 Overview

This project provides a **REST API for sentiment analysis** using a pre-trained Transformer model.

👉 It also includes a **lightweight web UI** for easy interaction and demo.

Perfect for:

* AI backend services
* NLP applications
* portfolio demonstration

---

## ✨ Features

* ⚡ FastAPI backend
* 🧠 Transformer-based sentiment analysis (Hugging Face)
* 🖥️ Minimal UI demo (no frontend framework needed)
* 📊 JSON API output
* 🐳 Docker support
* 🧪 Testing & linting ready
* ⚙️ Configurable via `.env`

---

## 🧱 Architecture

```id="l3fhg8"
User → UI (HTML) → FastAPI → Sentiment Service → Model → Response
```

---

## 📁 Project Structure

```id="a6v2h1"
04_NLP_sentiment_analysis/
│
├── app/
│   ├── api/            # API routes
│   ├── core/           # Config & settings
│   ├── schemas/        # Request/response models
│   ├── services/       # NLP model logic
│   ├── templates/      # UI demo (HTML)
│   └── main.py         # FastAPI entrypoint
│
├── assets/             # figure for Demo
├── tests/              # Unit tests
├── .env                # Environment variables
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚙️ Requirements

* Python **3.11**
* pip
* Docker (optional)

---

# 🧪 Run Locally

## 1. Clone repository

```bash id="c4b9h2"
git clone <your-repo-url>
cd 04_NLP_sentiment_analysis
```

---

## 2. Create virtual environment

```bash id="3v0gqt"
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash id="d6kp5v"
pip install --upgrade pip
pip install -r requirements.txt
```


---

## 4. Run server

```bash id="r4x7jm"
uvicorn app.main:app --reload
```

---

# 🖥️ Demo UI

Open:

👉 http://127.0.0.1:8000/

![Demo UI](assets/demo_ui_4.png)

### Usage

1. Enter a sentence
2. Click **Analyze Sentiment**
3. View result (positive/negative + score)

---

# 📘 API Docs (Swagger)

👉 http://127.0.0.1:8000/docs

---

## 🔌 API Endpoints

### ✅ GET /health

Check if API is running

---

### ✅ GET /ready

Check if model is loaded

---

### ✍️ POST /predict

Input:

```json id="h1i6qp"
{
  "text": "I really enjoyed this product."
}
```

Output:

```json id="rf4p7y"
{
  "input_text": "I really enjoyed this product.",
  "sentiment": "positive",
  "score": 0.9987,
  "result": "The sentence \"I really enjoyed this product.\" is positive with score: 0.9987."
}
```

---

# 🐳 Run with Docker

## 1. Ensure `.env` exists

```bash id="g7e3lk"
ls .env
```

---

## 2. Build & run

```bash id="2ah9dn"
docker compose up --build
```

---

## 3. Access

* UI: http://127.0.0.1:8000/
* Docs: http://127.0.0.1:8000/docs

---

## ⚠️ Notes

* First run will download the model (~100MB)
* Docker build may take longer due to dependencies

---

# 🧪 Testing

```bash id="l2m7qg"
pytest
```

---

# 🧹 Lint

```bash id="c7h4kq"
ruff check .
```

---

# 🧠 Model

Default:

```id="t4k9x2"
distilbert-base-uncased-finetuned-sst-2-english
```

---

# 🚀 Future Improvements

* 🔐 Authentication (API Key / JWT)
* 🎨 Better UI (React / Next.js)
* 📊 Dashboard visualization
* ☁️ Cloud deployment
* 🔁 Model versioning

---

# 👨‍💻 Author

**Your Name**

GitHub: https://github.com/tatdattran

---

# ⭐ Star this repo if you find it useful!
