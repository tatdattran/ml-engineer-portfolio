# YOLO Object Detection API (FastAPI + YOLOv8)

A production-ready REST API for object detection using **YOLOv8 (Ultralytics)** and **FastAPI**.
This project follows clean architecture principles and is fully containerized with Docker — suitable for development, deployment, and portfolio demonstration.

---

## ⚙️ Requirements

* Python **3.11** (recommended)
* pip
* Docker (optional)


## ✨ Features

* ⚡ FastAPI high-performance backend
* 🎯 YOLOv8 object detection (Ultralytics)
* 🖼️ Upload image → get annotated image with bounding boxes
* 📊 JSON output with detection results
* 🧪 Unit testing with pytest
* 🧹 Linting with Ruff
* 🐳 Docker & docker-compose support
* ⚙️ Environment-based configuration
* 🧱 Clean architecture (API / Services / Core / Schemas)

---

## Project structure

```text
05_object_detection_yolo/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── schemas/
│   │   └── response.py
│   ├── services/
│   │   └── detector.py
│   └── main.py
├── assets/
│   ├── outputs/
│   │   └── contain the figure after prediction
│   ├── uploads/
│   │   └── figures uploaded
│   ├── demo_ui_5_1.jpg
│   ├── demo_ui_5_2.jpg
│   ├── demo_ui_5_3.jpg
│   ├── demo_ui_5_4.jpg
│   └── demo_ui_5_5.jpg
├── scripts/
│   └── download_sample_model.sh
├── tests/
│   ├── test_health.py
│   └── test_predict_schema.py
├── .dockerignore
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Quick start

### 1) Clone and enter the project

```bash
git clone https://github.com/tatdattran/ml-engineer-portfolio.git
cd 05_object_detection_yolo
```

### 2) Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Configure environment

Default model:

```env
MODEL_PATH=yolov8n.pt
```


### 5) Run the API

```bash
make run
```

Open:

- Swagger UI: `http://127.0.0.1:8000/docs`


## API endpoints

### `GET /health`
Returns service status.

### `GET /ready`
Checks whether the model is loaded.

### `POST /predict`
Detect objects from an uploaded image.

* Go to `POST/predict`
![Demo UI 5.1](assets/demo_ui_5_1.png)
* Try it out
![Demo UI 5.2](assets/demo_ui_5_2.png)
* Upload an image
![Demo UI 5.3](assets/demo_ui_5_3.png)
* Click **Execute**
![Demo UI 5.4](assets/demo_ui_5_4.png)
* View the result image with bounding boxes
![Demo UI 5.5](assets/demo_ui_5_5.png)


## Run tests

```bash
make test
```

## Lint and format

```bash
make lint
make format
```

## Run with Docker

```bash
docker compose up --build
```

## ⚠️ Notes

* First Docker build may take several minutes (PyTorch is large)
* Model (`yolov8n.pt`) will be downloaded automatically on first run

---



# 🚀 Production Notes

* Clean architecture for scalability
* Docker-ready deployment
* Easy integration with frontend or microservices
* Suitable for cloud deployment

---

# 🔮 Future Improvements

* 🔐 Authentication (API Key / JWT)
* 📡 Video / stream detection
* ☁️ Cloud deployment (AWS / GCP / Render)
* 📊 Monitoring & logging
* 🎯 Model versioning

---

# 👤 Author

**Dr. Tat Dat Tran**

GitHub: https://github.com/tatdattran

---

# ⭐ If you find this project useful, please give it a star!
