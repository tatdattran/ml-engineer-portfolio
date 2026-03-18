# 🏠 House Price Prediction ML App

An end-to-end machine learning project that predicts house prices using an **XGBoost model**, deployed as a web application.

---

## 🚀 Project Highlights

* 🔍 Feature selection (reduced from 81 → 9 features)
* 🤖 Model training using XGBoost
* ⚙️ Data preprocessing with Scikit-learn pipeline
* 🌐 Flask API for inference
* 🖥️ Web UI for prediction
* 🐳 Docker containerization
* ☁️ Cloud deployment ready

---

## 🧠 ML Pipeline

```
Kaggle Dataset
      ↓
Data Preprocessing
      ↓
Feature Importance Analysis
      ↓
Top 9 Features Selected
      ↓
XGBoost Training
      ↓
Pipeline Serialization
      ↓
Flask API
      ↓
Web UI
```

---

## 💻 Web Demo

### Input Features

* FullBath
* GarageQual
* CentralAir
* BsmtQual
* OverallQual
* GarageCars
* GrLivArea
* KitchenQual
* GarageType

### Output

* 🏷️ Predicted house price

---

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app/app.py
```

Open in your browser:

```
http://localhost:5000
```

---

## 🐳 Run with Docker

Build the image:

```bash
docker build -t house-price-ml .
```

Run the container:

```bash
docker run -p 5000:5000 house-price-ml
```

---

## 📊 Model Details

* **Algorithm:** XGBoost Regressor
* **Features:** Top 9 most important features
* **Evaluation Metric:** RMSE (Root Mean Squared Error)
