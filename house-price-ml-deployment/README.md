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

## ⚠️ Requirements

* Python **3.10** (recommended)
* pip
* Virtual environment (venv)

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

### 1. Clone repository

```
git clone https://github.com/tatdattran/house-price-ml-deployment.git
cd house-price-ml-deployment
```

---

### 2. Create virtual environment (Python 3.10)

#### macOS / Linux

```
python3.10 -m venv venv
source venv/bin/activate
```

#### Windows

```
py -3.10 -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Run the application

```
python app/app.py
```

---

### 5. Open in browser

```
http://localhost:5000
```

---

## 🐳 Run with Docker

Build the image:

```
docker build -t house-price-ml .
```

Run the container:

```
docker run -p 5000:5000 house-price-ml
```

Open in browser

```
http://localhost:5000
```

---


## 📸 Demo UI

![Demo UI](assets/demo_ui.png)


## 📊 Model Details

* **Algorithm:** XGBoost Regressor
* **Features:** Top 9 most important features
* **Evaluation Metric:** RMSE (Root Mean Squared Error)
