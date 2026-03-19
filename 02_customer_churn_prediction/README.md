# Customer Churn Prediction

## Problem
Predict whether a customer will churn (leave) a telecom company based on customer data using machine learning models. This project demonstrates an end-to-end machine learning pipeline including data preprocessing, feature engineering, model training, evaluation, and visualization.

## Dataset
[Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). 
Dataset contains customer information such as tenure, monthly charges, contract type, payment method, demographics. 
Target variable: `Churn`.

## Project Structure
02_customer_churn_prediction/
├─ data/
│ └─ telco_churn.csv
├─ src/
│ ├─ train.py
│ ├─ predict.py
├─ notebooks/
│ └─ churn_eda.ipynb
├─ models/
│ ├─ RandomForest.pkl
│ └─ LogisticRegression.pkl
└─ README.md


## Machine Learning Pipeline
1. Data cleaning (convert TotalCharges to numeric, handle missing values)
2. Feature preprocessing (OneHotEncoding for categorical variables)
3. Train/Test split
4. Model training (RandomForest, LogisticRegression)
5. Hyperparameter tuning (GridSearchCV)
6. Model evaluation (Accuracy, ROC AUC)
7. Save model and predictions
8. Exploratory Data Analysis (EDA) and visualization

## Models
Random Forest Classifier, Logistic Regression

## Evaluation Metrics
Accuracy, ROC AUC

## Run the project
### Environment Setup
Create environment:
or install dependencies:
pip install -r requirements.txt

### Run training
cd projects/02_customer_churn_prediction/src
python train.py

Output: `models/RandomForest.pkl`, `models/LogisticRegression.pkl`, `results/model_results.csv`

### Run prediction
python predict.py

Output: `results/predictions.csv`. By default RandomForest is used; change model in line 6 of `predict.py` to switch.

### Exploratory Data Analysis
Open `notebooks/churn_eda.ipynb` for dataset overview, missing values, churn distribution, tenure & monthly charges distributions, ROC curve with AUC, top 10 feature importances.

## Tech Stack
* Python, Pandas, scikit-learn, Joblib
* Random Forest, Logistic Regression
* Pipeline + GridSearchCV
* OneHotEncoding for categorical variables
* Matplotlib & Seaborn for visualization
* Jupyter Notebook for EDA
