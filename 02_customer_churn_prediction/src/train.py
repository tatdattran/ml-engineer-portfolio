import pandas as pd
import numpy as np
import time
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Start timer
# -------------------------

start_time = time.time()

print("Loading data...")

df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") 
# -------------------------
# Data cleaning
# -------------------------

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]

# -------------------------
# Feature types
# -------------------------

num_cols = X.select_dtypes(include=["int64", "float64"]).columns

cat_cols = X.select_dtypes(include=["object"]).columns

# -------------------------
# Preprocessing
# -------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# -------------------------
# Train test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Models
# -------------------------

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(random_state=42)
}

params = {
    "LogisticRegression": {
        "model__C": [0.1, 1, 10]
    },
    "RandomForest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10]
    }
}

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

results = []

# -------------------------
# Training loop
# -------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    grid = GridSearchCV(
        pipe,
        params[name],
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    roc = roc_auc_score(y_test, preds)

    print("Best params:", grid.best_params_)
    print("Accuracy:", acc)
    print("ROC AUC:", roc)

    joblib.dump(best_model, f"models/{name}.pkl")

    results.append({
        "model": name,
        "accuracy": acc,
        "roc_auc": roc
    })

# -------------------------
# Save results
# -------------------------

results_df = pd.DataFrame(results)

results_df.to_csv("results/model_results.csv", index=False)

# -------------------------
# End timer
# -------------------------

end_time = time.time()

print("\nTraining completed.")
print(f"Total runtime: {round(end_time - start_time,2)} seconds")
