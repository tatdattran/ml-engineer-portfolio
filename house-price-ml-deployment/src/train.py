import os
import time
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from utils import load_data, split_features_target, train_test_split_data

# -----------------------------
# TIMER
# -----------------------------
start_total = time.time()

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading data...")
df = load_data("../data/train.csv")
X, y = split_features_target(df)
print("Dataset shape:", X.shape)

# -----------------------------
# FEATURE TYPES
# -----------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()

# -----------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split_data(X, y)

# -----------------------------
# FULL PREPROCESSOR
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), numerical_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ]
)

# -----------------------------
# MODEL & PARAM GRID
# -----------------------------
model = XGBRegressor(objective="reg:squarederror", random_state=42)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.05, 0.1]
}

# -----------------------------
# PIPELINE
# -----------------------------
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nTraining XGBoost with GridSearchCV...")
grid = GridSearchCV(pipe, param_grid, cv=3,
                    scoring="neg_root_mean_squared_error", n_jobs=-1)

grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_
print("Best params:", grid.best_params_)

# -----------------------------
# VALIDATION SCORE
# -----------------------------
y_pred = best_pipeline.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print("Validation RMSE (full features):", round(rmse,2))

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
print("\nCalculating feature importance...")
preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
model_fitted = best_pipeline.named_steps["model"]

num_features = preprocessor_fitted.transformers_[0][2]
cat_features = preprocessor_fitted.transformers_[1][2]

encoder = preprocessor_fitted.named_transformers_["cat"]["onehot"]
encoded_cat_features = encoder.get_feature_names_out(cat_features)

all_features = np.concatenate([num_features, encoded_cat_features])
importances = model_fitted.feature_importances_

feature_importance = pd.Series(importances, index=all_features)
top_features_encoded = feature_importance.sort_values(ascending=False).head(10)

print("\nTop encoded features:")
print(top_features_encoded)

# -----------------------------
# MAP TO ORIGINAL FEATURES
# -----------------------------
# giữ feature gốc cho web app
top_features = list(set([f.split("_")[0] for f in top_features_encoded.index]))
print("\nTop original features (for web input):")
print(top_features)

# Lưu top features JSON
os.makedirs("../models", exist_ok=True)
with open("../models/features.json","w") as f:
    json.dump(top_features,f)
print("Top features saved to models/features.json")

# -----------------------------
# TRAIN PIPELINE VỚI TOP FEATURES
# -----------------------------
print("\nRetraining pipeline with top features only...")
X_train_small = X_train[top_features]
X_val_small = X_val[top_features]

categorical_small = X_train_small.select_dtypes(include=["object"]).columns.tolist()
numerical_small = X_train_small.select_dtypes(include=["int64","float64"]).columns.tolist()

preprocessor_small = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), numerical_small),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_small)
    ]
)

pipeline_small = Pipeline([
    ("preprocessor", preprocessor_small),
    ("model", model)
])

pipeline_small.fit(X_train_small, y_train)

y_pred_small = pipeline_small.predict(X_val_small)
rmse_small = mean_squared_error(y_val, y_pred_small, squared=False)
print("Validation RMSE (top features):", round(rmse_small,2))

# -----------------------------
# SAVE MODELS
# -----------------------------
joblib.dump(best_pipeline, "../models/XGB_full_model.pkl")
joblib.dump(pipeline_small, "../models/pipeline_XGB.pkl")
print("Models saved to /models")

end_total = time.time()
print("\nTotal training time:", round(end_total-start_total,2), "seconds")
