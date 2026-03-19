import pandas as pd
import joblib

print("Loading model...")

model = joblib.load("models/RandomForest.pkl")

print("Loading data...")

df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

X = df.drop(["Churn", "customerID"], axis=1)

preds = model.predict(X)

df["Prediction"] = preds

print(df[["Prediction"]].head())

df.to_csv("results/predictions.csv", index=False)

print("Predictions saved to results/predictions.csv")
