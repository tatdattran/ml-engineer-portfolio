from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ACTION_WEIGHTS = {
    "view": 1.0,
    "add_to_cart": 2.0,
    "purchase": 3.0,
}


def normalize_dict(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    max_val = max(values.values()) or 1.0
    return {k: round(v / max_val, 4) for k, v in values.items()}


def build_cooccurrence(interactions: pd.DataFrame) -> dict[int, dict[int, float]]:
    user_groups = interactions.groupby("user_id")["product_id"].apply(list)
    cooccurrence = defaultdict(lambda: defaultdict(float))

    for products in user_groups:
        unique_products = list(dict.fromkeys(products))
        for i in range(len(unique_products)):
            for j in range(len(unique_products)):
                if i == j:
                    continue
                cooccurrence[unique_products[i]][unique_products[j]] += 1.0

    return {pid: normalize_dict(dict(neighbors)) for pid, neighbors in cooccurrence.items()}


def build_popularity(products: pd.DataFrame, interactions: pd.DataFrame) -> dict[int, float]:
    interaction_score = interactions.groupby("product_id")["weight"].sum().to_dict()
    scores = {}
    for _, row in products.iterrows():
        pid = int(row["product_id"])
        rating_component = float(row["rating"]) / 5.0
        review_component = min(np.log1p(float(row["num_reviews"])) / 10.0, 1.0)
        interaction_component = interaction_score.get(pid, 0.0)
        scores[pid] = 0.45 * rating_component + 0.25 * review_component + 0.30 * interaction_component
    return normalize_dict(scores)


def main() -> None:
    products_path = Path("data/products.csv")
    interactions_path = Path("data/interactions.csv")
    output_path = Path("artifacts/recommender_bundle.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    products = pd.read_csv(products_path)
    interactions = pd.read_csv(interactions_path)

    required_product_cols = [
        "product_id", "title", "category", "brand", "price", "rating",
        "num_reviews", "description", "features",
    ]
    missing_products = [col for col in required_product_cols if col not in products.columns]
    if missing_products:
        raise ValueError(f"Missing required product columns: {missing_products}")

    required_interaction_cols = ["user_id", "product_id", "action"]
    missing_interactions = [col for col in required_interaction_cols if col not in interactions.columns]
    if missing_interactions:
        raise ValueError(f"Missing required interaction columns: {missing_interactions}")

    products = products.copy()
    interactions = interactions.copy()

    products["combined_text"] = (
        products["title"].fillna("") + " "
        + products["category"].fillna("") + " "
        + products["brand"].fillna("") + " "
        + products["description"].fillna("") + " "
        + products["features"].fillna("")
    )

    interactions["weight"] = interactions["action"].map(ACTION_WEIGHTS).fillna(1.0)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(products["combined_text"])

    cooccurrence = build_cooccurrence(interactions)
    popularity = build_popularity(products, interactions)

    bundle = {
        "products_df": products[
            [
                "product_id", "title", "category", "brand", "price", "rating",
                "num_reviews", "description", "features",
            ]
        ].reset_index(drop=True),
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "id_to_index": {int(product_id): idx for idx, product_id in enumerate(products["product_id"].tolist())},
        "cooccurrence": cooccurrence,
        "popularity": popularity,
    }

    joblib.dump(bundle, output_path)
    print(f"Saved recommender bundle to: {output_path}")


if __name__ == "__main__":
    main()
