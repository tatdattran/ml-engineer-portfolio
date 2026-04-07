import joblib
import numpy as np

from app.services.ranking import combine_scores


class SeniorHybridRecommender:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.bundle = None
        self.products_df = None
        self.tfidf_matrix = None
        self.id_to_index = None
        self.cooccurrence = None
        self.popularity = None
        self._load()

    def _load(self) -> None:
        self.bundle = joblib.load(self.model_path)
        self.products_df = self.bundle["products_df"]
        self.tfidf_matrix = self.bundle["tfidf_matrix"]
        self.id_to_index = self.bundle["id_to_index"]
        self.cooccurrence = self.bundle["cooccurrence"]
        self.popularity = self.bundle["popularity"]

    def is_ready(self) -> bool:
        return self.bundle is not None

    def _product_card(self, row) -> dict:
        return {
            "product_id": int(row["product_id"]),
            "title": str(row["title"]),
            "category": str(row["category"]),
            "brand": str(row["brand"]),
            "price": round(float(row["price"]), 2),
            "rating": round(float(row["rating"]), 2),
        }

    def get_catalog(self) -> list[dict]:
        return [self._product_card(row) for _, row in self.products_df.iterrows()]

    def recommend_product(self, product_id: int, top_k: int = 6) -> dict:
        if product_id not in self.id_to_index:
            raise KeyError(f"Product {product_id} not found.")

        idx = self.id_to_index[product_id]
        query_row = self.products_df.iloc[idx]
        query_vec = self.tfidf_matrix[idx]
        content_sim = (self.tfidf_matrix @ query_vec.T).toarray().ravel()

        ranked_indices = np.argsort(-content_sim)
        recommendations = []

        for rec_idx in ranked_indices:
            if rec_idx == idx:
                continue

            row = self.products_df.iloc[rec_idx]
            rec_product_id = int(row["product_id"])

            content_score = float(content_sim[rec_idx])
            category_score = 1.0 if row["category"] == query_row["category"] else 0.0
            brand_score = 1.0 if row["brand"] == query_row["brand"] else 0.0

            max_price = max(float(query_row["price"]), float(row["price"]), 1.0)
            price_score = max(0.0, 1.0 - abs(float(query_row["price"]) - float(row["price"])) / max_price)

            popularity_score = float(self.popularity.get(rec_product_id, 0.0))
            collaborative_score = float(self.cooccurrence.get(product_id, {}).get(rec_product_id, 0.0))

            final_score = combine_scores(
                content_score=content_score,
                category_score=category_score,
                brand_score=brand_score,
                price_score=price_score,
                popularity_score=popularity_score,
                collaborative_score=collaborative_score,
            )

            recommendations.append(
                {
                    **self._product_card(row),
                    "final_score": round(final_score, 4),
                    "content_score": round(content_score, 4),
                    "collaborative_score": round(collaborative_score, 4),
                    "popularity_score": round(popularity_score, 4),
                }
            )

        recommendations = sorted(recommendations, key=lambda x: x["final_score"], reverse=True)[:top_k]

        return {
            "query_product": self._product_card(query_row),
            "recommendations": recommendations,
        }

    def get_trending(self, top_k: int = 6) -> list[dict]:
        ranked = sorted(self.popularity.items(), key=lambda x: x[1], reverse=True)[:top_k]
        items = []
        for product_id, pop_score in ranked:
            idx = self.id_to_index[product_id]
            row = self.products_df.iloc[idx]
            items.append(
                {
                    **self._product_card(row),
                    "final_score": round(float(pop_score), 4),
                    "content_score": 0.0,
                    "collaborative_score": 0.0,
                    "popularity_score": round(float(pop_score), 4),
                }
            )
        return items
