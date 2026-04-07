def combine_scores(
    content_score: float,
    category_score: float,
    brand_score: float,
    price_score: float,
    popularity_score: float,
    collaborative_score: float,
) -> float:
    return round(
        0.45 * content_score
        + 0.15 * category_score
        + 0.10 * brand_score
        + 0.10 * price_score
        + 0.10 * popularity_score
        + 0.10 * collaborative_score,
        4,
    )
