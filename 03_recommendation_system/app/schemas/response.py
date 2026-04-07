from pydantic import BaseModel, Field


class ProductCard(BaseModel):
    product_id: int = Field(..., examples=[101])
    title: str = Field(..., examples=["Wireless Noise Cancelling Headphones"])
    category: str = Field(..., examples=["Electronics"])
    brand: str = Field(..., examples=["Sony"])
    price: float = Field(..., examples=[199.99])
    rating: float = Field(..., examples=[4.6])


class CatalogResponse(BaseModel):
    items: list[ProductCard]


class RecommendationItem(ProductCard):
    final_score: float = Field(..., examples=[0.8123])
    content_score: float = Field(..., examples=[0.6021])
    collaborative_score: float = Field(..., examples=[0.35])
    popularity_score: float = Field(..., examples=[0.91])


class ProductRecommendResponse(BaseModel):
    query_product: ProductCard
    recommendations: list[RecommendationItem]


class TrendingResponse(BaseModel):
    items: list[RecommendationItem]
