import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.config import Settings, get_settings
from app.schemas.response import CatalogResponse, ProductRecommendResponse, TrendingResponse
from app.services.recommender import SeniorHybridRecommender

logger = logging.getLogger(__name__)
router = APIRouter()


def get_recommender() -> SeniorHybridRecommender:
    from app.main import recommender
    return recommender


@router.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness check")
def ready(recommender: SeniorHybridRecommender = Depends(get_recommender)) -> dict:
    return {"ready": recommender.is_ready()}


@router.get("/catalog", response_model=CatalogResponse, summary="Get product catalog")
def catalog(recommender: SeniorHybridRecommender = Depends(get_recommender)) -> CatalogResponse:
    return CatalogResponse(items=recommender.get_catalog())


@router.get(
    "/recommend/product/{product_id}",
    response_model=ProductRecommendResponse,
    summary="Recommend similar products",
)
def recommend_product(
    product_id: int,
    top_k: int = Query(default=6, ge=1, le=20),
    settings: Settings = Depends(get_settings),
    recommender: SeniorHybridRecommender = Depends(get_recommender),
) -> ProductRecommendResponse:
    if top_k <= 0:
        top_k = settings.top_k_default

    try:
        return ProductRecommendResponse(**recommender.recommend_product(product_id=product_id, top_k=top_k))
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Recommendation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation failed: {exc}",
        ) from exc


@router.get("/recommend/trending", response_model=TrendingResponse, summary="Get trending products")
def recommend_trending(
    top_k: int = Query(default=6, ge=1, le=20),
    recommender: SeniorHybridRecommender = Depends(get_recommender),
) -> TrendingResponse:
    try:
        return TrendingResponse(items=recommender.get_trending(top_k=top_k))
    except Exception as exc:
        logger.exception("Trending recommendation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trending recommendation failed: {exc}",
        ) from exc
