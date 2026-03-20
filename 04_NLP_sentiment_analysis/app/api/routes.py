import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import Settings, get_settings
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()


def get_analyzer() -> SentimentAnalyzer:
    from app.main import analyzer

    return analyzer


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready(analyzer: SentimentAnalyzer = Depends(get_analyzer)) -> dict:
    return {"ready": analyzer.is_ready()}


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    settings: Settings = Depends(get_settings),
    analyzer: SentimentAnalyzer = Depends(get_analyzer),
) -> PredictResponse:
    text = payload.text.strip()

    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text must not be empty.",
        )

    max_chars = getattr(settings, "max_text_length", 1000)
    if len(text) > max_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Input text exceeds max length of {max_chars} characters.",
        )

    try:
        result = analyzer.predict(text)

        label = str(result["label"]).lower()
        score = float(result["score"])

        if label == "label_0":
            label = "negative"
        elif label == "label_1":
            label = "positive"

        message = f'The sentence "{text}" is {label} with score: {score:.4f}.'

        return PredictResponse(
            text=text,
            label=label,
            score=round(score, 4),
            message=message,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc