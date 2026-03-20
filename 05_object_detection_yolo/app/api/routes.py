import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.schemas.response import PredictResponse
from app.services.detector import YOLODetector

logger = logging.getLogger(__name__)
router = APIRouter()

# ====== FOLDERS ======
BASE_DIR = Path("assets")
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ====== DEPENDENCY ======
def get_detector() -> YOLODetector:
    from app.main import detector
    return detector


# ====== HELPERS ======
def _validate_image_upload(file: UploadFile, content: bytes, settings: Settings) -> None:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image uploads are supported.",
        )

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds max size of {settings.max_image_size_mb} MB.",
        )


def _safe_stem(filename: str | None) -> str:
    if not filename:
        return "uploaded_image"
    return Path(filename).stem.replace(" ", "_")


# ====== ROUTES ======
@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready")
def ready(detector: YOLODetector = Depends(get_detector)) -> dict:
    return {"ready": detector.is_ready()}


# ====== MAIN: RETURN IMAGE ======
@router.post(
    "/predict",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Annotated image with bounding boxes.",
        }
    },
)
async def predict(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    detector: YOLODetector = Depends(get_detector),
):
    content = await file.read()
    _validate_image_upload(file, content, settings)

    try:
        # decode image
        image_array = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file.",
            )

        # filenames
        name = _safe_stem(file.filename)
        input_path = UPLOAD_DIR / f"{name}.jpg"
        output_path = OUTPUT_DIR / f"{name}_pred.jpg"

        # save original
        cv2.imwrite(str(input_path), image)

        # run detection
        if not hasattr(detector, "model"):
            raise RuntimeError(
                "Detector must expose 'model'. Make sure YOLODetector has self.model"
            )

        results = detector.model.predict(source=image, verbose=False)
        annotated = results[0].plot()

        # save output
        cv2.imwrite(str(output_path), annotated)

        # encode for Swagger preview
        success, encoded = cv2.imencode(".jpg", annotated)
        if not success:
            raise RuntimeError("Failed to encode image")

        return StreamingResponse(
            BytesIO(encoded.tobytes()),
            media_type="image/jpeg",
            headers={"Content-Disposition": f'inline; filename="{output_path.name}"'},
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


# ====== OPTIONAL: RETURN JSON ======
@router.post("/predict-json", response_model=PredictResponse)
async def predict_json(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    detector: YOLODetector = Depends(get_detector),
) -> PredictResponse:
    content = await file.read()
    _validate_image_upload(file, content, settings)

    try:
        result = detector.predict(content)
        return PredictResponse(**result)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc