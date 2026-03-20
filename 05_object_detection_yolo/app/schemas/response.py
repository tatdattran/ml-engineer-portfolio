from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(..., min_length=4, max_length=4)


class PredictResponse(BaseModel):
    image_width: int
    image_height: int
    model: str
    detections: list[DetectionItem]
    count: int
