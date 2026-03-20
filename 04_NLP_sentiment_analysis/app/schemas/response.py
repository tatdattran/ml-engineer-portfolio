from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    text: str = Field(..., description="Input text to analyze")
    label: str = Field(..., description="Predicted sentiment label")
    score: float = Field(..., description="Prediction confidence score")
    message: str = Field(..., description="Human-readable prediction result")