from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    request_id: Optional[str] = Field(default=None, description="Optional client request id")
    features: Dict[str, float] = Field(..., description="Feature name -> numeric value")


class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_stage: str
    # This model's "positive class" is the class for which probability is returned.
    positive_class: str
    probability_positive_class: float
    prediction: int
    prediction_label: str
    probability: float