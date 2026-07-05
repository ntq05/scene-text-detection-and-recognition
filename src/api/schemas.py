from pydantic import BaseModel

class Prediction(BaseModel):
    bbox: list[float]
    class_name: str
    confidence: float
    text: str


class PredictionResponse(BaseModel):
    predictions: list[Prediction]