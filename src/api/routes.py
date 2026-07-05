from fastapi import APIRouter
from ..models.detector import Detector
from ..models.recognizer import Recognizer
from ..models.pipeline import Pipeline
from ..config.settings import YOLOV11_M, CRNN_MODEL, DATA_TRANSFORMS, DEVICE
from .schemas import PredictionResponse
from fastapi import UploadFile, File
from PIL import Image

router = APIRouter()

detector = Detector(YOLOV11_M)
recognizer = Recognizer(CRNN_MODEL, DATA_TRANSFORMS["val"], DEVICE)
pipeline = Pipeline(detector, recognizer)

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    predictions = pipeline.process(image)
    return {
        "predictions": predictions
    }