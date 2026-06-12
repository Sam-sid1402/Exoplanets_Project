import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "model" / "xgboost_exoplanet_model.pkl"

sys.path.append(str(SRC_DIR))

from exoplanet_features import prepare_model_input


THRESHOLD = 0.7


app = FastAPI(
    title="Exoplanet Host-Likeness API",
    description=(
        "Batch prediction API returning host-likeness scores based on "
        "similarity to known exoplanet-host stars. This does not confirm "
        "planet presence."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.136:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class StarInput(BaseModel):
    star_metallicity: float
    star_mass_solar_units: float = Field(gt=0)
    star_age_billion_years: float = Field(ge=0)
    surface_gravity_log: float
    star_radius_solar_units: float = Field(gt=0)
    star_temperature_kelvin: float = Field(gt=0)
    star_luminosity_log: float


class BatchPredictionRequest(BaseModel):
    stars: list[StarInput] = Field(
        ...,
        description="List of raw star records."
    )


class BatchPredictionResponse(BaseModel):
    total_stars: int
    threshold: float
    predictions: list[dict[str, Any]]


def host_likeness_label(score: float) -> str:
    if score >= 0.9:
        return "Very host-like"
    if score >= 0.7:
        return "Host-like"
    if score >= 0.5:
        return "Moderately host-like"
    if score >= 0.3:
        return "Weak host similarity"
    return "Not host-like"


@app.get("/")
def root():
    return {
        "message": "Exoplanet Host-Likeness API is running.",
        "model_loaded": model is not None,
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Predict host-likeness scores for a batch of stars.
    """

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {MODEL_PATH}",
        )

    if not request.stars:
        raise HTTPException(
            status_code=400,
            detail="No stars provided.",
        )

    predictions = []

    for index, star in enumerate(request.stars):
        try:
            raw_data = pd.DataFrame([star.model_dump()])
            X = prepare_model_input(raw_data)
            score = float(model.predict_proba(X)[:, 1][0])

        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Prediction failed.",
                    "row_index": index,
                    "error": str(error),
                },
            )

        predictions.append(
            {
                "star_id": f"star_{index + 1}",
                "host_likeness_score": round(score, 6),
                "host_likeness_percent": round(score * 100, 2),
                "label": host_likeness_label(score),
                "prediction": (
                    "host-like"
                    if score >= THRESHOLD
                    else "not host-like"
                ),
                "note": (
                    "This score measures similarity to known exoplanet-host "
                    "stars from the training data. It does not confirm the "
                    "presence of planets."
                ),
            }
        )

    return {
        "total_stars": len(predictions),
        "threshold": THRESHOLD,
        "predictions": predictions,
    }