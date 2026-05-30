import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "model" / "xgboost_exoplanet_model.pkl"

sys.path.append(str(SRC_DIR))

from exoplanet_features import prepare_model_input


app = FastAPI(
    title="Exoplanet Host-Likeness API",
    description=(
        "Returns a host-likeness score based on similarity to known "
        "exoplanet-host stars. This does not confirm planet presence."
    ),
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)


class StarInput(BaseModel):
    star_metallicity: float
    star_mass_solar_units: float = Field(gt=0)
    star_age_billion_years: float = Field(ge=0)
    surface_gravity_log: float
    star_radius_solar_units: float = Field(gt=0)
    star_temperature_kelvin: float = Field(gt=0)
    star_luminosity_log: float


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
    return {"message": "Exoplanet Host-Likeness API is running"}


@app.post("/predict")
def predict_star(star: StarInput):
    raw_data = pd.DataFrame([star.model_dump()])
    X = prepare_model_input(raw_data)

    score = float(model.predict_proba(X)[:, 1][0])

    return {
        "host_likeness_score": round(score, 6),
        "host_likeness_percent": round(score * 100, 2),
        "label": host_likeness_label(score),
        "threshold_0_7_prediction": "host-like" if score >= 0.7 else "not host-like",
        "note": (
            "This score measures similarity to known exoplanet-host stars "
            "from the training data. It does not confirm the presence of planets."
        ),
    }