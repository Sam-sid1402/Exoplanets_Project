"""Feature engineering utilities for the exoplanet host prediction project.

"""

from __future__ import annotations

import numpy as np
import pandas as pd


COLUMN_NAMES = {
    "st_teff": "star_temperature_kelvin",
    "st_mass": "star_mass_solar_units",
    "st_rad": "star_radius_solar_units",
    "st_met": "star_metallicity",
    "st_logg": "surface_gravity_log",
    "st_lum": "star_luminosity_log",  # NASA Exoplanet Archive stores stellar luminosity as log10(L/L_sun)
    "st_age": "star_age_billion_years",
    "sy_dist": "distance_parsecs",
    "sy_plx": "parallax_milliarcsec",
    "is_exoplanet_host": "exoplanet_host",
}

SUN_TEMP_K = 5778.0
SUN_RADIUS_SOLAR = 1.0
SUN_MASS_SOLAR = 1.0
SUN_METALLICITY = 0.0


FINAL_FEATURES = [
    "star_metallicity",
    "star_mass_solar_units_log",
    "star_age_billion_years",
    "surface_gravity_log",
    "star_radius_solar_units_log",
    "star_temperature_kelvin",
    "star_luminosity_log",
    "sun_similarity",
    "stellar_density",
    # "stellar_type",
]


NUMERIC_FEATURES = [
    "star_metallicity",
    "star_mass_solar_units_log",
    "star_age_billion_years",
    "surface_gravity_log",
    "star_radius_solar_units_log",
    "star_temperature_kelvin",
    "star_luminosity_log",
    "sun_similarity",
    "stellar_density",
]


CATEGORICAL_FEATURES = []
# "stellar_type"


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw NASA-style columns into readable project column names."""
    return df.rename(columns=COLUMN_NAMES).copy()


def stellar_type_from_temperature(temperature_kelvin: float) -> str:
    """Assign Harvard spectral class from stellar effective temperature."""
    if pd.isna(temperature_kelvin):
        return "Unknown"
    if temperature_kelvin >= 33000:
        return "O"
    if temperature_kelvin >= 10000:
        return "B"
    if temperature_kelvin >= 7300:
        return "A"
    if temperature_kelvin >= 6000:
        return "F"
    if temperature_kelvin >= 5300:
        return "G"
    if temperature_kelvin >= 3900:
        return "K"
    return "M"


def compute_stellar_density(mass_solar: pd.Series, radius_solar: pd.Series) -> pd.Series:
    """Compute mean stellar density relative to the Sun: rho/rho_sun = M/R^3."""
    radius_safe = radius_solar.replace(0, np.nan)
    return mass_solar / (radius_safe ** 3)


def compute_sun_similarity(df: pd.DataFrame) -> pd.Series:
    """Create a simple 0-1 similarity score to Sun-like stellar properties."""
    temp_score = 1 - (np.abs(df["star_temperature_kelvin"] - SUN_TEMP_K) / SUN_TEMP_K)
    radius_score = 1 - (np.abs(df["star_radius_solar_units"] - SUN_RADIUS_SOLAR) / SUN_RADIUS_SOLAR)
    mass_score = 1 - (np.abs(df["star_mass_solar_units"] - SUN_MASS_SOLAR) / SUN_MASS_SOLAR)
    metallicity_score = 1 - np.abs(df["star_metallicity"] - SUN_METALLICITY)

    score = (
        0.4 * temp_score
        + 0.2 * radius_score
        + 0.2 * mass_score
        + 0.2 * metallicity_score
    )

    return np.clip(score, 0, 1).round(4)


def compute_habitable_zone(log_luminosity_solar: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute approximate habitable zone boundaries from log10 stellar luminosity.

    Input must be log10(L/L_sun), as in NASA Exoplanet Archive st_lum.
    Returns inner boundary, outer boundary, and width in AU.
    """
    luminosity_solar = 10 ** log_luminosity_solar
    hz_inner = np.sqrt(luminosity_solar / 1.1)
    hz_outer = np.sqrt(luminosity_solar / 0.53)
    hz_width = hz_outer - hz_inner
    return hz_inner, hz_outer, hz_width


def engineer_features(df: pd.DataFrame, include_distance_features: bool = True) -> pd.DataFrame:
    """Apply all deterministic feature engineering used by the model.

    Expected raw input columns after standardization:
    - star_temperature_kelvin
    - star_mass_solar_units
    - star_radius_solar_units
    - star_metallicity
    - surface_gravity_log
    - star_luminosity_log
    - star_age_billion_years

    Optional columns:
    - distance_parsecs
    - parallax_milliarcsec
    - exoplanet_host
    """
    df = standardize_column_names(df)

    required = [
        "star_temperature_kelvin",
        "star_mass_solar_units",
        "star_radius_solar_units",
        "star_metallicity",
        "surface_gravity_log",
        "star_luminosity_log",
        "star_age_billion_years",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    # Keep np.log1p to match the original training logic.
    df["star_radius_solar_units_log"] = np.log1p(df["star_radius_solar_units"].clip(lower=0))
    df["star_mass_solar_units_log"] = np.log1p(df["star_mass_solar_units"].clip(lower=0))

    if include_distance_features:
        if "distance_parsecs" in df.columns:
            df["distance_parsecs_log"] = np.log1p(df["distance_parsecs"].clip(lower=0))
        if "parallax_milliarcsec" in df.columns:
            df["parallax_milliarcsec_log"] = np.log1p(df["parallax_milliarcsec"].clip(lower=0))

    df["stellar_type"] = df["star_temperature_kelvin"].apply(stellar_type_from_temperature)
    df["stellar_density"] = compute_stellar_density(
        df["star_mass_solar_units"],
        df["star_radius_solar_units"],
    )
    df["sun_similarity"] = compute_sun_similarity(df)

    # Optional astrophysical features for EDA. They are not part of FINAL_FEATURES by default.
    df["hz_inner"], df["hz_outer"], df["hz_width"] = compute_habitable_zone(
        df["star_luminosity_log"]
    )

    return df


def prepare_model_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw user/API input into the exact feature table expected by the model."""
    engineered = engineer_features(raw_df, include_distance_features=False)
    return engineered[FINAL_FEATURES]


