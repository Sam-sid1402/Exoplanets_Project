"""Reusable data-fetching utilities for the exoplanet host prediction project.

Supported Gaia use cases:
1. mode="ids"        -> fetch Gaia rows for known source IDs, used for known host stars.
2. mode="negatives"  -> fetch Gaia rows for negative training samples and add target = 0.
3. mode="prediction" -> fetch unseen Gaia rows without target labels for model inference/top candidates.

Important:
- Call Gaia.login() in the notebook/session if needed.
- Gaia `lum_flame` is converted to log10 luminosity so it matches the project/NASA `st_lum` style.
"""

from __future__ import annotations
import math
import time
from io import StringIO
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import requests
from astroquery.gaia import Gaia





GAIA_COLUMN_RENAME = {
    "source_id": "gaia_dr3_id",
    "teff_gspphot": "st_teff",
    "radius_gspphot": "st_rad",
    "mass_flame": "st_mass",
    "mh_gspphot": "st_met",
    "logg_gspphot": "st_logg",
    "lum_flame": "st_lum",
    "age_flame": "st_age",
    "distance_gspphot": "sy_dist",
    "parallax": "sy_plx",
}

STELLAR_FEATURE_COLUMNS = [
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met",
    "st_logg",
    "st_lum",
    "st_age",
    "sy_dist",
    "sy_plx",
]

NASA_STELLAR_COLUMNS = ["gaia_dr3_id", *STELLAR_FEATURE_COLUMNS]



def download_nasa_stellar(output_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Download confirmed exoplanet-host stellar parameters from NASA Exoplanet Archive.

    Parameters
    ----------
    output_path:
        Optional CSV path where the raw NASA data should be saved.

    Returns
    -------
    pd.DataFrame
        NASA `pscomppars` stellar data with Gaia DR3 IDs where available.
    """
    query = """
    SELECT
        gaia_dr3_id,
        st_teff,
        st_rad,
        st_mass,
        st_met,
        st_logg,
        st_lum,
        st_age,
        sy_dist,
        sy_plx
    FROM pscomppars
    """

    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    params = {"query": query, "format": "csv"}

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))


    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path.resolve()}")

    return df


def extract_gaia_ids(nasa_df: pd.DataFrame) -> list[int]:
    """Extract clean integer Gaia DR3 source IDs from NASA `gaia_dr3_id` column."""
    if "gaia_dr3_id" not in nasa_df.columns:
        raise KeyError("Input dataframe must contain a 'gaia_dr3_id' column.")

    df = nasa_df.dropna(subset=["gaia_dr3_id"]).copy()
    df["gaia_dr3_id"] = df["gaia_dr3_id"].astype(str).str.extract(r"(\d{10,})")
    df = df.dropna(subset=["gaia_dr3_id"])

    return df["gaia_dr3_id"].astype("int64").drop_duplicates().tolist()


def _base_gaia_select() -> str:
    return """
        gs.source_id,
        gs.teff_gspphot,
        gs.logg_gspphot,
        gs.mh_gspphot,
        gs.distance_gspphot,
        gs.parallax,
        ap.radius_gspphot,
        ap.mass_flame,
        ap.lum_flame,
        ap.age_flame
    """


def _base_gaia_from() -> str:
    return """
    FROM gaiadr3.gaia_source AS gs
    LEFT JOIN gaiadr3.astrophysical_parameters AS ap
        ON gs.source_id = ap.source_id
    """


def _base_gaia_where() -> str:
    return """
    WHERE gs.teff_gspphot IS NOT NULL
    AND gs.logg_gspphot IS NOT NULL
    AND gs.mh_gspphot IS NOT NULL
    AND gs.distance_gspphot IS NOT NULL
    AND gs.parallax > 0
    AND ap.radius_gspphot IS NOT NULL
    AND ap.mass_flame IS NOT NULL
    AND ap.lum_flame IS NOT NULL
    AND ap.lum_flame > 0
    AND ap.age_flame IS NOT NULL
    """


def _standardize_gaia_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Gaia result table and rename columns to project schema."""
    if df.empty:
        return df

    df = df.drop_duplicates(subset="source_id").copy()
    df["lum_flame"] = np.log10(df["lum_flame"])

    df = df.rename(columns=GAIA_COLUMN_RENAME)
    return df


def fetch_gaia(
    mode: str,
    gaia_ids: Optional[Sequence[int]] = None,
    n_stars: int = 15_000,
    batch_size: int = 1_500,
    output_path: Optional[str | Path] = None,
    add_negative_label: bool = False,
    random_index_min: Optional[int] = None,
    random_index_max: Optional[int] = None,
    sleep_seconds: float = 1.0,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None
) -> pd.DataFrame:
    """Fetch Gaia DR3 stellar data for the project.

    Parameters
    ----------
    mode:
        One of:
        - "ids": fetch specific Gaia source IDs, usually known host stars.
        - "negatives": fetch Gaia stars for negative training samples and add `is_exoplanet_host = 0`.
        - "prediction": fetch unseen Gaia stars without a target label for inference/top candidates.
    gaia_ids:
        Gaia source IDs used only when `mode="ids"`.
    n_stars:
        Number of Gaia stars to fetch for `mode="negatives"` or `mode="prediction"`.
    batch_size:
        Number of IDs per query batch for `mode="ids"`.
    output_path:
        Optional CSV path to save the result.
    add_negative_label:
        If True, add `is_exoplanet_host = 0`. Automatically enabled for `mode="negatives"`.
    random_index_min, random_index_max:
        Optional Gaia `random_index` range for faster diverse chunk fetching.
        Useful for prediction chunks/top-candidate search.
    sleep_seconds:
        Pause between ID batches to avoid stressing Gaia service.
    temp_min, temp_max:
        Optional Gaia `teff_gspphot` range for controlled temperature fetching.

    Returns
    -------
    pd.DataFrame
        Standardized Gaia table using project column names.
    """
    mode = mode.lower().strip()
    valid_modes = {"ids", "negatives", "prediction"}

    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got: {mode}")

    all_results: list[pd.DataFrame] = []

    if mode == "ids":
        if gaia_ids is None:
            raise ValueError("For mode='ids', provide gaia_ids.")
        if len(gaia_ids) == 0:
            raise ValueError("gaia_ids is empty.")

        total_batches = math.ceil(len(gaia_ids) / batch_size)

        for start in range(0, len(gaia_ids), batch_size):
            batch = gaia_ids[start : start + batch_size]
            ids_str = ", ".join(str(int(source_id)) for source_id in batch)
            batch_num = start // batch_size + 1

            query = f"""
            SELECT
                {_base_gaia_select()}
            {_base_gaia_from()}
            {_base_gaia_where()}
            AND gs.source_id IN ({ids_str})
            """

            try:
                job = Gaia.launch_job(query)
                results = job.get_results().to_pandas()
                all_results.append(results)
                print(f"Batch {batch_num}/{total_batches} done — {len(results)} stars")
                time.sleep(sleep_seconds)
            except Exception as exc:
                print(f"Batch {batch_num}/{total_batches} failed: {exc}")

    else:
        random_filter = ""
        random_filter1 = ""
        if random_index_min is not None and random_index_max is not None:
            random_filter = f"""
            AND gs.random_index BETWEEN {int(random_index_min)} AND {int(random_index_max)}
            """
        if temp_min is not None and temp_max is not None:
            random_filter1 = f"""
            AND gs.teff_gspphot BETWEEN {float(temp_min)} AND {int(temp_max)}
            """

        query = f"""
        SELECT TOP {int(n_stars)}
            {_base_gaia_select()}
        {_base_gaia_from()}
        {_base_gaia_where()}
        {random_filter1}
        {random_filter}

        """

        print(f"Fetching {n_stars} Gaia stars for mode='{mode}'...")
        job = Gaia.launch_job(query)
        results = job.get_results().to_pandas()
        all_results.append(results)

    if not all_results:
        print("No results fetched.")
        return pd.DataFrame()

    df = pd.concat(all_results, ignore_index=True)
    df = _standardize_gaia_dataframe(df)

    if mode == "negatives" or add_negative_label:
        df["is_exoplanet_host"] = 0

    print(f"\nFetched: {len(df)} unique stars")
    print(f"Null counts:\n{df.isnull().sum()}\n")


    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path.resolve()}")

    return df


def build_labeled_training_dataset(
    nasa_df: pd.DataFrame,
    gaia_hosts_df: pd.DataFrame,
    gaia_negatives_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Create final labeled dataset from NASA hosts + Gaia fallback values + Gaia negatives."""
    nasa_df = nasa_df.copy()
    nasa_df["gaia_dr3_id"] = nasa_df["gaia_dr3_id"].astype(str).str.extract(r"(\d{10,})")
    nasa_df = nasa_df.dropna(subset=["gaia_dr3_id"]).copy()
    nasa_df["gaia_dr3_id"] = nasa_df["gaia_dr3_id"].astype("int64")

    gaia_hosts = gaia_hosts_df.copy()
    gaia_hosts = gaia_hosts.rename(
        columns={col: f"{col}_gaia" for col in STELLAR_FEATURE_COLUMNS}
    )

    stellar_df = nasa_df.merge(gaia_hosts, on="gaia_dr3_id", how="left")

    for col in STELLAR_FEATURE_COLUMNS:
        gaia_col = f"{col}_gaia"
        if gaia_col in stellar_df.columns:
            stellar_df[col] = stellar_df[col].fillna(stellar_df[gaia_col])

    gaia_fallback_cols = [col for col in stellar_df.columns if col.endswith("_gaia")]
    stellar_df = stellar_df.drop(columns=gaia_fallback_cols)

    stellar_df["is_exoplanet_host"] = 1
    stellar_df = stellar_df.drop_duplicates(subset="gaia_dr3_id")
    stellar_df = stellar_df.dropna().copy()

    negatives = gaia_negatives_df.copy()
    if "is_exoplanet_host" not in negatives.columns:
        negatives["is_exoplanet_host"] = 0

    labeled_df = pd.concat([stellar_df, negatives], ignore_index=True)
    labeled_df = labeled_df.drop_duplicates(subset="gaia_dr3_id").copy()

    print(f"Total stars: {len(labeled_df)}")
    print(f"Hosts: {int(labeled_df['is_exoplanet_host'].sum())}")
    print(f"Non-hosts: {int(len(labeled_df) - labeled_df['is_exoplanet_host'].sum())}")


    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_csv(output_path, index=False)
        print(f"Saved to: {output_path.resolve()}")

    return labeled_df
