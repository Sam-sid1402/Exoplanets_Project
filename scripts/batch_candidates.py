#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import sys
import pandas as pd
import joblib
import numpy as np


PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from fetch_data import (
    download_nasa_stellar,
    extract_gaia_ids,
    fetch_gaia,
    build_labeled_training_dataset,
)

from exoplanet_features import prepare_model_input


# In[ ]:


xgb_pipeline = joblib.load(
    PROJECT_ROOT/'model'/'xgboost_exoplanet_model.pkl'
)


# In[ ]:


def probability_label(prob):
    if prob >= 0.9:
        return "Very High Chance"
    elif prob >= 0.7:
        return "High Chance"
    elif prob >= 0.5:
        return "Possible Candidate"
    elif prob >= 0.3:
        return "Low Probability"
    else:
        return "Unlikely Host"


# In[ ]:


def find_top_gaia_candidates(
    model_pipeline,
    fetch_function,
    prepare_model_input,
    n_iterations=10,
    n_stars_per_batch=100_000,
    top_n=100,
    random_index_start=1_000_000,
    random_index_step=1_000_000,
    output_path=None
):
    top_candidates = pd.DataFrame()
    seen_ids = set()

    for i in range(n_iterations):
        random_min = random_index_start + i * random_index_step
        random_max = random_min + random_index_step

        print("=" * 60)
        print(f"Batch {i + 1}/{n_iterations}")
        print(f"Random index range: {random_min} - {random_max}")
        print("=" * 60)

        batch_df = fetch_function(
            mode="prediction",
            n_stars=n_stars_per_batch,
            random_index_min=random_min,
            random_index_max=random_max,
            output_path=None
        )

        if batch_df is None or batch_df.empty:
            print("No data fetched. Skipping batch.")
            continue

        batch_df = batch_df[
            ~batch_df["gaia_dr3_id"].isin(seen_ids)
        ].copy()

        if batch_df.empty:
            print("All stars already processed. Skipping.")
            continue

        seen_ids.update(batch_df["gaia_dr3_id"].tolist())

        X_batch = prepare_model_input(batch_df)

        batch_df["host_probability"] = model_pipeline.predict_proba(X_batch)[:, 1]

        batch_df["candidate_label"] = batch_df["host_probability"].apply(
            probability_label
        )

        batch_top = batch_df.sort_values(
            "host_probability",
            ascending=False
        ).head(top_n)

        top_candidates = pd.concat(
            [top_candidates, batch_top],
            ignore_index=True
        )

        top_candidates = top_candidates.drop_duplicates(
            subset="gaia_dr3_id"
        )

        top_candidates = top_candidates.sort_values(
            "host_probability",
            ascending=False
        ).head(top_n)

        print(f"Batch stars scored: {len(batch_df)}")
        print(f"Current best probability: {top_candidates['host_probability'].max():.4f}")
        print(f"Current top candidates stored: {len(top_candidates)}")

    if output_path is not None:
        top_candidates.to_csv(output_path, index=False)
        print(f"\nSaved top {top_n} candidates to: {output_path}")

    return top_candidates


# In[ ]:


top_100 = find_top_gaia_candidates(
    model_pipeline=xgb_pipeline,
    fetch_function=fetch_gaia,
    prepare_model_input=prepare_model_input,
    n_iterations=10,
    n_stars_per_batch=100_000,
    top_n=100,
    random_index_start=1_000_000,
    random_index_step=1_000_000,
    output_path=PROJECT_ROOT/'data'/'processed'/'top_100_gaia_candidates.csv'
)


# In[ ]:


top_100.info()


# In[ ]:


top_100.groupby('gaia_dr3_id')['host_probability'].max()


# In[ ]:


top = top_100.sort_values(
    "host_probability",
    ascending=False
)

top[[
    "gaia_dr3_id",
    "host_probability"
]].head(20)


# In[ ]:


from astroquery.gaia import Gaia
import pandas as pd

source_id = 5923445416658860160
# Uncomment when needed. 
# Gaia.login(user="", password="")
query = f"""
SELECT
    gs.source_id,
    gs.ra,
    gs.dec,
    gs.parallax,
    gs.phot_g_mean_mag,
    gs.bp_rp,
    gs.teff_gspphot,
    gs.logg_gspphot,
    gs.mh_gspphot,
    gs.distance_gspphot,
    ap.radius_gspphot,
    ap.mass_flame,
    ap.lum_flame,
    ap.age_flame
    FROM gaiadr3.gaia_source AS gs
    LEFT JOIN gaiadr3.astrophysical_parameters AS ap
        ON gs.source_id = ap.source_id
    WHERE gs.source_id = {source_id}
"""

job = Gaia.launch_job(query)

result = job.get_results().to_pandas()

if result.empty:
    print("Star not found in Gaia DR3")
else:
    print(result)


# In[ ]:


from astroquery.simbad import Simbad

result = Simbad.query_object("Gaia DR3 5923445416658860160")

print(result)


# In[ ]:




