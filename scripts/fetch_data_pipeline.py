#!/usr/bin/env python
# coding: utf-8

# # Gaia + NASA Data Fetching Workflow
# 
# This notebook prepares the raw data for the exoplanet-host prediction project.
# 
# Main steps:
# 1. Download confirmed exoplanet-host stellar data from NASA Exoplanet Archive.
# 2. Extract Gaia DR3 source IDs from the NASA table.
# 3. Fetch Gaia DR3 stellar parameters for known host IDs.
# 4. Fetch Gaia DR3 negative samples for training.
# 5. Build the final labeled training dataset.
# 6. Optionally fetch unseen Gaia stars for future prediction/top-candidate search.
# 
# 

# In[ ]:


from pathlib import Path
import sys
import pandas as pd

# If this notebook is inside /notebooks, make imports from project root work.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from src.fetch_data import (
    download_nasa_stellar,
    extract_gaia_ids,
    fetch_gaia,
    build_labeled_training_dataset,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR, PROCESSED_DIR


# ## Gaia login
# 
# For small public queries, Gaia may work without login, but login is useful for larger jobs.
# 

# In[ ]:


from astroquery.gaia import Gaia

# Uncomment when needed. 
# Gaia.login(user="", password="")


# ## 1. Download NASA confirmed host-star data
# 
# NASA `pscomppars` gives confirmed exoplanet systems and stellar parameters. This is the positive class source.
# 

# In[ ]:


nasa_df = download_nasa_stellar(
    output_path=RAW_DIR / "Stellar_Host.csv"
)

nasa_df.head()


# In[ ]:


nasa_df.info()
nasa_df.isna().sum()


# ## 2. Extract Gaia DR3 IDs from NASA data
# 
# These IDs allow us to query Gaia DR3 for additional/fallback stellar parameters for known exoplanet-host stars.
# 

# In[ ]:


gaia_ids = extract_gaia_ids(nasa_df)

print(f"Extracted Gaia IDs: {len(gaia_ids)}")
print(gaia_ids[:5])


# ## 3. Fetch Gaia data for known host stars
# 
# `mode="ids"` queries Gaia only for the known Gaia source IDs extracted from NASA.
# 
# 

# In[ ]:


gaia_hosts_df = fetch_gaia(
    mode="ids",
    gaia_ids=gaia_ids,
    batch_size=1500,
    output_path=RAW_DIR / "Gaia_Hosts.csv",
)

gaia_hosts_df.head()


# ## 4. Fetch Gaia negative training sample
# 
# `mode="negatives"` fetches Gaia stars with complete parameters and adds:
# 
# ```python
# is_exoplanet_host = 0
# ```
# 

# In[ ]:


cool_m_negatives = fetch_gaia(
    mode="negatives",
    n_stars=3000,
    temp_min=2500,
    temp_max=3300,
    output_path=RAW_DIR/"gaia_negatives_cool_m.csv"
)

m_negatives = fetch_gaia(
    mode="negatives",
    n_stars=3000,
    temp_min=3300,
    temp_max=3900,
    output_path=RAW_DIR/"gaia_negatives_m.csv"
)

k_negatives = fetch_gaia(
    mode="negatives",
    n_stars=3000,
    temp_min=3900,
    temp_max=5300,
    output_path=RAW_DIR/"gaia_negatives_k.csv"
)

g_negatives = fetch_gaia(
    mode="negatives",
    n_stars=3000,
    temp_min=5300,
    temp_max=6000,
    output_path=RAW_DIR/"gaia_negatives_g.csv"
)

f_negatives = fetch_gaia(
    mode="negatives",
    n_stars=3000,
    temp_min=6000,
    temp_max=7300,
    output_path=RAW_DIR/"gaia_negatives_f.csv"
)

a_negatives = fetch_gaia(
    mode="negatives",
    n_stars=3000,
    temp_min=7300,
    temp_max=10_000,
    output_path=RAW_DIR/"gaia_negatives_a.csv"
)

b_negatives = fetch_gaia(
    mode="negatives",
    n_stars=3000,
    temp_min=10000,
    temp_max=33000,
    output_path=RAW_DIR/"gaia_negatives_b.csv"
)

balanced_negatives = pd.concat(
    [
        cool_m_negatives,
        m_negatives,
        k_negatives,
        g_negatives,
        f_negatives,
        b_negatives,
        a_negatives,
    ],
    ignore_index=True
).drop_duplicates(subset="gaia_dr3_id")

balanced_negatives["is_exoplanet_host"] = 0

balanced_negatives.to_csv(
    PROCESSED_DIR/"gaia_negatives_balanced_by_temperature.csv",
    index=False
)



# ## 5. Build final labeled training dataset
# 
# 
# 

# In[ ]:


# balanced_negatives = balanced_negatives[
#     (balanced_negatives["st_rad"] < 5) &
#     (balanced_negatives["st_met"] > -0.5)
# ].copy()

labeled_stars_df = build_labeled_training_dataset(
    nasa_df=nasa_df,
    gaia_hosts_df=gaia_hosts_df,
    gaia_negatives_df=balanced_negatives,
    output_path=PROCESSED_DIR / "Labeled_Stars.csv",
)

balanced_negatives.shape


# In[ ]:


labeled_stars_df.info()
labeled_stars_df["is_exoplanet_host"].value_counts()


# In[ ]:




