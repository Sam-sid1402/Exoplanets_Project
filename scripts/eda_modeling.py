#!/usr/bin/env python
# coding: utf-8

# # Exoplanet Host Prediction — EDA and Modeling
# 
# This notebook explores stellar properties, creates reusable engineered features, compares classification models, and selects XGBoost as the final model for predicting whether a star is likely to be an exoplanet host.
# 
# Main target: `exoplanet_host`  
# Main selected model: `XGBoost`  
# Important note: this model estimates similarity to known exoplanet-host stars. It does not confirm that a star truly has planets.
# 

# ## 1. Imports and configuration

# In[1]:


import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.plotting import scatter_matrix
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

set_config(display="diagram")
pd.set_option("display.max_columns", None)
get_ipython().run_line_magic('matplotlib', 'inline')

# Make local utilities importable when this notebook is opened from the notebooks folder.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from src.exoplanet_features import (
    CATEGORICAL_FEATURES,
    FINAL_FEATURES,
    NUMERIC_FEATURES,
    engineer_features,
    prepare_model_input
)


# ## 2. Load data
# 
# Update `DATA_PATH` if your file is in a different location.

# In[2]:


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "Labeled_Stars.csv"

raw_stars = pd.read_csv(DATA_PATH, index_col=False)
raw_stars.head()


# ## 3. Basic cleaning and feature engineering
# 
# 
# 

# In[3]:


stars_df = raw_stars.drop(columns=["gaia_dr3_id"], errors="ignore").copy()
stars_df = engineer_features(stars_df)

print(f"Dataset shape after feature engineering: {stars_df.shape}")
stars_df.info()


# In[4]:


stars_df.head()


# In[5]:


stars_df.describe().T


# ## 4. Target balance

# In[6]:


target_counts = stars_df["exoplanet_host"].value_counts()
target_share = stars_df["exoplanet_host"].value_counts(normalize=True).round(3)

print(target_counts)
print(target_share)

sns.countplot(data=stars_df, x="exoplanet_host")
plt.title("Target Balance: Exoplanet Host vs Non-host")
plt.xlabel("Exoplanet host")
plt.ylabel("Count")
plt.show()


# ## 5. Distribution analysis

# In[7]:


hist_features = [
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

stars_df[hist_features].hist(bins=50, figsize=(18, 12))
plt.suptitle("Feature Distributions")
plt.show()


# ## 6. Correlation and mutual information

# In[8]:


corr_matrix = stars_df.corr(numeric_only=True)
corr_with_target = corr_matrix["exoplanet_host"].sort_values(ascending=False)
corr_with_target


# In[9]:


mi_X = stars_df[FINAL_FEATURES]
# .drop(columns=["stellar_type"])
mi_y = stars_df["exoplanet_host"]

mi_scores = mutual_info_classif(mi_X, mi_y, random_state=42)
mi_scores = pd.Series(mi_scores, index=mi_X.columns).sort_values(ascending=False)
mi_scores


# In[10]:


plt.figure(figsize=(8, 5))
mi_scores.sort_values().plot(kind="barh")
plt.title("Mutual Information Scores")
plt.xlabel("MI score")
plt.show()


# ## 7. Pairplot / scatter matrix
# 
# Use a sample to keep the plot readable and fast.

# In[11]:


pairplot_features = [
    "star_metallicity",
    "star_mass_solar_units_log",
    "surface_gravity_log",
    "star_temperature_kelvin",
]

sns.pairplot(
    stars_df.sample(min(1000, len(stars_df)), random_state=42),
    vars=pairplot_features,
    hue="exoplanet_host",
)
plt.show()


# ## 8. Train/test split
# 
# Distance and parallax features are intentionally excluded from `FINAL_FEATURES` to reduce observational bias.

# In[12]:


X = stars_df[FINAL_FEATURES]

y = stars_df["exoplanet_host"]

strat_X_train, strat_X_test, strat_y_train, strat_y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print(strat_X_train.shape, strat_X_test.shape)


# ## 9. Preprocessing pipeline

# In[ ]:


num_pipeline = Pipeline([
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessing = ColumnTransformer([
    ("num", num_pipeline, NUMERIC_FEATURES),
    # ("cat", cat_pipeline, CATEGORICAL_FEATURES),
])

preprocessing


# ## 10. Model comparison

# In[14]:


neg_ratio = (strat_y_train == 0).sum() / (strat_y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        scale_pos_weight=neg_ratio,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    ),
}

cv_scoring = ["roc_auc", "f1", "recall", "precision"]
results = {}
trained_pipelines = {}
cv_storage = {}
THRESHOLD = 0.70 # used for higher presicion for xgboost model

for model_name, model in models.items():
    print(f"{'=' * 50}")
    print(f"Training: {model_name}")
    print(f"{'=' * 50}")

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])

    cv_results = cross_validate(
        full_pipeline,
        strat_X_train,
        strat_y_train,
        cv=5,
        n_jobs=-1,
        scoring=cv_scoring,
        return_train_score=False,
    )

    full_pipeline.fit(strat_X_train, strat_y_train)

    y_pred_default = full_pipeline.predict(strat_X_test)
    y_proba = full_pipeline.predict_proba(strat_X_test)[:, 1]
    y_pred_threshold = (y_proba > THRESHOLD).astype(int)

    test_metrics = {
        "roc_auc": round(roc_auc_score(strat_y_test, y_proba), 4),
        "f1": round(f1_score(strat_y_test, y_pred_threshold), 4),
        "recall": round(recall_score(strat_y_test, y_pred_threshold), 4),
        "precision": round(precision_score(strat_y_test, y_pred_threshold), 4),
    }

    cv_metrics = {
        "cv_roc_auc_mean": round(cv_results["test_roc_auc"].mean(), 4),
        "cv_roc_auc_std": round(cv_results["test_roc_auc"].std(), 4),
        "cv_f1_mean": round(cv_results["test_f1"].mean(), 4),
        "cv_recall_mean": round(cv_results["test_recall"].mean(), 4),
        "cv_precision_mean": round(cv_results["test_precision"].mean(), 4),
    }

    cm = confusion_matrix(strat_y_test, y_pred_threshold)

    results[model_name] = {**test_metrics, **cv_metrics, "confusion_matrix": cm.tolist()}
    trained_pipelines[model_name] = full_pipeline
    cv_storage[model_name] = cv_results

    print("Test Results:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name:12}: {metric_value}")

    print("Cross Validation:")
    for metric_name, metric_value in cv_metrics.items():
        print(f"{metric_name:24}: {metric_value}")

    print(f"Confusion Matrix at threshold {THRESHOLD}:{cm}")
    print("Classification Report at default threshold 0.5:")
    print(classification_report(strat_y_test, y_pred_default))


# In[15]:


comparison_df = pd.DataFrame(results).T.drop(columns="confusion_matrix")
comparison_df


# ## 11. Final model: XGBoost

# In[16]:


xgb_pipeline = trained_pipelines["XGBoost"]
xgb_model = xgb_pipeline.named_steps["model"]

feature_names = xgb_pipeline.named_steps["preprocessing"].get_feature_names_out()

feature_importance = pd.Series(
    xgb_model.feature_importances_,
    index=feature_names,
).sort_values(ascending=False)

feature_importance


# In[17]:


plt.figure(figsize=(10, 7))
feature_importance.sort_values().plot(kind="barh")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# In[18]:


# =====================================
# SHAP FOR XGBOOST PIPELINE
# =====================================

import shap
import pandas as pd
import matplotlib.pyplot as plt

xgb_pipeline = trained_pipelines["XGBoost"]

preprocessor = xgb_pipeline.named_steps["preprocessing"]
model = xgb_pipeline.named_steps["model"]

X_test_processed = preprocessor.transform(strat_X_test)

feature_names = preprocessor.get_feature_names_out()

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

X_sample = X_test_processed.sample(
    min(500, len(X_test_processed)),
    random_state=42
)

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_sample)

print("SHAP calculation completed")
print("Sample shape:", X_sample.shape)


# In[ ]:


shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar"
)


# In[ ]:


shap.summary_plot(
    shap_values,
    X_sample
)


# In[21]:


row = 0

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[row],
        base_values=explainer.expected_value,
        data=X_sample.iloc[row],
        feature_names=X_sample.columns
    )
)


# ## 12. XGBoost evaluation plots

# In[22]:


RocCurveDisplay.from_estimator(
    xgb_pipeline,
    strat_X_test,
    strat_y_test,
)
plt.title("ROC Curve — XGBoost")
plt.show()


# In[23]:


PrecisionRecallDisplay.from_estimator(
    xgb_pipeline,
    strat_X_test,
    strat_y_test,
)
plt.title("Precision-Recall Curve — XGBoost")
plt.show()


# In[24]:


xgb_proba = xgb_pipeline.predict_proba(strat_X_test)[:, 1]
xgb_pred_threshold = (xgb_proba > THRESHOLD).astype(int)
xgb_cm = confusion_matrix(strat_y_test, xgb_pred_threshold)

ConfusionMatrixDisplay(
    confusion_matrix=xgb_cm,
    display_labels=["Non-host", "Host"],
).plot()
plt.title(f"XGBoost Confusion Matrix — Threshold {THRESHOLD}")
plt.show()


# ## 13. Prediction table for error analysis

# In[25]:


xgb_results = strat_X_test.copy()
xgb_results["true_label"] = strat_y_test
xgb_results["host_probability"] = xgb_proba
xgb_results["prediction_threshold_0_7"] = xgb_pred_threshold

xgb_results.sort_values("host_probability", ascending=False).head(20)


# ## 14. Sanity check: Sun-like star
# 
# 

# In[26]:


sun_star = pd.DataFrame({
    "star_metallicity": [0.0],
    "star_mass_solar_units": [1.0],
    "star_age_billion_years": [4.6],
    "surface_gravity_log": [4.44],
    "star_radius_solar_units": [1.0],
    "star_temperature_kelvin": [5778],
    "star_luminosity_log": [0.0]
})

host_like_star = pd.DataFrame({
    "star_metallicity": [0.25],
    "star_mass_solar_units": [1.05],
    "star_age_billion_years": [5.0],
    "surface_gravity_log": [4.4],
    "star_radius_solar_units": [1.05],
    "star_temperature_kelvin": [5900],
    "star_luminosity_log": [0.1]
})

metal_poor_star = pd.DataFrame({
    "star_metallicity": [-1.5],
    "star_mass_solar_units": [0.85],
    "star_age_billion_years": [11.0],
    "surface_gravity_log": [4.5],
    "star_radius_solar_units": [0.9],
    "star_temperature_kelvin": [5400],
    "star_luminosity_log": [-0.2]
})

red_dwarf = pd.DataFrame({
    "star_metallicity": [-0.3],
    "star_mass_solar_units": [0.25],
    "star_age_billion_years": [8.0],
    "surface_gravity_log": [5.0],
    "star_radius_solar_units": [0.28],
    "star_temperature_kelvin": [3200],
    "star_luminosity_log": [-2.2]
})

unlikely_host_star = pd.DataFrame({
    "star_metallicity": [-2.5],
    "star_mass_solar_units": [0.18],
    "star_age_billion_years": [12.5],
    "surface_gravity_log": [5.1],
    "star_radius_solar_units": [0.2],
    "star_temperature_kelvin": [2900],
    "star_luminosity_log": [-3.0]
})

for name, star in {
    "Sun": sun_star,
    "Host-like": host_like_star,
    "Metal-poor": metal_poor_star,
    "Red dwarf": red_dwarf,
    "Unlikely": unlikely_host_star,
}.items():
    star = prepare_model_input(star)
    prob = xgb_pipeline.predict_proba(star)[0, 1]
    sun_prediction = int(prob > THRESHOLD)
    print(f"{name}: {prob:.4f}")
    print("Prediction:", sun_prediction)





# In[27]:


pred_proba = xgb_pipeline.predict_proba(strat_X_test)[:, 1]

print(pd.Series(pred_proba).describe())


# In[28]:


pd.Series(pred_proba).hist(bins=50)
plt.show()


# ## 15. Save final model

# In[ ]:


MODEL_PATH = PROJECT_ROOT/ "model" /"xgboost_exoplanet_model.pkl"
joblib.dump(xgb_pipeline, MODEL_PATH)
print(f"Saved model to: {MODEL_PATH}")


# In[30]:


import json

best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_pipeline   = trained_pipelines[best_model_name]
print(f"\nBest model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']})")


metrics_to_save = {
    name: {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    for name, metrics in results.items()
}

metrics_to_save['best_model'] = best_model_name

with open(PROJECT_ROOT/'model'/'metrics.json', 'w') as f:
    json.dump(metrics_to_save, f, indent=4)
print(f"All metrics saved to model/metrics.json")


# In[ ]:




