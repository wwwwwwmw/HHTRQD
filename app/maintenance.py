from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

MAINTENANCE_FEATURES = [
    "manufacturer",
    "year",
    "mileage",
    "accidents_or_damage",
    "price",
]

MAX_TRAIN_ROWS = 6000


@dataclass(frozen=True)
class MaintenanceArtifacts:
    model: RandomForestRegressor
    features: list[str]
    mae: float


def _synthesize_target(df: pd.DataFrame) -> pd.Series:
    current_year = datetime.utcnow().year
    year = df.get("year", 0).fillna(current_year)
    age = np.clip(current_year - year, 0, 30)

    mileage = df.get("mileage", 0).fillna(0)
    accidents = df.get("accidents_or_damage", 0).fillna(0)
    price = df.get("price", 0).fillna(0)

    base = 300 + age * 25
    mileage_component = (mileage / 1000) * 4
    accident_component = accidents * 250
    premium_component = (price / 10000) * 60

    synthetic = base + mileage_component + accident_component + premium_component
    return pd.Series(np.clip(synthetic, 200, None), index=df.index, name="maintenance_cost")


@st.cache_resource(show_spinner=False, hash_funcs={pd.DataFrame: lambda _: 0})
def train_maintenance_model(df: pd.DataFrame, data_signature: str) -> MaintenanceArtifacts:
    _ = data_signature  # ensures cache invalidates when dataset changes
    df_train = df.copy()

    if len(df_train) > MAX_TRAIN_ROWS:
        df_train = df_train.sample(MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)

    for col in MAINTENANCE_FEATURES:
        if col not in df_train.columns:
            df_train[col] = 0

    if "maintenance_cost" in df_train.columns:
        target = df_train["maintenance_cost"].astype(float)
    else:
        target = _synthesize_target(df_train)

    cat_cols = df_train.select_dtypes(include="string").columns.intersection(MAINTENANCE_FEATURES)
    for col in cat_cols:
        df_train[col] = df_train[col].astype("category").cat.codes

    features = df_train[MAINTENANCE_FEATURES].fillna(0)

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
        max_depth=18,
        max_features="sqrt",
        min_samples_leaf=2,
        max_samples=0.8,
    )
    model.fit(features, target)

    preds = model.predict(features)
    mae = float(mean_absolute_error(target, preds))

    return MaintenanceArtifacts(model=model, features=MAINTENANCE_FEATURES, mae=mae)


def predict_maintenance_cost(df: pd.DataFrame, artifacts: MaintenanceArtifacts) -> pd.Series:
    df_pred = df.copy()
    for col in artifacts.features:
        if col not in df_pred.columns:
            df_pred[col] = 0

    cat_cols = df_pred.select_dtypes(include="string").columns.intersection(artifacts.features)
    for col in cat_cols:
        df_pred[col] = df_pred[col].astype("category").cat.codes

    feature_df = df_pred[artifacts.features].fillna(0)
    preds = artifacts.model.predict(feature_df)
    return pd.Series(preds, index=df.index, name="maintenance_cost_pred")
