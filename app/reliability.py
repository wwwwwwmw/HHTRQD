from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

RELIABILITY_FEATURES = [
    "price",
    "mileage",
    "year",
    "accidents_or_damage",
    "one_owner",
    "driver_rating",
]

MAX_TRAIN_ROWS = 6000


@dataclass(frozen=True)
class ReliabilityArtifacts:
    model: RandomForestClassifier
    features: list[str]
    auc: float


def _synthesize_labels(df: pd.DataFrame) -> pd.Series:
    current_year = datetime.utcnow().year
    year = df.get("year", 0).fillna(current_year)
    age = np.clip(current_year - year, 0, 40)

    mileage = df.get("mileage", 0).fillna(0)
    accidents = df.get("accidents_or_damage", 0).fillna(0)
    owner = df.get("one_owner", 0).fillna(0)

    risk_score = (
        (age / 20)
        + (mileage / 150000)
        + (accidents * 0.5)
        + (1 - owner) * 0.3
    )
    return pd.Series((risk_score >= 0.9).astype(int), index=df.index, name="major_repair")


@st.cache_resource(show_spinner=False, hash_funcs={pd.DataFrame: lambda _: 0})
def train_reliability_model(df: pd.DataFrame, data_signature: str) -> ReliabilityArtifacts:
    _ = data_signature
    df_train = df.copy()

    if len(df_train) > MAX_TRAIN_ROWS:
        df_train = df_train.sample(MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)

    for col in RELIABILITY_FEATURES:
        if col not in df_train.columns:
            df_train[col] = 0

    target = df_train.get("major_repair")
    if target is None or target.nunique() == 1:
        target = _synthesize_labels(df_train)

    cat_cols = df_train.select_dtypes(include="string").columns.intersection(RELIABILITY_FEATURES)
    for col in cat_cols:
        df_train[col] = df_train[col].astype("category").cat.codes

    features = df_train[RELIABILITY_FEATURES].fillna(0)

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=18,
        max_features="sqrt",
        min_samples_leaf=2,
        max_samples=0.85,
    )
    model.fit(features, target)

    try:
        auc = float(roc_auc_score(target, model.predict_proba(features)[:, 1]))
    except ValueError:
        auc = 0.5

    return ReliabilityArtifacts(model=model, features=RELIABILITY_FEATURES, auc=auc)


def predict_major_repair_risk(df: pd.DataFrame, artifacts: ReliabilityArtifacts) -> pd.Series:
    df_pred = df.copy()
    for col in artifacts.features:
        if col not in df_pred.columns:
            df_pred[col] = 0

    cat_cols = df_pred.select_dtypes(include="string").columns.intersection(artifacts.features)
    for col in cat_cols:
        df_pred[col] = df_pred[col].astype("category").cat.codes

    features = df_pred[artifacts.features].fillna(0)
    probs = artifacts.model.predict_proba(features)[:, 1]
    return pd.Series(probs, index=df.index, name="repair_risk_pred")
