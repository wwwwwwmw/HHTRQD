from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ModelArtifacts:
    model: RandomForestClassifier
    features_ai: list[str]
    accuracy: float
    feature_importance: pd.DataFrame


def _hash_dataframe(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values
    return sha1(hashed.tobytes()).hexdigest()


@st.cache_resource(show_spinner=False, hash_funcs={pd.DataFrame: _hash_dataframe})
def train_ai_model(df: pd.DataFrame) -> ModelArtifacts:
    df_ml = df.copy()

    cat_cols = df_ml.select_dtypes(include="string").columns
    for col in cat_cols:
        df_ml[col] = df_ml[col].astype("category").cat.codes

    features_ai = ["year", "mileage", "one_owner", "driver_rating", "seller_rating"]

    missing = [c for c in features_ai + ["risk"] if c not in df_ml.columns]
    if missing:
        raise ValueError(f"Thiếu cột cần thiết trong CSV: {missing}")

    X = df_ml[features_ai]
    y = df_ml["risk"].astype(int)

    if y.nunique() < 2:
        fallback = DummyClassifier(strategy="most_frequent")
        fallback.fit(X, y)
        fi = pd.DataFrame({"feature": features_ai, "importance": 0.0})
        return ModelArtifacts(model=fallback, features_ai=features_ai, accuracy=1.0, feature_importance=fi)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_depth=None,
    )
    rf.fit(X_train, y_train)

    acc = float(rf.score(X_test, y_test))

    fi = (
        pd.DataFrame({"feature": features_ai, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return ModelArtifacts(model=rf, features_ai=features_ai, accuracy=acc, feature_importance=fi)
