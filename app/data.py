from __future__ import annotations

import io
from hashlib import sha1
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def _load_csv(bio: io.BytesIO, dtype: dict[str, str]) -> pd.DataFrame:
    try:
        return pd.read_csv(bio, dtype=dtype, engine="pyarrow")
    except Exception:
        bio.seek(0)
        return pd.read_csv(bio, dtype=dtype)


@st.cache_data(show_spinner=False)
def load_data_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    column_types = {
        "manufacturer": "string",
        "model": "string",
        "year": "int32",
        "mileage": "float32",
        "price": "float32",
        "mpg": "string",
        "fuel_type": "string",
        "engine": "string",
        "accidents_or_damage": "float32",
        "one_owner": "float32",
        "driver_rating": "float32",
        "seller_rating": "float32",
        "price_drop": "float32",
    }

    bio = io.BytesIO(csv_bytes)
    df = _load_csv(bio, column_types)

    num_cols = df.select_dtypes(include="number").columns
    str_cols = df.select_dtypes(include="string").columns
    df[num_cols] = df[num_cols].fillna(0)
    df[str_cols] = df[str_cols].fillna("Unknown")
    return df


@st.cache_data(show_spinner=False)
def load_data_from_path(file_path: str) -> pd.DataFrame:
    return load_data_from_bytes(Path(file_path).read_bytes())


def add_risk_label(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "price" in df.columns:
        df = df[df["price"] > 1].copy()

    df["risk"] = np.where(
        (df.get("accidents_or_damage", 0) > 0) | (df.get("driver_rating", 0) < 3.5),
        1,
        0,
    )
    return df


def apply_filters(
    df: pd.DataFrame,
    manufacturer: str,
    year_range: tuple[int, int],
    price_range: tuple[float, float],
    max_mileage: float,
) -> pd.DataFrame:
    out = df
    if manufacturer != "(All)" and "manufacturer" in out.columns:
        out = out[out["manufacturer"] == manufacturer]

    if "year" in out.columns:
        out = out[(out["year"] >= year_range[0]) & (out["year"] <= year_range[1])]

    if "price" in out.columns:
        out = out[(out["price"] >= price_range[0]) & (out["price"] <= price_range[1])]

    if "mileage" in out.columns:
        out = out[out["mileage"] <= max_mileage]

    return out.copy()


def dataset_signature(df: pd.DataFrame) -> str:
    numeric_cols = [
        "price",
        "mileage",
        "year",
        "accidents_or_damage",
        "one_owner",
        "driver_rating",
    ]
    stats: list[float] = [float(len(df))]
    for col in numeric_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").fillna(0)
            stats.append(float(series.sum()))
            stats.append(float(series.mean()))
        else:
            stats.extend([0.0, 0.0])

    stats_array = np.array(stats, dtype="float64")
    return sha1(stats_array.tobytes()).hexdigest()
