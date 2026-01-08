from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_DEFAULT_PATH = Path(__file__).with_name("cars.csv")


@dataclass(frozen=True)
class ModelArtifacts:
    model: RandomForestClassifier
    features_ai: list[str]
    accuracy: float
    feature_importance: pd.DataFrame


def _safe_divide(numer: pd.Series | float, denom: pd.Series | float) -> pd.Series | float:
    return numer / (denom + 1e-9)


def parse_mpg(value) -> float:
    if pd.isna(value):
        return 0.0
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "unknown"}:
            return 0.0
        if "-" in text:
            low, high = text.split("-", 1)
            return (float(low) + float(high)) / 2.0
        return float(text)
    except Exception:
        return 0.0


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
    try:
        df = pd.read_csv(bio, dtype=column_types, engine="pyarrow")
    except Exception:
        bio.seek(0)
        df = pd.read_csv(bio, dtype=column_types)

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


@st.cache_data(show_spinner=False)
def train_ai_model(df: pd.DataFrame) -> ModelArtifacts:
    df_ml = df.copy()

    label_encoders: dict[str, LabelEncoder] = {}
    cat_cols = df_ml.select_dtypes(include="string").columns
    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype("string"))
        label_encoders[col] = le

    features_ai = ["year", "mileage", "one_owner", "driver_rating", "seller_rating"]

    missing = [c for c in features_ai + ["risk"] if c not in df_ml.columns]
    if missing:
        raise ValueError(f"Thiếu cột cần thiết trong CSV: {missing}")

    X = df_ml[features_ai]
    y = df_ml["risk"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    acc = float(rf.score(X_test, y_test))

    fi = (
        pd.DataFrame({"feature": features_ai, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return ModelArtifacts(model=rf, features_ai=features_ai, accuracy=acc, feature_importance=fi)


def default_ahp_matrix(criteria: list[str]) -> np.ndarray:
    # Thứ tự đúng với notebook: price mileage year accident owner rating seller mpg drop
    A = np.array(
        [
            [1, 3, 5, 5, 7, 3, 3, 5, 3],
            [1 / 3, 1, 3, 3, 5, 3, 3, 3, 3],
            [1 / 5, 1 / 3, 1, 3, 3, 3, 3, 3, 3],
            [1 / 5, 1 / 3, 1 / 3, 1, 3, 3, 3, 3, 3],
            [1 / 7, 1 / 5, 1 / 3, 1 / 3, 1, 1, 1, 3, 1],
            [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1, 1, 3, 3, 3],
            [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1, 1 / 3, 1, 3, 3],
            [1 / 5, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1, 1],
            [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1, 1 / 3, 1 / 3, 1, 1],
        ],
        dtype=float,
    )
    if A.shape != (len(criteria), len(criteria)):
        raise ValueError("Ma trận AHP không khớp số tiêu chí")
    return A


def compute_ahp_weights(A: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    # Chuẩn hóa theo cột + trung bình dòng (đúng notebook)
    col_sum = A.sum(axis=0)
    A_norm = A / col_sum
    weights = A_norm.mean(axis=1)
    weights = weights / weights.sum()

    n = A.shape[0]
    Aw = A.dot(weights)
    lambda_max = float(np.mean(Aw / weights))
    CI = float((lambda_max - n) / (n - 1))

    # RI theo Saaty; n=9 -> 1.45 (đúng notebook)
    ri_map = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = float(ri_map.get(n, 1.49))
    CR = float(CI / RI) if RI > 0 else 0.0

    return weights, lambda_max, CI, CR


def compute_ahp_score(df: pd.DataFrame, criteria: list[str], weights: dict[str, float]) -> pd.Series:
    df_work = df.copy()

    if "mpg" in df_work.columns:
        df_work["mpg"] = df_work["mpg"].apply(parse_mpg).astype(float)

    # Build a working AHP frame that is resilient to missing columns
    df_ahp = pd.DataFrame(index=df_work.index)
    for col in criteria:
        if col in df_work.columns:
            df_ahp[col] = df_work[col]
        else:
            df_ahp[col] = 0.0

    benefit = ["year", "one_owner", "driver_rating", "seller_rating", "mpg", "price_drop"]
    cost = ["price", "mileage", "accidents_or_damage"]

    for col in benefit:
        if col not in df_ahp.columns:
            df_ahp[col] = 0.0
        max_val = float(df_ahp[col].max()) if len(df_ahp) else 0.0
        df_ahp[col] = _safe_divide(df_ahp[col], max_val if max_val > 0 else 1.0)

    for col in cost:
        if col not in df_ahp.columns:
            df_ahp[col] = 0.0
        min_val = float(df_ahp[col].min()) if len(df_ahp) else 0.0
        df_ahp[col] = _safe_divide(min_val, df_ahp[col] + 1e-6) if min_val > 0 else 0.0

    score = 0.0
    for col in criteria:
        score = score + (df_ahp[col].astype(float) * float(weights.get(col, 0.0)))
    return pd.Series(score, index=df.index, name="ahp_score")


def apply_filters(df: pd.DataFrame, manufacturer: str, year_range: tuple[int, int], price_range: tuple[float, float], max_mileage: float) -> pd.DataFrame:
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


def main() -> None:
    st.set_page_config(page_title="Car DSS (AHP + AI)", layout="wide")

    st.title("Hệ thống hỗ trợ ra quyết định mua ô tô (AHP + AI)")

    with st.sidebar:
        st.header("Dữ liệu")
        uploaded = st.file_uploader("Upload cars.csv", type=["csv"], accept_multiple_files=False)

        if uploaded is not None:
            df_raw = load_data_from_bytes(uploaded.getvalue())
            st.caption("Đang dùng file CSV bạn upload")
        else:
            if not DATA_DEFAULT_PATH.exists():
                st.error("Không tìm thấy cars.csv trong thư mục HHTRQD")
                st.stop()
            df_raw = load_data_from_path(str(DATA_DEFAULT_PATH))
            st.caption("Đang dùng cars.csv mặc định")

    df = add_risk_label(df_raw)

    # UI filters
    with st.sidebar:
        st.header("Bộ lọc")

        manufacturer_values = ["(All)"]
        if "manufacturer" in df.columns:
            manufacturer_values += sorted(df["manufacturer"].astype("string").unique().tolist())
        manufacturer = st.selectbox("Hãng xe", manufacturer_values)

        if "year" in df.columns and len(df):
            y_min, y_max = int(df["year"].min()), int(df["year"].max())
        else:
            y_min, y_max = 1990, 2026
        year_range = st.slider("Năm sản xuất", min_value=y_min, max_value=y_max, value=(y_min, y_max))

        if "price" in df.columns and len(df):
            p_min, p_max = float(df["price"].min()), float(df["price"].quantile(0.99))
        else:
            p_min, p_max = 0.0, 100000.0
        price_range = st.slider("Giá", min_value=float(p_min), max_value=float(p_max), value=(float(p_min), float(p_max)))

        if "mileage" in df.columns and len(df):
            m_max_default = float(df["mileage"].quantile(0.99))
        else:
            m_max_default = 200000.0
        max_mileage = st.slider("Mileage tối đa", min_value=0.0, max_value=float(max(m_max_default, 1.0)), value=float(m_max_default))

    df_filtered = apply_filters(df, manufacturer, year_range, price_range, max_mileage)

    # AHP setup
    ahp_criteria = [
        "price",
        "mileage",
        "year",
        "accidents_or_damage",
        "one_owner",
        "driver_rating",
        "seller_rating",
        "mpg",
        "price_drop",
    ]

    with st.sidebar:
        st.header("AHP")
        st.caption("Chấm điểm độ quan trọng (1 = ít quan trọng, 9 = rất quan trọng). Ứng dụng sẽ tự tạo ma trận AHP từ tỉ lệ điểm và tính trọng số.")

        default_scores = {
            "price": 8,
            "mileage": 7,
            "year": 6,
            "accidents_or_damage": 9,
            "one_owner": 5,
            "driver_rating": 7,
            "seller_rating": 5,
            "mpg": 4,
            "price_drop": 4,
        }

        criteria_scores: dict[str, int] = {}
        for c in ahp_criteria:
            criteria_scores[c] = int(
                st.slider(
                    f"{c}",
                    min_value=1,
                    max_value=9,
                    value=int(default_scores.get(c, 5)),
                    step=1,
                )
            )

    # Create a reciprocal pairwise matrix from user scores: A[i,j] = score_i / score_j
    scores_vec = np.array([float(criteria_scores[c]) for c in ahp_criteria], dtype=float)
    if np.all(scores_vec > 0):
        A = scores_vec[:, None] / scores_vec[None, :]
    else:
        # Fallback: if something odd happens, use equal importance
        A = np.ones((len(ahp_criteria), len(ahp_criteria)), dtype=float)

    w, lambda_max, CI, CR = compute_ahp_weights(A)
    ahp_weights = dict(zip(ahp_criteria, w))

    # Train model on full df (not filtered) to keep stable; predict on filtered
    try:
        artifacts = train_ai_model(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Score + predict
    df_scored = df_filtered.copy()
    df_scored["ahp_score"] = compute_ahp_score(df_scored, ahp_criteria, ahp_weights)

    # Prepare df_ml for prediction with same columns used in notebook
    df_ml_pred = df_scored.copy()
    cat_cols = df_ml_pred.select_dtypes(include="string").columns
    for col in cat_cols:
        le = LabelEncoder()
        df_ml_pred[col] = le.fit_transform(df_ml_pred[col].astype("string"))

    df_scored["risk_pred"] = artifacts.model.predict(df_ml_pred[artifacts.features_ai])

    with st.sidebar:
        st.header("Đề xuất")
        q = st.slider("Ngưỡng AHP (quantile)", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
        top_n = st.number_input("Top N", min_value=5, max_value=50, value=10, step=1)

    df_scored["recommendation"] = np.where(
        (df_scored["ahp_score"] >= df_scored["ahp_score"].quantile(float(q))) & (df_scored["risk_pred"] == 0),
        "RECOMMENDED",
        "NOT_RECOMMENDED",
    )

    ranked = df_scored[df_scored["risk_pred"] == 0]["ahp_score"].rank(method="dense", ascending=False)
    df_scored["rank"] = ranked.fillna(0).astype(int)

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("AI (RandomForest)")
        st.metric("Accuracy (test split)", f"{artifacts.accuracy:.3f}")
        st.write("Feature importance")
        st.dataframe(artifacts.feature_importance, use_container_width=True)
        st.bar_chart(artifacts.feature_importance.set_index("feature")["importance"])

    with col2:
        st.subheader("AHP")
        st.write("Trọng số")
        w_df = pd.DataFrame({"criteria": list(ahp_weights.keys()), "weight": list(ahp_weights.values())}).sort_values(
            "weight", ascending=False
        )
        st.dataframe(w_df, use_container_width=True)
        st.write(f"λ_max = {lambda_max:.4f} | CI = {CI:.4f} | CR = {CR:.4f}")
        st.caption("CR < 0.10 thường được xem là chấp nhận được")

    st.subheader("Kết quả đề xuất")
    st.caption(f"Số xe sau lọc: {len(df_scored):,} | RECOMMENDED: {(df_scored['recommendation'] == 'RECOMMENDED').sum():,}")

    show_cols = [
        "rank",
        "manufacturer",
        "model",
        "year",
        "price",
        "mileage",
        "ahp_score",
        "risk_pred",
        "recommendation",
    ]
    show_cols = [c for c in show_cols if c in df_scored.columns]

    df_top = (
        df_scored[df_scored["rank"] > 0]
        .sort_values(["rank", "ahp_score"], ascending=[True, False])
        .head(int(top_n))
    )

    st.dataframe(df_top[show_cols], use_container_width=True)

    st.subheader("Xem nhanh dữ liệu")
    st.dataframe(df_scored.head(20), use_container_width=True)

    csv_out = df_scored.to_csv(index=False).encode("utf-8")
    st.download_button("Download kết quả (CSV)", data=csv_out, file_name="dss_result.csv", mime="text/csv")


if __name__ == "__main__":
    main()
