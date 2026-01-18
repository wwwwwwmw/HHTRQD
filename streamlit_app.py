from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.ahp import compute_ahp_score, compute_ahp_weights, parse_mpg
from app.auth import verify_password
from app.constants import (
    DEFAULT_SCORES,
    DISPLAY_COLUMN_LABELS,
    PRESET_SCORES,
    RISK_TEXT,
)
from app.data import add_risk_label, dataset_signature, normalize_vehicle_dataframe, load_data_from_bytes
from app.db import (
    bulk_insert_cars_from_dataframe,
    create_criterion,
    create_db_engine,
    delete_car,
    delete_criterion,
    delete_user,
    get_user_by_id,
    get_user_by_username,
    healthcheck,
    init_db,
    insert_car,
    list_criteria,
    list_recommendations,
    list_users,
    read_cars_df,
    save_recommendation,
    update_user,
    update_criterion,
    count_rows,
    create_user,
)
from app.maintenance import predict_maintenance_cost, train_maintenance_model
from app.model import train_ai_model
from app.reliability import predict_major_repair_risk, train_reliability_model
from app.seed import ensure_seed
from app.auth import hash_password


@st.cache_resource(show_spinner=False)
def get_engine():
    engine = create_db_engine()
    init_db(engine)
    ensure_seed(engine)
    return engine


def current_user() -> dict | None:
    return st.session_state.get("user")


def is_admin() -> bool:
    user = current_user()
    return bool(user and user.get("role") == "admin")


def sidebar_auth(engine) -> None:
    sidebar = st.sidebar
    sidebar.subheader("Tài khoản")
    sidebar.caption("Khách vẫn dùng bình thường; đăng nhập để có thêm chức năng.")

    user = current_user()
    if user:
        sidebar.success(f"Đang đăng nhập: {user['username']} ({user['role']})")
        if sidebar.button("Đăng xuất", width="stretch"):
            st.session_state.pop("user", None)
            st.rerun()
        return

    username = sidebar.text_input("Tên đăng nhập", key="login_username")
    password = sidebar.text_input("Mật khẩu", type="password", key="login_password")
    if sidebar.button("Đăng nhập", width="stretch"):
        u = get_user_by_username(engine, username.strip()) if username else None
        if u is None or not verify_password(password, u.password_hash):
            sidebar.error("Sai tài khoản hoặc mật khẩu")
            return
        st.session_state.user = {"id": u.id, "username": u.username, "role": u.role}
        st.rerun()


def apply_filter_criteria(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    out = df
    for f in filters:
        key = f["key"]
        value = f["value"]
        if key not in out.columns:
            continue

        if value is None:
            continue

        if isinstance(value, tuple) and len(value) == 2:
            lo, hi = value
            out = out[(out[key] >= lo) & (out[key] <= hi)]
        elif isinstance(value, str) and value != "(All)":
            out = out[out[key].astype("string") == value]
    return out.copy()


def page_recommend(engine) -> None:
    st.header("Gợi ý xe")
    st.caption("Kết quả chỉ được tính khi bạn nhấn nút 'Tìm xe phù hợp'.")

    with st.spinner("Đang tải dữ liệu xe..."):
        df_raw = read_cars_df(engine)
        df_raw = normalize_vehicle_dataframe(df_raw)
    if df_raw.empty:
        st.warning("Chưa có dữ liệu xe trong CSDL. Admin hãy vào trang 'Thêm dữ liệu' để nhập.")
        return

    df = add_risk_label(df_raw)
    data_key = dataset_signature(df)

    ahp_criteria_rows = list_criteria(engine, kind="ahp", enabled_only=True)
    filter_rows = list_criteria(engine, kind="filter", enabled_only=True)

    ahp_criteria = [c.key for c in ahp_criteria_rows]
    labels = {c.key: c.label for c in ahp_criteria_rows + filter_rows}
    benefit_criteria = [c.key for c in ahp_criteria_rows if c.direction == "benefit"]
    cost_criteria = [c.key for c in ahp_criteria_rows if c.direction == "cost"]

    if not ahp_criteria:
        st.error("Chưa cấu hình tiêu chí AHP. Admin hãy thêm tiêu chí ở trang 'Tiêu chí'.")
        return

    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = next(iter(PRESET_SCORES.keys()))

    if "criteria_initialized" not in st.session_state:
        for criterion in ahp_criteria:
            st.session_state[f"criteria_{criterion}"] = int(DEFAULT_SCORES.get(criterion, 5))
        st.session_state.criteria_initialized = True

    def _apply_preset_to_sliders() -> None:
        preset_name = st.session_state.get("selected_preset")
        preset_values = PRESET_SCORES.get(preset_name, DEFAULT_SCORES)
        for criterion in ahp_criteria:
            if criterion in preset_values:
                st.session_state[f"criteria_{criterion}"] = int(preset_values[criterion])

    sidebar = st.sidebar
    sidebar.header("1. Mục tiêu sử dụng")
    sidebar.caption("Chọn nhu cầu để tự gợi ý trọng số (bạn vẫn có thể chỉnh lại).")
    preset_options = list(PRESET_SCORES.keys())
    sidebar.radio(
        "Bạn đang tìm chiếc xe cũ kiểu nào?",
        preset_options,
        key="selected_preset",
        help="Chọn phong cách lái xe để hệ thống tự động điều chỉnh trọng số.",
        on_change=_apply_preset_to_sliders,
    )

    with sidebar.form("search_form"):
        st.subheader("2 · Điều kiện thực tế")
        st.caption("Điều kiện lọc xe. (Admin có thể chỉnh danh sách bộ lọc ở trang 'Tiêu chí'.)")

        filter_inputs: list[dict] = []
        for row in filter_rows:
            key = row.key
            if key not in df.columns:
                continue

            label = labels.get(key, key)
            if pd.api.types.is_numeric_dtype(df[key]):
                series = pd.to_numeric(df[key], errors="coerce").fillna(0)
                lo = float(series.min())
                hi = float(series.quantile(0.99)) if len(series) else float(series.max())
                if hi < lo:
                    hi = lo
                # Use integer sliders for common integer-like fields
                if key == "year":
                    lo_i = int(lo)
                    hi_i = int(hi)
                    value = st.slider(
                        f"{label}",
                        min_value=lo_i,
                        max_value=hi_i,
                        value=(lo_i, hi_i),
                        key=f"filter_{key}",
                    )
                    filter_inputs.append({"key": key, "value": value})
                else:
                    value = st.slider(
                        f"{label}",
                        min_value=float(lo),
                        max_value=float(hi),
                        value=(float(lo), float(hi)),
                        key=f"filter_{key}",
                    )
                    filter_inputs.append({"key": key, "value": value})
            else:
                options = ["(All)"] + sorted(df[key].astype("string").unique().tolist())
                value = st.selectbox(label, options, key=f"filter_{key}")
                filter_inputs.append({"key": key, "value": value})

        st.subheader("3 · Tuỳ chọn AI nâng cao")
        st.caption("Bật/tắt các gợi ý AI nâng cao nếu bạn quan tâm.")
        maintenance_focus = st.checkbox(
            "Quan tâm chi phí bảo dưỡng (AI dự đoán)",
            help="Chỉ giữ xe có chi phí bảo dưỡng dự đoán thấp hơn ngưỡng.",
            key="maintenance_focus",
        )
        maintenance_threshold = st.slider(
            "Ngưỡng chi phí tối đa (USD/năm)",
            min_value=300,
            max_value=4000,
            value=int(st.session_state.get("maintenance_threshold", 1200)),
            step=50,
            disabled=not maintenance_focus,
            key="maintenance_threshold",
        )

        repair_filter = st.checkbox(
            "Ẩn xe có nguy cơ sửa chữa lớn (AI)",
            help="Ẩn xe có rủi ro sửa chữa lớn vượt ngưỡng bạn chọn.",
            key="repair_filter",
        )
        repair_threshold = st.slider(
            "Ngưỡng rủi ro sửa chữa (%)",
            min_value=5,
            max_value=80,
            value=int(st.session_state.get("repair_threshold", 30)),
            step=5,
            format="%d%%",
            disabled=not repair_filter,
            key="repair_threshold",
        )

        st.subheader("4 · Điều gì quan trọng với bạn?")
        st.caption("Chấm điểm mức ưu tiên cho từng tiêu chí (1–9).")

        criteria_scores: dict[str, int] = {}
        for criterion in ahp_criteria:
            label = labels.get(criterion, criterion.replace("_", " ").title())
            slider_key = f"criteria_{criterion}"
            default_value = int(st.session_state.get(slider_key, DEFAULT_SCORES.get(criterion, 5)))
            criteria_scores[criterion] = int(
                st.slider(
                    f"{label} (1–9)",
                    min_value=1,
                    max_value=9,
                    value=default_value,
                    step=1,
                    key=slider_key,
                )
            )

        st.subheader("5 · Hiển thị gợi ý")
        st.caption("Chọn số lượng kết quả và mức lọc "
                   "khắt khe (giữ lại top % điểm phù hợp).")
        top_percent = st.slider(
            "Chỉ giữ lại nhóm xe phù hợp nhất (%)",
            min_value=10,
            max_value=50,
            value=int(st.session_state.get("top_percent", 30)),
            step=5,
            key="top_percent",
        )
        q = 1.0 - (float(top_percent) / 100.0)
        top_n = st.number_input(
            "Muốn xem tối đa bao nhiêu xe?",
            min_value=5,
            max_value=50,
            value=int(st.session_state.get("top_n", 10)),
            step=1,
            key="top_n",
        )

        submitted = st.form_submit_button("Tìm xe phù hợp")
    if not submitted:
        cached = st.session_state.get("recommendation_cache")
        if cached and cached.get("data_key") == data_key:
            st.info("Đang hiển thị kết quả lần chạy gần nhất. Nhấn 'Tìm xe phù hợp' để tính lại nếu bạn thay đổi điều kiện.")
            df_scored = cached["df_scored"].copy()
            df_top = cached["df_top"].copy()
            show_cols = list(cached["show_cols"])
            CR = float(cached["cr"])
            maintenance_note = str(cached.get("maintenance_note", ""))
            repair_note = str(cached.get("repair_note", ""))
        else:
            st.info("Điền bộ lọc ở thanh bên và nhấn 'Tìm xe phù hợp'.")
            st.subheader("Xem nhanh dữ liệu")
            st.dataframe(df.head(20), width="stretch")
            return

    with st.spinner("Đang tính toán điểm phù hợp và rủi ro..."):
        df_filtered = apply_filter_criteria(df, filter_inputs)
        if df_filtered.empty:
            st.warning("Không có xe nào thỏa bộ lọc hiện tại. Hãy nới lỏng điều kiện và thử lại.")
            return

        scores_vec = np.array([float(criteria_scores[c]) for c in ahp_criteria], dtype=float)
        pairwise = scores_vec[:, None] / scores_vec[None, :] if np.all(scores_vec > 0) else np.ones((len(ahp_criteria), len(ahp_criteria)), dtype=float)

        weights_raw, _, _, CR = compute_ahp_weights(pairwise)
        ahp_weights = dict(zip(ahp_criteria, weights_raw))

        artifacts = train_ai_model(df, data_key)

        df_scored = df_filtered.copy()
        df_scored["ahp_score"] = compute_ahp_score(
            df_scored,
            ahp_criteria,
            ahp_weights,
            benefit_criteria=benefit_criteria,
            cost_criteria=cost_criteria,
        )
        df_scored["risk_pred"] = artifacts.model.predict(df_scored[artifacts.features_ai])

        maintenance_note = ""
        if maintenance_focus:
            maintenance_artifacts = train_maintenance_model(df, data_key)
            df_scored["maintenance_cost_pred"] = predict_maintenance_cost(df_scored, maintenance_artifacts)
            before_count = len(df_scored)
            df_scored = df_scored[df_scored["maintenance_cost_pred"] <= float(maintenance_threshold)]
            removed = before_count - len(df_scored)
            maintenance_note = f"Áp dụng ngưỡng chi phí bảo dưỡng ≤ {maintenance_threshold:,.0f} USD/năm."
            if removed > 0:
                maintenance_note += f" Đã loại {removed} xe vượt ngưỡng."
            if df_scored.empty:
                st.warning("Không có xe nào đáp ứng ngưỡng chi phí bảo dưỡng hiện tại. Hãy tăng giới hạn và thử lại.")
                return
        else:
            df_scored["maintenance_cost_pred"] = np.nan

        repair_note = ""
        if repair_filter:
            reliability_artifacts = train_reliability_model(df, data_key)
            repair_probs = predict_major_repair_risk(df_scored, reliability_artifacts)
            repair_fraction = float(repair_threshold) / 100.0
            df_scored["repair_risk_pred"] = repair_probs * 100.0
            before_count = len(df_scored)
            df_scored = df_scored[repair_probs <= repair_fraction]
            removed = before_count - len(df_scored)
            repair_note = f"Ẩn xe có nguy cơ sửa chữa > {repair_threshold}% trong 12 tháng."
            if removed > 0:
                repair_note += f" Đã loại {removed} xe vượt ngưỡng."
            if df_scored.empty:
                st.warning("Không còn xe nào sau khi áp dụng ngưỡng rủi ro sửa chữa. Hãy tăng giới hạn và thử lại.")
                return
        else:
            df_scored["repair_risk_pred"] = np.nan

        threshold = df_scored["ahp_score"].quantile(float(q))
        df_scored["recommendation"] = np.where(
            (df_scored["ahp_score"] >= threshold) & (df_scored["risk_pred"] == 0),
            "NÊN MUA",
            "KHÔNG NÊN MUA",
        )

        ranked = df_scored[df_scored["risk_pred"] == 0]["ahp_score"].rank(method="dense", ascending=False)
        df_scored["rank"] = ranked.fillna(0).astype(int)

    total_after_filter = len(df_scored)
    recommended_count = int((df_scored["recommendation"] == "NÊN MUA").sum())

    st.subheader("Tổng quan kết quả")
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Xe phù hợp với bộ lọc", f"{total_after_filter:,}")
    with kpi2:
        st.metric("Xe được gợi ý mua", f"{recommended_count:,}")
    with kpi3:
        st.metric("Độ ổn định ưu tiên (CR)", f"{CR:.3f}")
    if CR < 0.10:
        st.caption("CR < 0.10 ⇒ bảng so sánh đủ ổn định.")
    else:
        st.warning("CR ≥ 0.10 · Hãy xem lại thang điểm ưu tiên để kết quả chính xác hơn.")
    if maintenance_note:
        st.caption(maintenance_note)
    if repair_note:
        st.caption(repair_note)

    df_scored["risk_level"] = df_scored["risk_pred"].map(RISK_TEXT).fillna("Không rõ")
    show_maintenance_col = bool(maintenance_focus and "maintenance_cost_pred" in df_scored.columns)
    show_repair_col = bool(repair_filter and "repair_risk_pred" in df_scored.columns)

    base_cols = [
        "rank",
        "manufacturer",
        "model",
        "year",
        "price",
        "mileage",
        "mpg",
        "fuel_type",
        "engine",
        "accidents_or_damage",
        "one_owner",
        "driver_rating",
        "seller_rating",
        "price_drop",
        "ahp_score",
        "risk_pred",
        "risk_level",
        "recommendation",
    ]
    if "id" in df_scored.columns:
        base_cols.insert(1, "id")

    show_cols = [c for c in base_cols if c in df_scored.columns]
    if show_maintenance_col:
        show_cols.append("maintenance_cost_pred")
    if show_repair_col:
        show_cols.append("repair_risk_pred")

    df_top = (
        df_scored[df_scored["rank"] > 0]
        .sort_values(["rank", "ahp_score"], ascending=[True, False])
        .head(int(top_n))
    )

    # Persist results to avoid recomputation on widget reruns (e.g. comparison)
    st.session_state["recommendation_cache"] = {
        "data_key": data_key,
        "df_scored": df_scored.copy(),
        "df_top": df_top.copy(),
        "show_cols": show_cols,
        "cr": float(CR),
        "maintenance_note": maintenance_note,
        "repair_note": repair_note,
    }

    tab_top, tab_all = st.tabs(["Top đề xuất", "Danh sách sau chấm điểm"])
    with tab_top:
        if len(df_top):
            st.dataframe(
                df_top[show_cols].rename(columns=DISPLAY_COLUMN_LABELS),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Không có xe nào đạt tới nhóm điểm bạn đã chọn.")
    with tab_all:
        st.dataframe(
            df_scored[show_cols].rename(columns=DISPLAY_COLUMN_LABELS),
            width="stretch",
            hide_index=True,
        )

    st.subheader("So sánh xe")
    st.caption("Chọn tối đa 4 xe và nhấn 'So sánh' để tạo bảng. Giá trị tốt hơn sẽ được tô xanh theo từng tiêu chí (benefit: cao hơn tốt hơn, cost: thấp hơn tốt hơn).")

    df_candidates = df_top if len(df_top) else df_scored
    if df_candidates.empty:
        st.info("Chưa có danh sách xe để so sánh.")
        return

    df_candidates = df_candidates.copy()
    key_col = "id" if "id" in df_candidates.columns else None
    df_candidates["__car_key"] = df_candidates[key_col].astype(str) if key_col else df_candidates.index.astype(str)

    def _car_label(row: pd.Series) -> str:
        manufacturer = str(row.get("manufacturer", ""))
        model = str(row.get("model", ""))
        year = row.get("year", "")
        price = row.get("price", None)
        rank = row.get("rank", None)

        parts: list[str] = []
        if pd.notna(rank) and int(rank) > 0:
            parts.append(f"#{int(rank)}")
        title = " ".join([p for p in [manufacturer, model] if p and p != "Unknown"]).strip()
        if title:
            parts.append(title)
        if pd.notna(year):
            try:
                parts.append(str(int(year)))
            except Exception:
                parts.append(str(year))
        if pd.notna(price):
            try:
                parts.append(f"${float(price):,.0f}")
            except Exception:
                parts.append(str(price))

        if key_col and pd.notna(row.get(key_col)):
            parts.append(f"id={row.get(key_col)}")
        return " · ".join([p for p in parts if p])

    df_candidates["__car_label"] = df_candidates.apply(_car_label, axis=1)
    label_to_key = dict(zip(df_candidates["__car_label"].tolist(), df_candidates["__car_key"].tolist()))

    with st.form("compare_form"):
        selected_labels = st.multiselect(
            "Chọn xe để so sánh",
            options=df_candidates["__car_label"].tolist(),
            default=st.session_state.get("compare_selected_labels", df_candidates["__car_label"].tolist()[:2]),
            max_selections=4,
            help="Trong form này, thay đổi lựa chọn sẽ KHÔNG tự chạy lại; chỉ chạy khi bạn bấm 'So sánh'.",
        )
        run_compare = st.form_submit_button("So sánh")

    st.session_state["compare_selected_labels"] = selected_labels
    if not run_compare:
        st.caption("Chọn xe và bấm 'So sánh' để tạo bảng (không tự chạy khi thêm/xoá lựa chọn).")
    elif len(selected_labels) < 2:
        st.info("Hãy chọn ít nhất 2 xe để bắt đầu so sánh.")
    else:
        with st.spinner("Đang tạo bảng so sánh..."):
            selected_keys = [label_to_key[lbl] for lbl in selected_labels if lbl in label_to_key]
            df_selected = df_candidates[df_candidates["__car_key"].isin(selected_keys)].copy()
            df_selected["__car_label"] = df_selected.apply(_car_label, axis=1)
            df_selected["__car_label"] = pd.Categorical(df_selected["__car_label"], categories=selected_labels, ordered=True)
            df_selected = df_selected.sort_values("__car_label")

            compare_fields = [
            "manufacturer",
            "model",
            "year",
            "price",
            "mileage",
            "mpg",
            "fuel_type",
            "engine",
            "accidents_or_damage",
            "one_owner",
            "driver_rating",
            "seller_rating",
            "price_drop",
            "maintenance_cost_pred",
            "repair_risk_pred",
            "ahp_score",
            "rank",
            "recommendation",
            "risk_level",
            ]
            compare_fields = [c for c in compare_fields if c in df_selected.columns]

            compare_df = pd.DataFrame(index=[DISPLAY_COLUMN_LABELS.get(c, c) for c in compare_fields])
            for _, row in df_selected.iterrows():
                col_name = str(row.get("__car_label"))
                values: list[object] = []
                for field in compare_fields:
                    val = row.get(field)
                    if field in {"price"} and pd.notna(val):
                        try:
                            values.append(f"${float(val):,.0f}")
                        except Exception:
                            values.append(val)
                    elif field in {"mileage", "maintenance_cost_pred"} and pd.notna(val):
                        try:
                            values.append(f"{float(val):,.0f}")
                        except Exception:
                            values.append(val)
                    elif field in {"repair_risk_pred", "price_drop"} and pd.notna(val):
                        try:
                            values.append(f"{float(val):.1f}%")
                        except Exception:
                            values.append(val)
                    elif field in {"ahp_score"} and pd.notna(val):
                        try:
                            values.append(f"{float(val):.4f}")
                        except Exception:
                            values.append(val)
                    else:
                        values.append(val)
                compare_df[col_name] = values

            # Determine comparison directions per field
            compare_direction: dict[str, str] = {}
            for c in benefit_criteria:
                compare_direction[c] = "benefit"
            for c in cost_criteria:
                compare_direction[c] = "cost"
            # Heuristics for AI/derived metrics
            compare_direction.setdefault("ahp_score", "benefit")
            compare_direction.setdefault("rank", "cost")
            compare_direction.setdefault("maintenance_cost_pred", "cost")
            compare_direction.setdefault("repair_risk_pred", "cost")

            def _numeric_series(field: str) -> pd.Series:
                series = df_selected[field]
                if field == "mpg":
                    return series.apply(parse_mpg).astype(float)
                return pd.to_numeric(series, errors="coerce")

            def _highlight_best(data: pd.DataFrame) -> pd.DataFrame:
                styles = pd.DataFrame("", index=data.index, columns=data.columns)
                for field in compare_fields:
                    direction = compare_direction.get(field)
                    if direction not in {"benefit", "cost"}:
                        continue
                    try:
                        numeric = _numeric_series(field)
                    except Exception:
                        continue
                    numeric = numeric.replace([np.inf, -np.inf], np.nan)
                    if numeric.notna().sum() == 0:
                        continue
                    best = float(numeric.max()) if direction == "benefit" else float(numeric.min())
                    winners = numeric.eq(best)
                    row_label = DISPLAY_COLUMN_LABELS.get(field, field)
                    for is_winner, col in zip(winners.tolist(), df_selected["__car_label"].astype(str).tolist()):
                        if is_winner and col in styles.columns and row_label in styles.index:
                            styles.loc[row_label, col] = "color: #118D57; font-weight: 700;"
                return styles

            st.dataframe(compare_df.style.apply(_highlight_best, axis=None), width="stretch")

    # Save history for logged-in users
    user = current_user()
    if user:
        inputs_payload = {
            "filters": filter_inputs,
            "criteria_scores": criteria_scores,
            "maintenance_focus": bool(maintenance_focus),
            "maintenance_threshold": int(maintenance_threshold),
            "repair_filter": bool(repair_filter),
            "repair_threshold": int(repair_threshold),
            "top_percent": int(top_percent),
            "top_n": int(top_n),
        }
        results_payload = {
            "top": df_top[show_cols].to_dict(orient="records"),
            "counts": {
                "after_filter": int(total_after_filter),
                "recommended": int(recommended_count),
                "cr": float(CR),
            },
        }
        save_recommendation(engine, int(user["id"]), inputs=inputs_payload, results=results_payload)


def page_history(engine) -> None:
    user = current_user()
    if not user:
        st.info("Bạn cần đăng nhập để xem lịch sử đề xuất.")
        return

    st.header("Lịch sử đề xuất")
    recs = list_recommendations(engine, int(user["id"]), limit=50)
    if not recs:
        st.caption("Chưa có lịch sử. Hãy chạy 'Gợi ý xe' để lưu lại.")
        return

    rows = [
        {
            "id": r.id,
            "created_at": str(r.created_at),
            "after_filter": r.results.get("counts", {}).get("after_filter"),
            "recommended": r.results.get("counts", {}).get("recommended"),
            "cr": r.results.get("counts", {}).get("cr"),
        }
        for r in recs
    ]
    df_hist = pd.DataFrame(rows)
    st.dataframe(df_hist, width="stretch", hide_index=True)

    pick = st.selectbox("Xem chi tiết lần chạy", options=[r.id for r in recs], format_func=lambda x: f"#{x}")
    selected = next((r for r in recs if r.id == pick), None)
    if selected:
        top = selected.results.get("top", [])
        st.subheader("Top kết quả đã lưu")
        if top:
            st.dataframe(pd.DataFrame(top), width="stretch", hide_index=True)
        else:
            st.caption("Không có dữ liệu top đã lưu.")


def page_admin_dashboard(engine) -> None:
    st.header("Admin · Dashboard")
    counts = count_rows(engine)
    c1, c2, c3 = st.columns(3)
    c1.metric("Users", counts["users"])
    c2.metric("Cars", counts["cars"])
    c3.metric("Recommendations", counts["recommendations"])

    st.subheader("Tạo tài khoản")
    with st.form("create_user"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["user", "admin"])
        submitted = st.form_submit_button("Tạo")
        if submitted:
            if not username or not password:
                st.error("Cần nhập username và password")
            elif get_user_by_username(engine, username) is not None:
                st.error("Username đã tồn tại")
            else:
                create_user(engine, username, hash_password(password), role=role)
                st.success("Đã tạo tài khoản")


def page_admin_users(engine) -> None:
    st.header("Admin · Quản lý người dùng")
    st.caption("Tạo/sửa/xoá tài khoản. Lưu ý: không nên xoá tài khoản đang đăng nhập.")

    users = list_users(engine, limit=500)
    if users:
        df_users = pd.DataFrame(
            [
                {
                    "id": u.id,
                    "username": u.username,
                    "role": u.role,
                    "created_at": str(u.created_at),
                }
                for u in users
            ]
        )
        st.dataframe(df_users, width="stretch", hide_index=True)
    else:
        st.caption("Chưa có người dùng")

    st.subheader("Tạo người dùng")
    with st.form("admin_create_user"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["user", "admin"])
        submitted = st.form_submit_button("Tạo")
        if submitted:
            if not username or not password:
                st.error("Cần nhập username và password")
            elif get_user_by_username(engine, username) is not None:
                st.error("Username đã tồn tại")
            else:
                create_user(engine, username, hash_password(password), role=role)
                st.success("Đã tạo")
                st.rerun()

    st.subheader("Cập nhật / Xoá")
    uid = st.number_input("User ID", min_value=0, value=0, step=1)
    uobj = get_user_by_id(engine, int(uid)) if int(uid) > 0 else None
    if uobj is not None:
        st.caption(f"Đang chọn: {uobj.username} ({uobj.role})")
    col1, col2 = st.columns(2)
    with col1:
        new_username = st.text_input("Username mới (bỏ trống nếu không đổi)", value="")
        new_role = st.selectbox("Role mới", ["(no change)", "user", "admin"])
        new_password = st.text_input("Mật khẩu mới (bỏ trống nếu không đổi)", type="password")
        if st.button("Cập nhật"):
            if int(uid) <= 0:
                st.error("Nhập User ID")
            else:
                try:
                    update_user(
                        engine,
                        int(uid),
                        username=new_username or None,
                        role=None if new_role == "(no change)" else new_role,
                        password_hash=None if not new_password else hash_password(new_password),
                    )
                    st.success("Đã cập nhật")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with col2:
        if st.button("Xoá", type="secondary"):
            cu = current_user()
            if int(uid) <= 0:
                st.error("Nhập User ID")
            elif cu and int(uid) == int(cu.get("id")):
                st.error("Không thể xoá tài khoản đang đăng nhập")
            else:
                try:
                    delete_user(engine, int(uid))
                    st.success("Đã xoá")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def page_admin_add_data(engine) -> None:
    st.header("Admin · Thêm dữ liệu")
    st.caption("Nhập xe vào CSDL (thay cho cars.csv).")

    st.subheader("Import từ CSV")
    csv_file = st.file_uploader("Chọn file cars.csv", type=["csv"], accept_multiple_files=False)
    skip_dups = st.checkbox("Bỏ qua dữ liệu trùng", value=True)
    if csv_file is not None:
        try:
            df_csv = load_data_from_bytes(csv_file.getvalue())
            df_csv = normalize_vehicle_dataframe(df_csv)
            st.caption(f"Đã đọc {len(df_csv):,} dòng từ CSV")
            st.dataframe(df_csv.head(30), width="stretch", hide_index=True)

            if st.button("Import vào CSDL", width="stretch"):
                with st.spinner("Đang import..."):
                    inserted, skipped = bulk_insert_cars_from_dataframe(engine, df_csv, skip_duplicates=skip_dups)
                st.success(f"Import xong. Thêm mới: {inserted:,} · Bỏ qua trùng: {skipped:,}")
                st.rerun()
        except Exception as exc:
            st.error(str(exc))

    with st.form("add_car"):
        manufacturer = st.text_input("Hãng (manufacturer)", value="")
        model = st.text_input("Mẫu xe (model)", value="")
        year = st.number_input("Năm (year)", min_value=1950, max_value=2100, value=2018, step=1)
        price = st.number_input("Giá (price)", min_value=0.0, value=15000.0, step=100.0)
        mileage = st.number_input("Số dặm (mileage)", min_value=0.0, value=50000.0, step=500.0)
        mpg = st.text_input("MPG (mpg)", value="")
        fuel_type = st.text_input("Loại nhiên liệu (fuel_type)", value="")
        engine_txt = st.text_input("Động cơ (engine)", value="")
        accidents = st.number_input("Tai nạn/Hư hại (accidents_or_damage)", min_value=0.0, value=0.0, step=1.0)
        one_owner = st.number_input("Số chủ (one_owner)", min_value=0.0, value=1.0, step=1.0)
        driver_rating = st.number_input("Điểm người lái (driver_rating)", min_value=0.0, max_value=5.0, value=4.2, step=0.1)
        seller_rating = st.number_input("Điểm người bán (seller_rating)", min_value=0.0, max_value=5.0, value=4.2, step=0.1)
        price_drop = st.number_input("Giảm giá % (price_drop)", min_value=0.0, value=0.0, step=0.1)
        submitted = st.form_submit_button("Thêm xe")
        if submitted:
            insert_car(
                engine,
                {
                    "manufacturer": manufacturer or "Unknown",
                    "model": model or "Unknown",
                    "year": int(year),
                    "price": float(price),
                    "mileage": float(mileage),
                    "mpg": mpg or "Unknown",
                    "fuel_type": fuel_type or "Unknown",
                    "engine": engine_txt or "Unknown",
                    "accidents_or_damage": float(accidents),
                    "one_owner": float(one_owner),
                    "driver_rating": float(driver_rating),
                    "seller_rating": float(seller_rating),
                    "price_drop": float(price_drop),
                },
            )
            st.success("Đã thêm xe")

    st.subheader("Danh sách xe")
    df = read_cars_df(engine)
    if df.empty:
        st.caption("Chưa có xe")
        return
    st.dataframe(df, width="stretch", hide_index=True)

    car_id = st.number_input("ID xe cần xoá", min_value=0, value=0, step=1)
    if st.button("Xoá xe", type="secondary"):
        if int(car_id) > 0:
            delete_car(engine, int(car_id))
            st.success("Đã xoá")
            st.rerun()


def page_admin_criteria(engine) -> None:
    st.header("Admin · Tiêu chí")
    st.caption("Thêm/xoá và bật/tắt tiêu chí tính AHP và tiêu chí lọc.")

    criteria = list_criteria(engine)
    if criteria:
        df = pd.DataFrame(
            [
                {
                    "id": c.id,
                    "key": c.key,
                    "label": c.label,
                    "kind": c.kind,
                    "direction": c.direction,
                    "enabled": c.enabled,
                }
                for c in criteria
            ]
        )
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        st.caption("Chưa có tiêu chí")

    st.subheader("Thêm tiêu chí")
    allowed_keys = [
        "manufacturer",
        "model",
        "year",
        "mileage",
        "price",
        "mpg",
        "fuel_type",
        "engine",
        "accidents_or_damage",
        "one_owner",
        "driver_rating",
        "seller_rating",
        "price_drop",
    ]
    with st.form("add_criterion"):
        key = st.selectbox("Cột (key)", allowed_keys)
        label = st.text_input("Nhãn hiển thị (label)", value=key)
        kind = st.selectbox("Loại (kind)", ["ahp", "filter"])
        direction = st.selectbox("Hướng (direction)", ["benefit", "cost", "none"])
        enabled = st.checkbox("Enabled", value=True)
        submitted = st.form_submit_button("Thêm")
        if submitted:
            try:
                create_criterion(engine, key=key, label=label, kind=kind, direction=direction, enabled=enabled)
                st.success("Đã thêm tiêu chí")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.subheader("Cập nhật / Xoá")
    cid = st.number_input("Criterion ID", min_value=0, value=0, step=1)
    col1, col2 = st.columns(2)
    with col1:
        new_label = st.text_input("Label mới (bỏ trống nếu không đổi)", value="")
        new_direction = st.selectbox("Direction", ["(no change)", "benefit", "cost", "none"])
        new_enabled = st.selectbox("Enabled", ["(no change)", "true", "false"])
        if st.button("Cập nhật"):
            if int(cid) <= 0:
                st.error("Nhập criterion id")
            else:
                update_criterion(
                    engine,
                    int(cid),
                    label=new_label or None,
                    direction=None if new_direction == "(no change)" else new_direction,
                    enabled=None if new_enabled == "(no change)" else (new_enabled == "true"),
                )
                st.success("Đã cập nhật")
                st.rerun()
    with col2:
        if st.button("Xoá", type="secondary"):
            if int(cid) <= 0:
                st.error("Nhập criterion id")
            else:
                delete_criterion(engine, int(cid))
                st.success("Đã xoá")
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Hệ thống hỗ trợ mua ô tô cũ", layout="wide")
    st.title("Hệ thống hỗ trợ mua ô tô cũ (AHP + AI)")
    st.caption("Dữ liệu lấy từ PostgreSQL. Khách vẫn dùng bình thường; user đăng nhập có lịch sử; admin có dashboard/quản trị.")

    try:
        engine = get_engine()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    ok, msg = healthcheck(engine)
    if not ok:
        st.error(f"Không kết nối được PostgreSQL: {msg}")
        st.stop()

    sidebar_auth(engine)

    # Top navigation (horizontal)
    pages = ["Gợi ý xe"]
    if current_user():
        pages.append("Lịch sử")
    if is_admin():
        pages.extend(["Dashboard", "Người dùng", "Thêm dữ liệu", "Tiêu chí"])

    page = st.radio("Điều hướng", pages, horizontal=True, label_visibility="collapsed")
    st.divider()
    if page == "Gợi ý xe":
        page_recommend(engine)
    elif page == "Lịch sử":
        page_history(engine)
    elif page == "Dashboard":
        page_admin_dashboard(engine)
    elif page == "Người dùng":
        page_admin_users(engine)
    elif page == "Thêm dữ liệu":
        page_admin_add_data(engine)
    elif page == "Tiêu chí":
        page_admin_criteria(engine)


if __name__ == "__main__":
    main()
