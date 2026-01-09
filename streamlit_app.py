from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.ahp import compute_ahp_score, compute_ahp_weights
from app.constants import (
    AHP_CRITERIA,
    CRITERIA_LABELS,
    DATA_DEFAULT_PATH,
    DEFAULT_SCORES,
    DISPLAY_COLUMN_LABELS,
    PRESET_SCORES,
    RISK_TEXT,
)
from app.data import add_risk_label, apply_filters, dataset_signature, load_data_from_path
from app.maintenance import predict_maintenance_cost, train_maintenance_model
from app.model import train_ai_model
from app.reliability import predict_major_repair_risk, train_reliability_model


def main() -> None:
    st.set_page_config(page_title="Hệ thống hỗ trợ mua ô tô cũ", layout="wide")

    st.title("Hệ thống hỗ trợ mua ô tô cũ (AHP + AI)")
    st.caption("Trả lời nhanh nhu cầu của bạn, hệ thống sẽ phân tích dữ liệu xe cũ có sẵn và đề xuất danh sách phù hợp.")

    if not DATA_DEFAULT_PATH.exists():
        st.error("Không tìm thấy cars.csv trong thư mục dự án")
        st.stop()

    df_raw = load_data_from_path(str(DATA_DEFAULT_PATH))
    df = add_risk_label(df_raw)
    data_key = dataset_signature(df)

    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = next(iter(PRESET_SCORES.keys()))

    if "criteria_initialized" not in st.session_state:
        for criterion in AHP_CRITERIA:
            st.session_state[f"criteria_{criterion}"] = int(DEFAULT_SCORES.get(criterion, 5))
        st.session_state.criteria_initialized = True

    def _apply_preset_to_sliders() -> None:
        preset_name = st.session_state.get("selected_preset")
        preset_values = PRESET_SCORES.get(preset_name, DEFAULT_SCORES)
        for criterion, value in preset_values.items():
            st.session_state[f"criteria_{criterion}"] = int(value)

    sidebar = st.sidebar
    sidebar.header("1. Mục tiêu sử dụng")
    preset_options = list(PRESET_SCORES.keys())
    sidebar.radio(
        "Bạn đang tìm chiếc xe cũ kiểu nào?",
        preset_options,
        key="selected_preset",
        help="Chọn phong cách lái xe để hệ thống tự động điều chỉnh trọng số.",
        on_change=_apply_preset_to_sliders,
    )

    sidebar.caption("Hệ thống tự áp dụng trọng số ngay khi bạn chọn phong cách; vẫn có thể tinh chỉnh thủ công ở bước 3.")

    maintenance_focus = False
    maintenance_threshold = None

    sidebar.subheader("2 · Điều kiện thực tế")
    sidebar.caption("Điền ngân sách, năm sản xuất và thông số cơ bản bạn mong muốn.")

    manufacturer_values = ["(All)"]
    if "manufacturer" in df.columns:
        manufacturer_values += sorted(df["manufacturer"].astype("string").unique().tolist())
    manufacturer = sidebar.selectbox("Hãng xe", manufacturer_values)

    if "year" in df.columns and len(df):
        y_min, y_max = int(df["year"].min()), int(df["year"].max())
    else:
        y_min, y_max = 1990, 2026
    year_range = sidebar.slider(
        "Khoảng năm sản xuất",
        min_value=y_min,
        max_value=y_max,
        value=(y_min, y_max),
        format="%d",
    )

    if "price" in df.columns and len(df):
        p_min = float(df["price"].min())
        p_max = float(df["price"].quantile(0.99))
    else:
        p_min, p_max = 0.0, 100000.0
    price_range = sidebar.slider(
        "Khoảng giá (USD)",
        min_value=float(p_min),
        max_value=float(p_max),
        value=(float(p_min), float(p_max)),
        format="$%.0f",
    )

    if "mileage" in df.columns and len(df):
        m_max_default = float(df["mileage"].quantile(0.99))
    else:
        m_max_default = 200000.0
    max_mileage = sidebar.slider(
        "Số dặm tối đa (mile)",
        min_value=0.0,
        max_value=float(max(m_max_default, 1.0)),
        value=float(m_max_default),
        format="%.0f mi",
    )

    sidebar.subheader("3 · Tuỳ chọn AI nâng cao")
    maintenance_focus = sidebar.checkbox(
        "Quan tâm chi phí bảo dưỡng (AI dự đoán)",
        help="Bật để chỉ giữ những xe có chi phí bảo dưỡng ước tính thấp hơn ngưỡng bạn chọn.",
        key="maintenance_focus",
    )
    maintenance_threshold = sidebar.slider(
        "Ngưỡng chi phí tối đa (USD/năm)",
        min_value=300,
        max_value=4000,
        value=int(st.session_state.get("maintenance_threshold", 1200)),
        step=50,
        disabled=not maintenance_focus,
        key="maintenance_threshold",
    )

    repair_filter = sidebar.checkbox(
        "Ẩn xe có nguy cơ sửa chữa lớn (AI)",
        help="AI dự đoán xác suất phải sửa chữa lớn trong 12 tháng tới.",
        key="repair_filter",
    )
    repair_threshold = sidebar.slider(
        "Ngưỡng rủi ro sửa chữa (%)",
        min_value=5,
        max_value=80,
        value=int(st.session_state.get("repair_threshold", 30)),
        step=5,
        format="%d%%",
        disabled=not repair_filter,
        key="repair_threshold",
    )

    sidebar.subheader("4 · Điều gì quan trọng với bạn?")
    sidebar.caption("Các trọng số đã được điều chỉnh nhờ preset, bạn vẫn có thể kéo lại từng tiêu chí.")

    criteria_scores: dict[str, int] = {}
    for criterion in AHP_CRITERIA:
        label = CRITERIA_LABELS.get(criterion, criterion.replace("_", " ").title())
        slider_key = f"criteria_{criterion}"
        default_value = int(st.session_state.get(slider_key, DEFAULT_SCORES.get(criterion, 5)))
        criteria_scores[criterion] = int(
            sidebar.slider(
                f"{label} (1–9)",
                min_value=1,
                max_value=9,
                value=default_value,
                step=1,
                key=slider_key,
            )
        )

    sidebar.subheader("5 · Hiển thị gợi ý")
    sidebar.caption("Chọn bạn muốn xem bao nhiêu xe và mức sàng lọc khắt khe tới đâu.")
    top_percent = sidebar.slider(
        "Chỉ giữ lại nhóm xe phù hợp nhất (%)",
        min_value=10,
        max_value=50,
        value=int(st.session_state.get("top_percent", 30)),
        step=5,
        help="Ví dụ 30% nghĩa là chỉ giữ nhóm xe có điểm phù hợp cao nhất 30%.",
        key="top_percent",
    )
    q = 1.0 - (float(top_percent) / 100.0)
    top_n = sidebar.number_input(
        "Muốn xem tối đa bao nhiêu xe?",
        min_value=5,
        max_value=50,
        value=int(st.session_state.get("top_n", 10)),
        step=1,
        key="top_n",
    )

    submitted = sidebar.button("Tìm xe phù hợp", use_container_width=True)

    if not submitted:
        st.info("Hoàn tất các bước ở thanh bên rồi nhấn 'Tìm xe phù hợp' để nhận gợi ý cá nhân hoá.")
        st.subheader("Xem nhanh dữ liệu")
        st.dataframe(df.head(20), use_container_width=True)
        return

    with st.spinner("Đang tính toán điểm phù hợp và rủi ro..."):
        df_filtered = apply_filters(df, manufacturer, year_range, price_range, max_mileage)

        if df_filtered.empty:
            st.warning("Không có xe nào thỏa bộ lọc hiện tại. Hãy nới lỏng điều kiện và thử lại.")
            st.stop()

        scores_vec = np.array([float(criteria_scores[c]) for c in AHP_CRITERIA], dtype=float)
        if np.all(scores_vec > 0):
            pairwise = scores_vec[:, None] / scores_vec[None, :]
        else:
            pairwise = np.ones((len(AHP_CRITERIA), len(AHP_CRITERIA)), dtype=float)

        weights_raw, _, _, CR = compute_ahp_weights(pairwise)
        ahp_weights = dict(zip(AHP_CRITERIA, weights_raw))

        try:
            artifacts = train_ai_model(df)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        df_scored = df_filtered.copy()
        df_scored["ahp_score"] = compute_ahp_score(df_scored, AHP_CRITERIA, ahp_weights)

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
                st.stop()
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
                st.stop()
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

    tab_top, tab_all = st.tabs(["Top đề xuất", "Danh sách sau chấm điểm"])
    column_config = {
        "price": st.column_config.NumberColumn("Giá (USD)", help="Giá niêm yết", format="$%0.0f"),
        "mileage": st.column_config.NumberColumn("Số dặm", format="%0.0f mi"),
        "mpg": st.column_config.TextColumn("Tiết kiệm nhiên liệu"),
        "fuel_type": st.column_config.TextColumn("Loại nhiên liệu"),
        "engine": st.column_config.TextColumn("Động cơ"),
        "accidents_or_damage": st.column_config.NumberColumn("Tai nạn/Hư hại", format="%0.0f"),
        "one_owner": st.column_config.NumberColumn("Số chủ sở hữu", format="%0.0f"),
        "driver_rating": st.column_config.NumberColumn("Điểm người lái", format="%0.1f"),
        "seller_rating": st.column_config.NumberColumn("Điểm người bán", format="%0.1f"),
        "price_drop": st.column_config.NumberColumn("Giảm giá (%)", format="%0.1f%%"),
        "ahp_score": st.column_config.NumberColumn("Điểm phù hợp", format="%0.2f"),
        "risk_pred": st.column_config.NumberColumn("Rủi ro (AI)", help="0 = Thấp, 1 = Cao"),
        "risk_level": st.column_config.TextColumn("Đánh giá rủi ro"),
        "maintenance_cost_pred": st.column_config.NumberColumn(
            "Ước tính bảo dưỡng (USD/năm)",
            format="$%0.0f",
            help="AI ước lượng chi phí bảo dưỡng mỗi năm",
        ),
        "repair_risk_pred": st.column_config.NumberColumn(
            "Nguy cơ sửa chữa (%)",
            format="%0.0f%%",
            help="Xác suất phải sửa chữa lớn trong 12 tháng",
        ),
    }

    with tab_top:
        st.caption("Những xe điểm phù hợp cao và bị AI đánh giá rủi ro thấp.")
        if len(df_top):
            st.dataframe(
                df_top[show_cols].rename(columns=DISPLAY_COLUMN_LABELS),
                use_container_width=True,
                column_config=column_config,
                hide_index=True,
            )
        else:
            st.info("Không có xe nào đạt tới nhóm điểm bạn đã chọn.")
    with tab_all:
        st.caption("Toàn bộ xe sau lọc, kèm điểm phù hợp và dự đoán rủi ro.")
        st.dataframe(
            df_scored[show_cols].rename(columns=DISPLAY_COLUMN_LABELS),
            use_container_width=True,
            column_config=column_config,
            hide_index=True,
        )

    csv_out = df_scored.to_csv(index=False).encode("utf-8")
    st.download_button("Tải danh sách đầy đủ (CSV)", data=csv_out, file_name="dss_result.csv", mime="text/csv")

    st.subheader("Phân tích chi tiết")
    col_ai, col_ahp = st.columns(2)

    with col_ai:
        st.write("Mô hình RandomForest AI dự đoán khả năng rủi ro của từng xe.")
        st.metric("Độ chính xác trên tập kiểm thử", f"{artifacts.accuracy:.3f}")
        st.write("Tầm quan trọng của từng thuộc tính")
        st.bar_chart(artifacts.feature_importance.set_index("feature")["importance"])

    with col_ahp:
        st.write("Các tiêu chí được bạn ưu tiên nhiều nhất")
        top_weights = pd.Series(ahp_weights).sort_values(ascending=False).head(3)
        if not top_weights.empty:
            bullet_lines = [
                f"- **{CRITERIA_LABELS.get(name, name)}** · {weight:.1%}"
                for name, weight in top_weights.items()
            ]
            st.markdown("\n".join(bullet_lines))
        st.caption("Điểm phù hợp (AHP) được chuẩn hóa 0–1 và kết hợp với dự đoán rủi ro để đưa ra đề xuất.")


if __name__ == "__main__":
    main()
