import os
from datetime import date
import mysql.connector
import pandas as pd
import streamlit as st

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "helmetwatch",
}

def get_review_status(ocr_mode: str, status: str, review_reason: str) -> str:
    """
    Reviewer-friendly label using status + reasons + OCR mode.
    """
    if status == "CONFIRMED":
        if ocr_mode == "strict":
            return "✅ CONFIRMED (strict)"
        return "✅ CONFIRMED"
    # REVIEW
    if ocr_mode == "non_strict_fallback":
        return "⚠️ REVIEW (fallback)"
    if ocr_mode == "none":
        return "❌ REVIEW (no plate)"
    if review_reason:
        return "⚠️ REVIEW"
    return "⚠️ REVIEW"


@st.cache_data(ttl=10)
def load_violations() -> pd.DataFrame:
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT * FROM violations ORDER BY id DESC", conn)
    conn.close()

    if df.empty:
        return df

    if "detection_time" in df.columns:
        df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")

    # Ensure columns exist even if DB was old
    if "ocr_mode" not in df.columns:
        df["ocr_mode"] = "unknown"
    if "status" not in df.columns:
        df["status"] = "CONFIRMED"
    if "review_reason" not in df.columns:
        df["review_reason"] = None

    df["review_status"] = df.apply(
        lambda r: get_review_status(
            str(r.get("ocr_mode", "unknown")),
            str(r.get("status", "CONFIRMED")),
            None if pd.isna(r.get("review_reason")) else str(r.get("review_reason"))
        ),
        axis=1
    )

    # Drop noisy raw JSON
    if "api_raw" in df.columns:
        df = df.drop(columns=["api_raw"])

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    st.sidebar.header("Filters")

    # Date range filter
    if "detection_time" in df.columns:
        min_date = df["detection_time"].dt.date.min()
        max_date = df["detection_time"].dt.date.max()

        start_date, end_date = st.sidebar.date_input(
            "Detection date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(start_date, date) and isinstance(end_date, date):
            mask = (df["detection_time"].dt.date >= start_date) & (
                df["detection_time"].dt.date <= end_date
            )
            df = df.loc[mask]

    # Plate filter
    plate_filter = st.sidebar.text_input("Filter by plate (contains)")
    if plate_filter and "plate_number" in df.columns:
        df = df[df["plate_number"].astype(str).str.contains(plate_filter.upper(), na=False)]

    # Helmet status filter
    if "helmet_status" in df.columns:
        statuses = sorted(df["helmet_status"].dropna().unique().tolist())
        selected_statuses = st.sidebar.multiselect("Helmet status", statuses, default=statuses)
        if selected_statuses:
            df = df[df["helmet_status"].isin(selected_statuses)]

    # Review filter
    if "status" in df.columns:
        st.sidebar.subheader("Review")
        show_review_only = st.sidebar.checkbox("Show only REVIEW", value=False)
        if show_review_only:
            df = df[df["status"] == "REVIEW"]

    return df


def fmt_float(x, decimals=3):
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{decimals}f}"
    except:
        return "—"


def main():
    st.set_page_config(page_title="HelmetWatch Dashboard", page_icon="📊", layout="wide")
    st.title("📊 HelmetWatch – Violation Dashboard")

    df = load_violations()
    if df.empty:
        st.info("No violations recorded yet. Run the detection script to log some.")
        return

    df_filtered = apply_filters(df)

    total_violations = len(df_filtered)
    unique_plates = df_filtered["plate_number"].nunique() if "plate_number" in df_filtered.columns else 0
    avg_conf = df_filtered["helmet_confidence"].mean() if "helmet_confidence" in df_filtered.columns else 0.0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Violations (filtered)", total_violations)
    kpi2.metric("Unique Plates", unique_plates)
    kpi3.metric("Avg Helmet Confidence", f"{avg_conf:.2f}" if avg_conf else "—")

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Violations Over Time")
        if "detection_time" in df_filtered.columns:
            df_time = df_filtered.copy()
            df_time["date"] = df_time["detection_time"].dt.date
            violations_per_day = df_time.groupby("date").size()
            st.bar_chart(violations_per_day)

        st.subheader("All Violations (filtered)")
        preferred_cols = [
            "id",
            "image_name",
            "image_path",
            "detection_time",
            "helmet_status",
            "helmet_confidence",
            "plate_number",
            "plate_score",
            "iou_with_vehicle",
            "ocr_mode",
            "status",
            "review_reason",
            "review_status",
        ]
        cols = [c for c in preferred_cols if c in df_filtered.columns]
        st.dataframe(df_filtered[cols], use_container_width=True)

        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    with right:
        st.subheader("Selected Violation")

        df_display = df_filtered.copy()
        df_display["label"] = df_display.apply(
            lambda row: f"#{row['id']} – {row.get('plate_number','?')} – {row['detection_time']:%Y-%m-%d %H:%M:%S}",
            axis=1,
        )

        selected_label = st.selectbox("Choose a record", df_display["label"].tolist())
        selected_row = df_display[df_display["label"] == selected_label].iloc[0]

        st.write("**Image name:**", selected_row.get("image_name"))
        st.write("**Image path:**", selected_row.get("image_path"))
        st.write("**Detection time:**", selected_row.get("detection_time"))
        st.write("**Helmet status:**", selected_row.get("helmet_status"))
        st.write("**Helmet confidence:**", fmt_float(selected_row.get("helmet_confidence"), 3))
        st.write("**Plate number:**", selected_row.get("plate_number"))
        st.write("**Plate score:**", fmt_float(selected_row.get("plate_score"), 3))
        st.write("**IoU with vehicle:**", fmt_float(selected_row.get("iou_with_vehicle"), 3))
        st.write("**OCR mode:**", selected_row.get("ocr_mode"))
        st.write("**Status:**", selected_row.get("status"))
        st.write("**Review reason:**", selected_row.get("review_reason") if pd.notna(selected_row.get("review_reason")) else "—")
        st.write("**Review label:**", selected_row.get("review_status"))

        img_path = selected_row.get("image_path")
        if isinstance(img_path, str) and os.path.exists(img_path):
            st.image(img_path, caption=str(selected_row.get("plate_number")), use_container_width=True)
        else:
            st.warning(f"Image not found at path: {img_path}")


if __name__ == "__main__":
    main()
