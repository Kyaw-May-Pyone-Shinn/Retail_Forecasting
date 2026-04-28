# =========================================================
# RETAIL DECISION INTELLIGENCE SYSTEM
# Balanced + Deeper Analytics + Store-Level XGBoost Forecast
# =========================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Retail Decision Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = r"C:/Users/kmpsh/Downloads/Msc/Projects/Retail_Forecasting_ML/data/processed/forecast_dataset.csv"
MODEL_PATH = r"C:/Users/kmpsh/Downloads/Msc/Projects/Retail_Forecasting_ML/model/xgb_sales_forecast_model.pkl"

MODEL_FEATURES = [
    "Store", "DayOfWeek", "Customers", "Open", "Promo", "StateHoliday",
    "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
    "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval", "year",
    "month", "day", "weekofyear", "dayofweek", "lag_1", "lab_7",
    "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_30", "lag_7"
]

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.5rem;
        max-width: 1500px;
    }
    .insight-box {
        padding: 14px 16px;
        border-radius: 12px;
        margin-top: 8px;
        margin-bottom: 12px;
        border: 1px solid #d1d5db;
    }
    .insight-green {
        background-color: rgba(16,185,129,0.10);
        border: 1px solid rgba(16,185,129,0.35);
    }
    .insight-amber {
        background-color: rgba(245,158,11,0.10);
        border: 1px solid rgba(245,158,11,0.35);
    }
    .insight-red {
        background-color: rgba(239,68,68,0.10);
        border: 1px solid rgba(239,68,68,0.35);
    }
    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HELPERS
# =========================================================
def safe_divide(a, b):
    if b is None or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def fmt_money(x):
    return f"€{x:,.0f}" if pd.notna(x) else "N/A"


def fmt_money2(x):
    return f"€{x:,.2f}" if pd.notna(x) else "N/A"


def fmt_pct(x):
    return f"{x:.2f}%" if pd.notna(x) else "N/A"


def fmt_num(x):
    return f"{x:,.0f}" if pd.notna(x) else "N/A"


def normalize_series(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
    min_val = s.min()
    max_val = s.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_val) / (max_val - min_val)


def inverse_normalize_series(s):
    return 1 - normalize_series(s)


def get_period_comparison(df_in):
    if df_in.empty:
        return np.nan, np.nan, np.nan

    max_date = df_in["Date"].max()
    min_date = df_in["Date"].min()
    total_days = max((max_date - min_date).days + 1, 1)
    window = min(30, total_days)

    current_end = max_date
    current_start = current_end - pd.Timedelta(days=window - 1)
    previous_end = current_start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=window - 1)

    current_df = df_in[(df_in["Date"] >= current_start) & (df_in["Date"] <= current_end)]
    previous_df = df_in[(df_in["Date"] >= previous_start) & (df_in["Date"] <= previous_end)]

    current_sales = current_df["Sales"].sum()
    previous_sales = previous_df["Sales"].sum()
    growth = safe_divide(current_sales - previous_sales, previous_sales) * 100

    return current_sales, previous_sales, growth


def compute_promo_uplift(data):
    if "Promo" not in data.columns or data.empty:
        return np.nan

    promo_mean = data.loc[data["Promo"] == 1, "Sales"].mean()
    non_mean = data.loc[data["Promo"] == 0, "Sales"].mean()

    if pd.isna(promo_mean) or pd.isna(non_mean) or non_mean == 0:
        return np.nan

    return ((promo_mean - non_mean) / non_mean) * 100


def infer_store_type_name(x):
    if pd.isna(x):
        return "Unknown"
    mapping_num = {0: "Type A", 1: "Type B", 2: "Type C", 3: "Type D"}
    mapping_str = {"a": "Type A", "b": "Type B", "c": "Type C", "d": "Type D"}
    if isinstance(x, str):
        return mapping_str.get(x.strip().lower(), x)
    try:
        return mapping_num.get(int(float(x)), str(x))
    except Exception:
        return str(x)


def infer_assortment_name(x):
    if pd.isna(x):
        return "Unknown"
    mapping_num = {0: "Basic", 1: "Extra", 2: "Extended"}
    mapping_str = {"a": "Basic", "b": "Extra", "c": "Extended"}
    if isinstance(x, str):
        return mapping_str.get(x.strip().lower(), x)
    try:
        return mapping_num.get(int(float(x)), str(x))
    except Exception:
        return str(x)


def priority_from_risk(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 65:
        return "P1"
    if score >= 40:
        return "P2"
    return "P3"


def risk_label(priority):
    if priority == "P1":
        return "Intervention Needed"
    if priority == "P2":
        return "Watchlist"
    if priority == "P3":
        return "Stable"
    return "Unknown"


def css_from_priority(priority):
    if priority == "P1":
        return "insight-red"
    if priority == "P2":
        return "insight-amber"
    return "insight-green"


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns:
        st.error("Dataset must contain a Date column.")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    numeric_cols = [
        "Store", "DayOfWeek", "Customers", "Open", "Promo", "StateHoliday",
        "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
        "Promo2SinceWeek", "Promo2SinceYear", "year", "month", "day",
        "weekofyear", "dayofweek", "lag_1", "lab_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_30", "lag_7", "Sales"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Open" in df.columns:
        df = df[df["Open"] == 1].copy()

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1

    df["Weekday"] = df["Date"].dt.day_name()
    df["MonthName"] = df["Date"].dt.month_name()

    if "StoreType" in df.columns:
        df["StoreTypeLabel"] = df["StoreType"].apply(infer_store_type_name)
    else:
        df["StoreTypeLabel"] = "Unknown"

    if "Assortment" in df.columns:
        df["AssortmentLabel"] = df["Assortment"].apply(infer_assortment_name)
    else:
        df["AssortmentLabel"] = "Unknown"

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    if "lag_1" not in df.columns:
        df["lag_1"] = df.groupby("Store")["Sales"].shift(1)
    if "lag_7" not in df.columns:
        df["lag_7"] = df.groupby("Store")["Sales"].shift(7)
    if "lab_7" not in df.columns:
        df["lab_7"] = df["lag_7"]
    if "lag_14" not in df.columns:
        df["lag_14"] = df.groupby("Store")["Sales"].shift(14)
    if "lag_28" not in df.columns:
        df["lag_28"] = df.groupby("Store")["Sales"].shift(28)
    if "rolling_mean_7" not in df.columns:
        df["rolling_mean_7"] = (
            df.groupby("Store")["Sales"]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )
    if "rolling_mean_30" not in df.columns:
        df["rolling_mean_30"] = (
            df.groupby("Store")["Sales"]
            .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
        )

    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    for col in MODEL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


df = load_data()

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        return e


model_obj = load_model()
model = None if isinstance(model_obj, Exception) else model_obj
model_error = model_obj if isinstance(model_obj, Exception) else None

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Retail Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Store Intelligence",
        "Forecast & Scenarios",
        "Business Chatbot"
    ]
)

stores = ["All"] + sorted(df["Store"].dropna().unique().tolist())
selected_store = st.sidebar.selectbox("Store", stores)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = pd.to_datetime(min_date), pd.to_datetime(max_date)

if model is not None:
    st.sidebar.success("Model loaded")
else:
    st.sidebar.error("Model not loaded")

# =========================================================
# FILTER
# =========================================================
def apply_filters(data):
    d = data.copy()
    d = d[(d["Date"] >= start_date) & (d["Date"] <= end_date)]

    if selected_store != "All":
        d = d[d["Store"] == selected_store]

    return d


df_filtered = apply_filters(df)

if df_filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# =========================================================
# FORECASTING
# =========================================================
def model_forecast(data, days=30):
    if model is None or data.empty:
        return None

    df_sorted = data.sort_values(["Store", "Date"]).copy()
    forecasts = []

    for store_id, store_df in df_sorted.groupby("Store"):
        store_df = store_df.sort_values("Date").copy()

        last_row = store_df.iloc[-1:].copy()
        history_sales = store_df["Sales"].astype(float).tolist()

        if len(history_sales) == 0:
            continue

        future_dates = pd.date_range(
            store_df["Date"].max() + pd.Timedelta(days=1),
            periods=days,
            freq="D"
        )

        store_preds = []

        for d in future_dates:
            row = last_row.copy()

            row["year"] = d.year
            row["month"] = d.month
            row["day"] = d.day
            row["weekofyear"] = int(d.isocalendar().week)
            row["dayofweek"] = d.dayofweek
            row["DayOfWeek"] = d.dayofweek + 1

            row["lag_1"] = history_sales[-1]
            row["lag_7"] = history_sales[-7] if len(history_sales) >= 7 else history_sales[-1]
            row["lab_7"] = row["lag_7"]
            row["lag_14"] = history_sales[-14] if len(history_sales) >= 14 else row["lag_7"].iloc[0]
            row["lag_28"] = history_sales[-28] if len(history_sales) >= 28 else row["lag_14"].iloc[0]

            row["rolling_mean_7"] = float(np.mean(history_sales[-7:]))
            row["rolling_mean_30"] = float(np.mean(history_sales[-30:]))

            X = row[MODEL_FEATURES].copy()
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

            pred = float(model.predict(X)[0])
            pred = max(pred, 0)

            recent_median = np.nanmedian(history_sales[-30:])
            upper_limit = max(recent_median * 2.5, history_sales[-1] * 1.5, 100)
            pred = min(pred, upper_limit)

            store_preds.append(pred)
            history_sales.append(pred)

            row["Sales"] = pred
            last_row = row

        forecasts.append(pd.DataFrame({
            "Date": future_dates,
            "Store": store_id,
            "Forecast": store_preds
        }))

    if not forecasts:
        return None

    return pd.concat(forecasts, ignore_index=True)


def aggregate_forecast(forecast_df):
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame(), np.nan

    daily_forecast = (
        forecast_df
        .groupby("Date", as_index=False)["Forecast"]
        .sum()
    )

    total_forecast = daily_forecast["Forecast"].sum()
    return daily_forecast, total_forecast


# =========================================================
# STORE SCORING
# =========================================================
def build_store_scores(data, forecast_df=None):
    if data.empty:
        return pd.DataFrame()

    rows = []

    forecast_by_store = pd.DataFrame()
    if forecast_df is not None and not forecast_df.empty:
        forecast_by_store = (
            forecast_df
            .groupby("Store", as_index=False)["Forecast"]
            .sum()
            .rename(columns={"Forecast": "ForecastTotal"})
        )

    for store_id, g in data.groupby("Store"):
        g = g.sort_values("Date").copy()

        total_sales = g["Sales"].sum()
        customers = g["Customers"].sum() if "Customers" in g.columns else np.nan
        sales_per_customer = safe_divide(total_sales, customers)

        current_sales, previous_sales, growth = get_period_comparison(g)
        promo_uplift = compute_promo_uplift(g)
        volatility = g["Sales"].std()

        forecast_total = np.nan
        if not forecast_by_store.empty:
            match = forecast_by_store[forecast_by_store["Store"] == store_id]
            if not match.empty:
                forecast_total = match["ForecastTotal"].iloc[0]

        forecast_growth = safe_divide(forecast_total - current_sales, current_sales) * 100

        rows.append({
            "Store": store_id,
            "Sales": total_sales,
            "Customers": customers,
            "SalesPerCustomer": sales_per_customer,
            "Growth": growth if pd.notna(growth) else 0,
            "PromoUplift": promo_uplift if pd.notna(promo_uplift) else 0,
            "Volatility": volatility if pd.notna(volatility) else 0,
            "ForecastTotal": forecast_total,
            "ForecastGrowth": forecast_growth if pd.notna(forecast_growth) else 0,
            "StoreType": g["StoreTypeLabel"].mode().iloc[0] if "StoreTypeLabel" in g.columns else "Unknown",
            "Assortment": g["AssortmentLabel"].mode().iloc[0] if "AssortmentLabel" in g.columns else "Unknown"
        })

    scores = pd.DataFrame(rows)

    if scores.empty:
        return scores

    scores["SalesScore"] = normalize_series(scores["Sales"])
    scores["GrowthScore"] = normalize_series(scores["Growth"])
    scores["PromoScore"] = normalize_series(scores["PromoUplift"])
    scores["CustomerScore"] = normalize_series(scores["Customers"])

    scores["PerformanceScore"] = (
        0.40 * scores["SalesScore"] +
        0.25 * scores["GrowthScore"] +
        0.20 * scores["PromoScore"] +
        0.15 * scores["CustomerScore"]
    ) * 100

    scores["NegativeGrowthRisk"] = inverse_normalize_series(scores["Growth"])
    scores["ForecastDeclineRisk"] = inverse_normalize_series(scores["ForecastGrowth"])
    scores["WeakPromoRisk"] = inverse_normalize_series(scores["PromoUplift"])
    scores["VolatilityRisk"] = normalize_series(scores["Volatility"])

    scores["RiskScore"] = (
        0.35 * scores["NegativeGrowthRisk"] +
        0.30 * scores["ForecastDeclineRisk"] +
        0.20 * scores["WeakPromoRisk"] +
        0.15 * scores["VolatilityRisk"]
    ) * 100

    scores["PerformanceScore"] = scores["PerformanceScore"].round(1)
    scores["RiskScore"] = scores["RiskScore"].round(1)
    scores["Priority"] = scores["RiskScore"].apply(priority_from_risk)
    scores["RiskLevel"] = scores["Priority"].apply(risk_label)

    def recommend(row):
        if row["Priority"] == "P1" and row["PromoUplift"] > 5:
            return "Run targeted promotion and monitor recovery."
        if row["Priority"] == "P1":
            return "Review pricing, assortment, staffing, and local execution."
        if row["Priority"] == "P2":
            return "Place on watchlist and compare against peer stores."
        return "Maintain strategy and protect margin."

    scores["RecommendedAction"] = scores.apply(recommend, axis=1)

    return scores.sort_values(["RiskScore", "PerformanceScore"], ascending=[False, False])


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================
def executive_page():
    st.title("Retail Decision Intelligence System")
    st.caption("Executive view of sales performance, forecast direction, promotion effectiveness, store ranking, and risk signals.")

    total_sales = df_filtered["Sales"].sum()
    customers = df_filtered["Customers"].sum() if "Customers" in df_filtered.columns else np.nan
    avg_daily_sales = df_filtered.groupby("Date")["Sales"].sum().mean()
    sales_per_customer = safe_divide(total_sales, customers)

    current_sales, previous_sales, growth_pct = get_period_comparison(df_filtered)
    promo_uplift = compute_promo_uplift(df_filtered)

    forecast_df = model_forecast(df_filtered, 30)
    daily_forecast, forecast_total = aggregate_forecast(forecast_df)
    forecast_growth = safe_divide(forecast_total - current_sales, current_sales) * 100

    store_scores = build_store_scores(df_filtered, forecast_df)
    p1_count = int((store_scores["Priority"] == "P1").sum()) if not store_scores.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", fmt_money(total_sales), fmt_pct(growth_pct))
    c2.metric("Forecast (30d)", fmt_money(forecast_total), fmt_pct(forecast_growth))
    c3.metric("Promo Uplift", fmt_pct(promo_uplift))
    c4.metric("P1 Risk Stores", fmt_num(p1_count))

    c5, c6, c7 = st.columns(3)
    c5.metric("Customers", fmt_num(customers))
    c6.metric("Average Daily Sales", fmt_money(avg_daily_sales))
    c7.metric("Sales per Customer", fmt_money2(sales_per_customer))

    actual = df_filtered.groupby("Date", as_index=False)["Sales"].sum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual["Date"], y=actual["Sales"],
        mode="lines", name="Actual Sales",
        line=dict(width=2)
    ))

    if not daily_forecast.empty:
        fig.add_trace(go.Scatter(
            x=daily_forecast["Date"], y=daily_forecast["Forecast"],
            mode="lines", name="Forecast",
            line=dict(width=2, dash="dot")
        ))

    fig.update_layout(
        title="Actual Sales vs 30-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Sales",
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        weekday_perf = (
            df_filtered.groupby("Weekday")["Sales"]
            .mean()
            .reindex(WEEKDAY_ORDER)
            .reset_index()
            .dropna()
        )
        fig_week = px.bar(
            weekday_perf,
            x="Weekday",
            y="Sales",
            title="Average Sales by Weekday"
        )
        st.plotly_chart(fig_week, use_container_width=True)

    with col2:
        monthly_sales = (
            df_filtered.groupby(df_filtered["Date"].dt.to_period("M"))["Sales"]
            .sum()
            .reset_index()
        )
        monthly_sales["Date"] = monthly_sales["Date"].astype(str)

        fig_month = px.line(
            monthly_sales,
            x="Date",
            y="Sales",
            title="Monthly Sales Trend"
        )
        st.plotly_chart(fig_month, use_container_width=True)

    if selected_store == "All" and not store_scores.empty:
        st.subheader("Store Performance Summary")

        top10 = store_scores.sort_values("PerformanceScore", ascending=False).head(10).reset_index(drop=True)
        bottom10 = store_scores.sort_values("PerformanceScore", ascending=True).head(10).reset_index(drop=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟢 Top 10 Performing Stores")
            for i, row in top10.iterrows():
                st.markdown(
                    f"""
                    <div class="insight-box insight-green">
                    <b>#{i+1} Store {int(row["Store"])}</b><br>
                    Sales: {fmt_money(row["Sales"])}<br>
                    Growth: {fmt_pct(row["Growth"])}<br>
                    Promo Uplift: {fmt_pct(row["PromoUplift"])}<br>
                    Performance Score: <b>{row["PerformanceScore"]:.1f}</b><br>
                    <span class="small-note">
                    Strong performer. Protect margin and replicate successful practices.
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            st.markdown("### 🔴 Bottom 10 Performing Stores")
            for i, row in bottom10.iterrows():
                st.markdown(
                    f"""
                    <div class="insight-box insight-red">
                    <b>#{i+1} Store {int(row["Store"])}</b><br>
                    Sales: {fmt_money(row["Sales"])}<br>
                    Growth: {fmt_pct(row["Growth"])}<br>
                    Promo Uplift: {fmt_pct(row["PromoUplift"])}<br>
                    Risk Score: <b>{row["RiskScore"]:.1f}</b> | Priority: <b>{row["Priority"]}</b><br>
                    <span class="small-note">
                    Action: {row["RecommendedAction"]}
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.subheader("Key Business Signals")

    signals = []

    if pd.notna(growth_pct) and growth_pct < 0:
        signals.append("📉 Sales declined versus the previous comparable period.")
    if pd.notna(forecast_growth) and forecast_growth < 0:
        signals.append("⚠️ Forecast indicates near-term commercial pressure.")
    if pd.notna(promo_uplift) and promo_uplift < 5:
        signals.append("🎯 Promotion uplift is modest; broad discounting should be avoided.")
    if p1_count > 0:
        signals.append(f"🔴 {p1_count} stores are classified as P1 intervention priorities.")

    if not signals:
        signals.append("✅ No major risk signals detected in the selected scope.")

    for signal in signals:
        st.markdown(f"- {signal}")

    if pd.notna(forecast_growth) and forecast_growth < 0:
        css = "insight-red"
        headline = "Forecast indicates commercial pressure."
        detail = "Near-term projected sales are below the recent period baseline. Management should review underperforming stores and test targeted interventions."
    elif pd.notna(promo_uplift) and promo_uplift < 5:
        css = "insight-amber"
        headline = "Promotions are not strongly lifting performance."
        detail = "Sales are not collapsing, but promotion efficiency looks modest. Focus on selective timing and store-level targeting."
    else:
        css = "insight-green"
        headline = "Business trend is stable to positive."
        detail = "Current performance and forecast direction are healthy enough to protect margin and focus on selective improvement."

    st.markdown(
        f"""
        <div class="insight-box {css}">
            <b>{headline}</b><br>
            {detail}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# STORE INTELLIGENCE
# =========================================================
def store_page():
    st.title("Store Intelligence")
    st.caption("Store-level diagnosis with trend, forecast, promotion response, weekday pattern, risk score, and recommendation.")

    if selected_store == "All":
        st.warning("Select a specific store from the sidebar.")
        return

    store_df = df_filtered.copy()

    total_sales = store_df["Sales"].sum()
    customers = store_df["Customers"].sum() if "Customers" in store_df.columns else np.nan
    sales_per_customer = safe_divide(total_sales, customers)

    current_sales, previous_sales, growth_pct = get_period_comparison(store_df)
    promo_uplift = compute_promo_uplift(store_df)

    forecast_df = model_forecast(store_df, 30)
    daily_forecast, forecast_total = aggregate_forecast(forecast_df)
    forecast_growth = safe_divide(forecast_total - current_sales, current_sales) * 100

    store_scores = build_store_scores(store_df, forecast_df)
    score_row = store_scores.iloc[0] if not store_scores.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Store Sales", fmt_money(total_sales), fmt_pct(growth_pct))
    c2.metric("Forecast (30d)", fmt_money(forecast_total), fmt_pct(forecast_growth))
    c3.metric("Promo Uplift", fmt_pct(promo_uplift))
    c4.metric("Sales per Customer", fmt_money2(sales_per_customer))

    if score_row is not None:
        c5, c6, c7 = st.columns(3)
        c5.metric("Performance Score", f"{score_row['PerformanceScore']:.1f}")
        c6.metric("Risk Score", f"{score_row['RiskScore']:.1f}")
        c7.metric("Priority", score_row["Priority"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=store_df["Date"], y=store_df["Sales"],
        mode="lines", name="Actual Sales",
        line=dict(width=2)
    ))

    if "rolling_mean_7" in store_df.columns:
        fig.add_trace(go.Scatter(
            x=store_df["Date"], y=store_df["rolling_mean_7"],
            mode="lines", name="Rolling 7",
            line=dict(width=2)
        ))

    if "rolling_mean_30" in store_df.columns:
        fig.add_trace(go.Scatter(
            x=store_df["Date"], y=store_df["rolling_mean_30"],
            mode="lines", name="Rolling 30",
            line=dict(width=2)
        ))

    if not daily_forecast.empty:
        fig.add_trace(go.Scatter(
            x=daily_forecast["Date"], y=daily_forecast["Forecast"],
            mode="lines", name="Forecast",
            line=dict(width=2, dash="dot")
        ))

    fig.update_layout(
        title=f"Store {selected_store}: Sales Trend and Forecast",
        xaxis_title="Date",
        yaxis_title="Sales",
        height=430
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        weekday_avg = (
            store_df.groupby("Weekday")["Sales"]
            .mean()
            .reindex(WEEKDAY_ORDER)
            .reset_index()
            .dropna()
        )
        fig_wd = px.bar(
            weekday_avg,
            x="Weekday",
            y="Sales",
            title="Average Sales by Weekday"
        )
        st.plotly_chart(fig_wd, use_container_width=True)

    with col2:
        if "Promo" in store_df.columns:
            promo_box = store_df.copy()
            promo_box["PromoLabel"] = promo_box["Promo"].map({0: "Non-Promo", 1: "Promo"})

            fig_box = px.box(
                promo_box,
                x="PromoLabel",
                y="Sales",
                title="Promo vs Non-Promo Sales Distribution"
            )
            st.plotly_chart(fig_box, use_container_width=True)

    strongest_day = (
        weekday_avg.sort_values("Sales", ascending=False)["Weekday"].iloc[0]
        if not weekday_avg.empty else "N/A"
    )
    weakest_day = (
        weekday_avg.sort_values("Sales", ascending=True)["Weekday"].iloc[0]
        if not weekday_avg.empty else "N/A"
    )

    if score_row is not None:
        css = css_from_priority(score_row["Priority"])
        headline = f"Store {selected_store}: {score_row['RiskLevel']}"
        detail = (
            f"Strongest day: {strongest_day}. Weakest day: {weakest_day}. "
            f"Recommended action: {score_row['RecommendedAction']}"
        )
    else:
        css = "insight-amber"
        headline = f"Store {selected_store}: limited scoring available."
        detail = "Not enough information to generate a complete score."

    st.markdown(
        f"""
        <div class="insight-box {css}">
            <b>{headline}</b><br>
            {detail}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FORECAST & SCENARIOS
# =========================================================
def forecast_page():
    st.title("Forecast & Scenarios")
    st.caption("Store-level XGBoost forecasts aggregated into business planning scenarios.")

    forecast_days = st.slider("Forecast horizon (days)", 7, 60, 30)

    forecast_df = model_forecast(df_filtered, forecast_days)

    if forecast_df is None:
        st.error("Model not loaded, so forecast cannot be generated.")
        if model_error is not None:
            st.code(str(model_error))
        return

    daily_forecast, baseline_total = aggregate_forecast(forecast_df)

    promo_scenario_total = baseline_total * 1.08
    worst_case_total = baseline_total * 0.92
    best_case_total = baseline_total * 1.08

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline Forecast", fmt_money(baseline_total))
    c2.metric("Promo Scenario", fmt_money(promo_scenario_total))
    c3.metric("Best-Case", fmt_money(best_case_total))
    c4.metric("Worst-Case", fmt_money(worst_case_total))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_forecast["Date"], y=daily_forecast["Forecast"],
        mode="lines", name="Baseline Forecast",
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=daily_forecast["Date"], y=daily_forecast["Forecast"] * 1.08,
        mode="lines", name="Promo Scenario",
        line=dict(width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=daily_forecast["Date"], y=daily_forecast["Forecast"] * 0.92,
        mode="lines", name="Worst Case",
        line=dict(width=2, dash="dot")
    ))

    fig.update_layout(
        title=f"{forecast_days}-Day Forecast Projection",
        xaxis_title="Date",
        yaxis_title="Forecast Sales",
        height=430
    )
    st.plotly_chart(fig, use_container_width=True)

    risk_days = daily_forecast.copy()
    threshold = risk_days["Forecast"].median()
    risk_days["DemandBand"] = np.where(
        risk_days["Forecast"] < threshold,
        "Lower Demand",
        "Higher Demand"
    )

    fig_bar = px.bar(
        risk_days,
        x="Date",
        y="Forecast",
        color="DemandBand",
        title="Forecast Demand Profile"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box insight-amber">
            <b>Scenario interpretation</b><br>
            The baseline forecast is generated at store-day level using the trained XGBoost model and then aggregated.
            The promo, best-case, and worst-case views are planning scenarios, not separately trained models.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# BUSINESS CHATBOT
# =========================================================
def business_chatbot_page():
    st.title("Business Chatbot")
    st.caption("Ask KPI, forecast, promotion, and store-performance questions without external APIs.")

    query = st.text_input(
        "Ask a question",
        placeholder="Examples: summarize performance, best stores, worst stores, highest sales, lowest sales, promo uplift, forecast summary, KPIs"
    )

    if not query:
        st.info("Try: 'summarize performance', 'best stores', 'worst stores', 'highest sales', 'forecast summary', or 'promotion analysis'.")
        return

    q = query.lower().strip()

    total_sales = df_filtered["Sales"].sum()
    customers = df_filtered["Customers"].sum() if "Customers" in df_filtered.columns else np.nan
    avg_daily_sales = df_filtered.groupby("Date")["Sales"].sum().mean()

    current_sales, previous_sales, growth_pct = get_period_comparison(df_filtered)
    promo_uplift = compute_promo_uplift(df_filtered)

    forecast_df = model_forecast(df_filtered, 30)
    daily_forecast, forecast_total = aggregate_forecast(forecast_df)
    forecast_growth = safe_divide(forecast_total - current_sales, current_sales) * 100

    store_scores = build_store_scores(df_filtered, forecast_df)

    st.markdown("### Answer")

    # IMPORTANT: worst/bottom must come BEFORE best/top
    if "worst" in q or "bottom" in q or "lowest performing" in q or "underperform" in q:
        bottom = store_scores.sort_values("RiskScore", ascending=False).head(10)
        st.markdown("**Stores Needing the Most Attention**")
        st.dataframe(
            bottom[[
                "Store", "Sales", "Customers", "Growth", "PromoUplift",
                "ForecastGrowth", "PerformanceScore", "RiskScore",
                "Priority", "RecommendedAction"
            ]],
            use_container_width=True,
            hide_index=True
        )

    elif "best" in q or "top" in q or "highest performing" in q:
        top = store_scores.sort_values("PerformanceScore", ascending=False).head(10)
        st.markdown("**Top 10 Performing Stores**")
        st.dataframe(
            top[[
                "Store", "Sales", "Customers", "Growth", "PromoUplift",
                "ForecastGrowth", "PerformanceScore", "RiskScore",
                "Priority"
            ]],
            use_container_width=True,
            hide_index=True
        )

    elif "highest sales" in q or "highest selling" in q:
        top_sales = (
            df_filtered.groupby("Store", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=False)
            .head(10)
        )
        st.markdown("**Highest Sales Stores**")
        st.dataframe(top_sales, use_container_width=True, hide_index=True)

    elif "lowest sales" in q or "lowest selling" in q:
        low_sales = (
            df_filtered.groupby("Store", as_index=False)["Sales"]
            .sum()
            .sort_values("Sales", ascending=True)
            .head(10)
        )
        st.markdown("**Lowest Sales Stores**")
        st.dataframe(low_sales, use_container_width=True, hide_index=True)

    elif "summary" in q or "summarize" in q or "overview" in q:
        st.markdown(
            f"""
**Business Performance Summary**

- Total sales: **{fmt_money(total_sales)}**
- Customers: **{fmt_num(customers)}**
- Average daily sales: **{fmt_money(avg_daily_sales)}**
- Recent growth: **{fmt_pct(growth_pct)}**
- 30-day forecast: **{fmt_money(forecast_total)}**
- Forecast growth: **{fmt_pct(forecast_growth)}**
- Promotion uplift: **{fmt_pct(promo_uplift)}**

**Interpretation:**  
The selected scope shows {'commercial pressure' if pd.notna(forecast_growth) and forecast_growth < 0 else 'stable to positive forecast direction'}.
            """
        )

    elif "promo" in q or "promotion" in q:
        st.markdown(
            f"""
**Promotion Analysis**

- Promotion uplift: **{fmt_pct(promo_uplift)}**

**Interpretation:**  
{'Promotions are creating meaningful uplift and should be targeted selectively.' if pd.notna(promo_uplift) and promo_uplift >= 5 else 'Promotion impact is weak or unclear. Broad discounting should be avoided.'}
            """
        )

    elif "forecast" in q or "future" in q or "outlook" in q:
        st.markdown(
            f"""
**Forecast Summary**

- 30-day forecast: **{fmt_money(forecast_total)}**
- Forecast growth: **{fmt_pct(forecast_growth)}**

**Interpretation:**  
{'Forecast suggests a slowdown and should be monitored closely.' if pd.notna(forecast_growth) and forecast_growth < 0 else 'Forecast is stable or positive.'}
            """
        )

    elif "kpi" in q or "metric" in q:
        kpi_df = pd.DataFrame({
            "KPI": [
                "Total Sales",
                "Customers",
                "Average Daily Sales",
                "Recent Growth",
                "30-Day Forecast",
                "Forecast Growth",
                "Promotion Uplift"
            ],
            "Value": [
                fmt_money(total_sales),
                fmt_num(customers),
                fmt_money(avg_daily_sales),
                fmt_pct(growth_pct),
                fmt_money(forecast_total),
                fmt_pct(forecast_growth),
                fmt_pct(promo_uplift)
            ]
        })
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    else:
        st.warning(
            "I can answer questions about summary, KPIs, best stores, worst stores, highest sales, lowest sales, promotions, and forecasts."
        )


# =========================================================
# ROUTER
# =========================================================
if page == "Executive Overview":
    executive_page()

elif page == "Store Intelligence":
    store_page()

elif page == "Forecast & Scenarios":
    forecast_page()

elif page == "Business Chatbot":
    business_chatbot_page()