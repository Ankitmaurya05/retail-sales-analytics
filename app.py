from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Retail Sales Analytics", page_icon="📊", layout="wide")

PALETTE = ["#2563eb", "#06b6d4", "#8b5cf6", "#f59e0b", "#10b981"]


@st.cache_data
def make_demo_data(days: int = 730, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    products = [
        ("Classic Shirt", "Apparel", 899, 420),
        ("Running Shoes", "Footwear", 2499, 1320),
        ("Travel Backpack", "Accessories", 1599, 760),
        ("Wireless Earbuds", "Electronics", 1999, 980),
        ("Cotton Bedsheet", "Home", 1299, 610),
        ("Denim Jeans", "Apparel", 1499, 690),
        ("Sports Sandals", "Footwear", 999, 470),
        ("Smart Watch", "Electronics", 3299, 1900),
    ]
    regions = ["West", "North", "South", "East", "Central"]
    rows = []
    for day_idx, date in enumerate(dates):
        season = 1 + 0.20 * np.sin(2 * np.pi * date.dayofyear / 365.25)
        weekend = 1.18 if date.dayofweek >= 5 else 1.0
        growth = 0.75 + 0.35 * day_idx / max(days, 1)
        for product, category, price, cost in products:
            for region in regions:
                base = 1.9 if category == "Electronics" else 2.6
                units = max(0, int(rng.poisson(base * season * weekend * growth)))
                if units:
                    rows.append((date, product, category, region, units, price, cost))
    return pd.DataFrame(rows, columns=[
        "date", "product", "category", "region", "units_sold", "unit_price", "unit_cost"
    ])


def normalize_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    required = {"date", "product", "category", "region", "units_sold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["units_sold", "unit_price", "unit_cost", "revenue", "profit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "revenue" not in df.columns:
        if "unit_price" not in df.columns:
            raise ValueError("Provide either revenue or unit_price.")
        df["revenue"] = df["units_sold"] * df["unit_price"]
    if "profit" not in df.columns:
        if "unit_cost" in df.columns and "unit_price" in df.columns:
            df["profit"] = df["units_sold"] * (df["unit_price"] - df["unit_cost"])
        else:
            df["profit"] = 0.0
    df = df.dropna(subset=["date", "units_sold", "revenue"])
    df = df[(df["units_sold"] >= 0) & (df["revenue"] >= 0)]
    if df.empty:
        raise ValueError("No valid rows found after cleaning.")
    return df


def forecast_daily(daily: pd.Series, horizon: int = 30):
    series = daily.asfreq("D", fill_value=0).astype(float)
    if len(series) < 45:
        raise ValueError("At least 45 days of history are required for forecasting.")
    frame = pd.DataFrame({"y": series})
    for lag in [1, 7, 14, 28]:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    frame["rolling_7"] = frame["y"].shift(1).rolling(7).mean()
    frame["rolling_28"] = frame["y"].shift(1).rolling(28).mean()
    frame["dayofweek"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["dayofyear"] = frame.index.dayofyear
    frame = frame.dropna()
    features = [c for c in frame.columns if c != "y"]
    split = max(int(len(frame) * 0.8), 1)
    train, test = frame.iloc[:split], frame.iloc[split:]
    model = RandomForestRegressor(n_estimators=180, max_depth=10, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    model.fit(train[features], train["y"])
    metrics = {}
    if len(test):
        pred = np.maximum(model.predict(test[features]), 0)
        metrics = {
            "mae": mean_absolute_error(test["y"], pred),
            "rmse": np.sqrt(mean_squared_error(test["y"], pred)),
        }
    model.fit(frame[features], frame["y"])
    history = list(series.values)
    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    results = []
    for date in future_dates:
        row = {
            "lag_1": history[-1], "lag_7": history[-7], "lag_14": history[-14],
            "lag_28": history[-28], "rolling_7": float(np.mean(history[-7:])),
            "rolling_28": float(np.mean(history[-28:])), "dayofweek": date.dayofweek,
            "month": date.month, "dayofyear": date.dayofyear,
        }
        value = max(float(model.predict(pd.DataFrame([row])[features])[0]), 0)
        results.append(value)
        history.append(value)
    forecast = pd.DataFrame({"date": future_dates, "predicted_sales": results})
    return forecast, metrics


st.title("📊 Retail Sales Analytics & Demand Forecasting")
st.caption("Explore retail performance, discover trends, and estimate the next 30 days of sales.")
with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload retail CSV", type=["csv"])
    st.caption("Expected fields: date, product, category, region, units_sold, unit_price, unit_cost.")
    st.divider()
    st.header("Dashboard filters")

try:
    raw = pd.read_csv(uploaded) if uploaded else make_demo_data()
    df = normalize_data(raw)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

min_date, max_date = df["date"].min().date(), df["date"].max().date()
with st.sidebar:
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    categories = st.multiselect("Category", sorted(df["category"].dropna().unique()), default=sorted(df["category"].dropna().unique()))
    regions = st.multiselect("Region", sorted(df["region"].dropna().unique()), default=sorted(df["region"].dropna().unique()))

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = min_date, max_date
view = df[
    df["date"].dt.date.between(start, end)
    & df["category"].isin(categories)
    & df["region"].isin(regions)
].copy()
if view.empty:
    st.warning("No records match the selected filters.")
    st.stop()

revenue = float(view["revenue"].sum())
profit = float(view["profit"].sum())
units = int(view["units_sold"].sum())
orders = int(len(view))
margin = (profit / revenue * 100) if revenue else 0
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Revenue", f"₹{revenue:,.0f}")
c2.metric("Profit", f"₹{profit:,.0f}")
c3.metric("Units sold", f"{units:,}")
c4.metric("Transactions", f"{orders:,}")
c5.metric("Profit margin", f"{margin:.1f}%")

st.divider()
left, right = st.columns(2)
with left:
    trend = view.groupby(view["date"].dt.to_period("M"))["revenue"].sum().reset_index()
    trend["month"] = trend["date"].astype(str)
    fig = px.line(trend, x="month", y="revenue", markers=True, title="Monthly revenue trend",
                  labels={"month": "Month", "revenue": "Revenue (₹)"})
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)
with right:
    category_perf = view.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    fig = px.pie(category_perf, names="category", values="revenue", hole=0.55,
                 title="Revenue by category", color_discrete_sequence=PALETTE)
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    top_products = view.groupby("product", as_index=False)["revenue"].sum().nlargest(10, "revenue")
    fig = px.bar(top_products.sort_values("revenue"), x="revenue", y="product", orientation="h",
                 title="Top 10 products by revenue", labels={"revenue": "Revenue (₹)", "product": "Product"})
    st.plotly_chart(fig, use_container_width=True)
with right:
    region_perf = view.groupby("region", as_index=False)["profit"].sum().sort_values("profit", ascending=False)
    fig = px.bar(region_perf, x="region", y="profit", title="Profit by region",
                 labels={"profit": "Profit (₹)", "region": "Region"})
    st.plotly_chart(fig, use_container_width=True)

st.subheader("30-day sales forecast")
st.caption("Forecast uses daily revenue as the target. It is a demo model; validate on your own data before business decisions.")
daily = view.groupby(view["date"].dt.normalize())["revenue"].sum().sort_index()
try:
    forecast, metrics = forecast_daily(daily, 30)
    hist = daily.asfreq("D", fill_value=0).tail(90).rename("actual_sales").reset_index()
    hist.columns = ["date", "actual_sales"]
    hist["predicted_sales"] = np.nan
    future = forecast.copy()
    future["actual_sales"] = np.nan
    chart_data = pd.concat([hist, future[["date", "actual_sales", "predicted_sales"]]], ignore_index=True)
    fig = px.line(chart_data, x="date", y=["actual_sales", "predicted_sales"],
                  title="Actual daily revenue and predicted next 30 days",
                  labels={"value": "Revenue (₹)", "date": "Date", "variable": "Series"})
    st.plotly_chart(fig, use_container_width=True)
    if metrics:
        m1, m2 = st.columns(2)
        m1.metric("Holdout MAE", f"₹{metrics['mae']:,.0f}")
        m2.metric("Holdout RMSE", f"₹{metrics['rmse']:,.0f}")
    st.download_button("Download 30-day forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                       file_name="retail_sales_forecast.csv", mime="text/csv")
except ValueError as exc:
    st.info(str(exc))

st.subheader("Filtered transaction data")
st.dataframe(view.sort_values("date", ascending=False), use_container_width=True, hide_index=True)
st.download_button("Download filtered data", view.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_retail_sales.csv", mime="text/csv")
st.caption("Demo data is synthetic. Currency is displayed in INR (₹).")
