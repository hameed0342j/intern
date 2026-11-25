"""
Sales Forecasting & Demand Intelligence Dashboard
Internship Project (Weeks 3 & 4) - XYlofy AI
Author: Muhammed Al Hameed M

Run:  streamlit run app.py
"""
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Demand Intelligence System", page_icon="📈", layout="wide")

NAVY, ORANGE, GREEN, RED, GREY = "#1b365d", "#e67e22", "#27ae60", "#c0392b", "#7f8c8d"
ORDER = (1, 1, 1)
SEASONAL = (1, 1, 1, 12)


# ----------------------------------------------------------------- data
APP_DIR = Path(__file__).resolve().parent
UPLOAD_PATH = Path(tempfile.gettempdir()) / "superstore_train.csv"
CANDIDATES = [APP_DIR / "train.csv", APP_DIR / "data" / "train.csv", Path.cwd() / "train.csv", UPLOAD_PATH]
REQUIRED = {"Order Date", "Ship Date", "Sales", "Category", "Sub-Category", "Region", "Order ID", "Customer ID"}


def find_data():
    for p in CANDIDATES:
        if p.is_file():
            return str(p)
    return None


def parse_dates(col: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(col, format="%d/%m/%Y")
    except (ValueError, TypeError):
        return pd.to_datetime(col, dayfirst=True, format="mixed")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df["Order Date"] = parse_dates(df["Order Date"])
    df["Ship Date"] = parse_dates(df["Ship Date"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


def monthly(frame: pd.DataFrame) -> pd.Series:
    s = frame.groupby(frame["Order Date"].dt.to_period("M"))["Sales"].sum()
    s.index = s.index.to_timestamp()
    return s


def segment_frame(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    if segment == "All sales":
        return df
    if segment in df["Category"].unique():
        return df[df["Category"] == segment]
    return df[df["Region"] == segment]


@st.cache_data
def sarima_results(segment: str):
    """Backtest on the last 3 months, then refit on all data and forecast 3 months ahead."""
    df = load_data(DATA_PATH)
    s = monthly(segment_frame(df, segment))
    train, test = s.iloc[:-3], s.iloc[-3:]
    fit = SARIMAX(train, order=ORDER, seasonal_order=SEASONAL,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    bt = fit.forecast(3)
    mae = mean_absolute_error(test, bt)
    rmse = float(np.sqrt(mean_squared_error(test, bt)))
    full = SARIMAX(s, order=ORDER, seasonal_order=SEASONAL,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    g = full.get_forecast(3)
    fc = g.predicted_mean
    ci = g.conf_int()
    ci.columns = ["lower", "upper"]
    ci["lower"] = ci["lower"].clip(lower=0)
    return s, fc, ci, mae, rmse


@st.cache_data
def weekly_anomalies() -> pd.DataFrame:
    df = load_data(DATA_PATH)
    w = df.groupby(df["Order Date"].dt.to_period("W"))["Sales"].sum()
    w.index = w.index.to_timestamp()
    wd = w.reset_index()
    wd.columns = ["week", "sales"]
    wd["iso"] = IsolationForest(contamination=0.05, random_state=42).fit_predict(wd[["sales"]]) == -1
    wd["rolling_mean"] = wd["sales"].rolling(8, center=True).mean()
    wd["rolling_std"] = wd["sales"].rolling(8, center=True).std()
    wd["z"] = (wd["sales"] - wd["rolling_mean"]) / wd["rolling_std"]
    wd["zflag"] = wd["z"].abs() > 2
    return wd


@st.cache_data
def clusters() -> tuple:
    df = load_data(DATA_PATH)
    sub = df.groupby("Sub-Category").agg(
        total_sales=("Sales", "sum"), avg_order_value=("Sales", "mean"),
        order_count=("Order ID", "nunique")).reset_index()
    ys = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack(fill_value=0)
    sub["growth_rate"] = (((ys.iloc[:, -1] - ys.iloc[:, 0]) / (ys.iloc[:, 0] + 1)) * 100).values
    ms = df.groupby(["Sub-Category", df["Order Date"].dt.to_period("M")])["Sales"].sum().unstack(fill_value=0)
    sub["volatility"] = ms.std(axis=1).values
    cols = ["total_sales", "avg_order_value", "growth_rate", "volatility", "order_count"]
    X = StandardScaler().fit_transform(sub[cols])
    sub["cluster"] = KMeans(4, random_state=42, n_init=10).fit_predict(X)
    names, strategy = {}, {}
    for c, g in sub.groupby("cluster"):
        members = set(g["Sub-Category"])
        if "Copiers" in members:
            names[c], strategy[c] = "Low volume, high value", "Order against confirmed demand; minimal stock"
        elif "Chairs" in members:
            names[c], strategy[c] = "High volume, high volatility", "Larger safety stock; flexible replenishment"
        elif "Binders" in members:
            names[c], strategy[c] = "Growing, frequently bought", "Raise base stock with growth; frequent small reorders"
        else:
            names[c], strategy[c] = "Low volume, mature", "Lean, just-in-time stock; bundle slow lines"
    sub["segment"] = sub["cluster"].map(names)
    sub["strategy"] = sub["cluster"].map(strategy)
    pca = PCA(2)
    P = pca.fit_transform(X)
    sub["PC1"], sub["PC2"] = P[:, 0], P[:, 1]
    return sub, pca.explained_variance_ratio_


DATA_PATH = find_data()
if DATA_PATH is None:
    st.title("Demand Intelligence System")
    st.warning("train.csv was not found next to app.py. Upload the Superstore sales file (train.csv) to continue.")
    up = st.file_uploader("Upload train.csv", type="csv")
    if up is not None:
        UPLOAD_PATH.write_bytes(up.getvalue())
        st.rerun()
    st.stop()

df = load_data(DATA_PATH)
missing = REQUIRED - set(df.columns)
if missing:
    st.error(f"This CSV is missing required columns: {', '.join(sorted(missing))}. Use the Superstore train.csv.")
    st.stop()

# ----------------------------------------------------------------- sidebar
st.sidebar.markdown("## 📈 Demand Intelligence")
st.sidebar.caption("Superstore sales · Jan 2015 – Dec 2018")
page = st.sidebar.radio("Navigate", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])
st.sidebar.markdown("---")
st.sidebar.caption("Model: SARIMA(1,1,1)(1,1,1,12)\n\nBacktest window: Oct – Dec 2018")

# ----------------------------------------------------------------- page 1
if page == "Sales Overview":
    st.title("Sales Overview")
    c1, c2 = st.columns(2)
    regions = c1.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    cats = c2.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    f = df[df["Region"].isin(regions) & df["Category"].isin(cats)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total sales", f"${f['Sales'].sum():,.0f}")
    k2.metric("Orders", f"{f['Order ID'].nunique():,}")
    k3.metric("Customers", f"{f['Customer ID'].nunique():,}")
    yr = f.groupby("Year")["Sales"].sum()
    k4.metric("2018 vs 2017", f"{(yr.get(2018, 0) / yr.get(2017, 1) - 1) * 100:+.1f}%")

    a, b = st.columns([1, 2])
    fig = px.bar(x=yr.index.astype(str), y=yr.values, labels={"x": "Year", "y": "Sales ($)"},
                 title="Total sales by year", color_discrete_sequence=[NAVY])
    a.plotly_chart(fig, width='stretch')
    m = monthly(f)
    fig = px.line(x=m.index, y=m.values, markers=True, labels={"x": "Month", "y": "Sales ($)"},
                  title="Monthly sales trend", color_discrete_sequence=[NAVY])
    b.plotly_chart(fig, width='stretch')

    a, b = st.columns(2)
    rc = f.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig = px.bar(rc, x="Region", y="Sales", color="Category", barmode="group", title="Sales by region and category",
                 color_discrete_sequence=[NAVY, ORANGE, GREEN])
    a.plotly_chart(fig, width='stretch')
    sc = f.groupby("Sub-Category")["Sales"].sum().sort_values().tail(10)
    fig = px.bar(x=sc.values, y=sc.index, orientation="h", title="Top 10 sub-categories",
                 labels={"x": "Sales ($)", "y": ""}, color_discrete_sequence=[ORANGE])
    b.plotly_chart(fig, width='stretch')

# ----------------------------------------------------------------- page 2
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    c1, c2 = st.columns([2, 1])
    options = ["All sales", "Furniture", "Office Supplies", "Technology", "Central", "East", "South", "West"]
    seg = c1.selectbox("Category or region", options, index=0)
    h = c2.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)
    s, fc, ci, mae, rmse = sarima_results(seg)
    fc, ci = fc.iloc[:h], ci.iloc[:h]

    fig = go.Figure()
    hist = s.iloc[-24:]
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual", mode="lines+markers", line=dict(color=NAVY)))
    fig.add_trace(go.Scatter(x=list(ci.index) + list(ci.index[::-1]), y=list(ci["upper"]) + list(ci["lower"][::-1]),
                             fill="toself", fillcolor="rgba(230,126,34,0.18)", line=dict(width=0), mode="lines", name="95% interval"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name="SARIMA forecast", mode="lines+markers",
                             line=dict(color=ORANGE, dash="dash")))
    fig.update_layout(title=f"{seg}: next {h} month(s)", yaxis_title="Sales ($)", height=430)
    st.plotly_chart(fig, width='stretch')

    out = pd.DataFrame({"Month": fc.index.strftime("%b %Y"), "Forecast ($)": fc.values.round(0),
                        "Lower 95% ($)": ci["lower"].values.round(0), "Upper 95% ($)": ci["upper"].values.round(0)})
    st.dataframe(out, hide_index=True, width='stretch', column_config={c: st.column_config.NumberColumn(format="%,.0f") for c in ["Forecast ($)", "Lower 95% ($)", "Upper 95% ($)"]})
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (backtest)", f"${mae:,.0f}")
    m2.metric("RMSE (backtest)", f"${rmse:,.0f}")
    m3.metric("Forecast total", f"${fc.sum():,.0f}")
    st.caption("Backtest: model trained to Sep 2018 and scored on Oct–Dec 2018. "
               "Forecast: model refitted on all 48 months.")

# ----------------------------------------------------------------- page 3
elif page == "Anomaly Report":
    st.title("Anomaly Report")
    wd = weekly_anomalies()
    k1, k2, k3 = st.columns(3)
    k1.metric("Weeks analysed", len(wd))
    k2.metric("Isolation Forest flags", int(wd["iso"].sum()))
    k3.metric("Z-score flags", int(wd["zflag"].sum()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wd["week"], y=wd["sales"], name="Weekly sales", line=dict(color="#34495e", width=1)))
    fig.add_trace(go.Scatter(x=wd["week"], y=wd["rolling_mean"], name="8-week rolling mean",
                             line=dict(color="#2980b9", dash="dash")))
    a = wd[wd["iso"]]
    fig.add_trace(go.Scatter(x=a["week"], y=a["sales"], mode="markers", name="Isolation Forest",
                             marker=dict(color=RED, symbol="x", size=11)))
    z = wd[wd["zflag"]]
    fig.add_trace(go.Scatter(x=z["week"], y=z["sales"], mode="markers", name="Z-score (|z| > 2)",
                             marker=dict(color=ORANGE, size=10)))
    fig.update_layout(height=430, yaxis_title="Sales ($)", title="Weekly sales with detected anomalies")
    st.plotly_chart(fig, width='stretch')
    flagged = wd[wd["iso"] | wd["zflag"]].copy()
    flagged["Flagged by"] = np.where(flagged["iso"] & flagged["zflag"], "Both",
                                     np.where(flagged["iso"], "Isolation Forest", "Z-score"))
    flagged["Week starting"] = flagged["week"].dt.strftime("%d %b %Y")
    flagged["Sales ($)"] = flagged["sales"].round(0)
    st.dataframe(flagged[["Week starting", "Sales ($)", "Flagged by"]], hide_index=True, width='stretch',
                 height=(len(flagged) + 1) * 35 + 3, column_config={"Sales ($)": st.column_config.NumberColumn(format="%,.0f")})

# ----------------------------------------------------------------- page 4
else:
    st.title("Product Demand Segments")
    sub, ev = clusters()
    colors = {"Growing, frequently bought": GREEN, "High volume, high volatility": RED,
              "Low volume, mature": GREY, "Low volume, high value": NAVY}
    fig = px.scatter(sub, x="PC1", y="PC2", color="segment", text="Sub-Category", color_discrete_map=colors,
                     title=f"K-Means clusters (k = 4) on PCA axes  ·  PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}")
    fig.update_traces(textposition="top center", marker=dict(size=14), textfont=dict(size=11))
    fig.update_layout(height=480)
    st.plotly_chart(fig, width='stretch')
    t = sub[["Sub-Category", "segment", "total_sales", "growth_rate", "volatility", "strategy"]].copy()
    t.columns = ["Sub-category", "Demand cluster", "Total sales ($)", "Growth 2015→18 (%)", "Monthly volatility ($)", "Stocking strategy"]
    t = t.sort_values(["Demand cluster", "Total sales ($)"], ascending=[True, False])
    st.dataframe(t.round(1), hide_index=True, width='stretch', height=(len(t) + 1) * 35 + 3,
                 column_config={"Total sales ($)": st.column_config.NumberColumn(format="%,.0f"),
                                "Monthly volatility ($)": st.column_config.NumberColumn(format="%,.0f")})
