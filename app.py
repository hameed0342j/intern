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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Demand Intelligence System", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------- palette & constants
NAVY, BLUE, ORANGE, GREEN, RED, GREY = "#1b365d", "#2f6fb0", "#e67e22", "#27ae60", "#c0392b", "#94a3b8"
CAT_COLORS = {"Furniture": NAVY, "Office Supplies": ORANGE, "Technology": GREEN}
ORDER = (1, 1, 1)
SEASONAL = (1, 1, 1, 12)

# ----------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 3.4rem; padding-bottom: 2.5rem; max-width: 1400px; }

    /* hero header */
    .hero { background: linear-gradient(120deg, #1b365d 0%, #2f6fb0 100%); color: #fff;
            padding: 1.35rem 1.7rem; border-radius: 16px; margin-bottom: 1.2rem;
            box-shadow: 0 8px 24px rgba(27,54,93,.18); }
    .hero h1 { color: #fff; font-size: 1.85rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: -.02em; }
    .hero p { color: #dbe7f5; margin: .35rem 0 0 0; font-size: .98rem; }
    .hero .tag { display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
                 padding: .15rem .6rem; border-radius: 999px; font-size: .75rem; margin-right: .4rem;
                 margin-top: .6rem; color: #fff; }

    /* KPI cards */
    .kpi { background: #fff; border: 1px solid #e5e9f0; border-radius: 14px; padding: 1rem 1.15rem;
           box-shadow: 0 2px 8px rgba(15,23,42,.05); height: 100%; border-top: 4px solid var(--accent); }
    .kpi .label { color: #64748b; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
    .kpi .value { color: #0f172a; font-size: 1.75rem; font-weight: 800; margin-top: .2rem; letter-spacing: -.02em; }
    .kpi .sub { color: #64748b; font-size: .8rem; margin-top: .15rem; }
    .kpi .up { color: #15803d; font-weight: 700; } .kpi .down { color: #b91c1c; font-weight: 700; }

    /* insight callout */
    .insight { background: #fff7ed; border-left: 5px solid #e67e22; border-radius: 10px;
               padding: .85rem 1.1rem; color: #7c2d12; font-size: .93rem; margin: .4rem 0 1rem 0; }
    .insight b { color: #9a3412; }

    /* segment cards */
    .seg { background: #fff; border: 1px solid #e5e9f0; border-radius: 14px; padding: .95rem 1.05rem;
           border-left: 6px solid var(--accent); height: 100%; box-shadow: 0 2px 8px rgba(15,23,42,.05); }
    .seg h4 { margin: 0 0 .3rem 0; font-size: 1rem; color: #0f172a; }
    .seg .items { color: #334155; font-size: .86rem; margin-bottom: .45rem; }
    .seg .rule { color: #475569; font-size: .82rem; background: #f1f5f9; border-radius: 8px; padding: .4rem .55rem; }

    .section-title { font-size: 1.08rem; font-weight: 700; color: #0f172a; margin: .4rem 0 .2rem 0; }

    /* sidebar */
    [data-testid="stSidebar"] { background: #0f2340; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .brand { font-size: 1.25rem; font-weight: 800; color: #fff !important; margin-bottom: .1rem; }
    [data-testid="stSidebar"] .muted { color: #94a3b8 !important; font-size: .8rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label { padding: .45rem .6rem; border-radius: 10px; margin-bottom: .2rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,.08); }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: rgba(230,126,34,.22); }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15); }
    .model-box { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
                 border-radius: 10px; padding: .7rem .8rem; font-size: .8rem; line-height: 1.5; }

    [data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-radius: 14px; }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str, tags=()):
    chips = "".join(f'<span class="tag">{t}</span>' for t in tags)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips}</div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "", accent: str = NAVY):
    col.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def insight(text: str):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def style(fig, height=380, legend=True):
    fig.update_layout(
        template="plotly_white", height=height, font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        title=dict(font=dict(size=15, color="#0f172a"), x=0, xanchor="left", y=0.98, yref="container", yanchor="top"),
        margin=dict(l=10, r=10, t=88 if legend else 55, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text=""),
    )
    fig.update_xaxes(showgrid=False, linecolor="#cbd5e1")
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    return fig


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


@st.cache_data(show_spinner="Training SARIMA model…")
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
    mape = float(np.mean(np.abs((test.values - bt.values) / test.values)) * 100)
    full = SARIMAX(s, order=ORDER, seasonal_order=SEASONAL,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    g = full.get_forecast(3)
    fc = g.predicted_mean
    ci = g.conf_int()
    ci.columns = ["lower", "upper"]
    ci["lower"] = ci["lower"].clip(lower=0)
    return s, fc, ci, mae, rmse, mape, test, bt


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


# ----------------------------------------------------------------- load or ask for data
DATA_PATH = find_data()
if DATA_PATH is None:
    hero("Demand Intelligence System", "Upload the Superstore sales file to start.")
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
with st.sidebar:
    st.markdown('<div class="brand">📈 Demand Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Superstore retail sales · Jan 2015 – Dec 2018</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["📊  Sales Overview", "🔮  Forecast Explorer", "🚨  Anomaly Report",
                                 "🧩  Product Demand Segments"], label_visibility="collapsed")
    page = page.split("  ", 1)[1]
    st.markdown("---")
    st.markdown(
        '<div class="model-box"><b>Forecast model</b><br>SARIMA(1,1,1)(1,1,1,12)<br><br>'
        '<b>Backtest window</b><br>Oct – Dec 2018 (peak quarter)<br><br>'
        '<b>Anomalies</b><br>Isolation Forest + rolling z-score<br><br>'
        '<b>Segments</b><br>K-Means (k = 4) with PCA</div>', unsafe_allow_html=True)
    st.markdown('<br><div class="muted">Built by Muhammed Al Hameed M<br>AI & Data Science Intern · XYlofy AI</div>',
                unsafe_allow_html=True)

# ----------------------------------------------------------------- page 1: overview
if page == "Sales Overview":
    hero("Sales Overview", "Four years of retail orders at a glance. Filter by region and category.",
         ["9,800 order lines", "4 regions", "3 categories", "17 sub-categories"])
    with st.container(border=True):
        c1, c2 = st.columns(2)
        regions = c1.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
        cats = c2.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    f = df[df["Region"].isin(regions) & df["Category"].isin(cats)]
    if f.empty:
        st.info("No data for this selection. Pick at least one region and one category.")
        st.stop()

    yr = f.groupby("Year")["Sales"].sum()
    growth = (yr.get(2018, 0) / yr.get(2017, 1) - 1) * 100
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Total sales", f"${f['Sales'].sum():,.0f}", "Jan 2015 – Dec 2018", NAVY)
    kpi(k2, "Orders", f"{f['Order ID'].nunique():,}", f"Avg order ${f.groupby('Order ID')['Sales'].sum().mean():,.0f}", BLUE)
    kpi(k3, "Customers", f"{f['Customer ID'].nunique():,}", "Unique buyers", GREEN)
    cls = "up" if growth >= 0 else "down"
    kpi(k4, "2018 vs 2017", f"{growth:+.1f}%", f'<span class="{cls}">{"▲" if growth >= 0 else "▼"}</span> year-on-year sales',
        ORANGE)
    st.write("")

    a, b = st.columns([1, 2])
    with a.container(border=True):
        fig = px.bar(x=yr.index.astype(str), y=yr.values, labels={"x": "Year", "y": "Sales ($)"},
                     title="Total sales by year", color_discrete_sequence=[NAVY], text=yr.values)
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False,
                          hovertemplate="%{x}: $%{y:,.0f}<extra></extra>", marker_line_width=0)
        fig.update_yaxes(range=[0, yr.max() * 1.15])
        st.plotly_chart(style(fig, legend=False), width="stretch")
    with b.container(border=True):
        m = monthly(f)
        fig = go.Figure(go.Scatter(x=m.index, y=m.values, mode="lines+markers", line=dict(color=NAVY, width=2.5),
                                   marker=dict(size=5), fill="tozeroy", fillcolor="rgba(47,111,176,.08)",
                                   hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>"))
        fig.update_layout(title="Monthly sales trend", yaxis_title="Sales ($)")
        st.plotly_chart(style(fig, legend=False), width="stretch")

    a, b = st.columns(2)
    with a.container(border=True):
        rc = f.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        fig = px.bar(rc, x="Region", y="Sales", color="Category", barmode="group", title="Sales by region and category",
                     color_discrete_map=CAT_COLORS)
        fig.update_traces(hovertemplate="%{x}: $%{y:,.0f}<extra></extra>", marker_line_width=0)
        st.plotly_chart(style(fig), width="stretch")
    with b.container(border=True):
        sc = f.groupby("Sub-Category")["Sales"].sum().sort_values().tail(10)
        fig = px.bar(x=sc.values, y=sc.index, orientation="h", title="Top 10 sub-categories",
                     labels={"x": "Sales ($)", "y": ""}, color_discrete_sequence=[ORANGE], text=sc.values)
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}: $%{x:,.0f}<extra></extra>", marker_line_width=0)
        fig.update_xaxes(range=[0, sc.max() * 1.22])
        st.plotly_chart(style(fig, legend=False), width="stretch")

    peak = m.groupby(m.index.month).mean()
    insight(f"<b>Seasonality:</b> the busiest month on average is <b>{pd.Timestamp(2000, int(peak.idxmax()), 1):%B}</b> and the "
            f"quietest is <b>{pd.Timestamp(2000, int(peak.idxmin()), 1):%B}</b>, about "
            f"<b>{peak.max() / peak.min():.1f}×</b> apart. Stock and staffing should build from September.")

# ----------------------------------------------------------------- page 2: forecast
elif page == "Forecast Explorer":
    hero("Forecast Explorer", "SARIMA forecasts with a 95% confidence range, for the whole business, a category or a region.",
         ["Trained on 45 months", "Backtested on Oct – Dec 2018", "Refitted on all 48 months"])
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        options = ["All sales", "Furniture", "Office Supplies", "Technology", "Central", "East", "South", "West"]
        seg = c1.selectbox("Category or region", options, index=0)
        h = c2.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)
    s, fc, ci, mae, rmse, mape, test, bt = sarima_results(seg)
    fc, ci = fc.iloc[:h], ci.iloc[:h]

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Forecast total", f"${fc.sum():,.0f}", f"Next {h} month(s)", ORANGE)
    kpi(k2, "MAE (backtest)", f"${mae:,.0f}", "Average monthly error", NAVY)
    kpi(k3, "RMSE (backtest)", f"${rmse:,.0f}", "Penalises large misses", BLUE)
    kpi(k4, "MAPE (backtest)", f"{mape:.1f}%", "Average % error", GREEN)
    st.write("")

    with st.container(border=True):
        fig = go.Figure()
        hist = s.iloc[-24:]
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual", mode="lines+markers",
                                 line=dict(color=NAVY, width=2.5), hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>Actual</extra>"))
        bx = [hist.index[-1]] + list(ci.index)
        fig.add_trace(go.Scatter(x=bx + bx[::-1],
                                 y=[hist.values[-1]] + list(ci["upper"]) + list(ci["lower"][::-1]) + [hist.values[-1]],
                                 fill="toself", fillcolor="rgba(230,126,34,0.18)", line=dict(width=0), mode="lines",
                                 name="95% interval", hoverinfo="skip"))
        bridge_x = [hist.index[-1]] + list(fc.index)
        bridge_y = [hist.values[-1]] + list(fc.values)
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, name="SARIMA forecast", mode="lines+markers",
                                 line=dict(color=ORANGE, dash="dash", width=2.5), marker=dict(size=8),
                                 hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>Forecast</extra>"))
        fig.add_vline(x=hist.index[-1], line_width=1, line_dash="dot", line_color=GREY)
        fig.update_layout(title=f"{seg}: last 24 months and next {h} month(s)", yaxis_title="Sales ($)")
        st.plotly_chart(style(fig, height=440), width="stretch")

    a, b = st.columns([3, 2])
    with a.container(border=True):
        st.markdown('<div class="section-title">Forecast table</div>', unsafe_allow_html=True)
        out = pd.DataFrame({"Month": fc.index.strftime("%b %Y"), "Forecast ($)": fc.values.round(0),
                            "Lower 95% ($)": ci["lower"].values.round(0), "Upper 95% ($)": ci["upper"].values.round(0)})
        st.dataframe(out, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="$%,.0f")
                                    for c in ["Forecast ($)", "Lower 95% ($)", "Upper 95% ($)"]})
    with b.container(border=True):
        st.markdown('<div class="section-title">Backtest: actual vs predicted</div>', unsafe_allow_html=True)
        bt_df = pd.DataFrame({"Month": test.index.strftime("%b %Y"), "Actual ($)": test.values.round(0),
                              "Predicted ($)": bt.values.round(0)})
        st.dataframe(bt_df, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="$%,.0f") for c in ["Actual ($)", "Predicted ($)"]})
    st.caption("Backtest: model trained to Sep 2018 and scored on Oct–Dec 2018. Forecast: model refitted on all 48 months.")

# ----------------------------------------------------------------- page 3: anomalies
elif page == "Anomaly Report":
    hero("Anomaly Report", "Weeks where sales broke from the normal pattern, found by two independent methods.",
         ["Isolation Forest (5% contamination)", "8-week rolling z-score, |z| > 2"])
    wd = weekly_anomalies()
    both = int((wd["iso"] & wd["zflag"]).sum())
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Weeks analysed", f"{len(wd)}", "Weekly sales totals", NAVY)
    kpi(k2, "Isolation Forest flags", f"{int(wd['iso'].sum())}", "Unusual vs whole history", RED)
    kpi(k3, "Z-score flags", f"{int(wd['zflag'].sum())}", "Unusual vs nearby weeks", ORANGE)
    kpi(k4, "Flagged by both", f"{both}", "Highest-confidence anomalies", GREEN)
    st.write("")

    with st.container(border=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wd["week"], y=wd["sales"], name="Weekly sales", line=dict(color="#475569", width=1.2),
                                 hovertemplate="Week of %{x|%d %b %Y}: $%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=wd["week"], y=wd["rolling_mean"], name="8-week rolling mean",
                                 line=dict(color=BLUE, dash="dash"), hoverinfo="skip"))
        a_ = wd[wd["iso"]]
        fig.add_trace(go.Scatter(x=a_["week"], y=a_["sales"], mode="markers", name="Isolation Forest",
                                 marker=dict(color=RED, symbol="x", size=12, line=dict(width=2)),
                                 hovertemplate="%{x|%d %b %Y}: $%{y:,.0f}<extra>Isolation Forest</extra>"))
        z = wd[wd["zflag"]]
        fig.add_trace(go.Scatter(x=z["week"], y=z["sales"], mode="markers", name="Z-score (|z| > 2)",
                                 marker=dict(color=ORANGE, size=11, line=dict(color="white", width=1.5)),
                                 hovertemplate="%{x|%d %b %Y}: $%{y:,.0f}<extra>Z-score</extra>"))
        fig.update_layout(yaxis_title="Sales ($)", title="Weekly sales with detected anomalies")
        st.plotly_chart(style(fig, height=450), width="stretch")

    flagged = wd[wd["iso"] | wd["zflag"]].copy()
    flagged["Flagged by"] = np.where(flagged["iso"] & flagged["zflag"], "🔴 Both",
                                     np.where(flagged["iso"], "Isolation Forest", "Z-score"))
    flagged["Week starting"] = flagged["week"].dt.strftime("%d %b %Y")
    flagged["Sales ($)"] = flagged["sales"].round(0)
    pct = (flagged["sales"] / flagged["rolling_mean"] - 1) * 100
    flagged["vs rolling mean"] = pct.map(lambda v: "–" if pd.isna(v) else f"{v:+.0f}%")
    with st.container(border=True):
        st.markdown('<div class="section-title">Flagged weeks</div>', unsafe_allow_html=True)
        st.dataframe(flagged[["Week starting", "Sales ($)", "vs rolling mean", "Flagged by"]], hide_index=True,
                     width="stretch", height=(len(flagged) + 1) * 35 + 3,
                     column_config={"Sales ($)": st.column_config.NumberColumn(format="$%,.0f")})
    insight("<b>How to use this:</b> spikes flagged by both methods are worth checking against promotions, bulk orders or "
            "data-entry errors before they are fed into the next forecast.")

# ----------------------------------------------------------------- page 4: segments
else:
    hero("Product Demand Segments", "17 sub-categories grouped by how they sell, each with its own stocking rule.",
         ["K-Means, k = 4", "5 standardised features", "PCA for the 2-D view"])
    sub, ev = clusters()
    colors = {"Growing, frequently bought": GREEN, "High volume, high volatility": RED,
              "Low volume, mature": GREY, "Low volume, high value": NAVY}

    cols = st.columns(4)
    for col, name in zip(cols, ["High volume, high volatility", "Growing, frequently bought",
                                "Low volume, high value", "Low volume, mature"]):
        g = sub[sub["segment"] == name]
        if g.empty:
            continue
        col.markdown(f'<div class="seg" style="--accent:{colors[name]}"><h4>{name}</h4>'
                     f'<div class="items">{", ".join(g.sort_values("total_sales", ascending=False)["Sub-Category"])}</div>'
                     f'<div class="rule">📦 {g["strategy"].iloc[0]}</div></div>', unsafe_allow_html=True)
    st.write("")

    with st.container(border=True):
        sub["label"] = np.where(sub["total_sales"] >= 30000, sub["Sub-Category"], "")
        fig = px.scatter(sub, x="PC1", y="PC2", color="segment", text="label", color_discrete_map=colors,
                         size="total_sales", size_max=38,
                         custom_data=["Sub-Category", "total_sales", "growth_rate"],
                         title=f"Demand clusters on the first two principal components  ·  PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}")
        fig.update_traces(textposition="top center", textfont=dict(size=11), marker=dict(line=dict(color="white", width=1)),
                          hovertemplate="<b>%{customdata[0]}</b><br>Sales $%{customdata[1]:,.0f}"
                                        "<br>Growth %{customdata[2]:.0f}%<extra></extra>")
        pos = {"Bookcases": "bottom center", "Appliances": "top left", "Chairs": "bottom center",
               "Accessories": "bottom center", "Supplies": "bottom center", "Storage": "middle right"}
        for tr in fig.data:
            tr.textposition = [pos.get(n, "top center") for n in sub.loc[sub["segment"] == tr.name, "Sub-Category"]]
        fig.update_layout(xaxis_title=f"PC1 ({ev[0]:.1%} of variance)", yaxis_title=f"PC2 ({ev[1]:.1%} of variance)")
        st.plotly_chart(style(fig, height=500), width="stretch")

    t = sub[["Sub-Category", "segment", "total_sales", "growth_rate", "volatility", "strategy"]].copy()
    t.columns = ["Sub-category", "Demand cluster", "Total sales ($)", "Growth 2015→18 (%)", "Monthly volatility ($)",
                 "Stocking strategy"]
    t = t.sort_values(["Demand cluster", "Total sales ($)"], ascending=[True, False])
    with st.container(border=True):
        st.markdown('<div class="section-title">Sub-category detail</div>', unsafe_allow_html=True)
        st.dataframe(t.round(1), hide_index=True, width="stretch", height=(len(t) + 1) * 35 + 3,
                     column_config={"Total sales ($)": st.column_config.NumberColumn(format="$%,.0f"),
                                    "Monthly volatility ($)": st.column_config.NumberColumn(format="$%,.0f"),
                                    "Stocking strategy": st.column_config.TextColumn(width="large"),
                                    "Growth 2015→18 (%)": st.column_config.ProgressColumn(
                                        format="%.0f%%", min_value=float(min(0, t["Growth 2015→18 (%)"].min())),
                                        max_value=float(t["Growth 2015→18 (%)"].max()))})
