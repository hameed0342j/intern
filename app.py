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
st.set_page_config(page_title="Demand Intelligence", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------- tokens
# Flat and ruled: white ground, #eef1f5 panels, 2px borders, one navy brand colour.
BRAND, BLUE, AMBER, RED, GREEN = "#0b2e6b", "#0b55b3", "#c98500", "#a30014", "#17621f"
INK, INK2, LINE, LINE_STRONG, SURFACE = "#0a0a0a", "#2a2f37", "#8b95a3", "#4a5462", "#eef1f5"
CAT_COLORS = {"Furniture": BRAND, "Office Supplies": AMBER, "Technology": GREEN}
SEG_COLORS = {"High volume, high volatility": RED, "Growing, frequently bought": GREEN,
              "Low volume, high value": BRAND, "Low volume, mature": LINE}
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ORDER = (1, 1, 1)
SEASONAL = (1, 1, 1, 12)

st.markdown(
    f"""
    <style>
    html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {{ font-family: {FONT}; }}
    .stApp {{ background: #fff; color: {INK}; }}
    .block-container {{ padding: 3.2rem 2rem 3.5rem; max-width: 1360px; }}
    .stApp ::selection {{ background: #c9dbff; color: {INK}; }}

    /* rail */
    [data-testid="stSidebar"] {{ background: {BRAND}; }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: .5rem; }}
    [data-testid="stSidebar"] * {{ color: #fff; }}
    .rail-brand {{ font-size: 20px; font-weight: 800; letter-spacing: .01em; padding: 2px 4px 2px; }}
    .rail-sub {{ font-size: 12.5px; color: #c8d5ee !important; padding: 0 4px 14px; }}
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px; display: flex; flex-direction: column; align-items: stretch; width: 100%; }}
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
    [data-testid="stSidebar"] [data-testid="stRadio"], [data-testid="stSidebar"] [data-testid="stRadioGroup"],
    [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {{ width: 100% !important; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {{
        display: flex; align-items: center; min-height: 44px; padding: 0 12px; margin: 0; border-radius: 10px;
        width: auto; align-self: stretch; max-width: none; transition: background-color .15s ease-out, color .15s ease-out; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]) {{ display: none; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div {{ padding: 0; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ color: #dbe5f7; font-weight: 650; font-size: 15px; margin: 0; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{ background: rgba(255,255,255,.1); }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background: #fff; }}
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: {BRAND}; }}
    .rail-source {{ margin-top: 22px; border: 2px solid rgba(255,255,255,.4); border-radius: 10px; padding: 10px 12px;
                    display: grid; gap: 8px; }}
    .rail-source div {{ display: grid; gap: 1px; }}
    .rail-source b {{ font-size: 13.5px; }}
    .rail-source span {{ font-size: 12.5px; color: #c8d5ee !important; }}
    .rail-foot {{ margin-top: 18px; font-size: 12px; color: #c8d5ee !important; line-height: 1.5; padding: 0 4px; }}

    /* page head */
    .desk-head {{ display: flex; align-items: center; gap: 12px; margin: 0 0 4px; }}
    .desk-head h1 {{ margin: 0; padding: 0; font-size: 26px; font-weight: 800; letter-spacing: -.01em; color: {INK}; }}
    .source-pill {{ margin-left: auto; font-size: 13px; font-weight: 750; padding: 4px 10px; border-radius: 99px;
                    border: 2px solid {BRAND}; color: {BRAND}; background: #e7edf8; white-space: nowrap; }}
    .desk-lede {{ margin: 0 0 14px; font-size: 14.5px; color: {INK2}; }}

    /* report row: headline figures as one ruled band */
    .report {{ display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr); gap: 2px;
               border: 2px solid {LINE_STRONG}; border-radius: 12px; overflow: hidden; background: {LINE}; }}
    .report > div {{ padding: 10px 14px 12px; display: flex; flex-direction: column; gap: 4px; background: #fff; }}
    .report > div.reg {{ background: #f4f6f9; }}
    .report span {{ font-size: 12.5px; font-weight: 800; color: {INK2}; line-height: 1.2; }}
    .report b {{ font-size: 28px; line-height: 1.05; font-weight: 800; letter-spacing: -.01em; color: {INK};
                 font-variant-numeric: tabular-nums; }}
    .report em {{ font-style: normal; font-size: 12.5px; color: {INK2}; font-weight: 600; }}
    .report-note {{ margin: 10px 2px 20px !important; font-size: 13.5px; color: {INK2}; }}

    /* panels */
    [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
        border: 2px solid {LINE} !important; border-radius: 12px !important; background: #fff; }}
    .panel-h {{ font-size: 16px; font-weight: 800; color: {INK}; margin: 0 0 2px; }}
    .panel-sub {{ font-size: 12.5px; color: {INK2}; font-weight: 600; margin: 0 0 6px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12.5px; font-weight: 700; color: {INK2}; margin: 2px 0 4px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend i {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}

    /* clusters: one ruled band, a column per cluster */
    .segs {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 2px; border: 2px solid {LINE_STRONG};
             border-radius: 12px; overflow: hidden; background: {LINE}; margin-bottom: 18px; }}
    .segs > div {{ background: #fff; padding: 12px 14px; display: grid; gap: 6px; align-content: start; }}
    .segs h3 {{ margin: 0; padding: 0; font-size: 14.5px; font-weight: 800; color: {INK}; display: flex; gap: 8px; align-items: center; }}
    .segs h3 i {{ width: 10px; height: 10px; border-radius: 2px; flex: none; }}
    .segs p {{ margin: 0; font-size: 13.5px; color: {INK}; }}
    .segs small {{ font-size: 12.5px; color: {INK2}; font-weight: 600; border-top: 1px solid #d5dbe3; padding-top: 6px; }}

    /* widgets */
    [data-testid="stWidgetLabel"] p {{ font-size: 13px; font-weight: 800; color: {INK2}; }}
    [data-baseweb="select"] > div {{ border: 2px solid {LINE_STRONG} !important; border-radius: 10px !important; }}
    [data-baseweb="tag"] {{ background: {BRAND} !important; border-radius: 6px !important; }}
    [data-testid="stDataFrame"] {{ border: 1px solid #d5dbe3; border-radius: 8px; }}
    [data-testid="stCaptionContainer"] p {{ color: {INK2}; font-size: 13px; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def head(title: str, lede: str, pill: str = "Superstore · Jan 2015 – Dec 2018"):
    st.markdown(f'<div class="desk-head"><h1>{title}</h1><span class="source-pill">{pill}</span></div>'
                f'<p class="desk-lede">{lede}</p>', unsafe_allow_html=True)


def report(cells, note: str = ""):
    """cells: (label, value, sub, regular?) -> one ruled band like a printed report row."""
    html = "".join(f'<div class="{"reg" if reg else ""}"><span>{lab}</span><b>{val}</b><em>{sub}</em></div>'
                   for lab, val, sub, reg in cells)
    st.markdown(f'<section class="report">{html}</section>'
                + (f'<p class="report-note">{note}</p>' if note else '<div style="height:16px"></div>'),
                unsafe_allow_html=True)


def panel_head(title: str, sub: str = "", legend: dict | None = None):
    leg = ""
    if legend:
        leg = '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{k}</span>'
                                               for k, c in legend.items()) + "</div>"
    st.markdown(f'<div class="panel-h">{title}</div>' + (f'<div class="panel-sub">{sub}</div>' if sub else "") + leg,
                unsafe_allow_html=True)


def style(fig, height=340):
    fig.update_layout(
        template="simple_white", height=height, showlegend=False, title=None,
        font=dict(family=FONT, size=12.5, color=INK2),
        margin=dict(l=4, r=8, t=8, b=4), paper_bgcolor="#fff", plot_bgcolor="#fff",
        hoverlabel=dict(bgcolor="#fff", bordercolor=LINE_STRONG, font=dict(family=FONT, size=12.5, color=INK)),
        bargap=0.28,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE_STRONG, linewidth=1.5, ticks="", title_font=dict(size=12.5))
    fig.update_yaxes(showgrid=True, gridcolor="#e3e8ef", showline=False, ticks="", zeroline=False,
                     title_font=dict(size=12.5))
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
    head("Demand Intelligence", "train.csv was not found next to app.py. Upload the Superstore sales file to continue.",
         "No data")
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

fmt_money = st.column_config.NumberColumn(format="$%,.0f")

# ----------------------------------------------------------------- rail
with st.sidebar:
    st.markdown('<div class="rail-brand">Demand Intelligence</div>'
                '<div class="rail-sub">Sales forecasting for a US retailer</div>', unsafe_allow_html=True)
    page = st.radio("Page", ["Sales overview", "Forecast", "Anomalies", "Demand segments"],
                    label_visibility="collapsed")
    st.markdown(
        '<div class="rail-source">'
        '<div><b>Data</b><span>Superstore train.csv · 9,800 order lines</span></div>'
        '<div><b>Forecast model</b><span>SARIMA (1,1,1)(1,1,1,12)</span></div>'
        '<div><b>Backtest</b><span>Trained to Sep 2018, scored Oct – Dec 2018</span></div>'
        '</div>'
        '<div class="rail-foot">Muhammed Al Hameed M<br>AI &amp; Data Science Intern, XYlofy AI</div>',
        unsafe_allow_html=True)

# ----------------------------------------------------------------- sales overview
if page == "Sales overview":
    head("Sales overview", "Four years of orders by region and category. The filters scope every figure on this page.")
    c1, c2 = st.columns(2)
    regions = c1.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    cats = c2.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    f = df[df["Region"].isin(regions) & df["Category"].isin(cats)]
    if f.empty:
        st.info("Nothing matches this filter. Pick at least one region and one category.")
        st.stop()

    yr = f.groupby("Year")["Sales"].sum()
    growth = (yr.get(2018, 0) / yr.get(2017, 1) - 1) * 100
    m = monthly(f)
    peak = m.groupby(m.index.month).mean()
    hi, lo = pd.Timestamp(2000, int(peak.idxmax()), 1), pd.Timestamp(2000, int(peak.idxmin()), 1)
    report([
        ("Total sales", f"${f['Sales'].sum():,.0f}", "Jan 2015 – Dec 2018", False),
        ("Orders", f"{f['Order ID'].nunique():,}", f"${f.groupby('Order ID')['Sales'].sum().mean():,.0f} per order", False),
        ("Customers", f"{f['Customer ID'].nunique():,}", "Distinct customer IDs", False),
        ("2018 vs 2017", f"{growth:+.1f}%", "Change in yearly sales", True),
    ], note=f"{hi:%B} is the busiest month on average and {lo:%B} the quietest, "
            f"{peak.max() / peak.min():.1f}× apart. Stock should start building in September.")

    a, b = st.columns([1, 2], gap="medium")
    with a.container(border=True):
        panel_head("Sales by year", "US dollars")
        fig = go.Figure(go.Bar(x=yr.index.astype(str), y=yr.values, marker_color=BRAND, text=yr.values,
                               texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"))
        fig.update_yaxes(range=[0, yr.max() * 1.16], tickprefix="$")
        st.plotly_chart(style(fig), width="stretch")
    with b.container(border=True):
        panel_head("Sales by month", "Each point is one calendar month")
        fig = go.Figure(go.Scatter(x=m.index, y=m.values, mode="lines+markers", line=dict(color=BRAND, width=2),
                                   marker=dict(size=4), hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>"))
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(style(fig), width="stretch")

    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        panel_head("Sales by region and category", legend={k: v for k, v in CAT_COLORS.items() if k in cats})
        rc = f.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        fig = px.bar(rc, x="Region", y="Sales", color="Category", barmode="group", color_discrete_map=CAT_COLORS)
        fig.update_traces(hovertemplate="%{x}: $%{y:,.0f}<extra></extra>", marker_line_width=0)
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(style(fig), width="stretch")
    with b.container(border=True):
        panel_head("Top 10 sub-categories", "Total sales over four years")
        sc = f.groupby("Sub-Category")["Sales"].sum().sort_values().tail(10)
        fig = go.Figure(go.Bar(x=sc.values, y=sc.index, orientation="h", marker_color=BLUE, text=sc.values,
                               texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: $%{x:,.0f}<extra></extra>"))
        fig.update_xaxes(range=[0, sc.max() * 1.22], showticklabels=False, showline=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style(fig), width="stretch")

# ----------------------------------------------------------------- forecast
elif page == "Forecast":
    head("Forecast", "SARIMA forecast with a 95% range, for the whole business, one category or one region.")
    c1, c2 = st.columns([2, 1])
    options = ["All sales", "Furniture", "Office Supplies", "Technology", "Central", "East", "South", "West"]
    seg = c1.selectbox("Category or region", options, index=0)
    h = c2.select_slider("Months ahead", options=[1, 2, 3], value=3)
    s, fc, ci, mae, rmse, mape, test, bt = sarima_results(seg)
    fc, ci = fc.iloc[:h], ci.iloc[:h]

    report([
        ("Forecast total", f"${fc.sum():,.0f}", f"{fc.index[0]:%b} – {fc.index[-1]:%b %Y}", False),
        ("MAE", f"${mae:,.0f}", "Average miss per month", True),
        ("RMSE", f"${rmse:,.0f}", "Weights large misses more", True),
        ("MAPE", f"{mape:.1f}%", "Average miss as a share", True),
    ], note="Error figures come from the backtest: the model was trained to September 2018 and scored on "
            "October to December 2018, the peak quarter. The forecast itself uses all 48 months.")

    with st.container(border=True):
        panel_head(f"{seg}: last 24 months and the next {h}",
                   legend={"Actual": BRAND, "Forecast": AMBER, "95% range": "#f3dca8"})
        hist = s.iloc[-24:]
        bx = [hist.index[-1]] + list(ci.index)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bx + bx[::-1],
                                 y=[hist.values[-1]] + list(ci["upper"]) + list(ci["lower"][::-1]) + [hist.values[-1]],
                                 fill="toself", fillcolor="rgba(201,133,0,0.18)", line=dict(width=0), mode="lines",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", line=dict(color=BRAND, width=2),
                                 marker=dict(size=5), hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>Actual</extra>"))
        fig.add_trace(go.Scatter(x=[hist.index[-1]] + list(fc.index), y=[hist.values[-1]] + list(fc.values),
                                 mode="lines+markers", line=dict(color=AMBER, width=2, dash="dash"), marker=dict(size=7),
                                 hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra>Forecast</extra>"))
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(style(fig, height=400), width="stretch")

    a, b = st.columns([3, 2], gap="medium")
    with a.container(border=True):
        panel_head("Forecast by month", "With the 95% range")
        out = pd.DataFrame({"Month": fc.index.strftime("%b %Y"), "Forecast": fc.values.round(0),
                            "Low": ci["lower"].values.round(0), "High": ci["upper"].values.round(0)})
        st.dataframe(out, hide_index=True, width="stretch",
                     column_config={c: fmt_money for c in ["Forecast", "Low", "High"]})
    with b.container(border=True):
        panel_head("Backtest", "Actual against predicted, Oct – Dec 2018")
        bt_df = pd.DataFrame({"Month": test.index.strftime("%b %Y"), "Actual": test.values.round(0),
                              "Predicted": bt.values.round(0)})
        st.dataframe(bt_df, hide_index=True, width="stretch",
                     column_config={c: fmt_money for c in ["Actual", "Predicted"]})

# ----------------------------------------------------------------- anomalies
elif page == "Anomalies":
    head("Anomalies", "Weeks where sales broke from their normal pattern, found by two separate methods.")
    wd = weekly_anomalies()
    both = int((wd["iso"] & wd["zflag"]).sum())
    report([
        ("Weeks checked", f"{len(wd)}", "Weekly sales totals", False),
        ("Isolation Forest", f"{int(wd['iso'].sum())}", "Unusual against all four years", False),
        ("Rolling z-score", f"{int(wd['zflag'].sum())}", "Unusual against the 8 weeks around it", False),
        ("Flagged by both", f"{both}", "Check these first", True),
    ], note="Isolation Forest is set to expect 5% unusual weeks. The z-score flags a week more than two standard "
            "deviations from its 8-week rolling mean.")

    with st.container(border=True):
        panel_head("Weekly sales", legend={"Weekly sales": LINE_STRONG, "8-week mean": BLUE,
                                           "Isolation Forest": RED, "Z-score": AMBER})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wd["week"], y=wd["sales"], line=dict(color=LINE_STRONG, width=1.2),
                                 hovertemplate="Week of %{x|%d %b %Y}: $%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=wd["week"], y=wd["rolling_mean"], line=dict(color=BLUE, width=1.5, dash="dot"),
                                 hoverinfo="skip"))
        a_ = wd[wd["iso"]]
        fig.add_trace(go.Scatter(x=a_["week"], y=a_["sales"], mode="markers",
                                 marker=dict(color=RED, symbol="x-thin", size=12, line=dict(width=2.5, color=RED)),
                                 hovertemplate="%{x|%d %b %Y}: $%{y:,.0f}<extra>Isolation Forest</extra>"))
        z = wd[wd["zflag"]]
        fig.add_trace(go.Scatter(x=z["week"], y=z["sales"], mode="markers",
                                 marker=dict(color="#fff", size=10, line=dict(color=AMBER, width=2.5)),
                                 hovertemplate="%{x|%d %b %Y}: $%{y:,.0f}<extra>Z-score</extra>"))
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(style(fig, height=400), width="stretch")

    flagged = wd[wd["iso"] | wd["zflag"]].copy()
    flagged["Flagged by"] = np.where(flagged["iso"] & flagged["zflag"], "Both",
                                     np.where(flagged["iso"], "Isolation Forest", "Z-score"))
    flagged["Week of"] = flagged["week"].dt.strftime("%d %b %Y")
    flagged["Sales"] = flagged["sales"].round(0)
    pct = (flagged["sales"] / flagged["rolling_mean"] - 1) * 100
    flagged["Against 8-week mean"] = pct.map(lambda v: "No mean yet" if pd.isna(v) else f"{v:+.0f}%")
    with st.container(border=True):
        panel_head("Flagged weeks", f"{len(flagged)} weeks, oldest first")
        st.dataframe(flagged[["Week of", "Sales", "Against 8-week mean", "Flagged by"]], hide_index=True,
                     width="stretch", height=(len(flagged) + 1) * 35 + 3, column_config={"Sales": fmt_money})

# ----------------------------------------------------------------- demand segments
else:
    head("Demand segments", "The 17 sub-categories grouped by how they sell, each with its own stocking rule.")
    sub, ev = clusters()
    order = ["High volume, high volatility", "Growing, frequently bought", "Low volume, high value", "Low volume, mature"]
    cells = ""
    for name in order:
        g = sub[sub["segment"] == name].sort_values("total_sales", ascending=False)
        if g.empty:
            continue
        cells += (f'<div><h3><i style="background:{SEG_COLORS[name]}"></i>{name}</h3>'
                  f'<p>{", ".join(g["Sub-Category"])}</p><small>{g["strategy"].iloc[0]}</small></div>')
    st.markdown(f'<section class="segs">{cells}</section>', unsafe_allow_html=True)

    with st.container(border=True):
        panel_head("Clusters on the first two principal components",
                   f"K-Means with k = 4 on five standardised features. PC1 explains {ev[0]:.1%} of the variance, "
                   f"PC2 {ev[1]:.1%}. Bubble size is total sales.", legend={k: SEG_COLORS[k] for k in order})
        sub["label"] = np.where(sub["total_sales"] >= 30000, sub["Sub-Category"], "")
        fig = px.scatter(sub, x="PC1", y="PC2", color="segment", text="label", color_discrete_map=SEG_COLORS,
                         size="total_sales", size_max=34, custom_data=["Sub-Category", "total_sales", "growth_rate"])
        pos = {"Bookcases": "bottom center", "Appliances": "top left", "Chairs": "bottom center",
               "Accessories": "bottom center", "Supplies": "bottom center", "Storage": "middle right"}
        for tr in fig.data:
            tr.textposition = [pos.get(n, "top center") for n in sub.loc[sub["segment"] == tr.name, "Sub-Category"]]
        fig.update_traces(textfont=dict(size=12, color=INK), marker=dict(line=dict(color="#fff", width=1.5), opacity=.9),
                          hovertemplate="<b>%{customdata[0]}</b><br>Sales $%{customdata[1]:,.0f}"
                                        "<br>Growth %{customdata[2]:.0f}%<extra></extra>")
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(style(fig, height=470), width="stretch")

    t = sub[["Sub-Category", "segment", "total_sales", "growth_rate", "volatility", "strategy"]].copy()
    t.columns = ["Sub-category", "Cluster", "Total sales", "Growth 2015–18", "Monthly spread", "Stocking rule"]
    t["Cluster"] = pd.Categorical(t["Cluster"], order)
    t = t.sort_values(["Cluster", "Total sales"], ascending=[True, False])
    t["Growth 2015–18"] = t["Growth 2015–18"].map(lambda v: f"{v:+.0f}%")
    with st.container(border=True):
        panel_head("Sub-category detail", "Monthly spread is the standard deviation of monthly sales")
        st.dataframe(t, hide_index=True, width="stretch", height=(len(t) + 1) * 35 + 3,
                     column_config={"Total sales": fmt_money, "Monthly spread": fmt_money,
                                    "Stocking rule": st.column_config.TextColumn(width="large")})
