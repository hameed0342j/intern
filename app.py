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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Demand Intelligence", layout="wide", initial_sidebar_state="auto")

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
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.stButton), [data-testid="stSidebar"] .stButton {{ width: 100% !important; }}
    [data-testid="stSidebar"] .stButton button {{ width: 100%; min-height: 44px; border-radius: 10px; border: 2px solid #fff;
        background: #fff; transition: background-color .15s ease-out; }}
    [data-testid="stSidebar"] .stButton button p {{ color: {BRAND}; font-weight: 750; font-size: 14.5px; }}
    [data-testid="stSidebar"] .stButton button:hover {{ background: #e7edf8; }}
    [data-testid="stSidebar"] .stButton button:disabled {{ opacity: .55; }}
    [data-testid="stSidebar"] .st-key-reset_sim button {{ background: transparent; border-color: rgba(255,255,255,.4); min-height: 36px; }}
    [data-testid="stSidebar"] .st-key-reset_sim button p {{ color: #fff; font-size: 13px; }}
    [data-testid="stSidebar"] .st-key-reset_sim button:hover {{ background: rgba(255,255,255,.08); }}
    .rail-h {{ font-size: 13px; font-weight: 800; color: #c8d5ee !important; margin: 18px 4px 6px; }}
    .live {{ display: grid; gap: 6px; margin: 8px 0 6px; font-size: 12.5px; }}
    .live span {{ color: #c8d5ee !important; }}
    .live-track {{ height: 8px; border-radius: 4px; background: rgba(255,255,255,.18); overflow: hidden; }}
    .live-track i {{ display: block; height: 100%; background: #ffb000; transition: width .5s ease-out; }}
    .live-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #ffb000; margin-right: 6px;
                 animation: pulse 1s ease-in-out infinite; }}
    @keyframes pulse {{ 50% {{ opacity: .3; }} }}
    @media (prefers-reduced-motion: reduce) {{ .live-dot {{ animation: none; }} .live-track i {{ transition: none; }} }}
    .sim-pill {{ font-size: 12px; font-weight: 800; padding: 2px 8px; border-radius: 99px; background: #fff4d6;
                 border: 2px solid #8a6100; color: #000; margin-left: 8px; vertical-align: 2px; }}
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
    .actions {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; margin: 6px 0 4px; }}
    .actions > div {{ border: 2px solid {LINE}; border-radius: 10px; padding: 12px 14px; background: #fff; }}
    .actions .t {{ margin: 0 0 4px; padding: 0; font-size: 15px; font-weight: 800; color: {INK}; display: flex; align-items: center;
                   gap: 8px; flex-wrap: wrap; }}
    .actions .t i {{ width: 10px; height: 10px; border-radius: 2px; flex: none; }}
    .actions .t span {{ font-size: 12px; font-weight: 750; color: {INK2}; border: 1.5px solid {LINE}; border-radius: 99px; padding: 0 8px; }}
    .actions .why {{ font-size: 13.5px; color: {INK2}; margin: 0 0 6px; }}
    .actions ul {{ margin: 0; padding-left: 18px; font-size: 13.5px; color: {INK}; }}
    .actions li {{ margin-bottom: 3px; }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* phones */
    @media (max-width: 700px) {{
        .block-container {{ padding: 3.2rem .75rem 2.5rem; }}
        .desk-head {{ flex-wrap: wrap; gap: 6px; }}
        .desk-head h1 {{ font-size: 22px; }}
        .source-pill {{ margin-left: 0; font-size: 11.5px; }}
        .desk-lede {{ font-size: 13.5px; }}
        .report {{ grid-auto-flow: row; grid-template-columns: 1fr 1fr; }}
        .report > div {{ padding: 8px 10px 10px; }}
        .report b {{ font-size: 21px; }}
        .report em, .report span {{ font-size: 11.5px; }}
        .report-note {{ font-size: 12.5px; }}
        .segs {{ grid-template-columns: 1fr; }}
        .actions {{ grid-template-columns: 1fr; }}
        .panel-h {{ font-size: 15px; }}
        .legend {{ font-size: 11.5px; gap: 2px 10px; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ padding: 10px 8px !important; }}
        [data-testid="stHorizontalBlock"] {{ gap: .75rem !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def head(title: str, lede: str, pill: str | None = None):
    pill = pill or DATA_PILL
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


PCFG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


def show(fig, height=340, where=None):
    (where or st).plotly_chart(style(fig, height), width="stretch", config=PCFG)


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
def sarima_results(segment: str, frame: pd.DataFrame):
    """Backtest on the last 3 months, then refit on all data and forecast 3 months ahead."""
    s = monthly(segment_frame(frame, segment))
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
def weekly_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame
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
def clusters(frame: pd.DataFrame) -> tuple:
    df = frame
    sub = df.groupby("Sub-Category").agg(
        total_sales=("Sales", "sum"), avg_order_value=("Sales", "mean"),
        order_count=("Order ID", "nunique")).reset_index()
    full_years = [y for y, n in df.groupby("Year")["Month"].nunique().items() if n == 12]
    ys = df[df["Year"].isin(full_years)].groupby(["Sub-Category", "Year"])["Sales"].sum().unstack(fill_value=0)
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


@st.cache_data
def weak_spots(frame: pd.DataFrame):
    """Sub-categories that grew 10+ points slower than their category (or shrank), with a reason and actions."""
    full_years = sorted(y for y, n in frame.groupby("Year")["Month"].nunique().items() if n == 12)
    f0, f1 = full_years[0], full_years[-1]
    fy = frame[frame["Year"].isin([f0, f1])]
    sub = fy.pivot_table(index=["Category", "Sub-Category"], columns="Year", values="Sales", aggfunc="sum", fill_value=0)
    cat = fy.pivot_table(index="Category", columns="Year", values="Sales", aggfunc="sum")
    cat_growth = (cat[f1] / cat[f0] - 1) * 100
    w = sub.reset_index()
    w["growth"] = (w[f1] / w[f0].replace(0, np.nan) - 1) * 100
    w["cat_growth"] = w["Category"].map(cat_growth)
    w["gap"] = w["growth"] - w["cat_growth"]
    w["weak"] = (w["gap"] <= -10) | (w["growth"] < 0)

    reg = fy.pivot_table(index=["Category", "Sub-Category"], columns=["Region", "Year"], values="Sales",
                         aggfunc="sum", fill_value=0)
    regions = sorted(frame["Region"].unique())
    heat = pd.DataFrame({r: (reg[(r, f1)] / reg[(r, f0)].replace(0, np.nan) - 1) * 100 for r in regions}).fillna(0)
    heat = heat.sort_index()

    ms = frame.groupby(["Sub-Category", frame["Order Date"].dt.to_period("M")])["Sales"].sum().unstack(fill_value=0)
    cv = ms.std(axis=1) / ms.mean(axis=1)
    w["cv"] = w["Sub-Category"].map(cv)
    leader = w.sort_values("growth").groupby("Category").tail(1).set_index("Category")["Sub-Category"]
    whys, acts = [], []
    for _, r in w.iterrows():
        h = heat.loc[(r["Category"], r["Sub-Category"])]
        wr, wv, br, bv = h.idxmin(), h.min(), h.idxmax(), h.max()
        lead = leader[r["Category"]]
        weak_region = f"weakest in {wr} ({wv:+.0f}%)"
        if r["growth"] < 0:
            whys.append(f"Sales fell {abs(r['growth']):.0f}% while {r['Category']} grew {r['cat_growth']:+.0f}%; {weak_region}.")
            a = ["Review price and range against competitors and drop the slowest-selling products.",
                 f"Bundle it with {lead}, the fastest grower in {r['Category']}."]
        else:
            whys.append(f"Grew {r['growth']:+.0f}% against {r['cat_growth']:+.0f}% for {r['Category']}, "
                        f"{abs(r['gap']):.0f} points behind; {weak_region}.")
            a = [f"Cross-sell it with {lead} at checkout and in promotions."]
        a.append(f"Target {wr} first; copy what works in {br} ({bv:+.0f}%).")
        a.append("Demand is erratic: hold extra safety stock and promote in quiet months." if r["cv"] > cv.median()
                 else "Demand is steady: a price or range refresh will help more than stocking changes.")
        acts.append(a)
    w["why"], w["actions"] = whys, acts
    return w, cat_growth, heat


CHURN_DAYS = 180
CHURN_FEATS = ["Recency (days)", "Orders", "Spend", "Average order", "Tenure (days)", "Categories"]


def rfm(frame: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Recency, frequency, monetary and related features using only orders on or before `cut`."""
    h = frame[frame["Order Date"] <= cut]
    g = h.drop_duplicates("Order ID").groupby("Customer ID")
    spend = h.groupby("Customer ID")["Sales"].sum()
    n = g.size()
    first, last = g["Order Date"].min(), g["Order Date"].max()
    X = pd.DataFrame({"Recency (days)": (cut - last).dt.days, "Orders": n, "Spend": spend, "Average order": spend / n,
                      "Tenure (days)": (cut - first).dt.days,
                      "Categories": h.groupby("Customer ID")["Category"].nunique()})
    X["Usual gap (days)"] = (last - first).dt.days / (n - 1).replace(0, np.nan)
    return X


@st.cache_data(show_spinner="Scoring customers…")
def churn(frame: pd.DataFrame) -> dict:
    """Churned = no order in the CHURN_DAYS after a cut-off. Features come only from before the cut-off."""
    end = frame["Order Date"].max()
    cut = end - pd.Timedelta(days=CHURN_DAYS)
    X = rfm(frame, cut)
    returned = frame.loc[frame["Order Date"] > cut, "Customer ID"].unique()
    y = (~X.index.isin(returned)).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    Xm = np.log1p(X[CHURN_FEATS])
    p = cross_val_predict(model, Xm, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, p)
    model.fit(Xm, y)
    info = frame.drop_duplicates("Customer ID").set_index("Customer ID")[["Customer Name", "Segment", "Region"]]
    bt = X.assign(churned=y * 100).join(info)
    bt["band"] = pd.qcut(bt["Recency (days)"].rank(method="first"), 4,
                         labels=["Most recent 25%", "Second 25%", "Third 25%", "Least recent 25%"])
    now = rfm(frame, end).join(info)
    now["Churn probability"] = model.predict_proba(np.log1p(now[CHURN_FEATS]))[:, 1] * 100
    now["Overdue ×"] = now["Recency (days)"] / now["Usual gap (days)"].fillna(365)
    now["Last 12 months"] = (frame[frame["Order Date"] > end - pd.Timedelta(days=365)]
                             .groupby("Customer ID")["Sales"].sum().reindex(now.index).fillna(0))
    now["At risk"] = (now["Recency (days)"] > 90) & (now["Overdue ×"] > 1.5)
    return dict(auc=roc_auc_score(y, p), fpr=fpr, tpr=tpr, rate=y.mean() * 100, cut=cut, end=end, n=len(y),
                by_band=bt.groupby("band", observed=True)["churned"].mean(),
                by_seg=bt.groupby("Segment")["churned"].mean().sort_values(),
                by_reg=bt.groupby("Region")["churned"].mean().sort_values(),
                coefs=pd.Series(model[-1].coef_[0], index=CHURN_FEATS).sort_values(), now=now)


@st.cache_data
def load_games():
    for p in [APP_DIR / "vgsales.csv", APP_DIR / "data" / "vgsales.csv", Path.cwd() / "vgsales.csv"]:
        if p.is_file():
            g = pd.read_csv(p).dropna(subset=["Year"])
            g["Year"] = g["Year"].astype(int)
            return g[g["Year"] <= 2016]
    return None


def genre_momentum(g: pd.DataFrame, col: str) -> pd.DataFrame:
    e = g[(g["Year"] >= 2005) & (g["Year"] <= 2010)].groupby("Genre")[col].sum()
    l_ = g[(g["Year"] >= 2011) & (g["Year"] <= 2016)].groupby("Genre")[col].sum()
    m = pd.DataFrame({"early": e / e.sum() * 100, "late": l_ / l_.sum() * 100}).fillna(0)
    m["change"] = m["late"] - m["early"]
    return m.sort_values("change")


# ----------------------------------------------------------------- simulated import
ORDER_LEVEL = ["Ship Mode", "Customer ID", "Customer Name", "Segment", "Country", "City", "State", "Postal Code", "Region"]
TICK_SECONDS = 0.7
TICKS_PER_MONTH = 14


def simulate_month(base: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Fake order lines for one month: resampled from the same calendar month in past years, grown ~8-22%
    year on year, with new order IDs, dates inside the month and noise on every line's sales value.
    Seeded by the month, so the same month always produces the same orders."""
    rng = np.random.default_rng(int(month.strftime("%Y%m")))
    hist = base[base["Order Date"].dt.month == month.month]
    recent = hist[hist["Year"] == hist["Year"].max()]
    n_lines = max(40, int(len(recent) * rng.uniform(1.08, 1.22)))
    lines = hist.iloc[rng.integers(0, len(hist), n_lines)].copy().reset_index(drop=True)
    lines["Sales"] = (lines["Sales"] * rng.lognormal(0, 0.18, n_lines)).round(2)

    sizes = []
    while sum(sizes) < n_lines:
        sizes.append(int(rng.choice([1, 2, 3, 4, 5], p=[.45, .25, .15, .1, .05])))
    sizes[-1] -= sum(sizes) - n_lines
    sizes = [k for k in sizes if k > 0]
    first = np.cumsum([0] + sizes[:-1])
    order_of_line = np.repeat(np.arange(len(sizes)), sizes)
    for c in ORDER_LEVEL:
        if c in lines:
            lines[c] = lines[c].values[first][order_of_line]

    lag = (lines["Ship Date"] - lines["Order Date"]).dt.days.values[first]
    day = rng.integers(0, pd.Period(month, "M").days_in_month, len(sizes))
    order_date = month + pd.to_timedelta(day, unit="D")
    lines["Order Date"] = order_date[order_of_line]
    lines["Ship Date"] = (order_date + pd.to_timedelta(lag, unit="D"))[order_of_line]
    ids = rng.choice(np.arange(100000, 1000000), len(sizes), replace=False)
    lines["Order ID"] = [f"CA-{month.year}-{ids[k]}" for k in order_of_line]
    lines["Year"], lines["Month"] = month.year, month.month
    return lines.sort_values(["Order Date", "Order ID"]).reset_index(drop=True)


def init_sim():
    for k, v in {"sim_done": [], "sim_live": None, "sim_queue": None, "sim_month": None, "sim_tick": 0.0}.items():
        st.session_state.setdefault(k, v)


def start_import(model_frame: pd.DataFrame):
    month = (model_frame["Order Date"].max().to_period("M") + 1).to_timestamp()
    rows = simulate_month(BASE_DF, month)
    if "Row ID" in rows:
        rows["Row ID"] = np.arange(len(rows)) + int(model_frame["Row ID"].max()) + 1
    st.session_state.sim_month = month
    st.session_state.sim_queue = rows
    st.session_state.sim_live = rows.iloc[:0]
    st.session_state.sim_tick = 0.0


def pump():
    """Runs every tick while an import is streaming: moves the next batch of order lines into the data."""
    import time
    q = st.session_state.sim_queue
    if q is None:
        return
    now = time.time()
    if now - st.session_state.sim_tick < TICK_SECONDS * 0.9:
        return
    st.session_state.sim_tick = now
    n = max(1, int(np.ceil((len(q) + len(st.session_state.sim_live)) / TICKS_PER_MONTH)))
    st.session_state.sim_live = pd.concat([st.session_state.sim_live, q.iloc[:n]], ignore_index=True)
    st.session_state.sim_queue = q.iloc[n:]
    if st.session_state.sim_queue.empty:
        st.session_state.sim_done.append(st.session_state.sim_live)
        st.session_state.sim_live, st.session_state.sim_queue, st.session_state.sim_month = None, None, None
    st.rerun()


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

BASE_DF = load_data(DATA_PATH)
missing = REQUIRED - set(BASE_DF.columns)
if missing:
    st.error(f"This CSV is missing required columns: {', '.join(sorted(missing))}. Use the Superstore train.csv.")
    st.stop()

init_sim()
STREAMING = st.session_state.sim_queue is not None
# model_df: base + fully imported months (feeds the models); df: plus the month that is still streaming in
model_df = pd.concat([BASE_DF, *st.session_state.sim_done], ignore_index=True) if st.session_state.sim_done else BASE_DF
df = pd.concat([model_df, st.session_state.sim_live], ignore_index=True) if STREAMING else model_df
N_SIM = len(df) - len(BASE_DF)
D0, D1 = df["Order Date"].min(), df["Order Date"].max()
DATA_PILL = f"Superstore · {D0:%b %Y} – {D1:%b %Y}"

fmt_money = st.column_config.NumberColumn(format="$%,.0f")

# ----------------------------------------------------------------- rail
with st.sidebar:
    st.markdown('<div class="rail-brand">Demand Intelligence</div>'
                '<div class="rail-sub">Sales forecasting for a US retailer</div>', unsafe_allow_html=True)
    page = st.radio("Page", ["Sales overview", "Forecast", "Anomalies", "Demand segments", "Weak spots",
                             "Customer churn", "Games market"],
                    label_visibility="collapsed")
    st.markdown('<div class="rail-h">Live data</div>', unsafe_allow_html=True)
    nxt = (model_df["Order Date"].max().to_period("M") + 1).to_timestamp()
    if STREAMING:
        live, total = len(st.session_state.sim_live), len(st.session_state.sim_live) + len(st.session_state.sim_queue)
        st.markdown(f'<div class="live"><b><i class="live-dot"></i>Importing {st.session_state.sim_month:%B %Y}</b>'
                    f'<div class="live-track"><i style="width:{live / total:.0%}"></i></div>'
                    f'<span>{live} of {total} order lines · ${st.session_state.sim_live["Sales"].sum():,.0f}</span></div>',
                    unsafe_allow_html=True)
        st.button("Importing…", disabled=True, key="import_sim")
    elif st.button(f"Import {nxt:%B %Y} orders", key="import_sim"):
        start_import(model_df)
        st.rerun()
    if st.session_state.sim_done or STREAMING:
        if st.button("Remove imported data", key="reset_sim"):
            for k in ["sim_done", "sim_live", "sim_queue", "sim_month", "sim_tick"]:
                del st.session_state[k]
            st.rerun()
    sim_note = f" + {N_SIM:,} simulated" if N_SIM else ""
    st.markdown(
        '<div class="rail-source">'
        f'<div><b>Data</b><span>Superstore train.csv · {len(BASE_DF):,} order lines{sim_note}</span></div>'
        '<div><b>Forecast model</b><span>SARIMA (1,1,1)(1,1,1,12)</span></div>'
        f'<div><b>Backtest</b><span>Scored on the last 3 complete months</span></div>'
        '</div>'
        '<div class="rail-foot">Muhammed Al Hameed M<br>AI &amp; Data Science Intern, XYlofy AI</div>',
        unsafe_allow_html=True)

# ----------------------------------------------------------------- sales overview
if page == "Sales overview":
    head("Sales overview", f"Orders from {D0:%B %Y} to {D1:%B %Y} by region and category. The filters scope every figure on this page.")
    c1, c2 = st.columns(2)
    regions = c1.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    cats = c2.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    f = df[df["Region"].isin(regions) & df["Category"].isin(cats)]
    if f.empty:
        st.info("Nothing matches this filter. Pick at least one region and one category.")
        st.stop()

    yr = f.groupby("Year")["Sales"].sum()
    full_years = [y for y, n in f.groupby("Year")["Month"].nunique().items() if n == 12]
    y1, y0 = (full_years[-1], full_years[-2]) if len(full_years) >= 2 else (yr.index[-1], yr.index[0])
    growth = (yr.get(y1, 0) / yr.get(y0, 1) - 1) * 100
    m = monthly(f)
    peak = m.groupby(m.index.month).mean()
    hi, lo = pd.Timestamp(2000, int(peak.idxmax()), 1), pd.Timestamp(2000, int(peak.idxmin()), 1)
    report([
        ("Total sales", f"${f['Sales'].sum():,.0f}", f"{D0:%b %Y} – {D1:%b %Y}", False),
        ("Orders", f"{f['Order ID'].nunique():,}", f"${f.groupby('Order ID')['Sales'].sum().mean():,.0f} per order", False),
        ("Customers", f"{f['Customer ID'].nunique():,}", "Distinct customer IDs", False),
        (f"{y1} vs {y0}", f"{growth:+.1f}%", "Change in yearly sales, full years", True),
    ], note=f"{hi:%B} is the busiest month on average and {lo:%B} the quietest, "
            f"{peak.max() / peak.min():.1f}× apart. Stock should start building in September.")

    sims = [*st.session_state.sim_done] + ([st.session_state.sim_live] if STREAMING else [])
    if sims:
        cur = sims[-1]
        month = st.session_state.sim_month if STREAMING else cur["Order Date"].min().to_period("M").to_timestamp()
        cur = cur[cur["Region"].isin(regions) & cur["Category"].isin(cats)]
        status = "importing now" if STREAMING else "imported"
        with st.container(border=True):
            st.markdown(f'<div class="panel-h">{month:%B %Y}<span class="sim-pill">Simulated · {status}</span></div>'
                        f'<div class="panel-sub">{cur["Order ID"].nunique():,} orders · {len(cur):,} order lines · '
                        f'${cur["Sales"].sum():,.0f} so far. Every chart and figure on every page includes these rows.</div>',
                        unsafe_allow_html=True)
            a, b = st.columns([3, 2], gap="medium")
            days = pd.date_range(month, month + pd.offsets.MonthEnd(0), freq="D")
            daily = cur.groupby(cur["Order Date"].dt.normalize())["Sales"].sum().reindex(days, fill_value=0)
            fig = go.Figure(go.Bar(x=daily.index, y=daily.values, marker_color=AMBER,
                                   hovertemplate="%{x|%d %b}: $%{y:,.0f}<extra></extra>"))
            fig.update_yaxes(tickprefix="$")
            fig.update_xaxes(range=[days[0] - pd.Timedelta(hours=18), days[-1] + pd.Timedelta(hours=18)], tickformat="%d %b")
            a.plotly_chart(style(fig, height=230), width="stretch", config=PCFG)
            latest = cur.tail(7).iloc[::-1]
            b.dataframe(pd.DataFrame({"Date": latest["Order Date"].dt.strftime("%d %b"), "Order": latest["Order ID"],
                                      "Sub-category": latest["Sub-Category"], "Region": latest["Region"],
                                      "Sales": latest["Sales"].round(0)}),
                        hide_index=True, width="stretch", height=8 * 35 + 3, column_config={"Sales": fmt_money})
        st.write("")

    a, b = st.columns([1, 2], gap="medium")
    with a.container(border=True):
        panel_head("Sales by year", "US dollars" + (" · latest year is partial" if len(full_years) < len(yr) else ""))
        fig = go.Figure(go.Bar(x=yr.index.astype(str), y=yr.values, marker_color=BRAND, text=yr.values,
                               texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: $%{y:,.0f}<extra></extra>"))
        fig.update_yaxes(range=[0, yr.max() * 1.16], tickprefix="$")
        st.plotly_chart(style(fig), width="stretch", config=PCFG)
    with b.container(border=True):
        panel_head("Sales by month", "Each point is one calendar month" + (" · the last point is still importing" if STREAMING else ""))
        fig = go.Figure(go.Scatter(x=m.index, y=m.values, mode="lines+markers", line=dict(color=BRAND, width=2),
                                   marker=dict(size=4), hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>"))
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(style(fig), width="stretch", config=PCFG)

    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        panel_head("Sales by region and category", legend={k: v for k, v in CAT_COLORS.items() if k in cats})
        rc = f.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        fig = px.bar(rc, x="Region", y="Sales", color="Category", barmode="group", color_discrete_map=CAT_COLORS)
        fig.update_traces(hovertemplate="%{x}: $%{y:,.0f}<extra></extra>", marker_line_width=0)
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(style(fig), width="stretch", config=PCFG)
    with b.container(border=True):
        panel_head("Top 10 sub-categories", f"Total sales, {D0:%b %Y} – {D1:%b %Y}")
        sc = f.groupby("Sub-Category")["Sales"].sum().sort_values().tail(10)
        fig = go.Figure(go.Bar(x=sc.values, y=sc.index, orientation="h", marker_color=BLUE, text=sc.values,
                               texttemplate="$%{text:,.0f}", textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: $%{x:,.0f}<extra></extra>"))
        fig.update_xaxes(range=[0, sc.max() * 1.22], showticklabels=False, showline=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(style(fig), width="stretch", config=PCFG)

# ----------------------------------------------------------------- forecast
elif page == "Forecast":
    head("Forecast", "SARIMA forecast with a 95% range, for the whole business, one category or one region.")
    c1, c2 = st.columns([2, 1])
    options = ["All sales", "Furniture", "Office Supplies", "Technology", "Central", "East", "South", "West"]
    seg = c1.selectbox("Category or region", options, index=0)
    h = c2.select_slider("Months ahead", options=[1, 2, 3], value=3)
    s, fc, ci, mae, rmse, mape, test, bt = sarima_results(seg, model_df)
    fc, ci = fc.iloc[:h], ci.iloc[:h]

    report([
        ("Forecast total", f"${fc.sum():,.0f}", f"{fc.index[0]:%b} – {fc.index[-1]:%b %Y}", False),
        ("MAE", f"${mae:,.0f}", "Average miss per month", True),
        ("RMSE", f"${rmse:,.0f}", "Weights large misses more", True),
        ("MAPE", f"{mape:.1f}%", "Average miss as a share", True),
    ], note=f"Error figures come from the backtest: the model was trained to {s.index[-4]:%B %Y} and scored on "
            f"{test.index[0]:%B} to {test.index[-1]:%B %Y}. The forecast itself uses all {len(s)} months."
            + (" The month that is still importing joins the model once it is complete." if STREAMING else ""))

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
        st.plotly_chart(style(fig, height=400), width="stretch", config=PCFG)

    a, b = st.columns([3, 2], gap="medium")
    with a.container(border=True):
        panel_head("Forecast by month", "With the 95% range")
        out = pd.DataFrame({"Month": fc.index.strftime("%b %Y"), "Forecast": fc.values.round(0),
                            "Low": ci["lower"].values.round(0), "High": ci["upper"].values.round(0)})
        st.dataframe(out, hide_index=True, width="stretch",
                     column_config={c: fmt_money for c in ["Forecast", "Low", "High"]})
    with b.container(border=True):
        panel_head("Backtest", f"Actual against predicted, {test.index[0]:%b} – {test.index[-1]:%b %Y}")
        bt_df = pd.DataFrame({"Month": test.index.strftime("%b %Y"), "Actual": test.values.round(0),
                              "Predicted": bt.values.round(0)})
        st.dataframe(bt_df, hide_index=True, width="stretch",
                     column_config={c: fmt_money for c in ["Actual", "Predicted"]})

# ----------------------------------------------------------------- anomalies
elif page == "Anomalies":
    head("Anomalies", "Weeks where sales broke from their normal pattern, found by two separate methods.")
    wd = weekly_anomalies(df)
    both = int((wd["iso"] & wd["zflag"]).sum())
    report([
        ("Weeks checked", f"{len(wd)}", "Weekly sales totals", False),
        ("Isolation Forest", f"{int(wd['iso'].sum())}", "Unusual against the whole history", False),
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
        st.plotly_chart(style(fig, height=400), width="stretch", config=PCFG)

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

# ----------------------------------------------------------------- weak spots
elif page == "Weak spots":
    head("Weak spots", "Sub-categories growing slower than their own category, where they are weakest, and what to do about it.")
    w, cat_growth, heat = weak_spots(model_df)
    weak = w[w["weak"]].sort_values("gap")
    report([
        ("Furniture", f"{cat_growth['Furniture']:+.0f}%", "Category growth, first to last full year", False),
        ("Office Supplies", f"{cat_growth['Office Supplies']:+.0f}%", "Category growth, first to last full year", False),
        ("Technology", f"{cat_growth['Technology']:+.0f}%", "Category growth, first to last full year", False),
        ("Weak spots", f"{len(weak)}", "Behind their category by 10+ points, or shrinking", True),
    ], note="A sub-category is a weak spot when it grew at least 10 percentage points slower than its own category, "
            "or when its sales fell. Growth compares the first and last full years in the data.")

    with st.container(border=True):
        panel_head("Growth against its own category",
                   "Percentage points above or below the category's growth, grouped Furniture, Office Supplies, Technology. Red bars are weak spots.",
                   legend={"Weak spot": RED, "Keeping pace or ahead": GREEN})
        wc = w.sort_values(["Category", "gap"], ascending=[True, True])
        labels = list(wc["Sub-Category"])
        fig = go.Figure(go.Bar(x=wc["gap"], y=labels, orientation="h",
                               marker_color=[RED if k else GREEN for k in wc["weak"]],
                               customdata=np.stack([wc["growth"], wc["cat_growth"]], axis=1),
                               hovertemplate="<b>%{y}</b><br>Own growth %{customdata[0]:+.0f}%"
                                             "<br>Category %{customdata[1]:+.0f}%<br>Gap %{x:+.0f} pts<extra></extra>"))
        fig.update_xaxes(ticksuffix=" pts", zeroline=True, zerolinecolor=LINE_STRONG, zerolinewidth=1.5, showgrid=True,
                         gridcolor="#eef2f7")
        fig.update_yaxes(showgrid=False, autorange="reversed")
        show(fig, height=520)

    with st.container(border=True):
        panel_head("Growth by region", "Sales growth for each sub-category in each region, first to last full year. "
                                       "Red cells are shrinking.")
        fig = go.Figure(go.Heatmap(
            z=heat.values.clip(-100, 200), x=heat.columns, y=[s for c, s in heat.index],
            text=[[f"{v:+.0f}%" for v in row] for row in heat.values], texttemplate="%{text}",
            textfont=dict(size=10.5), colorscale=[[0, RED], [1 / 3, "#ffffff"], [1, GREEN]], zmin=-100, zmax=200,
            showscale=False, xgap=2, ygap=2, hovertemplate="%{y}, %{x}: %{text}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(side="top", showline=False)
        show(fig, height=560)

    with st.container(border=True):
        panel_head("What to do about each weak spot", "Actions follow from why each one is weak")
        st.markdown('<div class="actions">' + "".join(
            f'<div><div class="t"><i style="background:{RED}"></i>{r["Sub-Category"]}<span>{r["Category"]}</span></div>'
            f'<p class="why">{r["why"]}</p><ul>{"".join(f"<li>{a}</li>" for a in r["actions"])}</ul></div>'
            for _, r in weak.iterrows()) + "</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------- churn
elif page == "Customer churn":
    head("Customer churn", f"Which customers stop buying. Churned means no order in the {CHURN_DAYS} days after a cut-off date.")
    c = churn(model_df)
    now = c["now"]
    risk = now[now["At risk"]].sort_values("Last 12 months", ascending=False)
    report([
        ("Customers", f"{c['n']:,}", f"Scored at the cut-off, {c['cut']:%d %b %Y}", False),
        ("Churn rate", f"{c['rate']:.0f}%", f"No order from {c['cut']:%b %Y} to {c['end']:%b %Y}", False),
        ("Model AUC", f"{c['auc']:.2f}", "0.50 is chance, 1.00 is perfect", True),
        ("At risk now", f"{len(risk)}", f"${risk['Last 12 months'].sum():,.0f} spent in the last 12 months", True),
    ], note=(f"Backtest: a logistic regression on six order-history features, scored with 5-fold cross-validation and "
             f"using only orders before the cut-off. An AUC of {c['auc']:.2f} means "
             + ("order history barely predicts who comes back in this data, so the at-risk list below uses a "
                "transparent rule instead of the model: no order for 90+ days and 1.5× longer than the customer's "
                "usual gap between orders." if c["auc"] < 0.6 else
                "the model ranks churners above returners better than chance.")))

    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        panel_head("ROC curve", "How well the model separates churners from returners",
                   legend={"Model": BRAND, "Chance": LINE})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=LINE, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", line=dict(color=BRAND, width=2.5),
                                 hovertemplate="False alarms %{x:.0%}<br>Churners caught %{y:.0%}<extra></extra>"))
        fig.update_xaxes(title="False alarm rate", tickformat=".0%", range=[0, 1])
        fig.update_yaxes(title="Churners caught", tickformat=".0%", range=[0, 1])
        show(fig, height=320)
    with b.container(border=True):
        panel_head("Churn rate by recency", "Customers split into quarters by days since their last order at the cut-off")
        bb = c["by_band"]
        fig = go.Figure(go.Bar(x=bb.index.astype(str), y=bb.values, marker_color=[BLUE, BLUE, AMBER, RED], text=bb.values,
                               texttemplate="%{text:.0f}%", textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:.0f}% churned<extra></extra>"))
        fig.update_yaxes(ticksuffix="%", range=[0, max(bb.max() * 1.3, 10)])
        show(fig, height=320)

    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        panel_head("Churn rate by segment and region", "Share of each group with no order after the cut-off")
        grp = pd.concat([c["by_seg"].rename(lambda s: f"{s} (segment)"), c["by_reg"].rename(lambda s: f"{s} (region)")])
        fig = go.Figure(go.Bar(x=grp.values, y=grp.index, orientation="h", text=grp.values, texttemplate="%{text:.0f}%",
                               textposition="outside", cliponaxis=False,
                               marker_color=[BRAND if "segment" in i else BLUE for i in grp.index],
                               hovertemplate="%{y}: %{x:.0f}%<extra></extra>"))
        fig.update_xaxes(ticksuffix="%", range=[0, grp.max() * 1.25])
        fig.update_yaxes(showgrid=False)
        show(fig, height=320)
    with b.container(border=True):
        panel_head("What the model leans on", "Standardised coefficients: right pushes towards churn, left towards returning",
                   legend={"Raises churn": RED, "Lowers churn": GREEN})
        co = c["coefs"]
        fig = go.Figure(go.Bar(x=co.values, y=co.index, orientation="h",
                               marker_color=[RED if v > 0 else GREEN for v in co.values],
                               hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fig.update_xaxes(zeroline=True, zerolinecolor=LINE_STRONG, zerolinewidth=1.5)
        fig.update_yaxes(showgrid=False)
        show(fig, height=320)

    with st.container(border=True):
        top = risk.head(15).iloc[::-1]
        panel_head("At-risk customers by value", f"Top {len(top)} of {len(risk)} overdue customers, ranked by what they "
                                                 f"spent in the last 12 months")
        fig = go.Figure(go.Bar(x=top["Last 12 months"], y=top["Customer Name"], orientation="h", marker_color=RED,
                               text=top["Last 12 months"], texttemplate="$%{text:,.0f}", textposition="outside",
                               cliponaxis=False, customdata=np.stack([top["Recency (days)"], top["Segment"]], axis=1),
                               hovertemplate="<b>%{y}</b><br>%{customdata[1]}<br>Last order %{customdata[0]} days ago"
                                             "<br>Spent $%{x:,.0f} in 12 months<extra></extra>"))
        fig.update_xaxes(range=[0, top["Last 12 months"].max() * 1.25], showticklabels=False, showline=False)
        fig.update_yaxes(showgrid=False)
        show(fig, height=460)
        t = risk[["Customer Name", "Segment", "Region", "Recency (days)", "Usual gap (days)", "Last 12 months",
                  "Churn probability"]].copy()
        t["Usual gap (days)"] = t["Usual gap (days)"].round(0)
        st.dataframe(t, hide_index=True, width="stretch", height=8 * 35 + 3,
                     column_config={"Last 12 months": fmt_money,
                                    "Churn probability": st.column_config.NumberColumn("Model churn %", format="%.0f%%")})

    with st.container(border=True):
        panel_head("What to do")
        seg_hi = c["by_seg"].idxmax()
        st.markdown(
            '<div class="actions">'
            f'<div><div class="t"><i style="background:{RED}"></i>Win back the top accounts</div><ul>'
            f'<li>Call the {min(15, len(risk))} highest-value overdue customers this month.</li>'
            f'<li>Offer them a discount on the category they buy most.</li></ul></div>'
            f'<div><div class="t"><i style="background:{AMBER}"></i>Watch the {seg_hi} segment</div><ul>'
            f'<li>It has the highest churn rate, {c["by_seg"].max():.0f}%.</li>'
            f'<li>Send a reorder reminder when a customer passes their usual gap between orders.</li></ul></div>'
            f'<div><div class="t"><i style="background:{BLUE}"></i>Test before scaling</div><ul>'
            f'<li>Contact half of the at-risk list and hold back the other half.</li>'
            f'<li>Compare how many in each half order again within 90 days.</li></ul></div>'
            '</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- games
elif page == "Games market":
    head("Games market", "The second data source: 16,598 video game titles with unit sales by region, 1980 – 2016.",
         "VGChartz · units in millions")
    g = load_games()
    if g is None:
        st.info("vgsales.csv was not found next to app.py, so the games pages cannot be shown.")
        st.stop()
    REG = {"Global": "Global_Sales", "North America": "NA_Sales", "Europe": "EU_Sales", "Japan": "JP_Sales",
           "Other": "Other_Sales"}
    c1, c2 = st.columns([1, 2])
    region = c1.selectbox("Region", list(REG), index=0)
    y0, y1 = c2.select_slider("Release years", options=list(range(1980, 2017)), value=(1995, 2016))
    col = REG[region]
    gf = g[(g["Year"] >= y0) & (g["Year"] <= y1)]
    genre = gf.groupby("Genre")[col].sum().sort_values()
    plat = gf.groupby("Platform")[col].sum().sort_values().tail(10)
    report([
        ("Titles", f"{len(gf):,}", f"Released {y0} – {y1}", False),
        ("Units sold", f"{gf[col].sum():,.0f}M", f"{region} sales", False),
        ("Top genre", genre.index[-1], f"{genre.iloc[-1]:,.0f}M units, {genre.iloc[-1] / genre.sum():.0%} of the total", False),
        ("Top platform", plat.index[-1], f"{plat.iloc[-1]:,.0f}M units", True),
    ], note="Sales are in millions of units, not dollars. Titles from 2016 onwards are incomplete in the source data.")

    with st.container(border=True):
        panel_head("Units sold by release year", "Stacked by region",
                   legend={"North America": BRAND, "Europe": BLUE, "Japan": RED, "Other": LINE})
        yr = gf.groupby("Year")[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum()
        fig = go.Figure()
        for c, name, colr in [("NA_Sales", "North America", BRAND), ("EU_Sales", "Europe", BLUE),
                              ("JP_Sales", "Japan", RED), ("Other_Sales", "Other", LINE)]:
            fig.add_trace(go.Bar(x=yr.index, y=yr[c], name=name, marker_color=colr,
                                 hovertemplate=f"%{{x}} · {name}: %{{y:,.1f}}M<extra></extra>"))
        fig.update_layout(barmode="stack", bargap=0.15)
        fig.update_yaxes(ticksuffix="M")
        show(fig, height=330)

    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        panel_head("Units by genre", f"{region}, {y0} – {y1}")
        fig = go.Figure(go.Bar(x=genre.values, y=genre.index, orientation="h", marker_color=BRAND, text=genre.values,
                               texttemplate="%{text:,.0f}M", textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:,.1f}M<extra></extra>"))
        fig.update_xaxes(range=[0, genre.max() * 1.22], showticklabels=False, showline=False)
        fig.update_yaxes(showgrid=False)
        show(fig, height=400)
    with b.container(border=True):
        panel_head("Where each genre sells", "Share of each genre's units by region",
                   legend={"North America": BRAND, "Europe": BLUE, "Japan": RED, "Other": LINE})
        mix = gf.groupby("Genre")[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].sum()
        mix = mix.div(mix.sum(axis=1), axis=0).loc[genre.index] * 100
        fig = go.Figure()
        for c, name, colr in [("NA_Sales", "North America", BRAND), ("EU_Sales", "Europe", BLUE),
                              ("JP_Sales", "Japan", RED), ("Other_Sales", "Other", LINE)]:
            fig.add_trace(go.Bar(x=mix[c], y=mix.index, orientation="h", name=name, marker_color=colr,
                                 hovertemplate=f"%{{y}} · {name}: %{{x:.0f}}%<extra></extra>"))
        fig.update_layout(barmode="stack", bargap=0.25)
        fig.update_xaxes(ticksuffix="%", range=[0, 100])
        fig.update_yaxes(showgrid=False)
        show(fig, height=400)

    mom = genre_momentum(g, col)
    a, b = st.columns(2, gap="medium")
    with a.container(border=True):
        panel_head("Rising and falling genres",
                   "Change in share of units, 2011–16 against 2005–10. Red genres are losing share.",
                   legend={"Losing share": RED, "Gaining share": GREEN})
        fig = go.Figure(go.Bar(x=mom["change"], y=mom.index, orientation="h",
                               marker_color=[RED if v < 0 else GREEN for v in mom["change"]],
                               customdata=np.stack([mom["early"], mom["late"]], axis=1),
                               hovertemplate="<b>%{y}</b><br>2005–10: %{customdata[0]:.1f}%<br>2011–16: "
                                             "%{customdata[1]:.1f}%<br>Change %{x:+.1f} pts<extra></extra>"))
        fig.update_xaxes(ticksuffix=" pts", zeroline=True, zerolinecolor=LINE_STRONG, zerolinewidth=1.5)
        fig.update_yaxes(showgrid=False)
        show(fig, height=400)
    with b.container(border=True):
        panel_head("Top 10 platforms", f"{region}, {y0} – {y1}")
        fig = go.Figure(go.Bar(x=plat.values, y=plat.index, orientation="h", marker_color=BLUE, text=plat.values,
                               texttemplate="%{text:,.0f}M", textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:,.1f}M<extra></extra>"))
        fig.update_xaxes(range=[0, plat.max() * 1.22], showticklabels=False, showline=False)
        fig.update_yaxes(showgrid=False)
        show(fig, height=400)

    falling = mom[mom["change"] < 0].sort_values("change")
    rising = mom[mom["change"] > 0].sort_values("change", ascending=False)
    with st.container(border=True):
        panel_head("Weak genres and what to do", f"{region} sales")
        cards = ""
        for gen, r in falling.head(4).iterrows():
            best = mix.loc[gen].idxmax() if gen in mix.index else "NA_Sales"
            best_name = {"NA_Sales": "North America", "EU_Sales": "Europe", "JP_Sales": "Japan",
                         "Other_Sales": "other markets"}[best]
            cards += (f'<div><div class="t"><i style="background:{RED}"></i>{gen}<span>{r["change"]:+.1f} pts of share</span></div>'
                      f'<p class="why">Fell from {r["early"]:.1f}% to {r["late"]:.1f}% of units.</p><ul>'
                      f'<li>Cut new titles in this genre; keep only proven franchises.</li>'
                      f'<li>Focus launches on {best_name}, where it still sells best.</li>'
                      f'<li>Move budget towards {", ".join(rising.index[:2])}, which are gaining share.</li></ul></div>')
        st.markdown(f'<div class="actions">{cards}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- demand segments
else:
    head("Demand segments", "The 17 sub-categories grouped by how they sell, each with its own stocking rule.")
    sub, ev = clusters(model_df)
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
        st.plotly_chart(style(fig, height=470), width="stretch", config=PCFG)

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

# ----------------------------------------------------------------- live import ticker
if STREAMING:
    st.fragment(run_every=TICK_SECONDS)(pump)()
