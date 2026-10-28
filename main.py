"""
EstateIQ - Real Estate Investment Advisor (Streamlit app)

Predicting property profitability and future value.
  * Classification: "Is this a Good Investment?"       (Logistic Regression vs XGBoost)
  * Regression:     "Estimated price after 5 years"    (Ridge vs XGBoost)
  * 20-question EDA, filters, heatmaps, trend charts, model confidence, feature
    importance, and MLflow experiment tracking.

Run:
    1. pip install -r requirements.txt
    2. python train.py            (optional - or use the Train button on the Model Lab page)
    3. streamlit run main.py

Dataset: india_housing_prices.csv in this folder or in ./data
"""
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import Housing_pipeline as hp

warnings.filterwarnings("ignore")

st.set_page_config(page_title="EstateIQ | Investment Advisor", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ theme --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1020; color: #edf2ff; }
[data-testid="stHeader"] { background: rgba(11,16,32,0.88); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#111a30 0%,#0d1426 100%); border-right: 1px solid #25314d; }
[data-testid="stSidebar"] * { color: #e6ecff; }
h1,h2,h3,h4 { font-family: 'Manrope', sans-serif; letter-spacing: -0.02em; color: #f4f7ff; }
p, label, .stMarkdown { color: #c6d0e8; }
.hero { padding: 1.5rem 1.7rem; border-radius: 22px; margin-bottom: 1rem;
  background: radial-gradient(circle at 85% 10%, rgba(56,189,248,.22), transparent 30%),
              linear-gradient(135deg,#172746 0%,#111a30 55%,#18223d 100%);
  border: 1px solid #2a3b5d; box-shadow: 0 18px 55px rgba(0,0,0,.18); }
.eyebrow { color:#67e8f9; text-transform:uppercase; font-size:.74rem; font-weight:800; letter-spacing:.15em; }
.hero h1 { font-size:2.1rem; line-height:1.15; margin:.35rem 0 .6rem; }
.hero p { color:#bdc9e4; max-width:880px; margin:0; }
.kpi { padding: 1rem 1.1rem; border-radius: 16px; min-height: 112px;
  background: linear-gradient(145deg,#17233d,#121b30); border: 1px solid #2a3a59; }
.kpi-label { color:#9fb0d1; font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { color:#f7faff; font-size:1.6rem; font-weight:800; margin-top:.35rem; line-height:1.2; }
.kpi-foot { color:#7dd3fc; font-size:.78rem; margin-top:.35rem; }
.panel { padding: 1rem 1.2rem; border-radius: 16px; background:#111a2d; border:1px solid #253653; margin:.3rem 0 .8rem; }
.note { padding:.85rem 1rem; border-left:3px solid #38bdf8; background:#111e34; border-radius: 8px;
  color:#c8d7ef; font-size:.9rem; margin:.4rem 0; }
.warn { border-left-color:#fbbf24; }
.verdict { padding: 1.2rem 1.4rem; border-radius: 18px; margin: .4rem 0 1rem; border: 1px solid; }
.verdict.good { background: linear-gradient(135deg,#0f2f27,#10241f); border-color:#1f6f58; }
.verdict.bad  { background: linear-gradient(135deg,#33202a,#241821); border-color:#7a3a52; }
.verdict h2 { margin:0 0 .2rem; font-size:1.7rem; }
.small-muted { color:#8fa1c4; font-size:.82rem; }
.crit-ok { color:#34d399; font-weight:700; } .crit-no { color:#f87171; font-weight:700; }
div[data-testid="stMetric"] { background:#141f35; border:1px solid #293957; padding:14px 16px; border-radius:14px; }
div[data-testid="stMetricLabel"] { color:#9fb0d1; } div[data-testid="stMetricValue"] { color:#f7faff; }
.stButton>button { border-radius:10px; font-weight:700; }
hr { border-color:#263652; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"]
SCALE = ["#1e2a47", "#38bdf8"]
HEAT = ["#0f1a30", "#1d4e89", "#38bdf8", "#a5f3fc"]


# ---------------------------------------------------------------- helpers --
def _st_version() -> tuple[int, int]:
    try:
        parts = st.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (1, 0)


_NEW_WIDTH_API = _st_version() >= (1, 50)


def stretch(fn, *args, **kwargs):
    """Call a Streamlit element full-width on both old and new Streamlit versions."""
    options = ({"width": "stretch"}, {"use_container_width": True}) if _NEW_WIDTH_API \
        else ({"use_container_width": True}, {"width": "stretch"})
    for extra in options:
        try:
            return fn(*args, **kwargs, **extra)
        except Exception:
            continue
    return fn(*args, **kwargs)


def show(fig, height: int = 380):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=48, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#d7e2f5", family="DM Sans"),
                      title_font=dict(color="#f5f8ff", size=15), legend=dict(bgcolor="rgba(0,0,0,0)"))
    stretch(st.plotly_chart, fig)


def table(data, **kwargs):
    stretch(st.dataframe, data, hide_index=True, **kwargs)


def kpi(label: str, value: str, foot: str = ""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>',
                unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                f'<p>{subtitle}</p></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False):
    st.markdown(f'<div class="note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def lakh(x) -> str:
    try:
        x = float(x)
        return f"₹{x / 100:,.2f} Cr" if abs(x) >= 100 else f"₹{x:,.1f} L"
    except Exception:
        return "—"


def inr(x) -> str:
    try:
        return f"₹{float(x):,.0f}"
    except Exception:
        return "—"


def pfmt(p) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def samp(d: pd.DataFrame, n: int = 20000) -> pd.DataFrame:
    return d if len(d) <= n else d.sample(n, random_state=42)


def rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()


# ------------------------------------------------------------------- data --
@st.cache_resource(show_spinner=False)
def load_all(path_str: str, mtime: float):
    df, ref, report = hp.prepare_dataset(path_str)
    choices = {c: sorted(df[c].dropna().astype(str).unique().tolist())
               for c in ["State", "City", "Property_Type", "Furnished_Status", "Facing", "Owner_Type",
                         "Availability_Status", "Public_Transport_Accessibility", "Parking_Space", "Security"]}
    tokens = set()
    for s in df["Amenities"].dropna().head(20000).unique():
        tokens.update(t.strip() for t in str(s).split(",") if t.strip())
    choices["Amenities"] = sorted(tokens)
    return df, ref, report, choices


@st.cache_resource(show_spinner=False)
def _load_bundle(mtime: float):
    return hp.load_bundle()


def current_bundle():
    mt = hp.BUNDLE_PATH.stat().st_mtime if hp.BUNDLE_PATH.exists() else 0.0
    return _load_bundle(mt)


data_path = hp.find_dataset()
if data_path is None:
    st.sidebar.markdown("## 🏙️ EstateIQ")
    st.error("Dataset not found. Put `india_housing_prices.csv` in the project folder or in a `data/` folder.")
    st.code(
        "Real Estate Investment Advisor/\n├── main.py\n├── housing_pipeline.py\n├── train.py\n└── data/\n    └── india_housing_prices.csv",
        language="text")
    st.stop()

try:
    with st.spinner("Loading and preparing the dataset (first run only)…"):
        DF, REF, REPORT, CHOICES = load_all(str(data_path), data_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.stop()

if DF.empty:
    st.error("The dataset has no usable rows after cleaning.")
    st.stop()


def no_models_notice():
    note("No trained models were found yet. Open the <b>Model Lab</b> page and press "
         "<b>Train / retrain models</b> (about a minute), or run <code>python train.py</code> in the terminal.",
         warn=True)


# ============================================================ PAGE: Overview
def page_overview():
    hero("PROJECT OVERVIEW", "Predict property profitability and future value.",
         "Classify whether a listing is a Good Investment and estimate its price after 5 years, "
         "backed by EDA, tracked experiments and explainable models.")
    c = st.columns(5)
    with c[0]:
        kpi("Properties", f"{len(DF):,}", "After cleaning")
    with c[1]:
        kpi("Median price", lakh(DF["Price_in_Lakhs"].median()), "Asking price")
    with c[2]:
        kpi("Median ₹ / sq ft", inr(DF["Price_per_SqFt"].median()), "Recomputed from price and size")
    with c[3]:
        kpi("Good-investment share", f"{REPORT['good_share']:.1f}%", f"Score ≥ {REF['threshold']} of 5 rules")
    with c[4]:
        kpi("Cities", f"{DF['City'].nunique():,}", f"{DF['State'].nunique()} states")

    bundle = current_bundle()
    st.markdown("### Model status")
    if bundle is None:
        no_models_notice()
    else:
        m = st.columns(4)
        if bundle.get("clf"):
            cn = bundle["clf"]["champion"];
            cm = bundle["clf"]["metrics"][cn]
            with m[0]: kpi("Classifier", cn, "Champion model")
            with m[1]: kpi("F1 · Accuracy", f"{cm['f1']:.3f} · {cm['accuracy']:.1%}", "Held-out test set")
        rn = bundle["reg"]["champion"];
        rm = bundle["reg"]["metrics"][rn]
        with m[2]:
            kpi("Regressor", rn, "Champion model")
        with m[3]:
            kpi("RMSE · R²", f"{rm['rmse']:.1f} L · {rm['r2']:.3f}", "Held-out test set")

    st.markdown("### Market pulse")
    left, right = st.columns([1.25, 1])
    with left:
        g = DF.groupby("City").agg(Avg_Price=("Price_in_Lakhs", "mean"), Listings=("Price_in_Lakhs", "size"))
        g = g.reset_index().sort_values("Avg_Price", ascending=False).head(12).sort_values("Avg_Price")
        fig = px.bar(g, x="Avg_Price", y="City", orientation="h", color="Avg_Price",
                     color_continuous_scale=SCALE, hover_data=["Listings"],
                     title="Highest average property prices by city")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Average price (₹ lakh)", yaxis_title="")
        show(fig, 420)
    with right:
        tc = DF["Property_Type"].value_counts().reset_index()
        tc.columns = ["Property type", "Listings"]
        fig = px.pie(tc, names="Property type", values="Listings", hole=0.65,
                     color_discrete_sequence=PALETTE, title="Listing mix by property type")
        fig.update_traces(textinfo="percent")
        show(fig, 420)

    # ==================== NEW: Interactive 1-10 Year Market Forecast ====================
    st.markdown("### 📈 Interactive Market Price Forecast (1–10 Years)")
    st.markdown(
        "Adjust the forecast horizon to see how median property prices scale over time based on market growth models.")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        forecast_years = st.slider("Select Forecast Horizon (Years)", min_value=1, max_value=10, value=5, step=1)
        selected_city_forecast = st.selectbox(
            "Select Market / City Scope",
            ["All India / Overall Median"] + sorted(DF["City"].unique().tolist()),
            key="overview_city_forecast"
        )

    with col_f2:
        if selected_city_forecast == "All India / Overall Median":
            base_val = float(DF["Price_in_Lakhs"].median())
        else:
            city_df = DF[DF["City"] == selected_city_forecast]
            base_val = float(city_df["Price_in_Lakhs"].median()) if not city_df.empty else float(
                DF["Price_in_Lakhs"].median())

        growth_r = hp.BASE_GROWTH
        if selected_city_forecast != "All India / Overall Median" and "city_demand" in REF:
            city_idx_val = REF["city_demand"].get(selected_city_forecast, 0.0)
            growth_r = hp.BASE_GROWTH + 0.02 * city_idx_val

        future_val = hp.future_value(base_val, growth_r, forecast_years)
        gain_val = future_val - base_val
        total_return_pct = (future_val / base_val - 1) * 100 if base_val > 0 else 0

        fc_cols = st.columns(3)
        with fc_cols[0]:
            kpi("Current Baseline", lakh(base_val), "Starting median price")
        with fc_cols[1]:
            kpi(f"Price in {forecast_years} Year(s)", lakh(future_val), f"At ~{growth_r * 100:.1f}% annual rate")
        with fc_cols[2]:
            kpi("Projected Gain", lakh(gain_val), f"+{total_return_pct:.1f}% total growth")

        years_range = np.arange(0, forecast_years + 1)
        trend_prices = [hp.future_value(base_val, growth_r, int(y)) for y in years_range]
        fig_trend = px.line(
            x=years_range, y=trend_prices, markers=True,
            title=f"Price Trajectory Projection ({forecast_years}-Year View)"
        )
        fig_trend.update_layout(
            xaxis_title="Years from Now",
            yaxis_title="Projected Price (₹ Lakh)",
            margin=dict(l=10, r=10, t=35, b=10)
        )
        show(fig_trend, height=280)
    # ==================================================================================

    st.markdown("### How the project works")
    steps = [
        ("01", "Preprocess", "Clean, impute, remove duplicates, flag outliers, scale and encode, engineer features."),
        ("02", "Explore", "Answer 20 EDA questions with charts and statistical tests."),
        ("03", "Model", "Train Logistic/Ridge baselines and XGBoost for both tasks."),
        ("04", "Track", "Log every run to MLflow and register the best models."),
        ("05", "Advise", "Score a property, filter the market and read confidence plus drivers.")]
    for col, (n, t, d) in zip(st.columns(5), steps):
        with col:
            st.markdown(f'<div class="panel"><div class="eyebrow">{n}</div><h4>{t}</h4>'
                        f'<p class="small-muted">{d}</p></div>', unsafe_allow_html=True)

    with st.expander("Data preparation report (imputation, duplicates, outliers)"):
        a, b, c2, d2 = st.columns(4)
        a.metric("Raw rows", f"{REPORT['raw_rows']:,}")
        b.metric("Duplicates removed", f"{REPORT['duplicates_removed']:,}")
        c2.metric("Invalid price/size removed", f"{REPORT['invalid_removed']:,}")
        d2.metric("Clean rows", f"{REPORT['clean_rows']:,}")
        e, f = st.columns(2)
        e.metric("Price/sq ft outliers (IQR rule)", f"{REPORT['outliers_ppsf']:,}")
        f.metric("Size outliers (IQR rule)", f"{REPORT['outliers_size']:,}")
        miss = pd.DataFrame({"Column": list(REPORT["missing_before"].keys()),
                             "Missing values": list(REPORT["missing_before"].values())})
        if not miss.empty:
            miss["Missing %"] = (miss["Missing values"] / REPORT["raw_rows"] * 100).round(2)
            st.markdown(
                "**Missing values before cleaning** (numeric → median, text → most frequent inside the model pipeline; "
                "State/City/Locality/Property type → “Unknown”)")
            table(miss.sort_values("Missing values", ascending=False))
        else:
            st.write("No missing values found.")
        if REPORT["absent_columns"]:
            st.warning(f"Columns absent from the CSV (filled empty): {REPORT['absent_columns']}")


# ================================================================ PAGE: EDA
def anova(d: pd.DataFrame, group: str, value: str):
    groups = [g[value].dropna().to_numpy() for _, g in d.groupby(group) if g[value].notna().sum() >= 5]
    if len(groups) < 2:
        return None
    f, p = stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    sst = float(((allv - grand) ** 2).sum())
    return {"F": float(f), "p": float(p), "eta2": float(ssb / sst) if sst > 0 else np.nan, "k": len(groups)}


def effect_word(v: float, cuts=(0.01, 0.06, 0.14)) -> str:
    if v is None or not np.isfinite(v):
        return "unknown"
    return "negligible" if v < cuts[0] else "small" if v < cuts[1] else "medium" if v < cuts[2] else "large"


def anova_text(res) -> str:
    if not res:
        return "Not enough groups for a significance test."
    if res["p"] >= 0.05:
        return f"ANOVA: F = {res['F']:.2f}, p = {pfmt(res['p'])} → no statistically significant difference between groups."
    return (f"ANOVA: F = {res['F']:.2f}, p = {pfmt(res['p'])} → statistically significant, but the effect is "
            f"**{effect_word(res['eta2'])}** (η² = {res['eta2']:.3f}). With many rows, tiny differences become 'significant', so read the effect size.")


def corr_text(x: pd.Series, y: pd.Series, xl: str, yl: str) -> str:
    ok = x.notna() & y.notna()
    if ok.sum() < 10:
        return "Not enough data."
    r, p = stats.pearsonr(x[ok], y[ok])
    rho, _ = stats.spearmanr(x[ok], y[ok])
    strength = effect_word(abs(r), (0.1, 0.3, 0.5))
    direction = "positive" if r > 0 else "negative"
    return (f"Pearson r = {r:.3f} (p = {pfmt(p)}), Spearman ρ = {rho:.3f} → a **{strength} {direction}** "
            f"relationship between {xl} and {yl}. Correlation does not prove causation.")


def qhead(n: int, text: str):
    st.markdown(f"#### Q{n}. {text}")


def group_relation(d, group, value, value_label, numeric_order=False, height=340):
    sub = d[[group, value]].dropna()
    if sub.empty:
        st.info("Not enough data for this question.")
        return
    agg = sub.groupby(group)[value].agg(Listings="count", Mean="mean", Median="median").reset_index()
    agg = agg.sort_values(group if numeric_order else "Mean", ascending=numeric_order)
    left, right = st.columns([1.5, 1])
    with left:
        fig = px.bar(agg, x=group, y="Mean", text_auto=".3s", color="Mean", color_continuous_scale=SCALE)
        fig.update_layout(coloraxis_showscale=False, yaxis_title=f"Average {value_label}", xaxis_title="")
        fig.update_xaxes(type="category")
        show(fig, height)
    with right:
        table(agg.round(2))
    st.caption(anova_text(anova(sub, group, value)))


def numeric_relation(d, xcol, ycol, xl, yl, height=340):
    sub = d[[xcol, ycol]].dropna()
    if sub.empty:
        st.info("Not enough data for this question.")
        return
    agg = sub.groupby(xcol)[ycol].mean().reset_index()
    fig = px.bar(agg, x=xcol, y=ycol, text_auto=".3s", color=ycol, color_continuous_scale=SCALE)
    fig.update_layout(coloraxis_showscale=False, yaxis_title=f"Average {yl}", xaxis_title=xl)
    fig.update_xaxes(type="category")
    show(fig, height)
    st.caption(corr_text(sub[xcol], sub[ycol], xl, yl))


def page_eda():
    hero("EXPLORATORY DATA ANALYSIS", "Twenty questions about the market.",
         "Each answer pairs a chart with the statistic behind it: descriptive tables, correlations, "
         "ANOVA, t-tests and the IQR outlier rule.")
    d = DF
    part = st.radio("Question group", ["1–5 · Price & size", "6–10 · Location", "11–15 · Feature relationships",
                                       "16–20 · Investment, amenities & ownership"], horizontal=True)

    if part.startswith("1–5"):
        qhead(1, "What is the distribution of property prices?")
        a, b = st.columns([1.7, 1])
        with a:
            fig = px.histogram(samp(d, 30000), x="Price_in_Lakhs", nbins=50, marginal="box",
                               color_discrete_sequence=[PALETTE[0]])
            fig.update_layout(xaxis_title="Price (₹ lakh)")
            show(fig, 340)
        with b:
            table(d["Price_in_Lakhs"].describe().round(2).rename("Price (₹ lakh)").reset_index())
        st.caption(f"Skewness = {d['Price_in_Lakhs'].skew():.2f}. Values near 0 mean a symmetric distribution; "
                   "positive values mean a long tail of expensive properties.")

        qhead(2, "What is the distribution of property sizes?")
        a, b = st.columns([1.7, 1])
        with a:
            fig = px.histogram(samp(d, 30000), x="Size_in_SqFt", nbins=50, marginal="box",
                               color_discrete_sequence=[PALETTE[1]])
            fig.update_layout(xaxis_title="Size (sq ft)")
            show(fig, 340)
        with b:
            table(d["Size_in_SqFt"].describe().round(1).rename("Size (sq ft)").reset_index())

        qhead(3, "How does price per sq ft vary by property type?")
        group_relation(d, "Property_Type", "Price_per_SqFt", "₹ per sq ft")

        qhead(4, "Is there a relationship between property size and price?")
        s = samp(d, 8000)
        fig = px.scatter(s, x="Size_in_SqFt", y="Price_in_Lakhs", opacity=0.35,
                         color_discrete_sequence=[PALETTE[0]])
        lr = stats.linregress(d["Size_in_SqFt"], d["Price_in_Lakhs"])
        xs = np.array([d["Size_in_SqFt"].min(), d["Size_in_SqFt"].max()])
        fig.add_trace(go.Scatter(x=xs, y=lr.intercept + lr.slope * xs, mode="lines", name="OLS fit",
                                 line=dict(color=PALETTE[3], width=3)))
        fig.update_layout(xaxis_title="Size (sq ft)", yaxis_title="Price (₹ lakh)")
        show(fig, 380)
        st.caption(corr_text(d["Size_in_SqFt"], d["Price_in_Lakhs"], "size", "price")
                   + f" Regression: each extra sq ft adds {lr.slope:.4f} lakh on average (R² = {lr.rvalue ** 2:.3f}).")

        qhead(5, "Are there outliers in price per sq ft or property size? (IQR method)")
        a, b = st.columns(2)
        s = samp(d, 20000)
        with a:
            fig = px.box(s, y="Price_per_SqFt", color_discrete_sequence=[PALETTE[0]])
            fig.update_layout(yaxis_title="₹ per sq ft", title="Price per sq ft")
            show(fig, 340)
        with b:
            fig = px.box(s, y="Size_in_SqFt", color_discrete_sequence=[PALETTE[1]])
            fig.update_layout(yaxis_title="Size (sq ft)", title="Size")
            show(fig, 340)
        rows = []
        for col, flag in (("Price_per_SqFt", "Outlier_PPSF"), ("Size_in_SqFt", "Outlier_Size")):
            lo, hi = REF["fences"][col]
            rows.append({"Variable": col, "Lower fence": round(lo, 1), "Upper fence": round(hi, 1),
                         "Outliers": int(d[flag].sum()), "% of rows": round(d[flag].mean() * 100, 2)})
        table(pd.DataFrame(rows))
        st.caption(
            "Fences = Q1 − 1.5·IQR and Q3 + 1.5·IQR. Outliers are flagged, not deleted; tree-based models are robust to them.")

    elif part.startswith("6–10"):
        mode = st.radio("Group localities by", ["Locality name", "City + Locality"], horizontal=True)
        dd = d.assign(Loc_Key=d["Locality"] if mode == "Locality name" else d["Locality"] + " · " + d["City"])

        qhead(6, "What is the average price per sq ft by state?")
        g = d.groupby("State")["Price_per_SqFt"].mean().sort_values().reset_index()
        fig = px.bar(g, x="Price_per_SqFt", y="State", orientation="h", color="Price_per_SqFt",
                     color_continuous_scale=SCALE, text_auto=".0f")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Average ₹ per sq ft", yaxis_title="")
        show(fig, max(340, 22 * len(g) + 80))

        qhead(7, "What is the average property price by city?")
        g = d.groupby("City")["Price_in_Lakhs"].mean().sort_values().reset_index()
        fig = px.bar(g, x="Price_in_Lakhs", y="City", orientation="h", color="Price_in_Lakhs",
                     color_continuous_scale=SCALE, text_auto=".0f")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Average price (₹ lakh)", yaxis_title="")
        show(fig, max(340, 22 * len(g) + 80))

        qhead(8, "What is the median age of properties by locality?")
        ages = dd.groupby("Loc_Key")["Age_of_Property"].median().sort_values()
        a, b = st.columns(2)
        with a:
            top = ages.tail(12).reset_index()
            fig = px.bar(top, x="Age_of_Property", y="Loc_Key", orientation="h", title="Oldest localities",
                         color_discrete_sequence=[PALETTE[3]])
            fig.update_layout(xaxis_title="Median age (years)", yaxis_title="")
            show(fig, 380)
        with b:
            low = ages.head(12).reset_index()
            fig = px.bar(low, x="Age_of_Property", y="Loc_Key", orientation="h", title="Newest localities",
                         color_discrete_sequence=[PALETTE[2]])
            fig.update_layout(xaxis_title="Median age (years)", yaxis_title="")
            show(fig, 380)
        with st.expander(f"All {len(ages):,} localities"):
            table(ages.rename("Median age (years)").reset_index().rename(columns={"Loc_Key": "Locality"}))

        qhead(9, "How is BHK distributed across cities?")
        top_cities = d["City"].value_counts().head(20).index
        ct = pd.crosstab(d.loc[d["City"].isin(top_cities), "City"], d.loc[d["City"].isin(top_cities), "BHK"],
                         normalize="index") * 100
        fig = px.imshow(ct.round(1), text_auto=".0f", aspect="auto", color_continuous_scale=HEAT,
                        labels=dict(x="BHK", y="City", color="% of city listings"))
        show(fig, max(360, 24 * len(ct) + 100))
        st.caption("Share (%) of each city's listings by BHK, for the 20 cities with the most listings.")

        qhead(10, "What are the price trends for the top 5 most expensive localities?")
        stat = dd.groupby("Loc_Key")["Price_in_Lakhs"].agg(["mean", "size"])
        stat = stat[stat["size"] >= 30].sort_values("mean", ascending=False)
        if stat.empty:
            st.info("No locality has at least 30 listings.")
        else:
            top5 = stat.head(5).index.tolist()
            tr = dd[dd["Loc_Key"].isin(top5)].groupby(["Loc_Key", "Year_Built"])["Price_in_Lakhs"].mean().reset_index()
            fig = px.line(tr, x="Year_Built", y="Price_in_Lakhs", color="Loc_Key", markers=True,
                          color_discrete_sequence=PALETTE)
            fig.update_layout(xaxis_title="Year built", yaxis_title="Average price (₹ lakh)", legend_title="")
            show(fig, 380)
            st.caption("The dataset has no sale dates, so Year Built is used as the time axis. "
                       "Top 5 = highest average price among localities with ≥ 30 listings.")

    elif part.startswith("11–15"):
        qhead(11, "How are numeric features correlated with each other?")
        cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Nearby_Schools",
                "Nearby_Hospitals", "Floor_No", "Total_Floors", "Amenity_Count", "Transport_Score",
                "Infrastructure_Score"]
        corr = d[cols].corr()
        fig = px.imshow(corr.round(2), text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale=["#f472b6", "#18233d", "#38bdf8"])
        show(fig, 560)
        pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack().abs().sort_values(ascending=False)
        a, b = pairs.index[0]
        st.caption(f"Strongest pair: {a} and {b} (|r| = {pairs.iloc[0]:.2f}).")

        qhead(12, "How do nearby schools relate to price per sq ft?")
        numeric_relation(d, "Nearby_Schools", "Price_per_SqFt", "nearby schools", "₹ per sq ft")
        qhead(13, "How do nearby hospitals relate to price per sq ft?")
        numeric_relation(d, "Nearby_Hospitals", "Price_per_SqFt", "nearby hospitals", "₹ per sq ft")
        qhead(14, "How does price vary by furnished status?")
        group_relation(d, "Furnished_Status", "Price_in_Lakhs", "price (₹ lakh)")
        qhead(15, "How does price per sq ft vary by facing direction?")
        group_relation(d, "Facing", "Price_per_SqFt", "₹ per sq ft")

    else:
        qhead(16, "How many properties belong to each owner type?")
        vc = d["Owner_Type"].value_counts().reset_index()
        vc.columns = ["Owner type", "Properties"]
        fig = px.bar(vc, x="Owner type", y="Properties", text_auto=",", color_discrete_sequence=[PALETTE[0]])
        show(fig, 320)

        qhead(17, "How many properties are available under each availability status?")
        vc = d["Availability_Status"].value_counts().reset_index()
        vc.columns = ["Availability", "Properties"]
        fig = px.bar(vc, x="Availability", y="Properties", text_auto=",", color_discrete_sequence=[PALETTE[1]])
        show(fig, 320)

        qhead(18, "Does parking space affect property price?")
        with_p = d.loc[d["Parking_Flag"] == 1, "Price_in_Lakhs"]
        without = d.loc[d["Parking_Flag"] == 0, "Price_in_Lakhs"]
        if len(with_p) > 1 and len(without) > 1:
            t, p = stats.ttest_ind(with_p, without, equal_var=False)
            pooled = np.sqrt((with_p.var() + without.var()) / 2)
            dd_ = (with_p.mean() - without.mean()) / pooled if pooled else np.nan
            a, b = st.columns([1.4, 1])
            with a:
                fig = px.box(samp(d, 20000).assign(
                    Parking=lambda x: np.where(x["Parking_Flag"] == 1, "With parking", "No parking")),
                             x="Parking", y="Price_in_Lakhs", color="Parking", color_discrete_sequence=PALETTE)
                fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Price (₹ lakh)")
                show(fig, 340)
            with b:
                table(pd.DataFrame({"Group": ["With parking", "No parking"], "Listings": [len(with_p), len(without)],
                                    "Mean price (₹ lakh)": [with_p.mean(), without.mean()]}).round(2))
            word = effect_word(abs(dd_), (0.2, 0.5, 0.8))
            msg = "no statistically significant" if p >= 0.05 else "a statistically significant"
            st.caption(
                f"Welch t-test: t = {t:.2f}, p = {pfmt(p)} → {msg} difference; effect size (Cohen's d) = {dd_:.3f} ({word}).")
        else:
            st.info("Parking information is not available in a usable form for this dataset.")

        qhead(19, "How do amenities affect price per sq ft?")
        group_relation(d, "Amenity_Count", "Price_per_SqFt", "₹ per sq ft", numeric_order=True)
        qhead(20, "How does public transport accessibility relate to price per sq ft and investment potential?")
        group_relation(d, "Public_Transport_Accessibility", "Price_per_SqFt", "₹ per sq ft")
        g = d.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().mul(100).reset_index()
        fig = px.bar(g, x="Public_Transport_Accessibility", y="Good_Investment", text_auto=".1f",
                     color_discrete_sequence=[PALETTE[2]], title="Share of Good Investments by transport access")
        fig.update_layout(yaxis_title="% flagged Good Investment", xaxis_title="")
        show(fig, 320)

    note("The brief also mentions <b>crime rate</b>. The supplied dataset has no crime column, so that analysis is "
         "not possible; the app uses an infrastructure score (schools, hospitals, transport) instead.")


# ========================================================= PAGE: Advisor
def page_advisor():
    hero("INVESTMENT ADVISOR", "Is this property a good investment?",
         "Enter the property details. The classifier answers whether it is a Good Investment; "
         "the regressor estimates its price after 5 years, with confidence and the main drivers.")
    bundle = current_bundle()
    if bundle is None or not bundle.get("reg"):
        no_models_notice()
        return
    has_clf = bool(bundle.get("clf"))
    d = DF

    st.markdown("### 1 · Property details")
    c = st.columns(4)
    state = c[0].selectbox("State", CHOICES["State"])
    cities = sorted(d.loc[d["State"] == state, "City"].dropna().unique().tolist()) or CHOICES["City"]
    city = c[1].selectbox("City", cities)
    cdf = d[d["City"] == city]
    locs = sorted(cdf["Locality"].dropna().unique().tolist()) or ["Unknown"]
    locality = c[2].selectbox("Locality", locs)
    ptype = c[3].selectbox("Property type", CHOICES["Property_Type"])

    med = lambda col, default: float(cdf[col].median()) if len(cdf) and pd.notna(cdf[col].median()) else default
    c = st.columns(4)
    bhk = c[0].number_input("BHK", 1, 10, int(np.clip(round(med("BHK", 2)), 1, 10)), key=f"bhk_{city}")
    size = c[1].number_input("Size (sq ft)", 100, 20000, int(np.clip(round(med("Size_in_SqFt", 1200)), 100, 20000)),
                             step=50, key=f"size_{city}")
    price = c[2].number_input("Asking price (₹ lakh)", 1.0, 100000.0,
                              float(np.clip(round(med("Price_in_Lakhs", 80), 1), 1.0, 100000.0)),
                              step=1.0, key=f"price_{city}")
    furnished = c[3].selectbox("Furnishing", CHOICES["Furnished_Status"])

    c = st.columns(4)
    age = c[0].number_input("Property age (years)", 0, 120, int(np.clip(round(med("Age_of_Property", 10)), 0, 120)),
                            key=f"age_{city}")
    floor = c[1].number_input("Floor number", 0, 80, 3)
    total_floors = c[2].number_input("Total floors", 1, 100, 10)
    avail = c[3].selectbox("Availability", CHOICES["Availability_Status"])

    c = st.columns(4)
    schools = c[0].number_input("Nearby schools", 0, 50, int(np.clip(round(med("Nearby_Schools", 5)), 0, 50)),
                                key=f"sch_{city}")
    hospitals = c[1].number_input("Nearby hospitals", 0, 50, int(np.clip(round(med("Nearby_Hospitals", 5)), 0, 50)),
                                  key=f"hos_{city}")
    transport = c[2].selectbox("Public transport", CHOICES["Public_Transport_Accessibility"])
    parking = c[3].selectbox("Parking space", CHOICES["Parking_Space"])

    c = st.columns(4)
    security = c[0].selectbox("Security", CHOICES["Security"])
    facing = c[1].selectbox("Facing", CHOICES["Facing"])
    owner = c[2].selectbox("Owner type", CHOICES["Owner_Type"])
    amenities = c[3].multiselect("Amenities", CHOICES["Amenities"], default=CHOICES["Amenities"][:2])

    with st.expander("Model and scenario settings"):
        s1, s2, s3 = st.columns(3)
        clf_names = list(bundle["clf"]["models"]) if has_clf else []
        clf_name = s1.selectbox("Classification model", clf_names,
                                index=clf_names.index(bundle["clf"]["champion"]) if has_clf else 0) if has_clf else None
        reg_names = list(bundle["reg"]["models"])
        reg_name = s2.selectbox("Regression model", reg_names, index=reg_names.index(bundle["reg"]["champion"]))
        fixed_rate = s3.slider("Fixed growth rate for comparison (% / year)", 0.0, 20.0, hp.BASE_GROWTH * 100,
                               0.5) / 100

    row = pd.DataFrame([{
        "State": state, "City": city, "Locality": locality, "Property_Type": ptype, "BHK": float(bhk),
        "Size_in_SqFt": float(size), "Price_in_Lakhs": float(price), "Year_Built": hp.CURRENT_YEAR - age,
        "Furnished_Status": furnished, "Floor_No": float(min(floor, total_floors)), "Total_Floors": float(total_floors),
        "Age_of_Property": float(age), "Nearby_Schools": float(schools), "Nearby_Hospitals": float(hospitals),
        "Public_Transport_Accessibility": transport, "Parking_Space": parking, "Security": security,
        "Amenities": ", ".join(amenities), "Facing": facing, "Owner_Type": owner, "Availability_Status": avail,
    }])
    if floor > total_floors:
        st.warning("Floor number is above the total floors; the floor was capped at the top floor.")
    row = hp.add_label(hp.add_features(row, REF), REF)
    ppsf = float(row["Price_per_SqFt"].iloc[0])
    r0 = row.iloc[0]

    horizon = REF["horizon"]
    future_ml = float(hp.predict_future_price(bundle, reg_name, row)[0])
    cagr_ml = (future_ml / price) ** (1 / horizon) - 1 if price > 0 and future_ml > 0 else np.nan
    rel_q90 = bundle["reg"]["metrics"][reg_name]["rel_err_q90"]

    st.markdown("---")
    st.markdown("### 2 · Classification: is this a Good Investment?")
    if has_clf:
        proba = float(hp.predict_good_probability(bundle, clf_name, row)[0])
        good = proba >= 0.5
        conf = max(proba, 1 - proba)
        st.markdown(
            f'<div class="verdict {"good" if good else "bad"}"><h2>{"✅ Good Investment" if good else "⚠️ Not a Good Investment"}</h2>'
            f'<span>{clf_name} gives a {proba:.0%} probability of "Good Investment" '
            f'(confidence in this verdict: {conf:.0%}).</span></div>', unsafe_allow_html=True)
        left, right = st.columns([1, 1.2])
        with left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=proba * 100, number={"suffix": "%"},
                title={"text": "Probability of Good Investment"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#38bdf8"},
                       "steps": [{"range": [0, 50], "color": "#3b1d2a"}, {"range": [50, 100], "color": "#12392f"}],
                       "threshold": {"line": {"color": "#fbbf24", "width": 3}, "value": 50}}))
            show(fig, 300)
        with right:
            crit = hp.investment_criteria(row, REF).iloc[0]
            details = {
                "Value_PPSF": f"{inr(ppsf)} vs city median {inr(r0['City_Median_PPSF'])} per sq ft",
                "Value_Price": f"{lakh(price)} vs city median {lakh(r0['City_Median_Price'])}",
                "Family_Size": f"{int(bhk)} BHK",
                "Ready_To_Move": str(avail),
                "Infrastructure": f"score {r0['Infrastructure_Score']:.1f} vs median {REF['infra_median']:.1f}",
            }
            st.markdown(
                f"**Rule-based score: {int(crit.sum())} of 5** (a property is labelled Good at ≥ {REF['threshold']})")
            for key, label in hp.CRITERIA_LABELS.items():
                ok = bool(crit[key])
                st.markdown(f'<span class="{"crit-ok" if ok else "crit-no"}">{"✔" if ok else "✘"}</span> {label} '
                            f'<span class="small-muted">— {details[key]}</span>', unsafe_allow_html=True)
            rule_good = int(crit.sum()) >= REF["threshold"]
            if rule_good != good:
                st.caption("The model and the rule score disagree on this property; treat it as borderline.")
    else:
        st.info("The classifier could not be trained (the dataset has only one class).")

    st.markdown("### 3 · Regression: estimated price after 5 years")
    k = st.columns(4)
    with k[0]:
        kpi("Current price", lakh(price), f"{inr(ppsf)} per sq ft")
    with k[1]:
        kpi(f"Estimated in {horizon} years", lakh(future_ml), f"{reg_name}")
    with k[2]:
        kpi("Likely range", f"{lakh(future_ml * (1 - rel_q90))} – {lakh(future_ml * (1 + rel_q90))}",
            f"90% of test predictions within ±{rel_q90:.1%}")
    with k[3]:
        kpi("Implied growth", f"{cagr_ml * 100:.1f}% / year" if np.isfinite(cagr_ml) else "—",
            f"Gain {lakh(future_ml - price)}")

    methods = {
        f"Fixed rate ({fixed_rate:.0%})": float(hp.growth_rates(row, REF, "fixed", fixed_rate).iloc[0]),
        "By city and property type": float(hp.growth_rates(row, REF, "location", fixed_rate).iloc[0]),
        "Feature-based formula": float(hp.growth_rates(row, REF, "feature", fixed_rate).iloc[0]),
    }
    rows, years = [], np.arange(0, horizon + 1)
    fig = go.Figure()
    for i, (name, r) in enumerate(methods.items()):
        fv = hp.future_value(price, r, horizon)
        rows.append({"Method": name, "Growth / year": f"{r * 100:.1f}%", f"Price in {horizon}y (₹ lakh)": round(fv, 2),
                     "Gain (₹ lakh)": round(fv - price, 2)})
        fig.add_trace(go.Scatter(x=years, y=hp.future_value(price, r, years), mode="lines", name=name,
                                 line=dict(color=PALETTE[i + 1], width=2, dash="dot")))
    rows.append({"Method": f"ML model ({reg_name})",
                 "Growth / year": f"{cagr_ml * 100:.1f}%" if np.isfinite(cagr_ml) else "—",
                 f"Price in {horizon}y (₹ lakh)": round(future_ml, 2), "Gain (₹ lakh)": round(future_ml - price, 2)})
    if np.isfinite(cagr_ml):
        lo_r = ((future_ml * (1 - rel_q90)) / price) ** (1 / horizon) - 1
        hi_r = ((future_ml * (1 + rel_q90)) / price) ** (1 / horizon) - 1
        fig.add_trace(go.Scatter(x=years, y=hp.future_value(price, hi_r, years), mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=hp.future_value(price, lo_r, years), mode="lines", fill="tonexty",
                                 fillcolor="rgba(56,189,248,0.18)", line=dict(width=0), name="ML likely range"))
        fig.add_trace(go.Scatter(x=years, y=hp.future_value(price, cagr_ml, years), mode="lines+markers",
                                 name=f"ML model ({reg_name})", line=dict(color=PALETTE[0], width=3)))
    fig.update_layout(title="Projected value path", xaxis_title="Years from now", yaxis_title="Price (₹ lakh)")
    a, b = st.columns([1.3, 1])
    with a:
        show(fig, 360)
    with b:
        table(pd.DataFrame(rows))
    note("There are no observed 5-year-later prices in the dataset, so the regressor learns a documented growth "
         "scenario (location + property features), not real future sales. Use the estimates to compare "
         "properties, not as a guarantee.", warn=True)

    st.markdown("### 4 · Model confidence and feature importance")
    a, b = st.columns(2)
    with a:
        if has_clf:
            imp = bundle["clf"]["importance"][clf_name].head(10).iloc[::-1]
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[PALETTE[0]],
                         title=f"What drives the 'Good Investment' call ({clf_name})")
            fig.update_layout(xaxis_title="Drop in ROC-AUC when shuffled", yaxis_title="")
            show(fig, 380)
    with b:
        imp = bundle["reg"]["importance"][reg_name].head(10).iloc[::-1]
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[PALETTE[1]],
                     title=f"What drives the 5-year price ({reg_name})")
        fig.update_layout(xaxis_title="Drop in R² when shuffled", yaxis_title="")
        show(fig, 380)
    st.caption("Permutation importance on held-out data: how much the score falls when a feature is shuffled. "
               "Correlated features (for example price and price ratios) share credit.")

    out = row[["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs"]].copy()
    if has_clf:
        out["Good_Investment_Probability"] = round(proba, 4)
        out["Good_Investment"] = "Yes" if good else "No"
    out["Rule_Score_of_5"] = int(r0["Investment_Score"])
    out[f"Estimated_Price_{horizon}Y_Lakhs"] = round(future_ml, 2)
    out["Implied_Growth_Pct"] = round(cagr_ml * 100, 2) if np.isfinite(cagr_ml) else np.nan
    stretch(st.download_button, "Download this analysis (CSV)", out.to_csv(index=False).encode("utf-8"),
            file_name="property_analysis.csv", mime="text/csv")


# ============================================================ PAGE: Finder
def page_finder():
    hero("PROPERTY FINDER", "Filter the market and rank the best opportunities.",
         "Filter by area, price, BHK and more. Matching properties are scored by the classifier "
         "so the likeliest Good Investments appear first.")
    d = DF
    bundle = current_bundle()
    has_clf = bool(bundle and bundle.get("clf"))
    if not has_clf:
        no_models_notice()

    f = st.columns(4)
    states = f[0].multiselect("State", CHOICES["State"])
    pool = sorted(d.loc[d["State"].isin(states), "City"].unique().tolist()) if states else CHOICES["City"]
    cities = f[1].multiselect("City", pool)
    types = f[2].multiselect("Property type", CHOICES["Property_Type"])
    furn = f[3].multiselect("Furnishing", CHOICES["Furnished_Status"])
    g = st.columns(4)
    bmin, bmax = int(d["BHK"].min()), int(d["BHK"].max())
    bhk = g[0].slider("BHK", bmin, bmax, (bmin, bmax))
    pmin, pmax = float(np.floor(d["Price_in_Lakhs"].min())), float(np.ceil(d["Price_in_Lakhs"].max()))
    price = g[1].slider("Budget (₹ lakh)", pmin, pmax, (pmin, pmax))
    smin, smax = int(d["Size_in_SqFt"].min()), int(d["Size_in_SqFt"].max())
    size = g[2].slider("Size (sq ft)", smin, smax, (smin, smax))
    avail = g[3].multiselect("Availability", CHOICES["Availability_Status"])
    h = st.columns(3)
    only_good = h[0].checkbox("Only show properties the model flags as Good", value=False, disabled=not has_clf)
    sort_options = ["Good-investment probability"] if has_clf else []
    sort_options += ["Price (low to high)", "₹ per sq ft (low to high)", "Rule score"]
    sort_by = h[1].selectbox("Rank by", sort_options)
    show_n = h[2].selectbox("Rows to display", [50, 100, 200, 500], index=1)

    m = d["BHK"].between(*bhk) & d["Price_in_Lakhs"].between(*price) & d["Size_in_SqFt"].between(*size)
    if states: m &= d["State"].isin(states)
    if cities: m &= d["City"].isin(cities)
    if types: m &= d["Property_Type"].isin(types)
    if furn: m &= d["Furnished_Status"].isin(furn)
    if avail: m &= d["Availability_Status"].isin(avail)
    work = d[m]

    k = st.columns(4)
    with k[0]:
        kpi("Matching properties", f"{len(work):,}", "Current filters")
    with k[1]:
        kpi("Average price", lakh(work["Price_in_Lakhs"].mean()) if len(work) else "—", "Filtered")
    with k[2]:
        kpi("Median ₹ / sq ft", inr(work["Price_per_SqFt"].median()) if len(work) else "—", "Filtered")
    with k[3]:
        kpi("Rule-based Good share", f"{work['Good_Investment'].mean() * 100:.1f}%" if len(work) else "—",
            f"Score ≥ {REF['threshold']} of 5")
    if work.empty:
        st.warning("No properties match these filters. Widen the budget, BHK or location.")
        return

    cap = 20000
    scored = work if len(work) <= cap else work.sample(cap, random_state=42)
    if len(work) > cap:
        st.caption(
            f"{len(work):,} properties match; a random {cap:,} are scored and ranked. Narrow the filters to rank all of them.")
    scored = scored.copy()
    if has_clf:
        proba = hp.predict_good_probability(bundle, bundle["clf"]["champion"], scored)
        scored["Good probability %"] = (proba * 100).round(1)
        if only_good:
            scored = scored[scored["Good probability %"] >= 50]
    if sort_by == "Good-investment probability":
        scored = scored.sort_values("Good probability %", ascending=False)
    elif sort_by.startswith("Price"):
        scored = scored.sort_values("Price_in_Lakhs")
    elif sort_by.startswith("₹"):
        scored = scored.sort_values("Price_per_SqFt")
    else:
        scored = scored.sort_values("Investment_Score", ascending=False)

    cols = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt", "Age_of_Property", "Furnished_Status", "Availability_Status", "Investment_Score"]
    if has_clf:
        cols = ["Good probability %"] + cols
    out = scored[cols].head(show_n).copy()
    out["Price_per_SqFt"] = out["Price_per_SqFt"].round(0)
    cfg = {}
    if has_clf:
        cfg["Good probability %"] = st.column_config.ProgressColumn("Good probability", min_value=0, max_value=100,
                                                                    format="%.0f%%")
    st.markdown(f"### Top {min(show_n, len(scored)):,} properties")
    table(out, column_config=cfg)
    stretch(st.download_button, "Download ranked list (CSV)", scored[cols].to_csv(index=False).encode("utf-8"),
            file_name="ranked_properties.csv", mime="text/csv")

    a, b = st.columns(2)
    with a:
        fig = px.histogram(scored, x="Price_in_Lakhs", nbins=40, color_discrete_sequence=[PALETTE[0]],
                           title="Price distribution of the scored properties")
        fig.update_layout(xaxis_title="Price (₹ lakh)")
        show(fig, 320)
    with b:
        gg = scored.groupby("City")["Price_per_SqFt"].mean().sort_values(ascending=False).head(12).reset_index()
        fig = px.bar(gg, x="Price_per_SqFt", y="City", orientation="h", color_discrete_sequence=[PALETTE[1]],
                     title="Average ₹ per sq ft by city")
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="₹ per sq ft", yaxis_title="")
        show(fig, 320)


# ======================================================== PAGE: Insights
def page_insights():
    hero("MARKET INSIGHTS", "Heatmaps, trends and infrastructure effects.",
         "Location-wise heatmaps, price trends over construction year, and how infrastructure relates to value.")
    d = DF
    view = st.radio("View", ["Location heatmaps", "Trend charts", "Infrastructure and resale value",
                             "Investment signals"], horizontal=True)

    if view == "Location heatmaps":
        a, b, c = st.columns(3)
        metric = a.selectbox("Metric", ["Average ₹ per sq ft", "Average price (₹ lakh)", "Good-investment share (%)"])
        dim = b.selectbox("Compare cities across", ["Property_Type", "BHK", "Furnished_Status",
                                                    "Public_Transport_Accessibility", "Owner_Type"])
        top_n = c.slider("Cities to show (most listings)", 5, 40, 20)
        col = {"Average ₹ per sq ft": "Price_per_SqFt", "Average price (₹ lakh)": "Price_in_Lakhs",
               "Good-investment share (%)": "Good_Investment"}[metric]
        sub = d[d["City"].isin(d["City"].value_counts().head(top_n).index)]
        pv = sub.pivot_table(index="City", columns=dim, values=col, aggfunc="mean")
        if col == "Good_Investment":
            pv = pv * 100
        pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]
        fig = px.imshow(pv.round(1), text_auto=".0f", aspect="auto", color_continuous_scale=HEAT,
                        labels=dict(color=metric))
        show(fig, max(380, 26 * len(pv) + 120))

    elif view == "Trend charts":
        a, b = st.columns([2, 1])
        default = d["City"].value_counts().head(5).index.tolist()
        cities = a.multiselect("Cities", CHOICES["City"], default=default)
        metric = b.selectbox("Metric", ["Average ₹ per sq ft", "Average price (₹ lakh)"])
        col = "Price_per_SqFt" if "sq ft" in metric else "Price_in_Lakhs"
        if cities:
            tr = d[d["City"].isin(cities)].groupby(["City", "Year_Built"])[col].mean().reset_index()
            fig = px.line(tr, x="Year_Built", y=col, color="City", markers=True, color_discrete_sequence=PALETTE)
            fig.update_layout(xaxis_title="Year built", yaxis_title=metric, title=f"{metric} by construction year")
            show(fig, 400)
        age = d.assign(Age_Band=pd.cut(d["Age_of_Property"], [-1, 2, 5, 10, 15, 20, 30, 200],
                                       labels=["0-2", "3-5", "6-10", "11-15", "16-20", "21-30", "30+"]))
        g = age.groupby("Age_Band", observed=True)["Price_per_SqFt"].mean().reset_index()
        fig = px.bar(g, x="Age_Band", y="Price_per_SqFt", text_auto=".0f", color_discrete_sequence=[PALETTE[2]],
                     title="Average ₹ per sq ft by property age (years)")
        fig.update_layout(xaxis_title="Age band", yaxis_title="₹ per sq ft")
        show(fig, 340)
        st.caption("The dataset has no sale dates; Year Built stands in for time.")

    elif view == "Infrastructure and resale value":
        st.markdown(
            "The infrastructure score (0–10) blends nearby schools (35%), hospitals (25%) and public transport (40%).")
        bands = pd.cut(d["Infrastructure_Score"], [-0.01, 2, 4, 6, 8, 10],
                       labels=["0–2", "2–4", "4–6", "6–8", "8–10"])
        g = d.assign(Band=bands).groupby("Band", observed=True).agg(
            PPSF=("Price_per_SqFt", "mean"), Price=("Price_in_Lakhs", "mean"),
            Listings=("Price_in_Lakhs", "size")).reset_index()
        a, b = st.columns(2)
        with a:
            fig = px.bar(g, x="Band", y="PPSF", text_auto=".0f", color_discrete_sequence=[PALETTE[0]],
                         title="Average ₹ per sq ft by infrastructure score")
            fig.update_layout(xaxis_title="Infrastructure score", yaxis_title="₹ per sq ft")
            show(fig, 340)
        with b:
            fig = px.bar(g, x="Band", y="Price", text_auto=".0f", color_discrete_sequence=[PALETTE[1]],
                         title="Average price by infrastructure score")
            fig.update_layout(xaxis_title="Infrastructure score", yaxis_title="Price (₹ lakh)")
            show(fig, 340)
        st.caption(corr_text(d["Infrastructure_Score"], d["Price_per_SqFt"], "infrastructure score", "₹ per sq ft"))
        st.caption(corr_text(d["School_Density_Score"], d["Price_per_SqFt"], "school density score", "₹ per sq ft"))
        s = samp(d, 6000)
        fig = px.scatter(s, x="Infrastructure_Score", y="Price_per_SqFt", opacity=0.3, color="Property_Type",
                         color_discrete_sequence=PALETTE, title="Infrastructure score vs ₹ per sq ft (sample)")
        show(fig, 380)

    else:
        dim = st.selectbox("Break down the Good-Investment share by",
                           ["Public_Transport_Accessibility", "BHK", "Owner_Type", "Property_Type",
                            "Furnished_Status", "Availability_Status", "State", "City"])
        g = d.groupby(dim)["Good_Investment"].mean().mul(100).sort_values(ascending=False).reset_index()
        if len(g) > 25:
            g = g.head(25)
        fig = px.bar(g, x=dim, y="Good_Investment", text_auto=".1f", color="Good_Investment",
                     color_continuous_scale=SCALE, title=f"Share of Good Investments by {dim}")
        fig.update_layout(coloraxis_showscale=False, yaxis_title="% Good Investment", xaxis_title="")
        fig.update_xaxes(type="category")
        show(fig, 400)
        sc = d["Investment_Score"].value_counts().sort_index().reset_index()
        sc.columns = ["Rule score", "Properties"]
        fig = px.bar(sc, x="Rule score", y="Properties", text_auto=",", color_discrete_sequence=[PALETTE[3]],
                     title="Distribution of the 0–5 rule score")
        fig.update_xaxes(type="category")
        show(fig, 320)
        note("Crime rate is not in the supplied dataset, so its impact on the classification cannot be measured here.")


# ========================================================== PAGE: Model Lab
def page_model_lab():
    hero("MODEL LAB", "Train, compare and track the models.",
         "Logistic Regression and Ridge are the baselines; XGBoost is the challenger. "
         "Every run is logged to MLflow and the best model of each task is registered.")
    if "train_notice" in st.session_state:
        st.success(st.session_state.pop("train_notice"))

    st.markdown("### Training")
    a, b, c = st.columns([1.3, 1, 1])
    rows = a.slider("Rows used for training (random sample)", 5000, max(len(DF), 5001),
                    min(hp.DEFAULT_TRAIN_ROWS, len(DF)), step=5000,
                    help="Fewer rows train faster. 60,000 is usually enough.")
    use_mlflow = b.checkbox("Log runs to MLflow", value=True)
    b.caption(
        "XGBoost: " + ("available" if hp.xgboost_available() else "not installed - using HistGradientBoosting instead"))
    c.write("")
    if stretch(st.button, "Train / retrain models", type="primary"):
        bar = st.progress(0.0, text="Starting…")

        def cb(msg, frac):
            bar.progress(float(min(max(frac, 0.0), 1.0)), text=msg)

        try:
            bundle = hp.train_all(DF, REF, max_rows=rows, log_mlflow=use_mlflow, progress=cb)
            hp.save_bundle(bundle)
            _load_bundle.clear()
            st.session_state["train_notice"] = (
                    f"Models trained on {bundle['train_rows']:,} rows and saved. "
                    + (f"{bundle['mlflow']['runs']} MLflow runs logged." if bundle["mlflow"]["enabled"]
                       else (
                f"MLflow logging skipped: {bundle['mlflow']['error']}" if bundle["mlflow"]["error"] else "")))
            rerun()
        except Exception as exc:
            bar.empty()
            st.error(f"Training failed: {exc}")
            st.exception(exc)

    bundle = current_bundle()
    if bundle is None:
        no_models_notice()
        return
    st.caption(f"Last trained {bundle['created']} · {bundle['train_rows']:,} training rows, "
               f"{bundle['test_rows']:,} test rows (sampled from {bundle['total_rows']:,}).")

    st.markdown("### Classification: Good Investment")
    if bundle.get("clf"):
        cm_ = bundle["clf"]["metrics"]
        champ = bundle["clf"]["champion"]
        tbl = pd.DataFrame(
            [{"Model": n + (" ★" if n == champ else ""), "Accuracy": m["accuracy"], "Precision": m["precision"],
              "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["auc"], "Fit (s)": m["fit_seconds"]}
             for n, m in cm_.items()])
        table(tbl.round(4))
        sel = st.selectbox("Inspect classifier", list(cm_), index=list(cm_).index(champ))
        a, b = st.columns(2)
        with a:
            fig = px.imshow(cm_[sel]["cm"], text_auto=True, x=["Not good", "Good"],
                            y=["Actual: not good", "Actual: good"],
                            color_continuous_scale=SCALE, title=f"Confusion matrix · {sel}")
            fig.update_layout(coloraxis_showscale=False)
            show(fig, 340)
        with b:
            fig = go.Figure()
            for i, (n, m) in enumerate(cm_.items()):
                fig.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode="lines", name=f"{n} (AUC {m['auc']:.3f})",
                                         line=dict(color=PALETTE[i], width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#64748b", dash="dash"),
                                     name="Chance"))
            fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
            show(fig, 340)
        imp = bundle["clf"]["importance"][sel].head(12).iloc[::-1]
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[PALETTE[0]],
                     title=f"Feature importance · {sel}")
        fig.update_layout(xaxis_title="Drop in ROC-AUC when shuffled", yaxis_title="")
        show(fig, 400)
    else:
        st.info("The classifier could not be trained (only one class in the data).")

    st.markdown("### Regression: price after 5 years")
    rm_ = bundle["reg"]["metrics"]
    rchamp = bundle["reg"]["champion"]
    tbl = pd.DataFrame(
        [{"Model": n + (" ★" if n == rchamp else ""), "RMSE (₹ lakh)": m["rmse"], "MAE (₹ lakh)": m["mae"],
          "R²": m["r2"], "MAPE %": m["mape"], "Fit (s)": m["fit_seconds"]} for n, m in rm_.items()])
    table(tbl.round(4))
    rsel = st.selectbox("Inspect regressor", list(rm_), index=list(rm_).index(rchamp))
    a, b = st.columns(2)
    with a:
        m = rm_[rsel]
        lo, hi = float(min(m["y_test"].min(), m["y_pred"].min())), float(max(m["y_test"].max(), m["y_pred"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["y_test"], y=m["y_pred"], mode="markers", name="Test predictions",
                                 marker=dict(color=PALETTE[0], size=5, opacity=0.5)))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect",
                                 line=dict(color=PALETTE[3], dash="dash")))
        fig.update_layout(title=f"Predicted vs actual · {rsel}", xaxis_title="Scenario price (₹ lakh)",
                          yaxis_title="Predicted (₹ lakh)")
        show(fig, 340)
    with b:
        resid = (m["y_pred"] - m["y_test"]) / np.maximum(m["y_test"], 1e-9) * 100
        fig = px.histogram(x=resid, nbins=40, color_discrete_sequence=[PALETTE[1]], title="Prediction error (%)")
        fig.update_layout(xaxis_title="Error as % of actual", yaxis_title="Test properties")
        show(fig, 340)
    imp = bundle["reg"]["importance"][rsel].head(12).iloc[::-1]
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[PALETTE[1]],
                 title=f"Feature importance · {rsel}")
    fig.update_layout(xaxis_title="Drop in R² when shuffled", yaxis_title="")
    show(fig, 400)

    st.markdown("### MLflow experiment tracking")
    ml = bundle.get("mlflow", {})
    if ml.get("enabled"):
        st.success(f"{ml['runs']} runs logged to experiment “{ml['experiment']}”. Best models are registered as "
                   f"“{hp.MLFLOW_CLF_NAME}” and “{hp.MLFLOW_REG_NAME}”.")
    elif ml.get("error"):
        st.warning(f"MLflow logging did not run: {ml['error']}. Install it with `pip install mlflow` and retrain.")
    else:
        st.info("MLflow logging was switched off for the last training run.")
    st.markdown("Open the MLflow dashboard from a terminal in this folder:")
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")
    runs, err = hp.mlflow_runs()
    if runs is not None:
        table(runs.round(4))
    elif err:
        st.caption(err)

    with st.expander("Methodology, targets and limitations", expanded=True):
        st.markdown(f"""
**Good Investment label (multi-factor score).** A property scores one point for each rule it meets:
price per sq ft ≤ city median · price ≤ city median · 3+ BHK · ready to move · infrastructure score ≥ dataset median.
Score ≥ **{REF['threshold']}** means *Good Investment*. (The dataset has no RERA column, so it is not used.)

**5-year price target (scenario).** Future = Current × (1 + r)^5. The dataset has no observed future prices, so *r* is an
assumption: a fixed 8% (project brief), a city/property-type rate (8% ± 2 pp by market level, ± 1 pp by type), and the
feature-based rate the regressor learns (location rate ± infrastructure, transport, age and readiness, about 3–13%).

**Features.** Price per sq ft (recomputed in ₹), amenity count, parking/security flags, transport score (Low/Medium/High = 1/2/3),
school density score (schools relative to the city median), infrastructure score (0–10), floor ratio, price ratios to the city median.
Numeric features are median-imputed and scaled (linear models); categoricals are most-frequent-imputed and one-hot encoded (linear)
or ordinal encoded (XGBoost).

**Read the metrics carefully.** Both targets are computed from the same features the models see, so very high accuracy and R² show
that the models recovered the rule/formula, **not** that they predict real market outcomes. With real historical sale
prices and investment outcomes, replace the targets in `housing_pipeline.py` and retrain.
""")


# ==================================================================== router
PAGES = {
    "Overview": page_overview,
    "EDA · 20 Questions": page_eda,
    "Investment Advisor": page_advisor,
    "Property Finder": page_finder,
    "Market Insights": page_insights,
    "Model Lab": page_model_lab,
}

with st.sidebar:
    st.markdown("## 🏙️ EstateIQ")
    st.caption("REAL ESTATE INVESTMENT ADVISOR")
    st.markdown("---")
    page = st.radio("WORKSPACE", list(PAGES), index=0)
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"`{data_path.name}`")
    st.caption(f"{len(DF):,} rows after cleaning")
    st.markdown("---")
    st.markdown('<div class="small-muted">Estimates are scenario-based decision support, not financial advice.</div>',
                unsafe_allow_html=True)

PAGES[page]()
st.markdown("---")
st.markdown('<div class="small-muted">EstateIQ · Verify listings, legal status, financing and market conditions '
            'independently before investing.</div>', unsafe_allow_html=True)
st.sidebar.caption("DEGINED BY - ⍻⍿икαи")
