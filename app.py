"""Streamlit demo: Colombo land price forecast (Component 04).

Run:  python -m streamlit run app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from predict_price import (  # noqa: E402
    DEFAULT_MODEL, FORECAST_PERIODS, LAND_TYPES, REPORT_DIR, ROOT, Predictor,
)

st.set_page_config(page_title="Land Price Forecast", page_icon="🏞️", layout="wide")

TEAL = "#1fdf6e"      # accent green
ORANGE = "#d6a644"    # amber
SURFACE = "#222325"
BORDER = "#2f3032"
SOFT_TEXT = "#c9ccd1"

# Dark chart style (same look as the app cards)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BORDER, "axes.labelcolor": SOFT_TEXT, "axes.titlecolor": "#ffffff",
    "text.color": "#ffffff", "xtick.color": SOFT_TEXT, "ytick.color": SOFT_TEXT,
    "grid.color": BORDER, "legend.labelcolor": "#e5e7eb",
})

# ---------- Style ----------
st.markdown(
    """
    <style>
    :root {--bg:#191a1c; --surface:#222325; --inner:#18191b; --border:#2f3032;
           --accent:#1fdf6e; --amber:#d6a644; --muted:#9a9ea6;}
    html, body, [class*="css"] {font-family: Inter, "Segoe UI", system-ui, sans-serif;}
    .block-container {padding-top: 1.4rem; max-width: 1200px;}
    [data-testid="stAppDeployButton"], #MainMenu, footer {display: none !important;}
    [data-testid="stHeader"] {background: transparent;}
    [data-testid="stSidebar"] {background: var(--inner); border-right: 1px solid var(--border);}

    .topbar {display:flex; align-items:center; justify-content:space-between;
             padding: 4px 0 16px 0; border-bottom: 1px solid var(--border); margin-bottom: 22px;}
    .brand {display:flex; align-items:center; gap:10px; font-weight:600; font-size:0.95rem;}
    .logo {width:32px; height:32px; border-radius:9px; background:var(--accent);
           display:flex; align-items:center; justify-content:center;}
    .page-title {font-size:2.2rem; font-weight:800; line-height:1.15; margin:0 0 6px 0;}
    .page-sub {color:var(--muted); margin-bottom:18px; font-size:0.95rem;}

    .pill {display:inline-block; padding:2px 11px; border-radius:999px; font-size:0.75rem;
           font-weight:600; border:1px solid; margin:0 6px 8px 0;}
    .pill.green {color:#3ddc84; border-color:#1d6b3f; background:rgba(31,223,110,0.10);}
    .pill.amber {color:#d6a644; border-color:#6b5217; background:rgba(214,166,68,0.10);}

    [data-testid="stMetric"] {background:var(--surface); border:1px solid var(--border);
           border-radius:14px; padding:16px 18px;}
    [data-testid="stMetricValue"] {font-weight:700;}
    [data-testid="stExpander"] {background:var(--surface); border:1px solid var(--border); border-radius:12px;}
    [data-testid="stDataFrame"] {border:1px solid var(--border); border-radius:12px; overflow:hidden;}
    [data-baseweb="tab-list"] {gap:6px;}
    button[role="tab"] {border-radius:999px; padding:6px 16px;}

    .step {background:var(--inner); border:1px solid var(--border); border-left:3px solid var(--accent);
           padding:12px 14px; border-radius:10px; margin-bottom:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    return Predictor()


@st.cache_data
def load_lvi_history():
    """CBSL LVI history (Overall), used only for the chart in 'How it works'."""
    try:
        from load_cbsl import load_lvi_half_yearly
        df = load_lvi_half_yearly(ROOT / "data" / "raw" / "cbsl_half_yearly_dataset.xls")
        return df[["period", "Overall"]].dropna().reset_index(drop=True)
    except Exception:
        return None


@st.cache_data
def read_csv_or_none(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def rs(value):
    """Format Rupees as 'Rs 13.82 M' (or the full number below one million)."""
    if value >= 1_000_000:
        return f"Rs {value / 1_000_000:,.2f} M"
    return f"Rs {value:,.0f}"


def style_axis(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}M"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


pred = load_predictor()

# ---------- Header ----------
st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="logo"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0b1f12"
             stroke-width="2.4" stroke-linejoin="round"><path d="M12 2l9 5v10l-9 5-9-5V7z"/></svg></div>
        <span>Land price forecast</span>
      </div>
      <div><span class="pill green">Component 04 · AI Forecasting</span></div>
    </div>
    <div class="page-title">Colombo Land Price Forecast</div>
    <div class="page-sub">Price per perch = Part A land value model (base price) × Part B CBSL LVI trend forecast.</div>
    """,
    unsafe_allow_html=True,
)

# ---------- Inputs (sidebar) ----------
with st.sidebar:
    st.header("Land details")
    area = st.selectbox(
        "Area", pred.areas, index=pred.areas.index("narahenpita"),
        format_func=lambda a: a.title(),
    )
    perches = st.number_input("Land size (perches)", min_value=1.0, max_value=500.0,
                              value=10.0, step=0.5)
    land_types = st.multiselect("Land type", LAND_TYPES, default=["residential"])

    st.header("Target time")
    period = st.select_slider("Half-year", options=FORECAST_PERIODS, value="2028-H2")

    with st.expander("Advanced"):
        model_name = st.selectbox("Part A model", list(pred.models),
                                  index=list(pred.models).index(DEFAULT_MODEL))

if not land_types:
    st.warning("Choose at least one land type in the sidebar.")
    st.stop()

year, half = int(period[:4]), int(period[-1])
result = pred.predict_price(area, perches, land_types, year, half, model_name)
base = result["base_price_2025H1"]
mid = result["price_per_perch_mid"]

tab_forecast, tab_compare, tab_val, tab_how = st.tabs(
    ["Forecast", "Compare areas", "Validation", "How it works"])

# =====================================================================
# Tab 1: Forecast
# =====================================================================
with tab_forecast:
    st.subheader(f"{area.title()}  ·  {perches:g} perches  ·  {', '.join(land_types)}  ·  {period}")
    st.markdown(
        f'<span class="pill green">Part A · {model_name}</span>'
        f'<span class="pill green">Part B · Drift trend</span>'
        f'<span class="pill amber">{period}</span>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price per perch (mid)", rs(mid), f"{(mid / base - 1) * 100:+.1f}% vs 2025-H1",
              help="Mid forecast: Part A base price x LVI growth (Drift trend).")
    c2.metric("Total land price (mid)", rs(result["total_mid"]))
    c3.metric("Low case per perch", rs(result["price_per_perch_low"]),
              help="Low end of the LVI trend range. It does not include the Part A model error.")
    c4.metric("High case per perch", rs(result["price_per_perch_high"]),
              help="High end of the LVI trend range. It does not include the Part A model error.")

    rows = []
    for p in FORECAST_PERIODS:
        r = pred.predict_price(area, perches, land_types, int(p[:4]), int(p[-1]), model_name)
        rows.append({"period": p, "low": r["price_per_perch_low"], "mid": r["price_per_perch_mid"],
                     "high": r["price_per_perch_high"]})
    curve = pd.DataFrame(rows)

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(curve["period"], curve["mid"] / 1e6, marker="o", color=TEAL, linewidth=2, label="Mid forecast")
        ax.fill_between(curve["period"], curve["low"] / 1e6, curve["high"] / 1e6,
                        alpha=0.18, color=TEAL, label="Low to high range")
        sel = curve[curve["period"] == period].iloc[0]
        ax.scatter([period], [sel["mid"] / 1e6], s=150, color=ORANGE, zorder=5, label=f"Selected ({period})")
        ax.annotate(f"{sel['mid'] / 1e6:.2f}M", (period, sel["mid"] / 1e6), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=10, fontweight="bold", color=ORANGE)
        style_axis(ax, "Price per perch (Rs)")
        ax.set_title("Price per perch over time", loc="left", fontweight="bold")
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**How this price was calculated**")
        st.markdown(
            f"""
            <div class="step"><b>1 · Part A</b> ({model_name}) gives today's base price
            (2025-H1 level): <b>{rs(base)}</b> per perch.</div>
            <div class="step"><b>2 · Part B</b> (CBSL Land Valuation Indicator trend) says the
            market grows by <b>×{result['growth_mid']:.3f}</b> until {period}.</div>
            <div class="step"><b>3 · Future price</b> = {rs(base)} × {result['growth_mid']:.3f}
            = <b>{rs(mid)}</b> per perch.</div>
            """,
            unsafe_allow_html=True,
        )
        table = curve.copy()
        for col in ["low", "mid", "high"]:
            table[col] = table[col].map(lambda v: f"{v:,.0f}")
        st.dataframe(
            table.rename(columns={"period": "Period", "low": "Low (Rs)", "mid": "Mid (Rs)", "high": "High (Rs)"}),
            hide_index=True, width="stretch", height=250,
        )

    with st.expander("What do the three Part A models say for this land?"):
        cmp_rows = []
        for m in pred.models:
            r = pred.predict_price(area, perches, land_types, year, half, m)
            cmp_rows.append({"Part A model": m, "Base price today (Rs/perch)": f"{r['base_price_2025H1']:,}",
                             f"Mid price {period} (Rs/perch)": f"{r['price_per_perch_mid']:,}"})
        st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, width="stretch")

# =====================================================================
# Tab 2: Compare areas
# =====================================================================
with tab_compare:
    st.subheader("Compare areas")
    st.caption(f"Same land ({perches:g} perches, {', '.join(land_types)}) in different areas. "
               "Land details come from the sidebar.")
    defaults = [a for a in ["narahenpita", "bambalapitiya", "maharagama", "angulana"] if a in pred.areas]
    chosen = st.multiselect("Areas to compare", pred.areas, default=defaults,
                            format_func=lambda a: a.title())
    log_scale = st.checkbox("Log scale (easier when prices are very different)", value=False)

    if not chosen:
        st.info("Choose at least one area.")
    else:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        summary = []
        for a in chosen:
            mids = [pred.predict_price(a, perches, land_types, int(p[:4]), int(p[-1]), model_name)
                    ["price_per_perch_mid"] / 1e6 for p in FORECAST_PERIODS]
            ax.plot(FORECAST_PERIODS, mids, marker="o", linewidth=2, label=a.title())
            summary.append({"Area": a.title(), "Today (Rs/perch)": f"{mids[0] * 1e6:,.0f}",
                            f"{period} (Rs/perch)": f"{mids[FORECAST_PERIODS.index(period)] * 1e6:,.0f}",
                            "Growth": f"{(mids[-1] / mids[0] - 1) * 100:+.1f}% by 2028-H2"})
        if log_scale:
            ax.set_yscale("log")
        style_axis(ax, "Mid price per perch (Rs)")
        ax.set_title("Mid price per perch over time", loc="left", fontweight="bold")
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")

# =====================================================================
# Tab 3: How it works
# =====================================================================
with tab_how:
    st.subheader("How it works")
    st.markdown(
        """
        <div class="step"><b>Part A · Land value model</b> predicts today's base price per perch from the
        area, land size and land type. Prices are converted to the 2025-H1 level, so the model does not need time.
        Models compared: Ridge, Random Forest, XGBoost.</div>
        <div class="step"><b>Part B · Trend model</b> forecasts the CBSL Land Valuation Indicator (LVI),
        half-yearly, up to 2028-H2. Models compared: Naive, Drift, Linear, Holt. Drift won the rolling backtest.</div>
        <div class="step"><b>Combine</b> · Future price = Part A base price × (LVI future ÷ 256.4).
        The low / mid / high range comes from the Part B backtest error.</div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Part A: test results (untouched 20% test set)**")
        res_a = read_csv_or_none(REPORT_DIR / "tables" / "04_test_results.csv")
        if res_a is not None:
            res_a = res_a.rename(columns={res_a.columns[0]: "Model"})
            keep = [c for c in ["Model", "R2", "MAPE_%", "Median_APE_%", "Within_20%"] if c in res_a.columns]
            st.dataframe(res_a[keep].round(2), hide_index=True, width="stretch")
        st.caption("Training data is augmented and calibrated to the CBSL LVI.")
    with col_b:
        st.markdown("**Part B: rolling backtest (24 forecasts per model)**")
        res_b = read_csv_or_none(REPORT_DIR / "model_b" / "model_b_rolling_results.csv")
        if res_b is not None:
            st.dataframe(res_b.rename(columns={"model": "Model", "n_forecasts": "Forecasts"}),
                         hide_index=True, width="stretch")
        st.caption("MAPE = average % error. Lower is better.")

    hist = load_lvi_history()
    fc = pred.lvi_forecast
    if hist is not None:
        st.markdown("**CBSL Land Valuation Indicator (Colombo): history and forecast**")
        fig, ax = plt.subplots(figsize=(10, 3.8))
        n = len(hist)
        ax.plot(range(n), hist["Overall"], marker="o", color=TEAL, linewidth=2, label="History")
        xs = range(n, n + len(fc))
        ax.plot(xs, fc["mid"], marker="o", linestyle="--", color=ORANGE, linewidth=2, label="Drift forecast (mid)")
        ax.fill_between(xs, fc["low"], fc["high"], alpha=0.2, color=ORANGE, label="Low to high range")
        labels = list(hist["period"]) + list(fc["period"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, fontsize=8)
        ax.set_ylabel("LVI (2017-H1 = 100)")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =====================================================================
# Tab 3: Validation (regression charts, from the saved validation files)
# =====================================================================
with tab_val:
    st.subheader("Model validation")
    st.caption(
        "Predicting a price is a regression task, so classification charts such as the ROC curve do not apply. "
        "These are the regression versions: predicted vs real, error size, error by price level, "
        "unseen areas and the trend model backtest. Part A charts use the untouched 20% test set."
    )

    PART_A_COLS = {"Ridge": "pred_ridge", "Random Forest": "pred_random_forest", "XGBoost": "pred_xgboost"}
    COLORS = {"Ridge": "#5aa9ff", "Random Forest": TEAL, "XGBoost": ORANGE,
              "Drift": TEAL, "Holt": ORANGE, "Linear": "#5aa9ff", "Naive": "#9ca3af"}

    tables_dir = REPORT_DIR / "tables"
    preds_df = read_csv_or_none(tables_dir / "04_test_predictions.csv")
    band_df = read_csv_or_none(tables_dir / "04_test_by_price_band.csv")
    unseen_df = read_csv_or_none(tables_dir / "04_random_vs_unseen_areas.csv")
    horizon_df = read_csv_or_none(REPORT_DIR / "model_b" / "model_b_rolling_by_horizon.csv")

    st.markdown("#### Part A: land value model")
    val_model = st.selectbox("Model", list(PART_A_COLS), index=list(PART_A_COLS).index(DEFAULT_MODEL),
                             key="val_model")

    if preds_df is not None:
        real = preds_df["real_price"].to_numpy()
        guess = preds_df[PART_A_COLS[val_model]].to_numpy()
        log_real, log_guess = np.log(real), np.log(guess)
        r2 = 1 - ((log_real - log_guess) ** 2).sum() / ((log_real - log_real.mean()) ** 2).sum()
        ape = np.abs(guess - real) / real * 100

        v1, v2 = st.columns(2)
        with v1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(real / 1e6, guess / 1e6, s=8, alpha=0.35, color=COLORS[val_model])
            lims = [min(real.min(), guess.min()) / 1e6, max(real.max(), guess.max()) / 1e6]
            ax.plot(lims, lims, color="red", linewidth=1, label="Perfect prediction")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Real price per perch (million Rs, log scale)")
            ax.set_ylabel("Predicted (million Rs, log scale)")
            ax.set_title(f"Predicted vs real  (R² on log price = {r2:.3f})", loc="left", fontweight="bold", fontsize=10)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, loc="upper left")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Points close to the red line are good predictions.")

        with v2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(np.clip(ape, 0, 100), bins=40, color=COLORS[val_model], alpha=0.85)
            ax.axvline(np.median(ape), color="red", linewidth=1.5, label=f"Median error {np.median(ape):.1f}%")
            ax.set_xlabel("Absolute % error of each test row (the last bar means 100% or more)")
            ax.set_ylabel("Number of test rows")
            ax.set_title(f"How big are the errors?  ({(ape <= 20).mean() * 100:.0f}% within 20%)",
                         loc="left", fontweight="bold", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Most errors are small. A few very expensive lands have large errors.")

    v3, v4 = st.columns(2)
    with v3:
        if band_df is not None:
            order = ["Lowest", "Low", "Middle", "High", "Highest"]
            pivot = band_df.pivot(index="Price band", columns="Model", values="Median_APE_%").loc[order]
            fig, ax = plt.subplots(figsize=(6, 4))
            width = 0.27
            for i, name in enumerate(["Ridge", "Random Forest", "XGBoost"]):
                ax.bar([x + (i - 1) * width for x in range(len(order))], pivot[name], width=width,
                       label=name, color=COLORS[name])
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order)
            ax.set_xlabel("Price level (cheapest to most expensive 20% of lands)")
            ax.set_ylabel("Median absolute % error")
            ax.set_title("Error by price level", loc="left", fontweight="bold", fontsize=10)
            ax.set_ylim(0, pivot.to_numpy().max() * 1.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("The models are weaker on the most expensive lands.")

    with v4:
        if unseen_df is not None:
            unseen_df = unseen_df.rename(columns={unseen_df.columns[0]: "Model"}).set_index("Model")
            fig, ax = plt.subplots(figsize=(6, 4))
            width = 0.35
            xs = range(len(unseen_df))
            ax.bar([x - width / 2 for x in xs], unseen_df["R2_random_test"], width=width,
                   label="Random test (areas seen in training)", color=TEAL)
            ax.bar([x + width / 2 for x in xs], unseen_df["R2_unseen_areas"], width=width,
                   label="Unseen areas (area-held-out)", color=ORANGE)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(unseen_df.index)
            ax.set_ylim(0, 1.2)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_ylabel("R²")
            ax.set_title("Known areas vs unseen areas", loc="left", fontweight="bold", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, loc="upper center", ncol=1, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("On areas the model never saw, R² is lower. This is an honest check.")

    st.markdown("#### Part B: LVI trend model")
    if horizon_df is not None:
        w1, w2 = st.columns([3, 2])
        with w1:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            steps = [c for c in horizon_df.columns if c != "model"]
            for _, row in horizon_df.iterrows():
                ax.plot(steps, [row[s] for s in steps], marker="o", linewidth=2,
                        label=row["model"], color=COLORS.get(row["model"]))
            ax.set_xlabel("How far ahead the forecast is")
            ax.set_ylabel("Average % error (MAPE)")
            ax.set_title("Rolling backtest: error by forecast distance", loc="left", fontweight="bold", fontsize=10)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with w2:
            st.markdown(
                """
                <div class="step"><b>Drift</b> has the lowest error at every distance (MAPE 3.54% overall).</div>
                <div class="step">The further ahead we forecast, the larger the error. This is why the
                forecast range gets wider with time.</div>
                <div class="step">Rolling backtest: 24 forecasts per model, from 17 half-yearly points.</div>
                """,
                unsafe_allow_html=True,
            )
