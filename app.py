"""Streamlit demo: Colombo land price forecast (Component 04).

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from predict_price import DEFAULT_MODEL, FORECAST_PERIODS, LAND_TYPES, Predictor  # noqa: E402

st.set_page_config(page_title="Land Price Forecast", page_icon="🏞️", layout="wide")


@st.cache_resource
def load_predictor():
    return Predictor()


def rs(value):
    """Format Rupees as 'Rs 13.82 million' (or full number below one million)."""
    if value >= 1_000_000:
        return f"Rs {value / 1_000_000:,.2f} M"
    return f"Rs {value:,.0f}"


pred = load_predictor()

# ---------- Header ----------
st.title("🏞️ Colombo Land Price Forecast")
st.caption(
    "Component 04: AI-based asset performance prediction for the land tokenization platform. "
    "Price per perch = Part A land value model (base price) x Part B CBSL LVI trend forecast."
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
        model_name = st.selectbox("Part A model", list(pred.models), index=list(pred.models).index(DEFAULT_MODEL))

if not land_types:
    st.warning("Choose at least one land type in the sidebar.")
    st.stop()

year, half = int(period[:4]), int(period[-1])

# ---------- Predict ----------
result = pred.predict_price(area, perches, land_types, year, half, model_name)
base = result["base_price_2025H1"]
mid = result["price_per_perch_mid"]

st.subheader(f"{area.title()}  |  {perches:g} perches  |  {', '.join(land_types)}  |  {period}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Price per perch (mid)", rs(mid), f"{(mid / base - 1) * 100:+.1f}% vs 2025-H1",
          help="Mid forecast: Part A base price x LVI growth (Drift trend).")
c2.metric("Total land price (mid)", rs(result["total_mid"]))
c3.metric("Low case per perch", rs(result["price_per_perch_low"]),
          help="Low end of the LVI trend range. It does not include the Part A model error.")
c4.metric("High case per perch", rs(result["price_per_perch_high"]),
          help="High end of the LVI trend range. It does not include the Part A model error.")

# ---------- Chart over time ----------
rows = []
for p in FORECAST_PERIODS:
    r = pred.predict_price(area, perches, land_types, int(p[:4]), int(p[-1]), model_name)
    rows.append({"period": p, "low": r["price_per_perch_low"], "mid": r["price_per_perch_mid"],
                 "high": r["price_per_perch_high"]})
curve = pd.DataFrame(rows)

left, right = st.columns([3, 2])
with left:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(curve["period"], curve["mid"] / 1e6, marker="o", color="#1f77b4", label="Mid forecast")
    ax.fill_between(curve["period"], curve["low"] / 1e6, curve["high"] / 1e6,
                    alpha=0.2, color="#1f77b4", label="Low to high range")
    sel = curve[curve["period"] == period].iloc[0]
    ax.scatter([period], [sel["mid"] / 1e6], s=140, color="#d62728", zorder=5, label=f"Selected ({period})")
    ax.set_ylabel("Price per perch (million Rs)")
    ax.set_title("Price per perch over time")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)

with right:
    st.markdown("**How the price is calculated**")
    st.markdown(
        f"1. **Part A** (land value model, {model_name}) gives the base price today (2025-H1 level): "
        f"**{rs(base)}** per perch.\n"
        f"2. **Part B** (CBSL Land Valuation Indicator trend) says the market grows to "
        f"**x{result['growth_mid']:.3f}** by {period}.\n"
        f"3. Future price = {rs(base)} x {result['growth_mid']:.3f} = **{rs(mid)}** per perch."
    )
    table = curve.copy()
    for col in ["low", "mid", "high"]:
        table[col] = table[col].map(lambda v: f"{v:,.0f}")
    st.dataframe(table.rename(columns={"period": "Period", "low": "Low (Rs)", "mid": "Mid (Rs)", "high": "High (Rs)"}),
                 hide_index=True, width="stretch")
