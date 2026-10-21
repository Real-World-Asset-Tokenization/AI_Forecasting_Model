"""Combine Part A (land value model) and Part B (LVI trend) into one price prediction.

Same logic as notebooks/combine/01_combine_predict.ipynb, written as a module
so that the notebook, the Streamlit demo (app.py) and a future API can share it.

Future price = Part A base price (2025-H1 level) x (LVI_future / LVI_base)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

LAND_TYPES = ["residential", "commercial", "agricultural", "other"]
LAND_FEATURES = ["Land_size(Perches)", "type_residential", "type_commercial",
                 "type_agricultural", "type_other"]
FORECAST_PERIODS = ["2025-H1", "2025-H2", "2026-H1", "2026-H2",
                    "2027-H1", "2027-H2", "2028-H1", "2028-H2"]
DEFAULT_MODEL = "Random Forest"


class Predictor:
    """Loads everything once, then answers predict_price() calls."""

    def __init__(self):
        # Part A: three fitted models from Notebook 03
        self.models = {
            "Ridge": joblib.load(MODEL_DIR / "ridge_model.joblib"),
            "Random Forest": joblib.load(MODEL_DIR / "random_forest_model.joblib"),
            "XGBoost": joblib.load(MODEL_DIR / "xgboost_model.joblib"),
        }
        with open(MODEL_DIR / "split_info.json") as f:
            self.features = json.load(f)["features"]
        area_features = [f for f in self.features if f not in LAND_FEATURES]

        # One row per area (the 27 area features are fixed for each area)
        df = pd.read_csv(DATA_DIR / "land_model_data.csv")
        table = df.groupby("Address")[area_features].first()
        table.index = table.index.str.lower()
        self.area_table = table

        # Part B: LVI forecast
        self.lvi_forecast = pd.read_csv(REPORT_DIR / "model_b" / "model_b_lvi_forecast.csv")
        with open(REPORT_DIR / "model_b" / "model_b_info.json") as f:
            self.b_info = json.load(f)
        self.lvi_base = self.b_info["last_lvi"]

    @property
    def areas(self):
        return sorted(self.area_table.index)

    def predict_base(self, area, perches, land_type, model_name=DEFAULT_MODEL):
        """Part A: base price per perch (Rs) at the 2025-H1 level."""
        area = area.strip().lower()
        if area not in self.area_table.index:
            raise ValueError(f"Unknown area: {area}")
        types = [land_type] if isinstance(land_type, str) else list(land_type)
        if not types:
            raise ValueError("Choose at least one land type")
        for t in types:
            if t not in LAND_TYPES:
                raise ValueError(f"Unknown land type: {t}. Use one of {LAND_TYPES}")

        row = self.area_table.loc[area].copy()
        row["Land_size(Perches)"] = perches
        for t in LAND_TYPES:
            row["type_" + t] = 1 if t in types else 0

        X_one = pd.DataFrame([row])[self.features]
        return float(np.exp(self.models[model_name].predict(X_one)[0]))

    def lvi_growth(self, year, half):
        """Part B: LVI low / mid / high and the growth multiplier for a half-year."""
        period = f"{year}-H{half}"
        if period == "2025-H1":
            low = mid = high = self.lvi_base
        else:
            row = self.lvi_forecast[self.lvi_forecast["period"] == period]
            if row.empty:
                raise ValueError(f"No forecast for {period}. Use 2025-H1 to 2028-H2.")
            low, mid, high = row[["low", "mid", "high"]].iloc[0]
        return {
            "period": period,
            "lvi_low": float(low), "lvi_mid": float(mid), "lvi_high": float(high),
            "growth_low": float(low) / self.lvi_base,
            "growth_mid": float(mid) / self.lvi_base,
            "growth_high": float(high) / self.lvi_base,
        }

    def predict_price(self, area, perches, land_type, year, half, model_name=DEFAULT_MODEL):
        """Future price per perch (Rs) with a low / mid / high range, and the total price."""
        base = self.predict_base(area, perches, land_type, model_name)
        g = self.lvi_growth(year, half)
        low, mid, high = (base * g["growth_low"], base * g["growth_mid"], base * g["growth_high"])
        return {
            "area": area, "perches": perches, "land_type": land_type,
            "period": g["period"], "model": model_name,
            "base_price_2025H1": round(base),
            "growth_mid": g["growth_mid"],
            "price_per_perch_low": round(low),
            "price_per_perch_mid": round(mid),
            "price_per_perch_high": round(high),
            "total_low": round(low * perches),
            "total_mid": round(mid * perches),
            "total_high": round(high * perches),
        }
