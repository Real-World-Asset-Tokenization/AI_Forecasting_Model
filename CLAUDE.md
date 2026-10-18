# Component 04: AI-Based Asset Performance Prediction & Forecasting

Project: **Compliant Blockchain-Based Real-World Asset Tokenization on Ethereum** (fractional land ownership via SPV).
My part (Component 04): ML model that predicts future land prices (Rs per perch) for the Colombo District, to be integrated into the platform later.

## User preferences
- User is a beginner in Jupyter/ML. Chat explanations in simple Singlish are fine.
- Notebook comments: put short, **simple English** comments above every cell (NOT Singlish).
- Give notebook cells **one by one in chat** (user runs each in Jupyter and checks output before the next). Do not write whole notebooks at once.
- Be honest about data: the training data is augmented and must be labelled as such.

## Data
- **Training data (final choice):** `data/raw/cbsl_land_dataset.csv`
  6,000 rows x 35 cols, 154 areas, 2021-01 to 2025-12. Augmented from real listings and calibrated to CBSL LVI (supervisor approved).
  ALWAYS describe it as "augmented, calibrated to CBSL Land Valuation Indicator, supervisor-approved". Never present it as raw real data.
- `data/colombo_land_prices.csv` (Kaggle real listings, 4,494 rows, Jul-Sep 2021) is the SOURCE the 6,000-row set was built from (1,498 area+size overlaps). **Do NOT use it as an independent test set (leakage). Do not use it in PP1.** Keep it only for reference.
- **Evaluation (PP1):** (a) random 80/20 split, and (b) **area-held-out split** (GroupKFold by Address_ID, test areas never seen in training). Always label results: "augmented data, calibrated to CBSL LVI, supervisor-approved". Real-world validation with fresh 2025/26 listings (ikman.lk / lankapropertyweb) is planned for PP2.
- 1 row = 1 listing. Target = `Price per Perch`. Features: `Land_size(Perches)`, `Land_type`, `Distance from fort`, `min_dist_*` / `count_*` amenity features. Location features are constant per area (Address_ID).
- **CBSL LVI** (Colombo, half-yearly, 2017-H1 = 100), Overall:
  `[100,107,114.7,122.3,129.7,137.2,144.3,151.6,159.8,178.7,186.9,205.2,215.3,219.8,230.1,236.8,256.4]` (2017-H1 .. 2025-H1). 2025-H2 not yet published.

- **CBSL raw files (use these, do not use pre-cleaned CSVs):** `data/raw/cbsl_half_yearly_dataset.xls` (LVI, half-yearly 2017-H1..2025-H1; 2025-H2 empty) and `data/raw/cbsl_asking_price_index_monthly_raw.xls` (Asking Price Index, monthly 2021-Jan..2024-Sep). These .xls files are really HTML tables. Load and clean them inside the notebook with `src/load_cbsl.py` (`load_lvi_half_yearly`, `load_api_monthly`).
- **Decision:** monthly Asking Price Index NOT used for the trend: Lands API is noisy (corr -0.15 with LVI half-year changes), ends 2024-Sep, and is asking prices. Mention as evaluated-but-not-used in the notebook.

## Design (two parts)
- **Part A, land value model:** predicts base price per perch from area/size/type/amenities. Time is removed by converting every price to 2025-H1 level: `price_adj = price * LVI_2025H1 / LVI_at_listing_time`. Compare 3 models: Ridge, Random Forest, Gradient Boosting (HistGradientBoosting). Use log-price target.
- **Part B, trend model:** forecasts CBSL LVI half-yearly. Simple drift (about 12.1 index pts per half-year) beat Holt and linear in backtest.
- **Combine (PP2):** future price = Part A base price x (LVI_future / 256.4).
- Why two parts: tree models cannot extrapolate time (flat future prediction); value and trend are separate questions; LVI has longer history (17 half-years).
- Show a **range** (low/mid/high), not a single number. Uncertainty grows with horizon. Backtest MAPE about 19%.

## Limitations to state honestly
- Augmented data matches LVI by construction (not independent validation).
- Asking prices, not transaction prices.
- Assumes same growth for all areas and that the LVI drift continues.
- Only 154 known Colombo areas (no map click).

## Plan
- **PP1:** train Part A and Part B separately. Notebook 1.
- **PP2:** combine + `predict_price(area, perches, land_type, year, half)` returning low/mid/high + simple frontend (area dropdown, land size, land type, target year e.g. 2028). Extras: validation on real 2025/26 listings, macro features, platform API.

## Files
- `src/component4_pipeline.py`: earlier pipeline (Part A models, Part B backtest, Part C combine). Part C should use drift forecast.
- `results/`: earlier outputs and `best_price_model.joblib` (from the older Kaggle data run).

## Next task
Build **Notebook 1 (PP1)** with beginner-friendly simple English comments, delivered cell by cell: load + clean data, time-adjust to 2025-H1 level, compare 3 models with random split + area-held-out split (labelled as augmented data), then Part B LVI trend + backtest.

## Suggested repo name
`colombo-land-price-forecasting`