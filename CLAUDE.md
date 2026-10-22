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
  6,000 rows x 35 cols, 154 areas, 2021-01 to 2025-12. Augmented from real listings and calibrated to CBSL LVI.
  ALWAYS describe it as "augmented, calibrated to CBSL Land Valuation Indicator". Never present it as raw real data.
- `data/colombo_land_prices.csv` (Kaggle real listings, 4,494 rows, Jul-Sep 2021) is the SOURCE the 6,000-row set was built from (1,498 area+size overlaps). **Do NOT use it as an independent test set (leakage). Do not use it in PP1.** Keep it only for reference.
- **Evaluation (PP1):** (a) random 80/20 split, and (b) **area-held-out split** (GroupKFold by Address_ID, test areas never seen in training). Always label results: "augmented data, calibrated to CBSL LVI". Real-world validation with fresh 2025/26 listings (ikman.lk / lankapropertyweb) is planned for PP2.
- 1 row = 1 listing. Target = `Price per Perch`. Features: `Land_size(Perches)`, `Land_type`, `Distance from fort`, `min_dist_*` / `count_*` amenity features. Location features are constant per area (Address_ID).
- **CBSL LVI** (Colombo, half-yearly, 2017-H1 = 100), Overall:
  `[100,107,114.7,122.3,129.7,137.2,144.3,151.6,159.8,178.7,186.9,205.2,215.3,219.8,230.1,236.8,256.4]` (2017-H1 .. 2025-H1). 2025-H2 not yet published.

- **CBSL raw files (use these, do not use pre-cleaned CSVs):** `data/raw/cbsl_half_yearly_dataset.xls` (LVI, half-yearly 2017-H1..2025-H1; 2025-H2 empty) and `data/raw/cbsl_asking_price_index_monthly_raw.xls` (Asking Price Index, monthly 2021-Jan..2024-Sep). These .xls files are really HTML tables. Load and clean them inside the notebook with `src/load_cbsl.py` (`load_lvi_half_yearly`, `load_api_monthly`).
- **Decision:** monthly Asking Price Index NOT used for the trend: Lands API is noisy (corr -0.15 with LVI half-year changes), ends 2024-Sep, and is asking prices. Mention as evaluated-but-not-used in the notebook.

## Design (two parts)
- **Part A, land value model:** predicts base price per perch from area/size/type/amenities. Time is removed by converting every price to 2025-H1 level: `price_adj = price * LVI_2025H1 / LVI_at_listing_time`. Compare 3 models: Ridge, Random Forest, Gradient Boosting (HistGradientBoosting). Use log-price target.
- **Part B, trend model:** forecasts CBSL LVI half-yearly. Simple drift (9.775 index pts per half-year, full history) beat Holt, linear and naive in the rolling backtest (MAPE 3.54%).
- **Combine (PP2):** future price = Part A base price x (LVI_future / 256.4).
- Why two parts: tree models cannot extrapolate time (flat future prediction); value and trend are separate questions; LVI has longer history (17 half-years).
- Show a **range** (low/mid/high), not a single number. Uncertainty grows with horizon (error about 1.76% x horizon). Part B rolling backtest MAPE 3.54%. The range covers only Part B uncertainty, not the Part A error (test MAPE about 23%).

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

## Progress
- Notebook 1 (`notebooks/model_a_land_value/01_data_understanding.ipynb`) done.
- Notebook 2 (`notebooks/model_a_land_value/02_data_preparation.ipynb`) done. Half-year label + LVI join, `price_adj` = price x 256.4 / LVI at listing time, 4 Land_type flags, dropped Mentioned Price (leakage) and Price_Scale.
  Median check: raw +47.7%, adjusted almost flat (2021-H1 about 8% high; data grows slower than LVI, mention in limitations).
  2025-H2 rows (624) set aside, no LVI yet. Outputs: `data/processed/land_model_data.csv` (5,376 x 39: ids, 32 features, target `log_price_adj`) and `land_2025H2_aside.csv`.
- Notebook 3 (`notebooks/model_a_land_value/03_model_training.ipynb`) done. Models changed to Ridge, Random Forest, XGBoost (xgboost 3.4.1 installed). Random 80/20 split (seed 42, 4,300 train / 1,076 test); 111 of 113 test areas also in train (random test will look optimistic). Tuning by 5-fold GroupKFold on Address_ID inside train only. Validation (log RMSE / R2 / MAPE): Ridge 0.493 / 0.775 / 41.8%, RF 0.497 / 0.772 / 41.5%, XGBoost 0.502 / 0.767 / 41.5%; models are tied (gap 0.01 < fold std 0.05). RF best params at grid edge (could widen). Saved: `models/` (3 joblib, best_params.json, split_info.json), `data/processed/train_80.csv`, `test_20_untouched.csv`, `reports/tables/03_validation_*.csv`. Test set not used for any tuning.
- Notebook 4 (`notebooks/model_a_land_value/04_model_evaluation.ipynb`) done. Untouched test (R2 / MAPE): Ridge 0.822 / 35.3%, RF 0.893 / 23.4%, XGBoost 0.898 / 23.6%. Area-held-out 5 folds R2: Ridge 0.753, RF 0.657, XGBoost 0.661 (Ridge best on unseen areas). Do NOT report "100 - MAPE" as accuracy.
- Model B notebook (`notebooks/model_b_trend/01_lvi_trend_model.ipynb`) done. 17 LVI points; Naive/Drift/Linear/Holt. Fixed split (14 train / 3 test): Holt 1.33%, Drift 1.52%. Rolling backtest (24 forecasts): Drift 3.54%, Holt 4.48%, Linear 4.69%, Naive 10.17%. Chosen: Drift. Holt best alpha 0.9, beta 0.2 (grid edge; unstable across windows). Forecast 2028-H2: mid 324.8 (low 284.9, high 364.8). Outputs in `reports/model_b/`.
- Combine notebook (`notebooks/combine/01_combine_predict.ipynb`) done. `predict_base()` (Part A, default Random Forest, area table of 154 areas with 27 fixed area features + 5 land features) and `lvi_growth()` (Part B) joined in `predict_price(area, perches, land_type, year, half)`; valid periods 2025-H1 to 2028-H2. Example: narahenpita, 10 perch, residential, 2028-H2: base Rs 10.9M, mid Rs 13.8M per perch (range 12.1M to 15.5M). Sanity checks passed. Outputs in `reports/combine/`.
- Known gaps: models were trained on the 80% train only (not refit on all data); "Ridge + time feature" variant was not tested; range excludes Part A error; low case never goes below today's price.
- Streamlit demo done: `src/predict_price.py` (class `Predictor`, same logic as the combine notebook) and `app.py` (run: `streamlit run app.py`). Sidebar: area dropdown (154), perches, land type (multi-select), half-year slider 2025-H1..2028-H2, Part A model choice. Shows mid / total / low / high price, chart with range, calculation steps, notes. Checked with Streamlit AppTest; app output matches the notebook (narahenpita 10 perch residential 2028-H2: mid Rs 13.82M per perch). Screenshot: `reports/figures/streamlit_demo_narahenpita_2028H2.png`.
- App UI upgraded (4 tabs): Forecast, Compare areas, Validation (predicted vs real, error histogram, error by price level, known vs unseen areas, Part B error by horizon; no ROC because this is regression), How it works. Screenshots: `reports/figures/app_ui_*.png`.
- App restyled to match the investor marketplace UI (dark `#191a1c`, cards `#222325`, border `#2f3032`, green accent `#1fdf6e`, amber `#d6a644`, pill badges, top bar). Theme in `.streamlit/config.toml` (run the app from the project folder). Old light version kept as `app_classic_backup.py`. New screenshots: `reports/figures/app_dark_*.png`.
- Next: validation on real 2025/26 listings; optional FastAPI endpoint for the platform (reuse `src/predict_price.py`).

## Original task (Notebook 1 plan)
Build **Notebook 1 (PP1)** with beginner-friendly simple English comments, delivered cell by cell: load + clean data, time-adjust to 2025-H1 level, compare 3 models with random split + area-held-out split (labelled as augmented data), then Part B LVI trend + backtest.

## Suggested repo name
`colombo-land-price-forecasting`