# Land Price Forecasting (Component 04)

AI-based land price forecasting for the Colombo District, part of the research project
"Compliant Blockchain-Based Real-World Asset Tokenization on Ethereum".

## Approach
- **Model A:** estimates a reference land price (Rs per perch) from land attributes. Ridge, Random Forest and Gradient Boosting are compared.
- **Model B:** forecasts the CBSL Land Valuation Indicator (LVI) with simple trend methods (Drift, Linear, Holt), checked by backtesting.
- **PP2:** combine both to produce future land prices.

## Data sources
- 6,000-record Colombo land dataset (`data/raw/`), calibrated to the CBSL Land Valuation Indicator.
- CBSL Land Valuation Indicator, half-yearly (`data/raw/cbsl_half_yearly_dataset.xls`).

## Folder structure
| Path | Purpose |
|---|---|
| `data/raw/` | Original files. Never edited. |
| `data/processed/` | Cleaned data and saved train/test splits |
| `notebooks/model_a_land_value/` | Part A notebooks 01-04, run in numeric order |
| `notebooks/model_b_trend/` | Part B (LVI trend) notebooks |
| `models/` | Saved models and preprocessing pipelines |
| `reports/figures/` | Charts for slides |
| `reports/tables/` | Result tables (CSV) |
| `src/` | Helper code (`load_cbsl.py` reads the CBSL files) |
| `results/` | Outputs of an earlier prototype run (old) |

## Notebook order (planned)
1. `01_data_understanding`
2. `02_data_preparation`
3. `03_price_model_training`
4. `04_price_model_evaluation`
5. `05_lvi_trend_models`

The train/test split is made once in notebook 02 and reused by all later notebooks.

## Setup
```
pip install -r requirements.txt
```
Then open the notebooks in VS Code and select the Python kernel.
