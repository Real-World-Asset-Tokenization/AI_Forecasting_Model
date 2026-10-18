"""Helper to read the CBSL Land Valuation Indicator (LVI) raw file.

The raw .xls file is really an HTML table, so we read it with BeautifulSoup.
"""
import pandas as pd
from bs4 import BeautifulSoup


def load_lvi_half_yearly(path):
    """Return a tidy table: one row per half-year, one column per LVI type.

    Columns: period, Overall, Residential, Commercial, Industrial.
    2025-H2 is empty in the raw file, so its values are NaN.
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    table = soup.find("table", id="ContentPlaceHolder1_grdResult")
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(strip=True).replace("\xa0", "") for c in tr.find_all(["th", "td"])]
        rows.append(cells)

    header = rows[0]
    periods = header[4:]  # first 4 columns are id, name, unit, scale
    out = {"period": periods}
    for row in rows[1:]:
        name = row[1]
        if "LVI" not in name:
            continue
        label = name.split(" - ")[-1]  # Overall, Residential, ...
        out[label] = pd.to_numeric(pd.Series(row[4:]).replace("", None), errors="coerce").values
    return pd.DataFrame(out)
