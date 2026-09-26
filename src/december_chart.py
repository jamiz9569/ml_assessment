"""
Fill in data/december_chart_inputs.csv with predicted_rate for the trained
model, ready to hand to the provided score.py (which validates the file
and renders the official chart -- it does not do any prediction itself).

december_chart_inputs.csv has no market_index / quote_signal columns,
which the trained model needs, because the fixed lane's date is the only
thing that varies. This script fits a lightweight harmonic (day-of-week +
day-of-year) regression to the historical market_index/quote_signal series
to forecast plausible December values for those two columns, then scores
the fixed Lexington -> Fort Wayne / Dry Van / 32,000 lb lane for each
December date with the same trained rate model used for validation.csv.

Run: python3 src/december_chart.py
Then: python3 score.py --predictions outputs/validation_predictions.csv \
    --december-predictions outputs/december_chart_inputs_filled.csv \
    --output-dir outputs/scorer_results
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from features import load_raw, clean, engineer, build_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def harmonic_features(dates: pd.Series) -> pd.DataFrame:
    doy = dates.dt.dayofyear
    dow = dates.dt.dayofweek
    return pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    })


def main():
    warnings.filterwarnings("ignore")
    bundle = joblib.load(OUT / "model.joblib")
    model, stats, columns = bundle["model"], bundle["stats"], bundle["columns"]

    hist = load_raw(DATA / "train_test.csv")
    hist_c, _ = clean(hist, fit_stats=stats)
    daily = hist_c.groupby("date")[["market_index", "quote_signal"]].mean().reset_index()

    Xh = harmonic_features(daily["date"])
    mi_model = LinearRegression().fit(Xh, daily["market_index"])
    qs_model = LinearRegression().fit(Xh, daily["quote_signal"])

    # december_chart_inputs.csv has no lat/lon columns; look the fixed lane's
    # coordinates up from the historical city table (Lexington & Fort Wayne
    # both appear in train_test.csv).
    city_coords = pd.concat([
        hist[["pickup", "pickup_lat", "pickup_lon"]].rename(
            columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
        hist[["delivery", "delivery_lat", "delivery_lon"]].rename(
            columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}),
    ]).drop_duplicates("city").set_index("city")

    dec = pd.read_csv(DATA / "december_chart_inputs.csv")
    dec["date"] = pd.to_datetime(dec["date"])
    dec["pickup_lat"] = dec["pickup"].map(city_coords["lat"])
    dec["pickup_lon"] = dec["pickup"].map(city_coords["lon"])
    dec["delivery_lat"] = dec["delivery"].map(city_coords["lat"])
    dec["delivery_lon"] = dec["delivery"].map(city_coords["lon"])
    Xd = harmonic_features(dec["date"])
    dec["market_index"] = mi_model.predict(Xd)
    dec["quote_signal"] = qs_model.predict(Xd)
    dec["load_id"] = [f"DEC-{i+1:03d}" for i in range(len(dec))]

    dec_c, _ = clean(dec, fit_stats=stats)
    dec_f = engineer(dec_c)
    Xdec = build_matrix(dec_f)
    Xdec = Xdec.reindex(columns=columns, fill_value=0)

    dec["predicted_rate"] = np.round(np.expm1(model.predict(Xdec)), 2)
    dec[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]].to_csv(
        OUT / "december_chart_inputs_filled.csv", index=False
    )

    print(dec[["date", "predicted_rate"]].to_string(index=False))
    print(f"\nSaved filled CSV -> {OUT/'december_chart_inputs_filled.csv'}")
    print("Run score.py on this file to validate it and render the official chart.")


if __name__ == "__main__":
    main()
