"""
Run the trained model on data/validation.csv and fill
data/validation_predictions_template.csv -> outputs/validation_predictions.csv

Run: python3 src/predict.py
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import load_raw, clean, engineer, build_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"


def main():
    warnings.filterwarnings("ignore")
    bundle = joblib.load(OUT / "model.joblib")
    model, stats, columns = bundle["model"], bundle["stats"], bundle["columns"]

    val_raw = load_raw(DATA / "validation.csv")
    val_c, _ = clean(val_raw, fit_stats=stats)
    val_f = engineer(val_c)
    X_val = build_matrix(val_f)
    X_val = X_val.reindex(columns=columns, fill_value=0)

    pred = np.expm1(model.predict(X_val))
    val_f["predicted_rate"] = np.round(pred, 2)

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    out = template[["load_id"]].merge(val_f[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert out["predicted_rate"].isna().sum() == 0, "missing predictions for some load_id"
    assert len(out) == len(template) == 12000

    out.to_csv(OUT / "validation_predictions.csv", index=False)
    print(f"Wrote {len(out)} predictions -> {OUT/'validation_predictions.csv'}")
    print(out["predicted_rate"].describe())


if __name__ == "__main__":
    main()
