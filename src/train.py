"""
Train and evaluate rate-prediction models.

Validation strategy
--------------------
The task is to predict rates for November/December 2025, i.e. dates the
model has never seen, using training data from Jan-Oct 2025. That is a
forecasting problem, not an interpolation problem. A random row-wise
train/val split would let the model see market_index/quote_signal levels
from days immediately adjacent (even the same week) as the held-out rows,
which overstates how well it will generalize to genuinely future,
unseen months.

So the internal validation split here is TIME-BASED: the last ~20% of
calendar dates (roughly September-October) are held out, and the model is
trained only on the earlier ~80% (roughly January-August). This mimics the
real deployment gap between train_test.csv (through Oct 31) and
validation.csv (Nov 1 - Dec 31).

Run: python3 src/train.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from features import load_raw, clean, engineer, build_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"  {label:22s}  MAE=${mae:8.2f}   RMSE=${rmse:8.2f}   MAPE={mape:5.2f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def main():
    warnings.filterwarnings("ignore")
    raw = load_raw(DATA / "train_test.csv")
    raw = raw.sort_values("date").reset_index(drop=True)

    # --- time-based split: last ~20% of dates held out as internal validation
    cutoff = raw["date"].quantile(0.80, interpolation="nearest")
    train_raw = raw[raw["date"] <= cutoff].copy()
    val_raw = raw[raw["date"] > cutoff].copy()
    print(f"Time-based split: train {train_raw.date.min().date()}..{train_raw.date.max().date()} "
          f"({len(train_raw)} rows), holdout {val_raw.date.min().date()}..{val_raw.date.max().date()} "
          f"({len(val_raw)} rows)")

    train_c, stats = clean(train_raw)
    val_c, _ = clean(val_raw, fit_stats=stats)

    train_f = engineer(train_c)
    val_f = engineer(val_c)

    X_train = build_matrix(train_f)
    X_val = build_matrix(val_f)
    # align columns (equipment dummies must match)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    y_train = np.log1p(train_f["posted_rate"])
    y_val = val_f["posted_rate"]

    print("\nModel comparison on time-based holdout (Sep-Oct):")
    results = {}

    # 1. Ridge baseline
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    pred = np.expm1(ridge.predict(X_val))
    results["ridge"] = metrics(y_val, pred, "Ridge (log-target)")

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    pred = np.expm1(rf.predict(X_val))
    results["random_forest"] = metrics(y_val, pred, "Random Forest (log-target)")

    # 3. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.0, random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    pred = np.expm1(xgb_model.predict(X_val))
    results["xgboost"] = metrics(y_val, pred, "XGBoost (log-target)")

    # 5-fold CV on train portion only, for stability check (not model selection)
    cv_mae = -cross_val_score(xgb_model, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42),
                               scoring="neg_mean_absolute_error")
    print(f"\n  XGBoost 5-fold CV MAE (log-target, train portion): {cv_mae.mean():.4f} +/- {cv_mae.std():.4f}")

    best_name = min(results, key=lambda k: results[k]["mae"])
    print(f"\nBest model by holdout MAE: {best_name}")

    # feature importance (best model = xgboost expected)
    importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 10 feature importances (XGBoost):")
    print(importances.head(10).to_string())

    # --- Refit chosen model (XGBoost) on ALL available labeled data (train_test.csv)
    # for the final artifact used to predict on validation.csv / December.
    full_c, full_stats = clean(raw)
    full_f = engineer(full_c)
    X_full = build_matrix(full_f)
    y_full = np.log1p(full_f["posted_rate"])

    final_model = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=5,
        reg_lambda=1.0, random_state=42, n_jobs=-1,
    )
    final_model.fit(X_full, y_full)

    joblib.dump({"model": final_model, "stats": full_stats, "columns": list(X_full.columns)},
                OUT / "model.joblib")

    with open(OUT / "metrics.json", "w") as f:
        json.dump({"holdout_results": results, "cv_mae_log": {"mean": cv_mae.mean(), "std": cv_mae.std()},
                    "best_model": best_name,
                    "split": {"train_start": str(train_raw.date.min().date()),
                              "train_end": str(train_raw.date.max().date()),
                              "holdout_start": str(val_raw.date.min().date()),
                              "holdout_end": str(val_raw.date.max().date())}}, f, indent=2)

    print(f"\nSaved final model -> {OUT/'model.joblib'}")
    print(f"Saved metrics -> {OUT/'metrics.json'}")


if __name__ == "__main__":
    main()
