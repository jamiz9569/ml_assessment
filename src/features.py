"""
Data cleaning and feature engineering shared by training and inference.

Key design decision: pickup/delivery city NAMES are not used as categorical
features. The validation set contains 8 cities never seen in train/test
(Knoxville, Allentown, Laredo, Charlotte, Jackson, San Diego, Chicago,
Norfolk), so a model keyed on city identity cannot generalize to them.
Instead we use geographic (lat/lon) and route (distance) features, which
carry the same information in a form that generalizes to new lanes.
"""
import numpy as np
import pandas as pd

EQUIPMENT_LEVELS = ["Dry Van", "Reefer", "Flatbed"]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean(df: pd.DataFrame, fit_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values. `fit_stats` carries medians learned on the
    training data so the exact same imputation values are reused for
    validation / December inference (no leakage from val into train stats).
    """
    df = df.copy()
    stats = {} if fit_stats is None else dict(fit_stats)

    # weight: ~0.6% missing in train, ~1.4% in validation. Impute with the
    # per-equipment-type median learned on TRAIN (weight varies systematically
    # by equipment: reefers/flatbeds run lighter than dry van in this data).
    if "weight_median_by_equip" not in stats:
        stats["weight_median_by_equip"] = df.groupby("equipment")["weight"].median().to_dict()
    overall_weight_median = df["weight"].median() if "weight_overall_median" not in stats else stats["weight_overall_median"]
    stats.setdefault("weight_overall_median", overall_weight_median)
    fill_map = df["equipment"].map(stats["weight_median_by_equip"])
    df["weight"] = df["weight"].fillna(fill_map).fillna(stats["weight_overall_median"])
    df["weight_was_missing"] = df["weight"].isna().astype(int)  # (0 after fill; kept for symmetry)

    # market_index: ~0.8% missing in train, ~2% in validation. This is a
    # market-wide signal that moves with time, not with the individual load,
    # so the least-biased fill is the median for that same calendar date
    # (falls back to the global train median if a date has no observations).
    df["market_index_missing"] = df["market_index"].isna().astype(int)
    if "market_index_median_overall" not in stats:
        stats["market_index_median_overall"] = df["market_index"].median()
    if "market_index_by_date" not in stats:
        stats["market_index_by_date"] = df.groupby(df["date"].dt.date)["market_index"].median().to_dict()
    date_fill = df["date"].dt.date.map(stats["market_index_by_date"])
    df["market_index"] = df["market_index"].fillna(date_fill).fillna(stats["market_index_median_overall"])

    return df, stats


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Route / geography
    df["haversine_dist"] = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    df["circuity"] = df["distance"] / df["haversine_dist"].replace(0, np.nan)
    df["circuity"] = df["circuity"].fillna(df["circuity"].median())
    df["log_distance"] = np.log1p(df["distance"])
    df["weight_per_mile"] = df["weight"] / df["distance"].replace(0, np.nan)

    # Calendar / seasonality
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Equipment as an ordered category (fixed levels so train/val/Dec agree)
    df["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT_LEVELS)

    return df


FEATURE_COLS = [
    "distance", "log_distance", "haversine_dist", "circuity",
    "weight", "weight_per_mile",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "market_index", "market_index_missing", "quote_signal",
    "month", "day_of_week", "day_of_year", "is_weekend", "doy_sin", "doy_cos",
    "equipment",
]


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    X = pd.get_dummies(X, columns=["equipment"], prefix="equip", dtype=int)
    return X
