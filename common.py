"""Shared data loading and feature engineering for training (train.py) and the Streamlit app (app.py)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "Data"
MODEL_DIR = ROOT / "models"

# --- NASA POWER hourly data (wind speed forecasting) ---
SITES = {"Mersing": "Mersing.csv", "Kudat": "kudat.csv"}
NASA_HEADER_ROWS = 14  # header block ends at "-END HEADER-"
TARGET = "WS50M"  # wind speed at 50 m, closest to turbine hub height
HORIZONS = list(range(1, 25))  # hours ahead; one XGBoost model per horizon
VAL_START = pd.Timestamp("2023-01-01")
TEST_START = pd.Timestamp("2024-01-01")

# --- T1 turbine SCADA data (power estimation) ---
# Direction and hour-of-day were tried but hurt accuracy on held-out months (overfitting), so wind speed only.
T1_FEATURES = ["wind_speed"]
T1_VAL_START = pd.Timestamp("2018-09-01")
T1_TEST_START = pd.Timestamp("2018-11-01")


def load_nasa(site: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / SITES[site], skiprows=NASA_HEADER_ROWS)
    df.index = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY, hour=df.HR))
    df = df.drop(columns=["YEAR", "MO", "DY", "HR"]).sort_index()
    df = df.replace(-999, np.nan).asfreq("h")
    return df.interpolate(method="time", limit=3)


def _cyclic(name: str, values, period: float) -> dict:
    angle = 2 * np.pi * np.asarray(values) / period
    return {f"{name}_sin": np.sin(angle), f"{name}_cos": np.cos(angle)}


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for forecasting from time t using only information available at t."""
    f = pd.DataFrame(index=df.index)
    for c in ["WS10M", "WS50M", "T2M", "PS", "RH2M"]:
        f[c] = df[c]
    f["ws_ratio"] = df["WS50M"] / df["WS10M"].replace(0, np.nan)  # shear / stability proxy
    rad = np.deg2rad(df["WD10M"])
    f["wd_sin"], f["wd_cos"] = np.sin(rad), np.cos(rad)

    for lag in [1, 2, 3, 6, 12, 24, 48]:
        f[f"ws50_lag{lag}"] = df["WS50M"].shift(lag)
    for lag in [1, 3, 6, 24]:
        f[f"ws10_lag{lag}"] = df["WS10M"].shift(lag)
    for w in [3, 6, 24]:
        roll = df["WS50M"].rolling(w)
        f[f"ws50_mean{w}"] = roll.mean()
        f[f"ws50_std{w}"] = roll.std()
    f["ws50_diff1"] = df["WS50M"].diff(1)
    f["ps_diff3"] = df["PS"].diff(3)  # pressure tendency

    for name, values, period in [("hour", df.index.hour, 24), ("doy", df.index.dayofyear, 365.25)]:
        for k, v in _cyclic(name, values, period).items():
            f[k] = v
    return f


def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    return df[TARGET].shift(-horizon)


def load_t1() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "T1.csv", encoding="utf-8-sig")
    df.columns = ["time", "power_kw", "wind_speed", "theoretical_kw", "wind_dir"]
    df["time"] = pd.to_datetime(df["time"], format="%d %m %Y %H:%M")
    return df.set_index("time").sort_index()


def make_t1_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[T1_FEATURES]


def t1_row(wind_speed) -> pd.DataFrame:
    """Feature frame for a wind speed (scalar or array) for interactive what-if estimates."""
    return pd.DataFrame({"wind_speed": np.atleast_1d(wind_speed).astype(float)})[T1_FEATURES]
