"""Shared data loading and feature engineering for training (train.py) and the Streamlit app (app.py).

Follows the CRISP-DM data phases in the interim report (Sections 3.2.2-3.2.3):
  acquire  -> T1 turbine SCADA + NASA POWER (LaRC) weather
  merge    -> weather joined onto SCADA timestamps
  clean    -> sensor faults, out-of-range values, downtime / curtailment removed
  features -> cyclical wind-direction encoding, air density from temperature, pressure and humidity
  split    -> chronological train / validation / test (no data leakage)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "Data"
MODEL_DIR = ROOT / "models"

# ---------------------------------------------------------------- turbine (T1 SCADA, Yalova wind farm, Turkey)
# Kaggle "Wind Turbine Scada Dataset" (2018, 10-min). Location 40.655 N, 29.277 E.
T1_WEATHER_FILE = "Yalova_T1_weather.csv"  # NASA POWER hourly T2M/PS/RH2M for the T1 location, UTC
RATED_KW = 3600.0
CUT_IN = 3.0  # m/s, manufacturer curve first exceeds zero here
RATED_SPEED = 13.0  # m/s, manufacturer curve reaches rated power here
CUT_OUT = 25.0  # m/s

T1_VAL_START = pd.Timestamp("2018-09-01")
T1_TEST_START = pd.Timestamp("2018-11-01")

# Weather inputs named in the report (Section 2.2) plus the two derived features (Section 3.2.3).
POWER_FEATURES = [
    "wind_speed", "wind_dir_sin", "wind_dir_cos", "temperature_c", "pressure_kpa", "humidity_pct", "air_density",
]
# Feature-set ablation for Objective 1: which weather variables actually move the accuracy.
FEATURE_SETS = {
    "Wind speed only": ["wind_speed"],
    "+ wind direction": ["wind_speed", "wind_dir_sin", "wind_dir_cos"],
    "+ air density": ["wind_speed", "wind_dir_sin", "wind_dir_cos", "air_density"],
    "+ temperature, pressure, humidity": POWER_FEATURES[:6],
    "All (report feature set)": POWER_FEATURES,
}
FEATURE_LABELS = {
    "wind_speed": "Wind speed", "wind_dir_sin": "Wind direction (sin)", "wind_dir_cos": "Wind direction (cos)",
    "temperature_c": "Temperature", "pressure_kpa": "Pressure", "humidity_pct": "Humidity", "air_density": "Air density",
}

# ---------------------------------------------------------------- Malaysian study sites (NASA POWER hourly)
SITES = {"Mersing": "Mersing.csv", "Kudat": "kudat.csv"}
TARGET = "WS50M"  # wind speed at 50 m, closest to turbine hub height
HORIZONS = list(range(1, 25))  # hours ahead; one XGBoost model per horizon
VAL_START = pd.Timestamp("2023-01-01")
TEST_START = pd.Timestamp("2024-01-01")


# ---------------------------------------------------------------- models
def load_booster(path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def predict(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Predict with the early-stopping best iteration (native Booster.predict would use every tree)."""
    best = booster.attr("best_iteration")
    return booster.predict(xgb.DMatrix(X), iteration_range=(0, int(best) + 1) if best is not None else (0, 0))


# ---------------------------------------------------------------- loading
def read_nasa_csv(path: Path) -> pd.DataFrame:
    """NASA POWER CSV export: skip the -BEGIN HEADER- ... -END HEADER- block, index by timestamp."""
    with open(path, encoding="utf-8") as fh:
        header_rows = next(i for i, line in enumerate(fh) if line.startswith("-END HEADER-")) + 1
    df = pd.read_csv(path, skiprows=header_rows)
    df.index = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY, hour=df.HR))
    df = df.drop(columns=["YEAR", "MO", "DY", "HR"]).sort_index()
    return df.replace(-999, np.nan).asfreq("h")


def load_nasa(site: str) -> pd.DataFrame:
    return read_nasa_csv(DATA_DIR / SITES[site]).interpolate(method="time", limit=3)


def load_t1() -> pd.DataFrame:
    """Raw SCADA. Timestamps are treated as UTC: hourly SCADA wind speed correlates best with the
    NASA POWER wind at zero offset (r = 0.70 vs 0.66 at UTC+3)."""
    df = pd.read_csv(DATA_DIR / "T1.csv", encoding="utf-8-sig")
    df.columns = ["time", "power_kw", "wind_speed", "theoretical_kw", "wind_dir"]
    df["time"] = pd.to_datetime(df["time"], format="%d %m %Y %H:%M")
    return df.set_index("time").sort_index()


# ---------------------------------------------------------------- feature engineering
def air_density(temperature_c, pressure_kpa, humidity_pct):
    """Moist-air density (kg/m^3): dry-air + water-vapour partial pressures, Tetens saturation pressure."""
    t = np.asarray(temperature_c, dtype=float)
    p = np.asarray(pressure_kpa, dtype=float) * 1000.0
    e_sat = 610.78 * 10 ** (7.5 * t / (t + 237.3))
    p_v = np.clip(np.asarray(humidity_pct, dtype=float), 0, 100) / 100.0 * e_sat
    t_k = t + 273.15
    return (p - p_v) / (287.058 * t_k) + p_v / (461.495 * t_k)


def power_features(wind_speed, wind_dir, temperature_c, pressure_kpa, humidity_pct) -> pd.DataFrame:
    """Model input frame. Scalars or equal-length arrays; scalars are broadcast."""
    cols = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype=float))
                                 for v in (wind_speed, wind_dir, temperature_c, pressure_kpa, humidity_pct)))
    ws, wd, t, p, rh = cols
    rad = np.deg2rad(wd)  # sin/cos so that 0 and 360 degrees are neighbours
    return pd.DataFrame({
        "wind_speed": ws, "wind_dir_sin": np.sin(rad), "wind_dir_cos": np.cos(rad),
        "temperature_c": t, "pressure_kpa": p, "humidity_pct": rh, "air_density": air_density(t, p, rh),
    })[POWER_FEATURES]


def clean_t1(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove records that are not weather-driven: sensor faults, out-of-range values, downtime, derating."""
    n = len(df)
    df = df.copy()
    neg = df.power_kw < 0
    df.loc[neg, "power_kw"] = 0.0  # standby consumption, clip to zero
    downtime = (df.wind_speed >= CUT_IN + 0.5) & (df.power_kw <= 0)
    # Binned power-curve filter: per 0.5 m/s wind-speed bin, IQR fences fitted on the training period only.
    bins = (df.wind_speed / 0.5).round()
    ref = df[(df.index < T1_VAL_START) & ~downtime].groupby(bins).power_kw.quantile([0.25, 0.75]).unstack()
    iqr = ref[0.75] - ref[0.25]
    lo, hi = bins.map(ref[0.25] - 1.5 * iqr), bins.map(ref[0.75] + 1.5 * iqr)
    rules = {
        "Out-of-range wind speed (<0 or >40 m/s) or direction": ~df.wind_speed.between(0, 40) | ~df.wind_dir.between(0, 360),
        "Downtime: wind above cut-in but zero output": downtime,
        "Derating, icing or curtailment: outside 1.5 IQR of its wind-speed bin":
            (df.wind_speed >= CUT_IN) & ((df.power_kw < lo) | (df.power_kw > hi)),
    }
    drop = pd.Series(False, index=df.index)
    report = {"rows_raw": n, "Negative power clipped to 0": int(neg.sum())}
    for name, mask in rules.items():
        report[name] = int((mask & ~drop).sum())
        drop |= mask
    df = df[~drop]
    report["rows_clean"] = len(df)
    return df, report


def load_t1_weather() -> pd.DataFrame:
    w = read_nasa_csv(DATA_DIR / T1_WEATHER_FILE).interpolate(method="time", limit=3)
    return w.rename(columns={"T2M": "temperature_c", "PS": "pressure_kpa", "RH2M": "humidity_pct"})


def build_t1_dataset() -> tuple[pd.DataFrame, dict]:
    """Clean SCADA merged with hourly weather (linearly interpolated onto the 10-min timestamps), plus features."""
    scada, report = clean_t1(load_t1())
    w = load_t1_weather()[["temperature_c", "pressure_kpa", "humidity_pct"]]
    w = w.reindex(w.index.union(scada.index)).interpolate(method="time").loc[scada.index]
    df = scada.join(w)
    feats = power_features(df.wind_speed, df.wind_dir, df.temperature_c, df.pressure_kpa, df.humidity_pct)
    feats.index = df.index
    df = df.drop(columns=["temperature_c", "pressure_kpa", "humidity_pct"]).join(feats.drop(columns="wind_speed"))
    df = df.dropna(subset=POWER_FEATURES)
    report["rows_merged"] = len(df)
    return df, report


def split_t1(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "train": df.index < T1_VAL_START,
        "val": (df.index >= T1_VAL_START) & (df.index < T1_TEST_START),
        "test": df.index >= T1_TEST_START,
    }


def site_power_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Power-model inputs from a Malaysian site's NASA POWER record (50 m wind speed, 10 m direction)."""
    f = power_features(df.WS50M, df.WD10M, df.T2M, df.PS, df.RH2M)
    f.index = df.index
    return f


def turbine_status(wind_speed: float, power_kw: float) -> tuple[str, str]:
    """(label, colour) operating state for the dashboard status indicator."""
    if wind_speed >= CUT_OUT:
        return "Cut-out: parked for protection", "red"
    if wind_speed < CUT_IN:
        return "Idle: below cut-in speed", "gray"
    if power_kw >= 0.95 * RATED_KW:
        return "Rated output", "green"
    return "Generating: partial load", "blue"


# ---------------------------------------------------------------- 24 h wind-speed outlook features (Malaysian sites)
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
