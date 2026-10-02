"""Cached loaders and shared visuals for the Streamlit pages in app_pages/."""
import json

import numpy as np
import pandas as pd
import streamlit as st

import common as c

BLUE, ORANGE, GREY, GREEN, RED = "#2a6fdb", "#e8710a", "#8a8f98", "#1e8e3e", "#d93025"


# ---------------------------------------------------------------- data and models
@st.cache_resource
def booster(path: str):
    return c.load_booster(path)


@st.cache_data
def results() -> dict:
    return json.loads((c.MODEL_DIR / "power_results.json").read_text())


@st.cache_data
def tuning_trials() -> pd.DataFrame:
    return pd.read_csv(c.MODEL_DIR / "tuning_trials.csv")


@st.cache_data
def outlook_metrics() -> pd.DataFrame:
    return pd.read_csv(c.MODEL_DIR / "metrics.csv")


@st.cache_data
def t1_data() -> pd.DataFrame:
    return c.build_t1_dataset()[0]


@st.cache_data
def nasa(site: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = c.load_nasa(site)
    return df, c.make_features(df)


def predict_power(X: pd.DataFrame) -> np.ndarray:
    """Tuned XGBoost point prediction (kW)."""
    pred = c.predict(booster(str(c.MODEL_DIR / "power_xgb.json")), X[results()["features"]])
    return np.clip(pred, 0, c.RATED_KW)


def predict_interval(X: pd.DataFrame) -> np.ndarray:
    """P10 / P50 / P90 (kW) from the quantile model, sorted so the bands never cross."""
    q = c.predict(booster(str(c.MODEL_DIR / "power_quantile.json")), X[results()["features"]])
    return np.sort(np.clip(q, 0, c.RATED_KW), axis=1)


def apply_cut_out(wind_speed, power):
    """Turbine control logic, not learned: the rotor parks at or above the cut-out speed."""
    return np.where(np.asarray(wind_speed) >= c.CUT_OUT, 0.0, power)


@st.cache_data
def site_power(site: str) -> pd.DataFrame:
    """Estimated hourly output if the T1 turbine stood at a Malaysian site, from its NASA POWER weather."""
    df = c.load_nasa(site).dropna(subset=["WS50M", "WD10M", "T2M", "PS", "RH2M"])
    X = c.site_power_inputs(df)
    q = predict_interval(X)
    out = pd.DataFrame({"wind_speed": X.wind_speed, "air_density": X.air_density, "power_kw": predict_power(X),
                        "p10": q[:, 0], "p90": q[:, 2]}, index=X.index)
    for col in ("power_kw", "p10", "p90"):
        out[col] = apply_cut_out(out.wind_speed, out[col])
    return out


@st.cache_data
def typical_conditions() -> dict[str, dict]:
    """Median weather per site, used as dashboard presets."""
    def circ_mean(deg):
        rad = np.deg2rad(deg.dropna())
        return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)

    out = {}
    for site in c.SITES:
        df = c.load_nasa(site)
        out[site] = dict(ws=df.WS50M.median(), wd=circ_mean(df.WD10M), temp=df.T2M.median(), pres=df.PS.median(), rh=df.RH2M.median())
    t1 = t1_data()
    w = c.load_t1_weather()
    out["Yalova (training site)"] = dict(ws=t1.wind_speed.median(), wd=circ_mean(t1.wind_dir), temp=w.temperature_c.median(),
                                         pres=w.pressure_kpa.median(), rh=w.humidity_pct.median())
    return {k: {f: round(float(v), 1) for f, v in d.items()} for k, d in out.items()}


def require_models():
    needed = ["power_xgb.json", "power_quantile.json", "power_results.json", "tuning_trials.csv", "metrics.csv"]
    missing = [f for f in needed if not (c.MODEL_DIR / f).exists()]
    if missing:
        st.error(f"Missing trained models ({', '.join(missing)}). Run `python train.py` in the project folder first.")
        st.stop()


# ---------------------------------------------------------------- visuals
def style(fig, ytitle, height=380):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=ytitle,
        legend=dict(orientation="h", y=1.12), hovermode="x unified",
    )
    return fig


def rotor_rpm(wind_speed: float, running: bool) -> float:
    """Variable-speed rotor: about 5 rpm at cut-in, rising to about 14 rpm at rated wind speed."""
    if not running:
        return 0.0
    frac = (np.clip(wind_speed, c.CUT_IN, c.RATED_SPEED) - c.CUT_IN) / (c.RATED_SPEED - c.CUT_IN)
    return float(5 + 9 * frac)


def turbine_svg(rpm: float, wind_dir: float, colour: str) -> str:
    """Animated turbine. Blade rotation speed follows the rotor rpm (shown 3x faster so it reads on screen)."""
    period = 60 / (3 * rpm) if rpm > 0 else 0
    spin = f"animation: rotor-spin {period:.2f}s linear infinite;" if period else ""
    blade = "M0,-4 C6,-30 5,-80 0,-100 C-3,-80 -4,-30 0,-4 Z"
    return f"""
<style>
  @keyframes rotor-spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
  .rotor {{ transform-box: fill-box; transform-origin: center; {spin} }}
</style>
<div style="display:flex;justify-content:center;gap:24px;align-items:flex-end">
<svg viewBox="-120 -120 240 360" width="220" height="330" role="img" aria-label="Wind turbine, rotor at {rpm:.0f} rpm">
  <polygon points="-5,0 5,0 9,235 -9,235" fill="#9aa3ad"/>
  <rect x="-16" y="-9" width="40" height="16" rx="6" fill="#7d8792"/>
  <g class="rotor">
    <circle cx="0" cy="0" r="104" fill="none"/>
    <path d="{blade}" fill="#c3c9d0" stroke="#7d8792" stroke-width="1"/>
    <path d="{blade}" fill="#c3c9d0" stroke="#7d8792" stroke-width="1" transform="rotate(120)"/>
    <path d="{blade}" fill="#c3c9d0" stroke="#7d8792" stroke-width="1" transform="rotate(240)"/>
  </g>
  <circle cx="0" cy="0" r="8" fill="{colour}"/>
</svg>
<svg viewBox="-50 -50 100 100" width="90" height="90" role="img" aria-label="Wind from {wind_dir:.0f} degrees">
  <circle r="44" fill="none" stroke="#9aa3ad" stroke-width="2"/>
  <text y="-30" text-anchor="middle" font-size="12" fill="#8a8f98">N</text>
  <g transform="rotate({wind_dir:.0f})">
    <line x1="0" y1="-38" x2="0" y2="18" stroke="{colour}" stroke-width="4"/>
    <polygon points="0,28 -9,12 9,12" fill="{colour}"/>
  </g>
</svg>
</div>"""
