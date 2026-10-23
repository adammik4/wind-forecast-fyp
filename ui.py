"""Cached loaders and shared visuals for the Streamlit pages in app_pages/."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c

ASSETS = c.ROOT / "assets"
BLUE, ORANGE, GREY, GREEN, RED, TEAL = "#2563EB", "#EA580C", "#94A3B8", "#15803D", "#DC2626", "#0E7490"
BAND = "rgba(234,88,12,0.16)"  # P10-P90 fill, ORANGE at low opacity
TEAL_SCALE = ["#ECFEFF", "#A5F3FC", "#22D3EE", "#0891B2", "#0E7490", "#164E63"]


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
def site_power(site: str, wind_scale: float = 1.0) -> pd.DataFrame:
    """Estimated hourly output if the T1 turbine stood at a Malaysian site, from its NASA POWER weather.
    wind_scale multiplies every hourly wind speed (hotspot scenario); 1 is the recorded wind."""
    df = c.load_nasa(site).dropna(subset=["WS50M", "WD10M", "T2M", "PS", "RH2M"])
    df = df.assign(WS50M=df.WS50M * wind_scale)
    X = c.site_power_inputs(df)
    q = predict_interval(X)
    out = pd.DataFrame({"wind_speed": X.wind_speed, "air_density": X.air_density, "power_kw": predict_power(X),
                        "p10": q[:, 0], "p90": q[:, 2]}, index=X.index)
    for col in ("power_kw", "p10", "p90"):
        out[col] = apply_cut_out(out.wind_speed, out[col])
    return out


@st.cache_data(show_spinner="Estimating wind output for all 16 states...")
def state_table(hotspot: bool = False) -> pd.DataFrame:
    """One row per state: wind and estimated output of one T1 turbine at its representative site (full years only).
    hotspot=True rescales the paper's states so their mean wind is the paper's hotspot speed, keeping the hourly pattern."""
    rows = []
    for state, s in c.STATES.items():
        scale = 1.0
        if hotspot and state in c.PAPER_STATES:
            ws = c.load_nasa(s["site"]).WS50M
            scale = c.HOTSPOT_WS / ws[ws.index.year < ws.index.max().year].mean()
        est = site_power(s["site"], scale)
        est = est[est.index.year < est.index.max().year]
        monthly = est.groupby(est.index.month).power_kw.mean()
        rows.append(dict(
            state=state, site=s["site"], lat=s["lat"], lon=s["lon"], grid=s["grid"], households=s["households"],
            territory=s.get("territory", False), paper=state in c.PAPER_STATES, study=s["site"] in c.SITES,
            wind_speed=est.wind_speed.mean(), above_cut_in=(est.wind_speed >= c.CUT_IN).mean(),
            mean_kw=est.power_kw.mean(), p10_kw=est.p10.mean(), p90_kw=est.p90.mean(),
            monthly_kw=monthly.reindex(range(1, 13)).tolist(),
        ))
    df = pd.DataFrame(rows)
    df["capacity_factor"] = df.mean_kw / c.RATED_KW
    df["emission_factor"] = df.grid.map(c.GRID_EMISSION_FACTOR)
    return df


def contribution(table: pd.DataFrame, turbines: int) -> pd.DataFrame:
    """Annual value of a wind farm of `turbines` T1 machines in each state (before wake and availability losses)."""
    out = table.copy()
    hours = 8766  # average year
    for col, src in (("energy_gwh", "mean_kw"), ("energy_gwh_p10", "p10_kw"), ("energy_gwh_p90", "p90_kw")):
        out[col] = out[src] * turbines * hours / 1e6
    out["homes"] = out.energy_gwh * 1e6 / c.KWH_PER_HOUSEHOLD
    out["home_share"] = out.homes / out.households
    out["co2_kt"] = out.energy_gwh * out.emission_factor  # GWh x t/MWh = kt
    return out


@st.cache_data
def states_geojson() -> dict:
    return json.loads(c.STATES_GEOJSON.read_text())


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
    needed += [f"{c.slug(site)}/h24.json" for site in c.SITES]
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
    """Animated turbine and wind-direction compass as one standalone SVG, for st.image (st.html strips SVG).
    Blade rotation speed follows the rotor rpm (shown 3x faster so it reads on screen)."""
    period = 60 / (3 * rpm) if rpm > 0 else 0
    spin = f"animation: rotor-spin {period:.2f}s linear infinite;" if period else ""
    blade = "M0,-4 C6,-30 5,-80 0,-100 C-3,-80 -4,-30 0,-4 Z"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="-120 -120 345 360" width="345" height="360">
  <style>
    @keyframes rotor-spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    .rotor {{ transform-origin: 0px 0px; {spin} }}
  </style>
  <polygon points="-5,0 5,0 9,235 -9,235" fill="#94A3B8"/>
  <rect x="-16" y="-9" width="40" height="16" rx="6" fill="#64748B"/>
  <g class="rotor">
    <path d="{blade}" fill="#CBD5E1" stroke="#64748B" stroke-width="1"/>
    <path d="{blade}" fill="#CBD5E1" stroke="#64748B" stroke-width="1" transform="rotate(120)"/>
    <path d="{blade}" fill="#CBD5E1" stroke="#64748B" stroke-width="1" transform="rotate(240)"/>
  </g>
  <circle cx="0" cy="0" r="8" fill="{colour}"/>
  <g transform="translate(170,185)">
    <circle r="44" fill="none" stroke="#94A3B8" stroke-width="2"/>
    <text y="-28" text-anchor="middle" font-family="Inter, sans-serif" font-size="12" fill="#94A3B8">N</text>
    <g transform="rotate({wind_dir:.0f})">
      <line x1="0" y1="-38" x2="0" y2="18" stroke="{colour}" stroke-width="4"/>
      <polygon points="0,28 -9,12 9,12" fill="{colour}"/>
    </g>
  </g>
</svg>"""


# ---------------------------------------------------------------- 10-15 % accuracy target
def target_band(fig):
    """Shade the 10-15 % target band (y axis in percent)."""
    lo, hi = c.ERROR_STRETCH * 100, c.ERROR_TARGET * 100
    fig.add_hrect(y0=0, y1=lo, fillcolor=GREEN, opacity=0.08, line_width=0)
    fig.add_hrect(y0=lo, y1=hi, fillcolor=GREEN, opacity=0.16, line_width=0,
                  annotation_text=f"target {lo:.0f}-{hi:.0f}%", annotation_position="top right")
    return fig


def target_chart(ms: pd.DataFrame, height=300):
    """One site's normalised outlook errors by horizon against the 10-15 % target."""
    fig = go.Figure()
    fig.add_scatter(x=ms.horizon, y=ms.wind_nmae * 100, name="Wind speed nMAE (% of mean wind)",
                    line=dict(color=ORANGE, width=3))
    fig.add_scatter(x=ms.horizon, y=ms.power_nrmse * 100, name="Power nRMSE (% of rated)", line=dict(color=TEAL, width=3))
    fig.update_xaxes(title="Forecast horizon (hours ahead)")
    return style(target_band(fig), "Normalised error (%)", height)


def within(errors: pd.Series, limit: float) -> int:
    """Longest horizon, counted from +1 h, for which every error up to it is within the limit."""
    return int((errors.to_numpy() <= limit).cumprod().sum())


def target_summary(ms: pd.DataFrame) -> str:
    h15, h10 = within(ms.wind_nmae, c.ERROR_TARGET), within(ms.wind_nmae, c.ERROR_STRETCH)
    return (f"Wind-speed error is within {c.ERROR_TARGET:.0%} up to +{h15} h and within {c.ERROR_STRETCH:.0%} up to +{h10} h "
            f"(+24 h: {ms.wind_nmae.iloc[-1]:.0%}). As a share of rated power, the power error peaks at "
            f"{ms.power_nrmse.max():.1%}, inside the target at every horizon.")
