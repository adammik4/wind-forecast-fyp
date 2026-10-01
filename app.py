"""Streamlit app: XGBoost wind speed forecasts (Mersing, Kudat) and turbine power estimates (T1).

    streamlit run app.py      (run `python train.py` first)
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor

import common as c

st.set_page_config(page_title="Wind Forecast", layout="wide")

BLUE, ORANGE, GREY = "#2a6fdb", "#e8710a", "#8a8f98"


@st.cache_data
def nasa(site: str):
    df = c.load_nasa(site)
    return df, c.make_features(df)


@st.cache_data
def t1():
    return c.load_t1()


@st.cache_resource
def load_model(path: str) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(path)
    return model


@st.cache_data
def metrics() -> pd.DataFrame:
    return pd.read_csv(c.MODEL_DIR / "metrics.csv")


def predict_horizons(site: str, X: pd.DataFrame) -> pd.DataFrame:
    """Rows = origin times in X, columns = horizon (hours)."""
    return pd.DataFrame(
        {h: load_model(str(c.MODEL_DIR / site / f"h{h:02d}.json")).predict(X) for h in c.HORIZONS}, index=X.index
    )


def style(fig, ytitle, height=380):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=ytitle,
        legend=dict(orientation="h", y=1.1), hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------- tabs
def forecast_tab(site: str):
    df, X = nasa(site)
    test_index = df.index[df.index >= c.TEST_START]
    first, last = test_index[0], test_index[-1]
    label = "Forecast origin (test period, which the model never saw in training)"
    picker = st.segmented_control("Date picker", ["Slider", "Calendar + hour"], default="Slider", required=True)
    if picker == "Slider":
        origin = st.slider(
            label, min_value=first.to_pydatetime(), max_value=last.to_pydatetime(),
            value=last.to_pydatetime(), step=pd.Timedelta(hours=1).to_pytimedelta(), format="YYYY-MM-DD HH:mm",
        )
    else:
        d_col, h_col = st.columns([3, 1])
        day = d_col.date_input(label, value=last.date(), min_value=first.date(), max_value=last.date())
        hour = h_col.selectbox("Hour", range(24), index=last.hour, format_func=lambda h: f"{h:02d}:00")
        origin = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    origin = pd.Timestamp(origin)
    row = X.loc[[origin]]
    if row.isna().all(axis=None):
        st.warning("No data at this time.")
        return

    fc = predict_horizons(site, row).iloc[0]
    fc.index = origin + pd.to_timedelta(fc.index, unit="h")
    fc = pd.concat([pd.Series({origin: df.at[origin, c.TARGET]}), fc])  # join forecast to last observation

    past = df.loc[origin - pd.Timedelta(hours=48): origin, c.TARGET]
    future = df.loc[origin: origin + pd.Timedelta(hours=24), c.TARGET]
    persistence = pd.Series(df.at[origin, c.TARGET], index=fc.index)

    fig = go.Figure()
    fig.add_scatter(x=past.index, y=past, name="Observed", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=future.index[1:], y=future.iloc[1:], name="Actual (after origin)", line=dict(color=BLUE, dash="dot"))
    fig.add_scatter(x=persistence.index, y=persistence, name="Persistence baseline", line=dict(color=GREY, dash="dash"))
    fig.add_scatter(x=fc.index, y=fc, name="XGBoost forecast", line=dict(color=ORANGE, width=3))
    fig.add_vline(x=origin, line_color=GREY, line_width=1)
    st.plotly_chart(style(fig, "Wind speed at 50 m (m/s)"), width="stretch")

    actual = future.iloc[1:]
    if len(actual):
        aligned = fc.iloc[1:].loc[actual.index]
        c1, c2, c3 = st.columns(3)
        c1.metric("XGBoost MAE over this window", f"{(aligned - actual).abs().mean():.2f} m/s")
        c2.metric("Persistence MAE", f"{(persistence.iloc[1:].loc[actual.index] - actual).abs().mean():.2f} m/s")
        c3.metric("Hours compared", len(actual))
    else:
        st.info("This is the last observation in the dataset, so there are no actual values to compare against yet.")

    show = fc.iloc[1:].rename("Forecast WS50M (m/s)").round(2).to_frame()
    show.index.name = "Time"
    with st.expander("Forecast table"):
        st.dataframe(show)


def performance_tab(site: str):
    m = metrics()
    ms = m[m.site == site]
    c1, c2, c3 = st.columns(3)
    for col, h in zip((c1, c2, c3), (1, 6, 24)):
        r = ms[ms.horizon == h].iloc[0]
        col.metric(f"+{h}h RMSE", f"{r.xgb_rmse:.2f} m/s", f"{r.skill:+.0%} vs persistence", delta_color="normal")

    fig = go.Figure()
    fig.add_scatter(x=ms.horizon, y=ms.xgb_rmse, name="XGBoost", line=dict(color=ORANGE, width=3))
    fig.add_scatter(x=ms.horizon, y=ms.persistence_rmse, name="Persistence", line=dict(color=GREY, dash="dash"))
    fig.update_xaxes(title="Forecast horizon (hours ahead)")
    st.plotly_chart(style(fig, "Test RMSE (m/s)", 320), width="stretch")

    h = st.select_slider("Horizon for detail plots", options=c.HORIZONS, value=6, format_func=lambda v: f"+{v}h")
    df, X = nasa(site)
    y = c.make_target(df, h)
    t_target = X.index + pd.Timedelta(hours=h)
    mask = y.notna() & (t_target >= c.TEST_START) & X.iloc[:, 0].notna()
    model = load_model(str(c.MODEL_DIR / site / f"h{h:02d}.json"))
    pred = pd.Series(model.predict(X[mask]), index=X.index[mask])

    left, right = st.columns(2)
    sample = pred.sample(min(3000, len(pred)), random_state=0)
    fig = go.Figure(go.Scattergl(x=y[sample.index], y=sample, mode="markers", marker=dict(size=4, color=BLUE, opacity=0.35)))
    top = float(max(y[mask].max(), pred.max()))
    fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=GREY, dash="dash"), showlegend=False)
    fig.update_xaxes(title="Actual (m/s)")
    fig.update_layout(title=f"Predicted vs actual, +{h}h", showlegend=False)
    left.plotly_chart(style(fig, "Predicted (m/s)", 360), width="stretch")

    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).nlargest(12).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE))
    fig.update_layout(title=f"Top features (gain), +{h}h")
    right.plotly_chart(style(fig, "", 360), width="stretch")


def power_tab():
    path = c.MODEL_DIR / "t1_power.json"
    df = t1()
    model = load_model(str(path))
    res = json.loads((c.MODEL_DIR / "t1_metrics.json").read_text())

    left, right = st.columns([1, 2])
    with left:
        ws = st.slider("Wind speed (m/s)", 0.0, 25.0, 8.0, 0.1)
        est = float(model.predict(c.t1_row(ws))[0])
        st.metric("Estimated active power", f"{max(est, 0):,.0f} kW")
        st.caption(
            f"Held-out RMSE (Nov-Dec 2018): {res['xgb']['rmse']:.0f} kW for XGBoost vs "
            f"{res['theoretical_curve']['rmse']:.0f} kW for the manufacturer's theoretical curve."
        )

    speeds = np.arange(0, 25.01, 0.25)
    grid = c.t1_row(speeds)
    theo = df.assign(bin=(df.wind_speed / 0.5).round() * 0.5).groupby("bin").theoretical_kw.mean()
    sample = df.sample(3000, random_state=0)

    fig = go.Figure()
    fig.add_scattergl(x=sample.wind_speed, y=sample.power_kw, mode="markers", name="Measured (sample)",
                      marker=dict(size=3, color=GREY, opacity=0.35))
    fig.add_scatter(x=theo.index, y=theo, name="Theoretical curve", line=dict(color=BLUE, dash="dash"))
    fig.add_scatter(x=speeds, y=np.clip(model.predict(grid), 0, None), name="XGBoost", line=dict(color=ORANGE, width=3))
    fig.add_scatter(x=[ws], y=[max(est, 0)], name="Your input", mode="markers", marker=dict(size=12, color="black"))
    fig.update_xaxes(title="Wind speed (m/s)")
    fig.update_layout(hovermode="closest")
    right.plotly_chart(style(fig, "Active power (kW)", 420), width="stretch")


# ---------------------------------------------------------------- page
st.title("Wind forecasting with XGBoost")

if not (c.MODEL_DIR / "metrics.csv").exists():
    st.error("No trained models found. Run `python train.py` in the project folder first.")
    st.stop()

tab_fc, tab_perf, tab_power = st.tabs(["Wind speed forecast", "Model performance", "Turbine power (T1)"])
with tab_fc:
    site = st.radio("Site", list(c.SITES), horizontal=True, key="site_fc")
    forecast_tab(site)
with tab_perf:
    site = st.radio("Site", list(c.SITES), horizontal=True, key="site_perf")
    performance_tab(site)
with tab_power:
    power_tab()
