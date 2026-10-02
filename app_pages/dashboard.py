"""Objective 3: operator dashboard. Set the weather, see predicted power, turbine state and the live power curve."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import common as c
import ui

PRESETS = ui.typical_conditions()
FIELDS = ("ws", "wd", "temp", "pres", "rh")
for k, v in PRESETS["Mersing"].items():
    st.session_state.setdefault(k, v)


def apply_preset():
    site = st.session_state.preset
    if site:
        for k in FIELDS:
            st.session_state[k] = PRESETS[site][k]


st.title("Wind turbine power dashboard")
st.caption(
    "Tuned XGBoost model of a 3.6 MW onshore turbine's output from weather. "
    "Set the conditions, or load the typical (median) weather of a site."
)

controls, view = st.columns([1, 2], gap="large")
with controls:
    with st.container(border=True):
        st.segmented_control("Load typical weather", list(PRESETS), key="preset", on_change=apply_preset)
        st.slider("Wind speed at hub height (m/s)", 0.0, 30.0, step=0.1, key="ws")
        st.slider("Wind direction (degrees, from)", 0.0, 359.0, step=1.0, key="wd")
        st.slider("Air temperature (°C)", -10.0, 45.0, step=0.5, key="temp")
        st.slider("Surface pressure (kPa)", 90.0, 105.0, step=0.1, key="pres")
        st.slider("Relative humidity (%)", 0.0, 100.0, step=1.0, key="rh")

ss = st.session_state
X = c.power_features(ss.ws, ss.wd, ss.temp, ss.pres, ss.rh)
rho = float(X.air_density.iloc[0])
point = float(ui.apply_cut_out(ss.ws, ui.predict_power(X))[0])
p10, _, p90 = ui.apply_cut_out(ss.ws, ui.predict_interval(X))[0]
status, colour = c.turbine_status(ss.ws, point)
hex_colour = {"red": ui.RED, "gray": ui.GREY, "green": ui.GREEN, "blue": ui.BLUE}[colour]

with view:
    with st.container(horizontal=True):
        st.metric("Predicted power", f"{point:,.0f} kW", border=True,
                  help="Tuned XGBoost point prediction. Clipped to 0 and rated power.")
        st.metric("80% interval (P10 to P90)", f"{p10:,.0f} to {p90:,.0f} kW", border=True,
                  help="Quantile XGBoost with the same hyperparameters. On the held-out test set "
                       f"{ui.results()['interval']['coverage_80']:.0%} of actual values fell inside this band.")
        st.metric("Share of rated power", f"{point / c.RATED_KW:.0%}", border=True)
        st.metric("Air density", f"{rho:.3f} kg/m³", border=True,
                  help="Derived from temperature, pressure and humidity (moist-air equation).")

    turbine, state = st.columns([3, 2])
    with turbine:
        st.html(ui.turbine_svg(ui.rotor_rpm(ss.ws, colour in ("blue", "green")), ss.wd, hex_colour))
    with state:
        st.markdown("**Operating state**")
        st.badge(status, color=colour, icon=":material/circle:")
        st.markdown(
            f"- Cut-in: **{c.CUT_IN:.0f} m/s**\n- Rated: **{c.RATED_SPEED:.0f} m/s** "
            f"({c.RATED_KW:,.0f} kW)\n- Cut-out: **{c.CUT_OUT:.0f} m/s**\n"
            f"- Rotor: **{ui.rotor_rpm(ss.ws, colour in ('blue', 'green')):.1f} rpm**"
        )
        lo, hi = ui.results()["training_range"]["air_density"]
        if not lo <= rho <= hi:
            st.warning(f"Air density is outside the training range ({lo:.3f} to {hi:.3f} kg/m³), "
                       "so the model is extrapolating.", icon=":material/warning:")

with st.container(border=True):
    st.subheader("Live power curve")
    st.caption("Model response across wind speed with direction and air density held at the values set above.")
    speeds = np.arange(0, 30.01, 0.25)
    grid = c.power_features(speeds, ss.wd, ss.temp, ss.pres, ss.rh)
    curve = ui.apply_cut_out(speeds, ui.predict_power(grid))
    band = ui.predict_interval(grid)
    band = np.column_stack([ui.apply_cut_out(speeds, band[:, i]) for i in range(3)])
    t1 = ui.t1_data()
    sample = t1.sample(3000, random_state=0)
    maker = t1.groupby((t1.wind_speed / 0.5).round() * 0.5).theoretical_kw.mean()

    fig = go.Figure()
    fig.add_scattergl(x=sample.wind_speed, y=sample.power_kw, mode="markers", name="Measured SCADA (sample)",
                      marker=dict(size=3, color=ui.GREY, opacity=0.3), hoverinfo="skip")
    fig.add_scatter(x=speeds, y=band[:, 2], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=speeds, y=band[:, 0], line=dict(width=0), fill="tonexty", fillcolor="rgba(232,113,10,0.18)",
                    name="80% interval", hoverinfo="skip")
    fig.add_scatter(x=maker.index, y=maker, name="Manufacturer curve", line=dict(color=ui.BLUE, dash="dash"))
    fig.add_scatter(x=speeds, y=curve, name="XGBoost", line=dict(color=ui.ORANGE, width=3))
    fig.add_scatter(x=[ss.ws], y=[point], name="Current conditions", mode="markers",
                    marker=dict(size=14, color=hex_colour, line=dict(color="white", width=2)))
    for x, label in ((c.CUT_IN, "cut-in"), (c.RATED_SPEED, "rated"), (c.CUT_OUT, "cut-out")):
        fig.add_vline(x=x, line_color=ui.GREY, line_width=1, line_dash="dot", annotation_text=label)
    fig.update_xaxes(title="Wind speed (m/s)", range=[0, 30])
    fig.update_layout(hovermode="closest")
    st.plotly_chart(ui.style(fig, "Active power (kW)", 440), width="stretch")
    st.caption(
        "Temperature, pressure and humidity reach the model through air density; the feature-set ablation on "
        "the Weather drivers page shows this beats feeding them in raw. Shutdown above cut-out is the turbine's "
        "control logic, applied after the model."
    )
