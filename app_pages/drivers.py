"""Objective 1: which weather variables and interactions drive the prediction."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c
import ui

res = ui.results()
st.title("Weather drivers")
st.caption("Which weather variables, and which interactions between them, most influence predicted turbine output.")

# ---------------------------------------------------------------- ablation
with st.container(border=True):
    st.subheader("Feature-set ablation")
    st.caption(
        "The same XGBoost configuration retrained as weather variables are added. Scores are on the chronological "
        "validation (Sep to Oct 2018) and held-out test (Nov to Dec 2018) sets. The deployed feature set is chosen "
        "on validation only."
    )
    abl = pd.DataFrame([dict(feature_set=r["feature_set"], val=r["val"]["rmse"], test=r["test"]["rmse"],
                             test_mae=r["test"]["mae"], features=len(r["features"])) for r in res["ablation"]])
    chosen = next(r["feature_set"] for r in res["ablation"] if r["features"] == res["features"])
    fig = go.Figure()
    fig.add_bar(y=abl.feature_set, x=abl.val, name="Validation RMSE", orientation="h", marker_color=ui.GREY)
    fig.add_bar(y=abl.feature_set, x=abl.test, name="Test RMSE", orientation="h",
                marker_color=[ui.ORANGE if f == chosen else ui.BLUE for f in abl.feature_set])
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="RMSE (kW)")
    fig.update_layout(barmode="group", hovermode="y unified")
    st.plotly_chart(ui.style(fig, "", 330), width="stretch")

    base, best = abl.iloc[0], abl.set_index("feature_set").loc[chosen]
    st.markdown(
        f"Deployed: **{chosen}** ({', '.join(c.FEATURE_LABELS[f] for f in res['features'])}). "
        f"Against wind speed alone, test RMSE falls from **{base.test:.0f} to {best.test:.0f} kW** "
        f"({1 - best.test / base.test:.0%} lower). Feeding temperature, pressure and humidity in raw as well "
        "adds nothing beyond the air density they produce. Raw pressure also depends on site elevation "
        "(Yalova is about 300 m up, the Malaysian sites near sea level), so density is the input that "
        "transfers between sites."
    )

# ---------------------------------------------------------------- importance
left, right = st.columns(2)
imp = pd.DataFrame(res["importance"])
imp["label"] = imp.feature.map(c.FEATURE_LABELS)
with left, st.container(border=True):
    st.subheader("Permutation importance")
    st.caption("Increase in test RMSE when one feature's values are shuffled. Model-agnostic.")
    s = imp.sort_values("permutation_rmse_increase")
    fig = go.Figure(go.Bar(x=s.permutation_rmse_increase, y=s.label, orientation="h", marker_color=ui.ORANGE))
    fig.update_xaxes(title="RMSE increase (kW)", type="log")
    st.plotly_chart(ui.style(fig, "", 280), width="stretch")
with right, st.container(border=True):
    st.subheader("XGBoost gain")
    st.caption("Average loss reduction from splits on each feature, built into XGBoost.")
    s = imp.assign(share=imp.gain / imp.gain.sum()).sort_values("share")
    fig = go.Figure(go.Bar(x=s.share * 100, y=s.label, orientation="h", marker_color=ui.BLUE))
    fig.update_xaxes(title="Share of total gain (%)", type="log")
    st.plotly_chart(ui.style(fig, "", 280), width="stretch")
st.caption("Both axes are logarithmic: wind speed dominates by orders of magnitude, so the secondary drivers would "
           "be invisible on a linear scale.")

# ---------------------------------------------------------------- interactions
left, right = st.columns(2)
speeds = np.arange(0, 25.01, 0.25)
with left, st.container(border=True):
    st.subheader("Wind speed × air density")
    lo, hi = res["training_range"]["air_density"]
    rho_lo, rho_hi = round(lo + 0.02, 2), round(hi - 0.02, 2)
    # Build the grid directly on air density: temperature/pressure/humidity only matter through it.
    curves = {}
    for rho in (rho_lo, rho_hi):
        X = c.power_features(speeds, 0, 15, 98, 70).assign(air_density=rho)
        curves[rho] = ui.predict_power(X)
    fig = go.Figure()
    fig.add_scatter(x=speeds, y=curves[rho_lo], name=f"ρ = {rho_lo} kg/m³ (warm, humid)", line=dict(color=ui.ORANGE, width=3))
    fig.add_scatter(x=speeds, y=curves[rho_hi], name=f"ρ = {rho_hi} kg/m³ (cold, dry)", line=dict(color=ui.BLUE, width=3))
    fig.update_xaxes(title="Wind speed (m/s)")
    st.plotly_chart(ui.style(fig, "Predicted power (kW)", 320), width="stretch")
    mid = (speeds >= 6) & (speeds <= 11)
    gap = (curves[rho_hi][mid] - curves[rho_lo][mid]).mean()
    st.caption(
        f"Between 6 and 11 m/s the denser air yields on average **{gap:,.0f} kW** more at the same wind speed. "
        "The curves converge at rated power, where the controller caps output. That is the interaction the "
        "trees learn without being told P = ½ρAv³Cp."
    )
with right, st.container(border=True):
    st.subheader("Wind direction")
    dirs = np.arange(0, 360, 5)
    t1 = ui.t1_data()
    rho_med = float(t1.air_density.median())
    fig = go.Figure()
    by_dir = {}
    for ws, colour in ((6, ui.ORANGE), (9, ui.BLUE)):
        X = c.power_features(ws, dirs, 15, 98, 70).assign(air_density=rho_med)
        by_dir[ws] = ui.predict_power(X)
        fig.add_scatterpolar(r=by_dir[ws], theta=dirs, name=f"{ws} m/s", line=dict(color=colour, width=3))
    fig.update_layout(polar=dict(angularaxis=dict(direction="clockwise", rotation=90)), height=320,
                      margin=dict(l=30, r=30, t=30, b=10), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    sector = int(np.argmax(np.histogram(t1.wind_dir, bins=np.arange(0, 361, 30))[0]) * 30)
    st.caption(
        f"Predicted power by wind direction at fixed speed and median air density. At 9 m/s it varies by "
        f"{np.ptp(by_dir[9]):,.0f} kW across directions; the most frequent sector is {sector}° to {sector + 30}°. "
        "The turbine yaws into the wind, so this residual effect is specific to the training site's surroundings "
        "and should not be expected to carry over unchanged to Mersing or Kudat."
    )

with st.expander("Data understanding: the cubic region of the power curve"):
    t1 = ui.t1_data()
    part = t1[(t1.wind_speed > c.CUT_IN) & (t1.wind_speed < c.RATED_SPEED - 2)]
    k = np.polyfit(np.log(part.wind_speed), np.log(part.power_kw.clip(lower=1)), 1)[0]
    st.markdown(
        f"Fitting log(power) against log(wind speed) between cut-in and just below rated gives an exponent of "
        f"**{k:.2f}**, steeper than the ideal 3 in P = ½ρAv³Cp. Real output is zero at the {c.CUT_IN:.0f} m/s "
        "cut-in rather than at 0 m/s, and drivetrain losses take a fixed bite at low speed, so measured power "
        "climbs faster than v³ on this stretch. Above rated speed it is flat. A single straight line cannot "
        "follow either shape, which is why Linear Regression trails on the Model evaluation page."
    )
