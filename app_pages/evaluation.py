"""Objectives 1 and 2: tuned XGBoost benchmarked against comparator algorithms on a held-out, chronological test set,
and the weather variables and interactions that drive its predictions."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c
import ui

res = ui.results()
st.title("Model evaluation", icon=":material/fact_check:")
st.caption(
    "All models use the same features and the same chronological split: train Jan to Aug 2018, validate "
    "Sep to Oct, test Nov to Dec. Hyperparameters and feature choices were made on validation only. Downtime and "
    "curtailment records are removed from every split, test included, so the scores describe normal, weather-driven "
    "operation; weather cannot predict a turbine being switched off."
)

comp = pd.DataFrame([
    dict(Model=r["model"], Role=r["role"], **{f"{p} {m.upper() if m != 'r2' else 'R²'}": r[p.lower()][m]
                                             for p in ("Val", "Test") for m in ("mae", "rmse", "r2")})
    for r in res["comparison"]
])
# Accuracy = 100% minus RMSE as a share of rated power, so the 10-15 % error target reads as 85-90 % accuracy
for p in ("Val", "Test"):
    comp[f"{p} accuracy"] = 100 * (1 - comp[f"{p} RMSE"] / c.RATED_KW)
tuned = comp.set_index("Model").loc["XGBoost (tuned)"]
curve = comp.set_index("Model").loc["Manufacturer power curve"]
PASS, STRETCH = 100 * (1 - c.ERROR_TARGET), 100 * (1 - c.ERROR_STRETCH)

with st.container(horizontal=True):
    st.metric("Test accuracy", f"{tuned['Test accuracy']:.1f}%", border=True,
              help="100% minus test RMSE as a share of the 3,600 kW rating.")
    st.metric("Test MAE", f"{tuned['Test MAE']:.0f} kW", border=True)
    st.metric("Test RMSE", f"{tuned['Test RMSE']:.0f} kW", border=True,
              help=f"{tuned['Test RMSE'] / c.RATED_KW:.1%} of rated power")
    st.metric("Test R²", f"{tuned['Test R²']:.3f}", border=True)
    st.metric("RMSE vs manufacturer curve", f"{1 - tuned['Test RMSE'] / curve['Test RMSE']:.0%} lower", border=True)
    st.metric("80% interval coverage", f"{res['interval']['coverage_80']:.0%}", border=True,
              help=f"Share of test points inside P10 to P90 (nominal 80%). Mean width {res['interval']['mean_width_kw']:.0f} kW.")

# ---------------------------------------------------------------- 10-15 % accuracy target
with st.container(border=True):
    st.subheader(f"Accuracy target: {c.ERROR_STRETCH:.0%} to {c.ERROR_TARGET:.0%} error", icon=":material/target:")
    ne = res["normalised_error"]
    left, right = st.columns([2, 3], gap="medium")
    with left:
        st.caption(
            f"Pass mark {c.ERROR_TARGET:.0%}, stretch goal {c.ERROR_STRETCH:.0%}. Power errors are normalised by the "
            f"{c.RATED_KW:,.0f} kW rating, the usual convention in wind power forecasting; wind-speed errors by "
            "the mean observed wind speed."
        )
        with st.container(horizontal=True):
            st.metric("Power model nRMSE", f"{ne['nrmse']:.1%}", border=True, help="Test RMSE / rated power.")
            st.metric("Power model nMAE", f"{ne['nmae']:.1%}", border=True, help="Test MAE / rated power.")
            st.metric("RMSE / mean output", f"{ne['rmse_over_mean']:.1%}", border=True,
                      help="A stricter normalisation: test RMSE over the mean measured output in Nov to Dec 2018.")
        for limit, label in ((c.ERROR_TARGET, "target"), (c.ERROR_STRETCH, "stretch goal")):
            ok = ne["nrmse"] <= limit
            st.badge(f"Power model within the {limit:.0%} {label}", color="green" if ok else "red",
                     icon=":material/check:" if ok else ":material/close:")
    with right:
        m = ui.outlook_metrics()
        fig = go.Figure()
        for (site, g), colour in zip(m.groupby("site", sort=False), (ui.ORANGE, ui.BLUE, ui.TEAL, ui.GREEN, "#7C3AED")):
            fig.add_scatter(x=g.horizon, y=g.wind_nmae * 100, name=site, line=dict(color=colour, width=2.5))
        fig.update_xaxes(title="Forecast horizon (hours ahead)")
        st.plotly_chart(ui.style(ui.target_band(fig), "Wind-speed nMAE (%)", 300), width="stretch")
    reach = {site: ui.within(g.wind_nmae, c.ERROR_TARGET) for site, g in m.groupby("site", sort=False)}
    worst = m.loc[m.power_nrmse.idxmax()]
    st.markdown(
        f"The weather-to-power model meets the stretch goal on unseen data in normal operation. The 24 h wind-speed outlook meets the "
        f"{c.ERROR_TARGET:.0%} mark for the first **{min(reach.values())} to {max(reach.values())} hours** "
        f"({', '.join(f'{k} +{v} h' for k, v in reach.items())}); further ahead the error levels off near "
        f"{m[m.horizon == 24].wind_nmae.mean():.0%}, because hourly reanalysis wind carries little memory beyond "
        f"half a day. Converted to power, the outlook error peaks at {worst.power_nrmse:.1%} of rated power "
        f"({worst.site}, +{worst.horizon} h), within the {c.ERROR_TARGET:.0%} pass mark at every site and horizon."
    )

# ---------------------------------------------------------------- comparison
with st.container(border=True):
    st.subheader("Model comparison: accuracy on the test set", icon=":material/leaderboard:")
    ranked = comp.sort_values("Test accuracy", ascending=False).reset_index(drop=True)
    ranked.insert(0, "Rank", ranked.index + 1)
    ranked["Result"] = np.select(
        [ranked["Test accuracy"] >= STRETCH, ranked["Test accuracy"] >= PASS],
        [f"Meets {STRETCH:.0f}% stretch", f"Meets {PASS:.0f}% pass"], f"Below {PASS:.0f}%")
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.dataframe(
            ranked[["Rank", "Model", "Role", "Test accuracy", "Val accuracy", "Test MAE", "Test RMSE", "Test R²", "Result"]],
            hide_index=True,
            column_config={
                "Test accuracy": st.column_config.ProgressColumn("Test accuracy", format="%.1f%%", min_value=0, max_value=100),
                "Val accuracy": st.column_config.NumberColumn("Val accuracy", format="%.1f%%"),
                "Test MAE": st.column_config.NumberColumn("Test MAE (kW)", format="%.1f"),
                "Test RMSE": st.column_config.NumberColumn("Test RMSE (kW)", format="%.1f"),
                "Test R²": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    with right:
        s = ranked.iloc[::-1]
        colours = [ui.ORANGE if m == "XGBoost (tuned)" else ui.GREY if r == "Reference" else ui.BLUE
                   for m, r in zip(s.Model, s.Role)]
        fig = go.Figure(go.Scatter(
            x=s["Test accuracy"], y=s.Model, mode="markers+text", marker=dict(size=14, color=colours),
            text=[f"{v:.1f}%" for v in s["Test accuracy"]], textposition="middle right", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Test accuracy %{x:.1f}%<extra></extra>",
        ))
        fig.add_vrect(x0=PASS, x1=STRETCH, fillcolor=ui.GREEN, opacity=0.16, line_width=0,
                      annotation_text=f"target {PASS:.0f}-{STRETCH:.0f}%", annotation_position="top left")
        fig.add_vrect(x0=STRETCH, x1=100, fillcolor=ui.GREEN, opacity=0.08, line_width=0)
        fig.update_xaxes(title="Test accuracy (%)", range=[80, 100.5])
        fig.update_layout(hovermode="closest")
        st.plotly_chart(ui.style(fig, "", 330), width="stretch")

    runner_up = ranked[ranked.Model != "XGBoost (tuned)"].iloc[0]
    linear = comp.set_index("Model").loc["Linear Regression"]
    st.markdown(
        f"Tuned XGBoost scores **{tuned['Test accuracy']:.1f}%**, "
        f"{tuned['Test accuracy'] - curve['Test accuracy']:.1f} points above the manufacturer power curve "
        f"({curve['Test accuracy']:.1f}%) and {tuned['Test accuracy'] - linear['Test accuracy']:.1f} above "
        f"linear regression ({linear['Test accuracy']:.1f}%). The closest rival is {runner_up.Model} at "
        f"{runner_up['Test accuracy']:.1f}%. "
        f"{(ranked['Test accuracy'] >= PASS).sum()} of {len(ranked)} models clear the {PASS:.0f}% pass mark."
    )
    st.caption(
        f"Accuracy = 100% − test RMSE ÷ {c.RATED_KW:,.0f} kW rated power, so the {c.ERROR_STRETCH:.0%} to "
        f"{c.ERROR_TARGET:.0%} error target is {STRETCH:.0f}% to {PASS:.0f}% accuracy. Random Forest is bagged, "
        "unboosted trees built with XGBoost's random-forest mode, and Linear Regression and KNN use NumPy/SciPy. "
        "The development machine's Windows Smart App Control blocks scikit-learn's compiled tree, neighbour and "
        "linear-model modules."
    )

# ---------------------------------------------------------------- weather drivers (Objective 1): ablation
with st.container(border=True):
    st.subheader("Weather drivers: feature-set ablation", icon=":material/thermostat:")
    st.caption(
        "Which weather variables most influence predicted turbine output. The same XGBoost configuration is retrained "
        "as weather variables are added, scored on the chronological validation (Sep to Oct 2018) and held-out test "
        "(Nov to Dec 2018) sets. The deployed feature set is chosen on validation only."
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

# ---------------------------------------------------------------- weather drivers: importance
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

# ---------------------------------------------------------------- weather drivers: interactions
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

# ---------------------------------------------------------------- tuning
with st.container(border=True):
    st.subheader("Hyperparameter tuning")
    tr = ui.tuning_trials()
    tun = pd.DataFrame([dict(Method=t["method"], Trials=t["trials"], **{"Time (s)": t["seconds"]},
                             **{"Best val RMSE": t["best_val_rmse"], "Test RMSE": t["test"]["rmse"]},
                             Parameters=", ".join(f"{k}={v:.3g}" for k, v in t["params"].items())) for t in res["tuning"]])
    left, right = st.columns([2, 3])
    with left:
        fig = go.Figure()
        for method, colour in (("Grid", ui.GREY), ("Random", ui.BLUE), ("Bayesian", ui.ORANGE)):
            g = tr[tr.method == method]
            fig.add_scatter(x=g.trial, y=g.val_rmse.cummin(), name=method, line=dict(color=colour, width=3, shape="hv"))
        fig.update_xaxes(title="Trial")
        st.plotly_chart(ui.style(fig, "Best validation RMSE so far (kW)", 300), width="stretch")
    with right:
        st.dataframe(tun, hide_index=True, column_config={
            "Time (s)": st.column_config.NumberColumn(format="%.0f"),
            "Best val RMSE": st.column_config.NumberColumn(format="%.1f"),
            "Test RMSE": st.column_config.NumberColumn(format="%.1f"),
        })
        targets = res["error_targets"]
        st.markdown(f"Selected: **{res['best_method']} search**, chosen on validation RMSE.")
        for name, ok in targets.items():
            st.badge(name, color="green" if ok else "red", icon=":material/check:" if ok else ":material/close:")
        st.caption("Error targets from the flowchart decision step (Figure 4), including the 10 to 15% accuracy "
                   "target, checked on the validation set before the test set is used. If one fails, training "
                   "stops and the model goes back for retuning.")

# ---------------------------------------------------------------- held-out test detail
with st.container(border=True):
    st.subheader("Held-out test period")
    t1 = ui.t1_data()
    test = t1[t1.index >= c.T1_TEST_START]
    X = test[c.POWER_FEATURES]
    pred = ui.predict_power(X)
    q = ui.predict_interval(X)
    days = pd.date_range(test.index.min().normalize(), test.index.max().normalize() - pd.Timedelta(days=3), freq="D")
    start = st.select_slider("Window start (3 days shown)", options=list(days.date), value=days[10].date())
    win = (test.index >= pd.Timestamp(start)) & (test.index < pd.Timestamp(start) + pd.Timedelta(days=3))

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_scatter(x=test.index[win], y=q[win, 2], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=test.index[win], y=q[win, 0], line=dict(width=0), fill="tonexty",
                        fillcolor=ui.BAND, name="80% interval")
        fig.add_scatter(x=test.index[win], y=test.power_kw[win], name="Measured", line=dict(color=ui.BLUE))
        fig.add_scatter(x=test.index[win], y=pred[win], name="XGBoost", line=dict(color=ui.ORANGE, width=2))
        st.plotly_chart(ui.style(fig, "Active power (kW)", 360), width="stretch")
    with right:
        fig = go.Figure(go.Scattergl(x=test.power_kw, y=pred, mode="markers", marker=dict(size=3, color=ui.BLUE, opacity=0.3)))
        fig.add_scatter(x=[0, c.RATED_KW], y=[0, c.RATED_KW], mode="lines", line=dict(color=ui.GREY, dash="dash"))
        fig.update_xaxes(title="Measured (kW)")
        fig.update_layout(showlegend=False, hovermode="closest")
        st.plotly_chart(ui.style(fig, "Predicted (kW)", 360), width="stretch")

with st.expander("Data preparation summary"):
    cl = res["cleaning"]
    rows = [(k, v) for k, v in cl.items() if not k.startswith("rows_")]
    st.table(pd.DataFrame(rows + [("Rows kept", cl["rows_clean"]), ("Raw rows", cl["rows_raw"])],
                          columns=["Step", "Rows"]).set_index("Step"))
    st.table(pd.DataFrame(res["split"]).T.rename_axis("Split"))
    st.caption(
        "Weather (temperature, pressure, humidity) is NASA POWER hourly data for the turbine's location, "
        "interpolated to the 10-minute SCADA timestamps. The derating filter's per-bin fences come from the "
        "training months only, so no test statistics leak into cleaning."
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
        "follow either shape, which is why Linear Regression trails in the model comparison above."
    )
