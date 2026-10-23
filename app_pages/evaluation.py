"""Objective 2: tuned XGBoost benchmarked against comparator algorithms on a held-out, chronological test set."""
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
    "Sep to Oct, test Nov to Dec. Hyperparameters and feature choices were made on validation only."
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
    st.markdown(
        f"The weather-to-power model meets the stretch goal on unseen data. The 24 h wind-speed outlook meets the "
        f"{c.ERROR_TARGET:.0%} mark for the first **{min(reach.values())} to {max(reach.values())} hours** "
        f"({', '.join(f'{k} +{v} h' for k, v in reach.items())}); further ahead the error levels off near "
        f"{m[m.horizon == 24].wind_nmae.mean():.0%}, because hourly reanalysis wind carries little memory beyond "
        "half a day. Converted to power, the outlook error stays below "
        f"{m.power_nrmse.max():.0%} of rated power at every site and horizon, inside the target."
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
                   "target. If one fails, the model goes back for retuning.")

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
