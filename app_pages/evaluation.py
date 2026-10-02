"""Objective 2: tuned XGBoost benchmarked against comparator algorithms on a held-out, chronological test set."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c
import ui

res = ui.results()
st.title("Model evaluation")
st.caption(
    "All models use the same features and the same chronological split: train Jan to Aug 2018, validate "
    "Sep to Oct, test Nov to Dec. Hyperparameters and feature choices were made on validation only."
)

comp = pd.DataFrame([
    dict(Model=r["model"], Role=r["role"], **{f"{p} {m.upper() if m != 'r2' else 'R²'}": r[p.lower()][m]
                                             for p in ("Val", "Test") for m in ("mae", "rmse", "r2")})
    for r in res["comparison"]
])
tuned = comp.set_index("Model").loc["XGBoost (tuned)"]
curve = comp.set_index("Model").loc["Manufacturer power curve"]

with st.container(horizontal=True):
    st.metric("Test MAE", f"{tuned['Test MAE']:.0f} kW", border=True)
    st.metric("Test RMSE", f"{tuned['Test RMSE']:.0f} kW", border=True,
              help=f"{tuned['Test RMSE'] / c.RATED_KW:.1%} of rated power")
    st.metric("Test R²", f"{tuned['Test R²']:.3f}", border=True)
    st.metric("RMSE vs manufacturer curve", f"{1 - tuned['Test RMSE'] / curve['Test RMSE']:.0%} lower", border=True)
    st.metric("80% interval coverage", f"{res['interval']['coverage_80']:.0%}", border=True,
              help=f"Share of test points inside P10 to P90 (nominal 80%). Mean width {res['interval']['mean_width_kw']:.0f} kW.")

# ---------------------------------------------------------------- comparison
with st.container(border=True):
    st.subheader("Benchmark against comparator algorithms")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            comp.sort_values("Test RMSE"), hide_index=True,
            column_config={col: st.column_config.NumberColumn(format="%.3f" if "R²" in col else "%.1f")
                           for col in comp.columns if col not in ("Model", "Role")},
        )
        st.caption(
            "Random Forest is bagged, unboosted trees built with XGBoost's random-forest mode, and Linear "
            "Regression and KNN use NumPy/SciPy. The development machine's Windows Smart App Control blocks "
            "scikit-learn's compiled tree, neighbour and linear-model modules."
        )
    with right:
        s = comp.sort_values("Test RMSE", ascending=False)
        colours = [ui.ORANGE if m == "XGBoost (tuned)" else ui.GREY if r == "Reference" else ui.BLUE
                   for m, r in zip(s.Model, s.Role)]
        fig = go.Figure(go.Bar(x=s["Test RMSE"], y=s.Model, orientation="h", marker_color=colours,
                               text=s["Test RMSE"].round(0), textposition="outside"))
        fig.update_xaxes(title="Test RMSE (kW)")
        st.plotly_chart(ui.style(fig, "", 330), width="stretch")

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
        st.caption("Error targets from the flowchart decision step (Figure 4). If one fails, the model goes back for retuning.")

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
                        fillcolor="rgba(232,113,10,0.18)", name="80% interval")
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
