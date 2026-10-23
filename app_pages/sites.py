"""Study sites (report Section 1.4): estimated generation at the five Malaysian sites, and a 24 h power outlook."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c
import ui

st.title("Study sites", icon=":material/location_on:")
st.caption(
    "What the same 3.6 MW turbine would produce in Malaysia's marginal wind corridors, using NASA POWER (MERRA-2) "
    "hourly weather from 2015 onwards. Treat this as a site comparison, not a yield assessment: the wind is at 50 m from a "
    "0.5° reanalysis grid, which smooths out gusts and local terrain effects."
)
site = st.segmented_control("Site", list(c.SITES), default="Mersing", required=True, key="site")
state = c.STATE_OF[site]
st.markdown(f":violet-badge[:material/location_on: {state}]")

est = ui.site_power(site)
full_years = est[est.index.year < est.index.max().year]
yearly = full_years.power_kw.resample("YE").mean()

with st.container(horizontal=True):
    st.metric("Mean wind speed at 50 m", f"{est.wind_speed.mean():.1f} m/s", border=True)
    st.metric("Hours below cut-in", f"{(est.wind_speed < c.CUT_IN).mean():.0%}", border=True,
              help=f"Wind under {c.CUT_IN:.0f} m/s: the rotor is idle.")
    st.metric("Estimated capacity factor", f"{est.power_kw.mean() / c.RATED_KW:.1%}", border=True,
              chart_data=(yearly / c.RATED_KW * 100).round(1).tolist(), chart_type="bar",
              help="Mean predicted output divided by rated power. Bars: one per full year.")
    st.metric("Estimated annual energy", f"{est.power_kw.mean() * 8.766:,.0f} MWh", border=True,
              help="Mean predicted output × 8,766 h. For comparison, the training site (Yalova, Turkey) "
                   f"averaged {ui.t1_data().power_kw.mean():,.0f} kW in 2018 (cleaned SCADA).")

left, right = st.columns(2)
with left, st.container(border=True):
    st.subheader("Seasonal pattern")
    monthly = est.groupby(est.index.month).agg(cf=("power_kw", "mean"), p10=("p10", "mean"), p90=("p90", "mean"),
                                               ws=("wind_speed", "mean"))
    months = pd.to_datetime(monthly.index, format="%m").strftime("%b")
    fig = go.Figure()
    fig.add_bar(x=months, y=monthly.cf / c.RATED_KW * 100, name="Capacity factor", marker_color=ui.ORANGE,
                error_y=dict(type="data", symmetric=False, array=(monthly.p90 - monthly.cf) / c.RATED_KW * 100,
                             arrayminus=(monthly.cf - monthly.p10) / c.RATED_KW * 100, color=ui.GREY))
    fig.add_scatter(x=months, y=monthly.ws, name="Mean wind speed (m/s)", yaxis="y2", line=dict(color=ui.BLUE))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="m/s", showgrid=False, tickformat=".1f"))
    st.plotly_chart(ui.style(fig, "Capacity factor (%)", 340), width="stretch")
    st.caption("Error bars: mean P10 and P90. The north-east monsoon (roughly Nov to Mar) carries most of the output.")

with right, st.container(border=True):
    st.subheader("Wind speed distribution")
    t1 = ui.t1_data()
    fig = go.Figure()
    bins = dict(start=0, end=20, size=0.5)
    fig.add_histogram(x=est.wind_speed, name=site, histnorm="percent", xbins=bins, marker_color=ui.ORANGE, opacity=0.75)
    fig.add_histogram(x=t1.wind_speed, name="Yalova (training site, hub height)", histnorm="percent", xbins=bins,
                      marker_color=ui.BLUE, opacity=0.45)
    fig.add_vline(x=c.CUT_IN, line_dash="dot", line_color=ui.GREY, annotation_text="cut-in")
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(title="Wind speed (m/s)")
    st.plotly_chart(ui.style(fig, "Share of hours (%)", 340), width="stretch")
    st.caption("Most Malaysian hours sit on the steep low end of the power curve, where small forecast errors cost the most.")

# ---------------------------------------------------------------- 24 h outlook
st.subheader("24-hour power outlook")
st.caption(
    "Wind speed is forecast 1 to 24 h ahead by one XGBoost model per horizon (trained on 2015 to 2022, "
    "tested on 2024 onwards). Each forecast is converted to power with the quantile model; direction and air "
    "density are held at their values at the forecast origin."
)
df, X = ui.nasa(site)
test_index = df.index[df.index >= c.TEST_START]
first, last = test_index[0], test_index[-1]
origin = st.slider(
    "Forecast origin (test period, never seen in training)", min_value=first.to_pydatetime(),
    max_value=last.to_pydatetime(), value=(last - pd.Timedelta(hours=24)).to_pydatetime(),
    step=pd.Timedelta(hours=1).to_pytimedelta(), format="YYYY-MM-DD HH:mm",
)
origin = pd.Timestamp(origin)
row = X.loc[[origin]]
if row.isna().all(axis=None):
    st.warning("No weather data at this time.")
    st.stop()

ws_fc = pd.Series({origin + pd.Timedelta(hours=h): float(c.predict(ui.booster(str(c.MODEL_DIR / c.slug(site) / f"h{h:02d}.json")), row)[0])
                   for h in c.HORIZONS})
o = df.loc[origin]
q = ui.predict_interval(c.power_features(ws_fc.clip(lower=0).to_numpy(), o.WD10M, o.T2M, o.PS, o.RH2M))
q = np.column_stack([ui.apply_cut_out(ws_fc.to_numpy(), q[:, i]) for i in range(3)])
actual = df.loc[origin: origin + pd.Timedelta(hours=24), c.TARGET].iloc[1:]
actual_power = est.power_kw.reindex(actual.index)

with st.container(border=True):
    wind_col, power_col = st.columns(2)
    past = df.loc[origin - pd.Timedelta(hours=48): origin, c.TARGET]
    fig = go.Figure()
    fig.add_scatter(x=past.index, y=past, name="Observed", line=dict(color=ui.BLUE, width=2))
    fig.add_scatter(x=actual.index, y=actual, name="Actual (after origin)", line=dict(color=ui.BLUE, dash="dot"))
    fig.add_scatter(x=[origin, *ws_fc.index], y=[o[c.TARGET], *ws_fc], name="XGBoost forecast", line=dict(color=ui.ORANGE, width=3))
    fig.add_hline(y=c.CUT_IN, line_dash="dot", line_color=ui.GREY, annotation_text="cut-in")
    fig.add_vline(x=origin, line_color=ui.GREY, line_width=1)
    wind_col.plotly_chart(ui.style(fig, "Wind speed at 50 m (m/s)", 340), width="stretch")

    fig = go.Figure()
    fig.add_scatter(x=ws_fc.index, y=q[:, 2], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=ws_fc.index, y=q[:, 0], line=dict(width=0), fill="tonexty", fillcolor=ui.BAND, name="80% interval")
    fig.add_scatter(x=ws_fc.index, y=q[:, 1], name="Forecast power (P50)", line=dict(color=ui.ORANGE, width=3))
    fig.add_scatter(x=actual_power.index, y=actual_power, name="Power from actual weather", line=dict(color=ui.BLUE, dash="dot"))
    power_col.plotly_chart(ui.style(fig, "Power (kW)", 340), width="stretch")

    with st.container(horizontal=True):
        st.metric("Forecast energy, next 24 h", f"{q[:, 1].sum() / 1000:,.1f} MWh", border=True)
        st.metric("80% range", f"{q[:, 0].sum() / 1000:,.1f} to {q[:, 2].sum() / 1000:,.1f} MWh", border=True)
        if len(actual):
            st.metric("Wind speed MAE over this window", f"{(ws_fc.loc[actual.index] - actual).abs().mean():.2f} m/s", border=True)

with st.expander("Wind-speed outlook accuracy by horizon (test set, 2024 onwards)", icon=":material/insights:"):
    m = ui.outlook_metrics()
    ms = m[m.site == site]
    left, right = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=ms.horizon, y=ms.xgb_rmse, name="XGBoost", line=dict(color=ui.ORANGE, width=3))
    fig.add_scatter(x=ms.horizon, y=ms.persistence_rmse, name="Persistence (same as now)", line=dict(color=ui.GREY, dash="dash"))
    fig.update_xaxes(title="Forecast horizon (hours ahead)")
    left.plotly_chart(ui.style(fig, "Test RMSE (m/s)", 300), width="stretch")
    right.plotly_chart(ui.target_chart(ms, 300), width="stretch")
    st.caption(f"Skill over persistence: {ms.skill.iloc[0]:+.0%} at +1 h, {ms.skill.iloc[5]:+.0%} at +6 h, "
               f"{ms.skill.iloc[-1]:+.0%} at +24 h. " + ui.target_summary(ms))
