"""Predicted generation at a study site over a chosen month, year or date range, from its recorded weather."""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c
import ui

st.title("Generation by period", icon=":material/bolt:")
st.caption(
    "What one 3.6 MW turbine would have generated at a study site over any month, year or date range since 2015: the "
    "power model run on the NASA POWER weather recorded there, hour by hour, then added up. No wake, availability or "
    "grid losses."
)

with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
    site = st.segmented_control("Site", list(c.SITES), default="Mersing", required=True, key="gen_site")
    span = st.segmented_control("Period", ["Month", "Year", "Custom"], default="Year", required=True, key="gen_span")
    est = ui.site_power(site)
    first, last = est.index[0], est.index[-1]
    if span == "Month":
        months = pd.period_range(first, last, freq="M")[::-1]
        month = st.selectbox("Month", months, index=int(months[0].end_time.floor("h") > last),  # latest full month
                             format_func=lambda p: p.strftime("%b %Y"), width=180)
        start, end = month.start_time, (month + 1).start_time
    elif span == "Year":
        years = list(range(last.year, first.year - 1, -1))
        year = st.selectbox("Year", years, index=int(pd.Timestamp(last.year, 12, 31, 23) > last), width=140)
        start, end = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 1, 1)
    else:
        picked = st.date_input("Dates", (last.date() - dt.timedelta(days=29), last.date()), min_value=first.date(),
                               max_value=last.date(), format="DD/MM/YYYY", width=260)
        if not picked:
            st.stop()
        start = pd.Timestamp(picked[0])
        end = pd.Timestamp(picked[-1]) + pd.Timedelta(days=1)  # a half-picked range counts as one day

sub = est.loc[start: end - pd.Timedelta(hours=1)]
if sub.empty:
    st.warning("No weather data in this period.")
    st.stop()

hours = (end - start) / pd.Timedelta(hours=1)
freq, unit, time_format = ("D", "day", "DD MMM YYYY") if hours <= 92 * 24 else ("MS", "month", "MMM YYYY")
agg = sub.resample(freq).agg(energy=("power_kw", "sum"), p10=("p10", "sum"), p90=("p90", "sum"),
                             mean_kw=("power_kw", "mean"), wind=("wind_speed", "mean"))
agg[["energy", "p10", "p90"]] /= 1000  # kWh -> MWh
agg["capacity_factor"] = agg.mean_kw / c.RATED_KW

full_years = est[est.index.year < last.year]
months_in = sub.index.month.unique()
usual = full_years[full_years.index.month.isin(months_in)].power_kw.mean() / c.RATED_KW
cf = sub.power_kw.mean() / c.RATED_KW
month_names = ", ".join(pd.to_datetime(sorted(months_in), format="%m").strftime("%b")) if len(months_in) < 12 else "all months"

with st.container(horizontal=True):
    st.metric("Energy", f"{sub.power_kw.sum() / 1000:,.1f} MWh", border=True,
              help="Sum of the hourly predicted output.")
    st.metric("80% range", f"{sub.p10.sum() / 1000:,.1f} to {sub.p90.sum() / 1000:,.1f} MWh", border=True,
              help="Sum of the hourly P10 and P90 from the quantile model.")
    st.metric("Capacity factor", f"{cf:.1%}", delta=f"{(cf - usual) * 100:+.1f} pts vs usual", border=True,
              help=f"Mean output over the {c.RATED_KW:,.0f} kW rating. Usual: {usual:.1%}, the {site} average for "
                   f"{month_names} over {full_years.index.year.min()} to {full_years.index.year.max()}.")
    st.metric("Mean wind at 50 m", f"{sub.wind_speed.mean():.1f} m/s", border=True)
    st.metric("Hours above cut-in", f"{(sub.wind_speed >= c.CUT_IN).mean():.0%}", border=True,
              help=f"Wind at or above {c.CUT_IN:.0f} m/s, so the rotor turns.")

expected = int(hours)
if len(sub) < expected:
    st.caption(f"{len(sub):,} of {expected:,} hours in this period have weather data "
               f"(the record runs {first:%d %b %Y} to {last:%d %b %Y}); the totals cover those hours only.")

chart_col, table_col = st.columns([3, 2], gap="medium")
with chart_col, st.container(border=True):
    st.markdown(f"**Energy per {unit} at {site}**")
    fig = go.Figure()
    fig.add_bar(x=agg.index, y=agg.energy, name="Energy", marker_color=ui.ORANGE,
                error_y=dict(type="data", symmetric=False, array=np.clip(agg.p90 - agg.energy, 0, None),
                             arrayminus=np.clip(agg.energy - agg.p10, 0, None), color=ui.GREY))
    fig.add_scatter(x=agg.index, y=agg.wind, name="Mean wind speed (m/s)", yaxis="y2", line=dict(color=ui.BLUE))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="m/s", showgrid=False, tickformat=".1f"))
    st.plotly_chart(ui.style(fig, "Energy (MWh)", 380), width="stretch")
    st.caption("Error bars: 80% interval (P10 to P90).")

with table_col, st.container(border=True):
    st.markdown(f"**By {unit}**")
    st.dataframe(
        agg.drop(columns="mean_kw").rename_axis("time").reset_index(), hide_index=True, height=410,
        column_config={
            "time": st.column_config.DatetimeColumn(unit.capitalize(), format=time_format),
            "energy": st.column_config.NumberColumn("MWh", format="%,.2f"),
            "p10": st.column_config.NumberColumn("P10 MWh", format="%,.2f"),
            "p90": st.column_config.NumberColumn("P90 MWh", format="%,.2f"),
            "wind": st.column_config.NumberColumn("Wind (m/s)", format="%.1f"),
            "capacity_factor": st.column_config.ProgressColumn("Capacity factor", format="percent", min_value=0,
                                                               max_value=float(max(0.4, agg.capacity_factor.max()))),
        },
        column_order=["time", "energy", "p10", "p90", "capacity_factor", "wind"],
    )
