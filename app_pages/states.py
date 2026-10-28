"""Value creation: what a wind farm of the modelled turbine could give each Malaysian state and federal territory."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common as c
import ui

MONTHS = pd.to_datetime(range(1, 13), format="%m").strftime("%b")
HOURS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24
HOURS_IN_YEAR = 8766
STUDY_STATES = [c.STATE_OF[site] for site in c.SITES]
dark = st.context.theme.type == "dark"


def pick_from_map():
    """Clicking a state or a site marker on the map selects that state."""
    points = st.session_state.state_map.selection.points
    if points:
        data = points[0].get("customdata")
        state = data[0] if isinstance(data, list) else points[0].get("location")
        if state in STUDY_STATES:
            st.session_state.state = state


st.title("Wind contribution by state", icon=":material/map:")
st.caption(
    "Pick a study-site state and a farm size to see the energy, homes and CO₂ a wind farm of this project's 3.6 MW "
    "turbine could give it in a year, and how each figure is worked out."
)

with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
    state_slot = st.empty()  # filled once the table is known, so the default can be the strongest state
    turbines = st.slider("Wind farm size (turbines)", 1, 50, 10, key="turbines", width=320,
                         help=f"Each turbine is rated {c.RATED_KW / 1000:.1f} MW.")
    scenario = st.segmented_control(
        "Wind resource", ["Recorded", "Paper hotspot"], default="Recorded", required=True, key="scenario",
        help=f"Recorded uses each site's NASA POWER hourly wind, 2015 to 2025. Paper hotspot scales the wind of the "
             f"paper's seven states so it averages {c.HOTSPOT_WS:.0f} m/s, the onshore hotspot speed in Elias et al. (2024).",
    )

table = ui.contribution(ui.state_table(hotspot=scenario == "Paper hotspot"), turbines)
table = table[table.state.isin(STUDY_STATES)]
if st.session_state.get("state") not in STUDY_STATES:
    st.session_state.state = table.loc[table.homes.idxmax(), "state"]
state = state_slot.selectbox("State", STUDY_STATES, key="state", width=260,
                             format_func=lambda s: f"{s} ({c.STATES[s]['site']})")
row = table.set_index("state").loc[state]
farm_mw = turbines * c.RATED_KW / 1000

# ---------------------------------------------------------------- selected state
with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader(state)
        st.markdown(f":gray-badge[:material/location_on: {row.site}] :violet-badge[{row.grid} grid]")

    with st.container(horizontal=True):
        st.metric("Energy a year", f"{row.energy_gwh:,.1f} GWh", border=True,
                  help=f"80% range from the quantile model: {row.energy_gwh_p10:,.1f} to {row.energy_gwh_p90:,.1f} GWh.")
        st.metric("Homes supplied", f"{row.homes:,.0f}", border=True,
                  help=f"{row.home_share:.1%} of {state}'s {row.households:,.0f} households.")
        st.metric("CO₂ avoided a year", f"{row.co2_kt:,.1f} kt", border=True,
                  help=f"{row.emission_factor} t CO₂e per MWh on the {row.grid} grid.")
        st.metric("Capacity factor", f"{row.capacity_factor:.1%}", border=True,
                  help=f"Mean output over the {c.RATED_KW:,.0f} kW rating.")

    calc_col, month_col = st.columns([3, 2], gap="medium")
    with calc_col:
        st.markdown("**Check the numbers**")
        st.table(pd.DataFrame([
            ("Wind at 50 m", f"{row.wind_speed:.2f} m/s",
             f"Mean at {row.site}; above the {c.CUT_IN:g} m/s cut-in {row.above_cut_in:.0%} of hours"),
            ("Output per turbine", f"{row.mean_kw:,.0f} kW",
             f"Power model mean; ÷ {c.RATED_KW:,.0f} kW rated = {row.capacity_factor:.1%} capacity factor"),
            ("Farm size", f"{farm_mw:,.1f} MW", f"{turbines} turbines × {c.RATED_KW / 1000:.1f} MW"),
            ("Energy", f"{row.energy_gwh:,.1f} GWh",
             f"{farm_mw:,.1f} MW × {row.capacity_factor:.1%} × {HOURS_IN_YEAR:,} h "
             f"(80% range {row.energy_gwh_p10:,.1f} to {row.energy_gwh_p90:,.1f})"),
            ("Homes", f"{row.homes:,.0f}",
             f"Energy ÷ {c.KWH_PER_HOUSEHOLD:,.0f} kWh per home = {row.home_share:.1%} of {row.households:,.0f} homes"),
            ("CO₂ avoided", f"{row.co2_kt:,.1f} kt", f"Energy × {row.emission_factor} t CO₂e/MWh ({row.grid} grid)"),
        ], columns=["Step", "Value", "How it is worked out"]), hide_index=True, border="horizontal")

    with month_col:
        monthly_gwh = np.array(row.monthly_kw) * turbines * HOURS_IN_MONTH / 1e6
        fig = go.Figure(go.Bar(x=MONTHS, y=monthly_gwh, marker_color=ui.TEAL,
                               hovertemplate="%{x}: %{y:,.2f} GWh<extra></extra>"))
        st.plotly_chart(ui.style(fig, "GWh per month", 300), config=dict(displayModeBar=False))
        best, worst = MONTHS[np.argmax(monthly_gwh)], MONTHS[np.argmin(monthly_gwh)]
        st.caption(f"Strongest month {best}, weakest {worst}. Plan maintenance for the low months.")

# ---------------------------------------------------------------- all states
with st.container(border=True):
    st.markdown("**Energy a year by state**")
    custom = np.column_stack([table.state, table.site, table.homes, table.capacity_factor * 100])
    fig = go.Figure(go.Choropleth(
        geojson=ui.states_geojson(), locations=table.state, featureidkey="properties.name", z=table.energy_gwh,
        colorscale=ui.TEAL_SCALE, marker_line_color="#0B1220" if dark else "white", marker_line_width=0.8,
        colorbar=dict(title=dict(text="GWh/yr", side="right"), thickness=12, len=0.8, outlinewidth=0),
        customdata=custom,
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>%{z:,.1f} GWh/yr · "
                      "%{customdata[2]:,.0f} homes<br>Capacity factor %{customdata[3]:.1f}%<extra></extra>",
    ))
    fig.add_choropleth(  # outline the selected state
        geojson=ui.states_geojson(), locations=[state], featureidkey="properties.name", z=[1],
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
        marker_line_color=ui.ORANGE, marker_line_width=3, hoverinfo="skip",
    )
    fig.add_scattergeo(
        lat=table.lat, lon=table.lon, customdata=custom, mode="markers",
        marker=dict(size=12, symbol="star", color=ui.ORANGE, line=dict(color="white", width=1)),
        hovertemplate="<b>%{customdata[1]}</b>, %{customdata[0]}<extra></extra>",
    )
    fig.update_geos(
        # a fixed box about 3.3 times wider than tall, so the map fills the card at 400 px; fitbounds keeps the
        # world's square frame and leaves a wide card mostly empty
        lonaxis_range=[97.8, 121.0], lataxis_range=[0.6, 7.6], projection_type="mercator", resolution=50, showframe=False,
        showland=True, landcolor="#1E293B" if dark else "#E9EEF3", showocean=True, oceancolor="#0B1220" if dark else "#F4FAFC",
        showcountries=True, countrycolor="#334155" if dark else "#CBD5E1", showcoastlines=False, bgcolor="rgba(0,0,0,0)",
    )
    # static map: geo only pans in "pan" mode, and Streamlit turns dragmode=False into "pan" when on_select is set,
    # so use a 3D-only mode the map ignores; hover and click-to-select still work
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, dragmode="orbit")
    st.plotly_chart(fig, key="state_map", on_select=pick_from_map, selection_mode="points",
                    config=dict(scrollZoom=False, displayModeBar=False))
    st.caption(":orange[★] study sites · click a state to open it")

with st.container(border=True):
    st.markdown("**Study-site states, strongest first**")
    show = table[["state", "site", "wind_speed", "capacity_factor", "energy_gwh", "homes", "co2_kt"]]
    st.dataframe(
        show.sort_values("energy_gwh", ascending=False), hide_index=True,
        column_config={
            "state": "State", "site": "Site",
            "wind_speed": st.column_config.NumberColumn("Wind (m/s)", format="%.1f"),
            "capacity_factor": st.column_config.ProgressColumn("Capacity factor", format="percent", min_value=0,
                                                               max_value=float(max(0.4, show.capacity_factor.max()))),
            "energy_gwh": st.column_config.NumberColumn("GWh/yr", format="%.1f"),
            "homes": st.column_config.NumberColumn("Homes", format="%,.0f"),
            "co2_kt": st.column_config.NumberColumn("CO₂ kt/yr", format="%.1f"),
        },
    )
    st.caption(f"A {farm_mw:,.1f} MW farm at each of the {len(table)} study sites: **{table.energy_gwh.sum():,.0f} GWh**, "
               f"**{table.homes.sum():,.0f}** homes and **{table.co2_kt.sum():,.0f} kt** CO₂ avoided a year.")

with st.expander("Method and sources", icon=":material/menu_book:"):
    st.markdown(
        f"""
- **Turbine output**: the tuned XGBoost power model and its P10/P90 quantile model, driven by each site's hourly
  50 m wind, direction, temperature, pressure and humidity, with the cut-out rule applied. No wake, availability or
  grid losses, so treat the figures as an upper bound for the recorded wind.
- **Sites**: each state is represented by its study site: {", ".join(f"{s} ({c.STATE_OF[s]})" for s in c.SITES)}.
- **States in the paper**: {c.PAPER}. It found onshore spots above {c.HOTSPOT_WS:.0f} m/s in
  {", ".join(sorted(c.PAPER_STATES))} using the Global Wind Atlas (100 m, 250 m grid). NASA POWER's
  0.5° × 0.625° cells average those ridges away, so the recorded scenario understates them; the hotspot
  scenario is the opposite bound.
- **Homes**: {c.DOMESTIC_GWH_2023:,} GWh domestic consumption in 2023 (DOSM, data.gov.my `electricity_consumption`)
  over 8.23 million households (2020 census, `hh_profile_state`) = {c.KWH_PER_HOUSEHOLD:,.0f} kWh a year each.
- **CO₂**: Energy Commission, Grid Emission Factor in Malaysia 2022 (Peninsular, Sabah including Labuan, Sarawak).
- **Boundaries**: geoBoundaries gbOpen MYS ADM1 (OpenStreetMap, ODbL).
"""
    )
