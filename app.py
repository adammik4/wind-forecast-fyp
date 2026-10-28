"""Wind turbine energy generation prediction dashboard (interim report, Section 3.2.6 / Objective 3).

    python fetch_data.py     (only if Data/sites/ or the state boundaries are missing)
    python train.py          (once, to build models/)
    streamlit run app.py
"""
import streamlit as st

import ui

st.set_page_config(page_title="Wind turbine power prediction", page_icon=str(ui.ASSETS / "icon.svg"), layout="wide")
st.logo(str(ui.ASSETS / "logo.svg"), icon_image=str(ui.ASSETS / "icon.svg"), size="large")
ui.require_models()

page = st.navigation({
    "Operate": [
        st.Page("app_pages/dashboard.py", title="Power dashboard", icon=":material/wind_power:", default=True),
        st.Page("app_pages/sites.py", title="Study sites", icon=":material/location_on:"),
        st.Page("app_pages/generation.py", title="Generation by period", icon=":material/bolt:"),
    ],
    "Value for Malaysia": [
        st.Page("app_pages/states.py", title="Contribution by state", icon=":material/map:"),
    ],
    "Analysis": [
        st.Page("app_pages/evaluation.py", title="Model evaluation", icon=":material/fact_check:"),
    ],
})
page.run()
