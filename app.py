"""Wind turbine energy generation prediction dashboard (interim report, Section 3.2.6 / Objective 3).

    python train.py          (once, to build models/)
    streamlit run app.py
"""
import streamlit as st

import ui

st.set_page_config(page_title="Wind turbine power prediction", page_icon=":material/wind_power:", layout="wide")
ui.require_models()

page = st.navigation({
    "Operate": [
        st.Page("app_pages/dashboard.py", title="Power dashboard", icon=":material/wind_power:", default=True),
        st.Page("app_pages/sites.py", title="Mersing and Kudat", icon=":material/location_on:"),
    ],
    "Analysis": [
        st.Page("app_pages/drivers.py", title="Weather drivers", icon=":material/thermostat:"),
        st.Page("app_pages/evaluation.py", title="Model evaluation", icon=":material/fact_check:"),
    ],
})
page.run()
