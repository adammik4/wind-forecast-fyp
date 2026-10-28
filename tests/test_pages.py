"""Smoke tests: every page of the Streamlit app runs without raising, including its main interactions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app.py", *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "app_pages").glob("*.py"))]
TIMEOUT = 600  # the first run of a page estimates every site's output


def run(page: str) -> AppTest:
    return AppTest.from_file(str(ROOT / page), default_timeout=TIMEOUT).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_dashboard_loads_site_weather():
    at = run("app_pages/dashboard.py")
    at.selectbox(key="preset").set_value("Kudat").run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("span", ["Month", "Year", "Custom"])
def test_generation_periods(span):
    at = run("app_pages/generation.py")
    at.segmented_control(key="gen_span").set_value(span).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.metric[0].label == "Energy"


def test_states_hotspot_scenario():
    at = run("app_pages/states.py")
    at.segmented_control(key="scenario").set_value("Paper hotspot").run()
    assert not at.exception, [e.value for e in at.exception]
