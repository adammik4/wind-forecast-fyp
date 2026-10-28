"""Model checks: the saved models reproduce their reported scores, the method has no leakage, and the power
model behaves like a wind turbine over the weather it is used on."""
import functools
import json

import numpy as np
import pandas as pd
import pytest

import common as c
import ui

RESULTS = json.loads((c.MODEL_DIR / "power_results.json").read_text())
FEATURES = RESULTS["features"]
# Typical Malaysian site weather: temperature (°C), pressure (kPa), humidity (%), wind direction (degrees)
SITE_WEATHER = [(t, p, rh, wd) for t in (24, 28, 32) for p, rh in ((100.8, 70), (100.5, 90)) for wd in (45, 200)]


@pytest.fixture(scope="module")
def booster():
    return c.load_booster(c.MODEL_DIR / "power_xgb.json")


@pytest.fixture(scope="module")
def t1():
    df, _ = c.build_t1_dataset()
    return df, c.split_t1(df)


@functools.cache
def site_features(site: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = c.load_nasa(site)
    return df, c.make_features(df).iloc[48:]


# ---------------------------------------------------------------- reported scores
def test_power_model_reproduces_reported_test_score(booster, t1):
    df, mask = t1
    y = df.loc[mask["test"], "power_kw"]
    pred = np.clip(c.predict(booster, df.loc[mask["test"], FEATURES]), 0, None)  # as train.evaluate()
    reported = next(r for r in RESULTS["comparison"] if r["model"] == "XGBoost (tuned)")["test"]
    assert np.sqrt(np.mean((pred - y) ** 2)) == pytest.approx(reported["rmse"], abs=0.1)
    assert np.mean(np.abs(pred - y)) == pytest.approx(reported["mae"], abs=0.1)


def test_quantile_model_reproduces_reported_coverage(t1):
    df, mask = t1
    q = np.clip(c.predict(c.load_booster(c.MODEL_DIR / "power_quantile.json"), df.loc[mask["test"], FEATURES]), 0, None)
    y = df.loc[mask["test"], "power_kw"].to_numpy()
    assert np.mean((y >= q[:, 0]) & (y <= q[:, 2])) == pytest.approx(RESULTS["interval"]["coverage_80"], abs=1e-3)


@pytest.mark.parametrize("h", [1, 24])
@pytest.mark.parametrize("site", list(c.SITES))
def test_outlook_models_reproduce_reported_scores(site, h):
    df, X = site_features(site)
    y = c.make_target(df, h).loc[X.index]
    test = (X.index + pd.Timedelta(hours=h) >= c.TEST_START) & y.notna()
    fc = c.predict(c.load_booster(c.MODEL_DIR / c.slug(site) / f"h{h:02d}.json"), X[test])
    reported = pd.read_csv(c.MODEL_DIR / "metrics.csv").set_index(["site", "horizon"]).loc[(site, h)]
    assert np.sqrt(np.mean((fc - y[test]) ** 2)) == pytest.approx(reported.xgb_rmse, abs=1e-3)


# ---------------------------------------------------------------- method
def test_error_targets_were_checked_on_validation():
    """Flowchart decision step (Figure 4): the targets gate on validation, never on test."""
    targets = RESULTS["error_targets"]
    assert targets and all("validation" in name.lower() for name in targets)
    assert all(targets.values())


def test_power_split_is_chronological(t1):
    df, mask = t1
    train, val, test = (df.index[mask[k]] for k in ("train", "val", "test"))
    assert train.max() < val.min() and val.max() < test.min()
    assert (mask["train"].astype(int) + mask["val"] + mask["test"] == 1).all()


def test_outlook_features_use_only_the_past():
    df = c.load_nasa("Mersing").iloc[:2000]
    t = df.index[1000]
    changed = df.copy()
    changed.loc[changed.index > t] += 5.0  # rewrite the future
    np.testing.assert_allclose(c.make_features(df).loc[:t], c.make_features(changed).loc[:t])


# ---------------------------------------------------------------- physical behaviour
@pytest.mark.parametrize("t, p, rh, wd", SITE_WEATHER)
def test_power_curve_shape(booster, t, p, rh, wd):
    ws = np.arange(0, 25.01, 0.25)
    power = np.clip(c.predict(booster, c.power_features(ws, wd, t, p, rh)[FEATURES]), 0, c.RATED_KW)
    assert power[ws < c.CUT_IN - 0.5].max() < 0.01 * c.RATED_KW  # idle below cut-in
    assert power[ws >= c.RATED_SPEED + 1].min() > 0.95 * c.RATED_KW  # flat at rated power
    climb = power[(ws >= c.CUT_IN) & (ws <= c.RATED_SPEED)]
    assert np.diff(climb).min() > -0.01 * c.RATED_KW  # never falls noticeably as the wind rises


def test_cut_out_parks_the_rotor():
    assert ui.apply_cut_out([24.9, 25.0, 30.0], [3600.0] * 3).tolist() == [3600.0, 0.0, 0.0]


def test_air_density_matches_standard_atmosphere():
    assert c.air_density(15, 101.325, 0) == pytest.approx(1.225, abs=1e-3)
    assert c.air_density(30, 101.325, 90) < c.air_density(30, 101.325, 0)  # humid air is lighter


def test_wind_direction_wraps_around():
    f = c.power_features(8, [0, 360], 20, 100, 50)
    np.testing.assert_allclose(f.iloc[0], f.iloc[1], atol=1e-9)


@pytest.mark.parametrize("site", list(c.SITES))
def test_site_weather_inside_training_range(site):
    """The power model is applied to Malaysian weather; it should not have to extrapolate."""
    X = c.site_power_inputs(c.load_nasa(site)).dropna()
    lo, hi = RESULTS["training_range"]["air_density"]
    assert lo <= X.air_density.min() and X.air_density.max() <= hi
    assert X.wind_speed.max() <= RESULTS["training_range"]["wind_speed"][1]
