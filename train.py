"""Train the XGBoost models used by app.py.

    python train.py

Writes to models/:
  <site>/h01.json ... h24.json   wind speed (WS50M) forecast, one model per horizon
  t1_power.json                  turbine active power from wind speed / direction / hour
  metrics.csv, t1_metrics.json   test-set scores
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

import common as c

PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    early_stopping_rounds=30,
    n_jobs=-1,
    random_state=42,
)


def scores(y, pred):
    return dict(rmse=root_mean_squared_error(y, pred), mae=mean_absolute_error(y, pred), r2=r2_score(y, pred))


def train_wind_speed(site: str) -> list[dict]:
    df = c.load_nasa(site)
    X = c.make_features(df).iloc[48:]  # first rows lack the 48h lag
    out_dir = c.MODEL_DIR / site
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in c.HORIZONS:
        y = c.make_target(df, h).loc[X.index]
        ok = y.notna()
        # split on the *target* time so no training label overlaps the val/test period
        t_target = X.index + pd.Timedelta(hours=h)
        train = ok & (t_target < c.VAL_START)
        val = ok & (t_target >= c.VAL_START) & (t_target < c.TEST_START)
        test = ok & (t_target >= c.TEST_START)

        model = XGBRegressor(**PARAMS)
        model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
        model.save_model(out_dir / f"h{h:02d}.json")

        xgb = scores(y[test], model.predict(X[test]))
        pers = scores(y[test], X.loc[test, c.TARGET])  # persistence: "same as now"
        rows.append(
            dict(
                site=site, horizon=h, n_test=int(test.sum()), best_iteration=int(model.best_iteration),
                **{f"xgb_{k}": v for k, v in xgb.items()},
                **{f"persistence_{k}": v for k, v in pers.items()},
                skill=1 - xgb["rmse"] / pers["rmse"],
            )
        )
        print(f"{site} +{h:>2}h  RMSE {xgb['rmse']:.3f} vs persistence {pers['rmse']:.3f}  (skill {rows[-1]['skill']:+.1%})")
    return rows


def train_t1_power() -> dict:
    df = c.load_t1()
    X = c.make_t1_features(df)
    y = df["power_kw"]
    train = df.index < c.T1_VAL_START
    val = (df.index >= c.T1_VAL_START) & (df.index < c.T1_TEST_START)
    test = df.index >= c.T1_TEST_START

    model = XGBRegressor(**PARAMS)
    model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
    model.save_model(c.MODEL_DIR / "t1_power.json")

    result = dict(
        xgb=scores(y[test], model.predict(X[test])),
        theoretical_curve=scores(y[test], df.loc[test, "theoretical_kw"]),
        n_test=int(test.sum()),
    )
    print(f"T1 power  RMSE {result['xgb']['rmse']:.1f} kW vs theoretical curve {result['theoretical_curve']['rmse']:.1f} kW")
    return result


if __name__ == "__main__":
    c.MODEL_DIR.mkdir(exist_ok=True)
    all_rows = [r for site in c.SITES for r in train_wind_speed(site)]
    pd.DataFrame(all_rows).to_csv(c.MODEL_DIR / "metrics.csv", index=False)
    (c.MODEL_DIR / "t1_metrics.json").write_text(json.dumps(train_t1_power(), indent=2))
    print("Saved models to", c.MODEL_DIR)
