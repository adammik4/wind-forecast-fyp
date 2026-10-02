"""Train and evaluate the models used by app.py, following the CRISP-DM phases in the interim report.

    python train.py            # everything
    python train.py power      # turbine power model only (Objectives 1 and 2)
    python train.py outlook    # 24 h wind-speed outlook models for Mersing and Kudat only

Writes to models/:
  power_xgb.json          tuned XGBoost power model (point forecast)
  power_quantile.json     same hyperparameters, quantile loss (P10 / P50 / P90 interval)
  power_results.json      cleaning, ablation, tuning, baseline comparison, importance, interval coverage
  tuning_trials.csv       every grid / random / Bayesian trial
  <site>/h01..h24.json    wind-speed outlook, one model per horizon
  metrics.csv             outlook test scores vs persistence
"""
import itertools
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import common as c

SEED = 42
BASE = dict(tree_method="hist", seed=SEED)
MAX_ROUNDS, EARLY_STOP = 2000, 50
DEFAULT_XGB = dict(learning_rate=0.3, max_depth=6)  # XGBoost library defaults
N_RANDOM = N_BAYES = 30

PARAMS = dict(  # outlook models (unchanged from preliminary testing): 1000 rounds max, early stop 30
    learning_rate=0.05, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=SEED,
)


def scores(y, pred):
    return dict(mae=mean_absolute_error(y, pred), rmse=root_mean_squared_error(y, pred), r2=r2_score(y, pred))


# Windows Smart App Control on the development machine blocks scikit-learn's compiled tree, neighbours,
# linear-model and model-selection modules (and with them xgboost.XGBRegressor), so XGBoost and LightGBM are
# used through their native APIs and the simple baselines are implemented on NumPy / SciPy.
class XGB:
    """Native-API XGBoost, early-stopped on a validation set. predict() uses the best iteration."""
    def __init__(self, params: dict, rounds=MAX_ROUNDS, early_stop=EARLY_STOP):
        self.params, self.rounds, self.early_stop = params, rounds, early_stop

    def fit(self, X, y, X_val=None, y_val=None):
        evals = [(xgb.DMatrix(X_val, y_val), "val")] if X_val is not None else []
        self.booster = xgb.train(self.params, xgb.DMatrix(X, y), num_boost_round=self.rounds, evals=evals,
                                 early_stopping_rounds=self.early_stop if evals else None, verbose_eval=False)
        return self

    @property
    def best_iteration(self) -> int:
        return int(self.booster.attr("best_iteration") or self.booster.num_boosted_rounds() - 1)

    def predict(self, X):
        return c.predict(self.booster, X)


class LightGBM:
    def __init__(self, **params):
        self.params = dict(objective="regression", verbose=-1, seed=SEED, **params)

    def fit(self, X, y):
        rounds = self.params.pop("n_estimators", 100)
        self.model = lgb.train(self.params, lgb.Dataset(X, y), num_boost_round=rounds)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LinearRegression:
    """Ordinary least squares on standardised inputs."""
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        A = np.column_stack([np.ones(len(X)), self.scaler.transform(X)])
        self.coef_, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), self.scaler.transform(X)]) @ self.coef_


class KNNRegressor:
    """Mean of the k nearest training points (Euclidean, standardised inputs) via a KD-tree."""
    def __init__(self, k=25):
        self.k = k

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        self.tree, self.y = cKDTree(self.scaler.transform(X)), np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        _, idx = self.tree.query(self.scaler.transform(X), k=self.k, workers=-1)
        return self.y[idx].mean(axis=1)


def sample_params(space: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{k: dist.rvs(random_state=rng) for k, dist in space.items()} for _ in range(n)]


def permutation_importance(model, X: pd.DataFrame, y, n_repeats=5, seed=SEED) -> np.ndarray:
    """Mean increase in test RMSE when one feature's values are shuffled."""
    rng = np.random.default_rng(seed)
    base = root_mean_squared_error(y, model.predict(X))
    out = []
    for col in X.columns:
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            deltas.append(root_mean_squared_error(y, model.predict(Xp)) - base)
        out.append(np.mean(deltas))
    return np.array(out)


# ---------------------------------------------------------------- power model
class PowerData:
    def __init__(self):
        self.df, self.cleaning = c.build_t1_dataset()
        self.mask = c.split_t1(self.df)
        self.y = {k: self.df.loc[m, "power_kw"] for k, m in self.mask.items()}

    def X(self, part, features=c.POWER_FEATURES):
        return self.df.loc[self.mask[part], features]


def fit_xgb(d: PowerData, params: dict, features=c.POWER_FEATURES, **extra) -> XGB:
    return XGB({**BASE, **params, **extra}).fit(d.X("train", features), d.y["train"], d.X("val", features), d.y["val"])


def evaluate(model, d: PowerData, features=c.POWER_FEATURES) -> dict:
    return {part: scores(d.y[part], np.clip(model.predict(d.X(part, features)), 0, None)) for part in ("val", "test")}


def ablation(d: PowerData) -> list[dict]:
    """Objective 1: accuracy as weather variables are added, same tuned-free XGBoost each time."""
    rows = []
    for name, feats in c.FEATURE_SETS.items():
        model = fit_xgb(d, dict(learning_rate=0.05, max_depth=6), feats)
        r = evaluate(model, d, feats)
        rows.append(dict(feature_set=name, features=feats, val=r["val"], test=r["test"]))
        print(f"  {name:<36} val RMSE {r['val']['rmse']:6.1f}  test RMSE {r['test']['rmse']:6.1f}")
    return rows


def tune(d: PowerData, features) -> tuple[dict, list[dict], pd.DataFrame]:
    """Grid, random and Bayesian (TPE) search, all scored on the chronological validation set."""
    trials = []

    def objective(method, params):
        t0 = time.perf_counter()
        model = fit_xgb(d, params, features)
        rmse = root_mean_squared_error(d.y["val"], np.clip(model.predict(d.X("val", features)), 0, None))
        trials.append(dict(method=method, trial=sum(t["method"] == method for t in trials) + 1, val_rmse=rmse,
                           best_iteration=int(model.best_iteration), seconds=time.perf_counter() - t0, **params))
        return rmse

    grid = dict(max_depth=[3, 5, 7], learning_rate=[0.03, 0.1, 0.3], min_child_weight=[1, 10, 50])
    for values in itertools.product(*grid.values()):
        objective("Grid", dict(zip(grid, values), subsample=0.8))

    space = dict(
        max_depth=randint(2, 11), learning_rate=loguniform(0.01, 0.3), min_child_weight=loguniform(1, 100),
        subsample=uniform(0.5, 0.5), colsample_bytree=uniform(0.5, 0.5),
        reg_lambda=loguniform(1e-3, 10), reg_alpha=loguniform(1e-3, 10),
    )
    for params in sample_params(space, N_RANDOM, SEED):
        objective("Random", {k: (int(v) if k == "max_depth" else float(v)) for k, v in params.items()})

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective("Bayesian", dict(
        max_depth=t.suggest_int("max_depth", 2, 10),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_weight=t.suggest_float("min_child_weight", 1, 100, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
    )), n_trials=N_BAYES)

    tr = pd.DataFrame(trials)
    param_cols = [k for k in tr.columns if k not in ("method", "trial", "val_rmse", "best_iteration", "seconds")]
    summary = []
    for method, g in tr.groupby("method", sort=False):
        best = g.loc[g.val_rmse.idxmin()]
        params = {k: (int(best[k]) if k == "max_depth" else float(best[k])) for k in param_cols if pd.notna(best[k])}
        test = evaluate(fit_xgb(d, params, features), d, features)["test"]
        summary.append(dict(method=method, trials=len(g), seconds=float(g.seconds.sum()),
                            best_val_rmse=float(best.val_rmse), test=test, params=params))
        print(f"  {method:<9} {len(g):>2} trials  best val RMSE {best.val_rmse:6.1f}  test RMSE {test['rmse']:6.1f}")
    best = min(summary, key=lambda s: s["best_val_rmse"])  # chosen on validation only, never on test
    return best, summary, tr


def baselines(d: PowerData, features, tuned_params: dict) -> list[dict]:
    """Objective 2 benchmark: every model trained on the same features and chronological split."""
    Xtr, ytr = d.X("train", features), d.y["train"]
    sk = {
        "Linear Regression": LinearRegression(),
        "K-Nearest Neighbors (k=25)": KNNRegressor(k=25),
        # bagged, unboosted trees: one round of 300 parallel trees on bootstrap-sized row samples
        "Random Forest": XGB(dict(num_parallel_tree=300, learning_rate=1.0, max_depth=12, min_child_weight=5,
                                  subsample=0.632, colsample_bynode=0.8, tree_method="hist", seed=SEED), rounds=1),
        "LightGBM": LightGBM(n_estimators=500, learning_rate=0.05),
    }
    rows = []
    for name, model in sk.items():
        model.fit(Xtr, ytr)
        rows.append(dict(model=name, role="Baseline" if name != "LightGBM" else "Comparison", **evaluate(model, d, features)))
    rows.append(dict(model="XGBoost (default)", role="Primary, untuned", **evaluate(fit_xgb(d, DEFAULT_XGB, features), d, features)))
    rows.append(dict(model="XGBoost (tuned)", role="Primary", **evaluate(fit_xgb(d, tuned_params, features), d, features)))
    rows.append(dict(model="Manufacturer power curve", role="Reference", **{
        part: scores(d.y[part], d.df.loc[d.mask[part], "theoretical_kw"]) for part in ("val", "test")}))
    for r in rows:
        print(f"  {r['model']:<28} test MAE {r['test']['mae']:6.1f}  RMSE {r['test']['rmse']:6.1f}  R2 {r['test']['r2']:.3f}")
    return rows


def train_power():
    c.MODEL_DIR.mkdir(exist_ok=True)
    d = PowerData()
    print("Cleaning:", d.cleaning)

    print("Feature-set ablation (Objective 1)")
    abl = ablation(d)
    # Deployed feature set: best on validation. Ties (within 1%) go to the smaller set.
    best_val = min(r["val"]["rmse"] for r in abl)
    chosen = min((r for r in abl if r["val"]["rmse"] <= best_val * 1.01), key=lambda r: len(r["features"]))
    features = chosen["features"]
    print(f"  -> using '{chosen['feature_set']}' {features}")

    print("Hyperparameter tuning")
    best, tuning, trials = tune(d, features)
    trials.to_csv(c.MODEL_DIR / "tuning_trials.csv", index=False)
    params = best["params"]
    print(f"  -> {best['method']} search wins: {params}")

    print("Model comparison on held-out test set (Nov-Dec 2018)")
    comparison = baselines(d, features, params)

    model = fit_xgb(d, params, features)
    model.booster.save_model(c.MODEL_DIR / "power_xgb.json")
    tuned_val = evaluate(model, d, features)["val"]["rmse"]
    default_val = next(r for r in comparison if r["model"] == "XGBoost (default)")["val"]["rmse"]
    curve_val = next(r for r in comparison if r["model"] == "Manufacturer power curve")["val"]["rmse"]
    targets = {
        "Beats manufacturer curve on validation": tuned_val < curve_val,
        "Beats default XGBoost on validation": tuned_val < default_val,
    }
    print("Error targets:", targets)

    # Uncertainty: one model, three quantiles, same tuned hyperparameters.
    qmodel = fit_xgb(d, params, features, objective="reg:quantileerror", quantile_alpha=np.array([0.1, 0.5, 0.9]))
    qmodel.booster.save_model(c.MODEL_DIR / "power_quantile.json")
    q = np.clip(qmodel.predict(d.X("test", features)), 0, None)
    yt = d.y["test"].to_numpy()
    interval = dict(coverage_80=float(np.mean((yt >= q[:, 0]) & (yt <= q[:, 2]))), mean_width_kw=float(np.mean(q[:, 2] - q[:, 0])))
    print("P10-P90 interval on test:", interval)

    gain = model.booster.get_score(importance_type="gain")
    perm = permutation_importance(model, d.X("test", features), d.y["test"])
    importance = [dict(feature=f, gain=float(gain.get(f, 0.0)), permutation_rmse_increase=float(perm[i]))
                  for i, f in enumerate(features)]

    results = dict(
        features=features, cleaning=d.cleaning,
        split={k: dict(rows=int(m.sum()), start=str(d.df.index[m].min()), end=str(d.df.index[m].max())) for k, m in d.mask.items()},
        training_range={f: [float(d.df[f].min()), float(d.df[f].max())] for f in c.POWER_FEATURES},
        ablation=abl, tuning=tuning, best_method=best["method"], params=params, comparison=comparison,
        error_targets=targets, interval=interval, importance=importance,
    )
    (c.MODEL_DIR / "power_results.json").write_text(json.dumps(results, indent=2))


# ---------------------------------------------------------------- 24 h wind-speed outlook (Mersing, Kudat)
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

        model = XGB(PARAMS, rounds=1000, early_stop=30).fit(X[train], y[train], X[val], y[val])
        model.booster.save_model(out_dir / f"h{h:02d}.json")

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


def train_outlook():
    c.MODEL_DIR.mkdir(exist_ok=True)
    all_rows = [r for site in c.SITES for r in train_wind_speed(site)]
    pd.DataFrame(all_rows).to_csv(c.MODEL_DIR / "metrics.csv", index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "power"):
        train_power()
    if what in ("all", "outlook"):
        train_outlook()
    print("Saved models to", c.MODEL_DIR)
