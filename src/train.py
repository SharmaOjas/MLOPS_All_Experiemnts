"""End-to-end training pipeline: ETL -> feature store -> CV experiments (MLflow) -> best model.

Run:  python -m src.train
"""
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config, path
from src.data import add_test_rul, add_train_rul, data_summary, load_split, load_test_rul
from src.drift import build_reference
from src.features import FeatureStore, build_features, feature_columns, last_cycle_rows, select_sensors
from src.metrics import regression_report

log = logging.getLogger("train")

ESTIMATORS = {
    "linear_regression": LinearRegression,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
}


def make_pipeline(name: str, params: dict, seed: int) -> Pipeline:
    est = ESTIMATORS[name]
    if "random_state" in est().get_params():
        params = {"random_state": seed, **params}
    return Pipeline([("scaler", StandardScaler()), ("model", est(**params))])


def run_etl(cfg: dict) -> dict:
    """Extract, validate, label, engineer features and load them into the feature store."""
    subset = cfg["data"]["subset"]
    cap = cfg["labels"]["rul_cap"]
    window = cfg["features"]["rolling_window"]

    train_raw = load_split("train", subset)
    test_raw = load_split("test", subset)
    test_rul = load_test_rul(subset)
    if set(test_raw["unit"]) != set(test_rul.index):
        raise ValueError("Test units and RUL file do not match")

    train = add_train_rul(train_raw, cap)
    test = add_test_rul(test_raw, test_rul)  # uncapped true RUL for honest evaluation
    sensors = select_sensors(train_raw, cfg["features"]["min_std"])
    dropped = sorted(set(c for c in train_raw.columns if c.startswith("s")) - set(sensors), key=lambda s: int(s[1:]))

    store = FeatureStore(path(cfg["features"]["feature_store_dir"]))
    spec = {"subset": subset, "sensors": sensors, "dropped_sensors": dropped, "window": window, "rul_cap": cap,
            "feature_columns": feature_columns(sensors)}
    train_f = build_features(train, sensors, window)
    test_f = build_features(test, sensors, window)
    store.write(f"{subset.lower()}_train_features", train_f, {**spec, "label": f"rul capped at {cap}"})
    store.write(f"{subset.lower()}_test_features", test_f, {**spec, "label": "true rul (uncapped)"})
    log.info("ETL done: %d sensors kept, dropped %s", len(sensors), dropped)
    return {"train_raw": train_raw, "train_f": train_f, "test_f": test_f, "spec": spec,
            "summary": {"train": data_summary(train_raw), "test": data_summary(test_raw)}}


def setup_mlflow(cfg: dict) -> str:
    db = path(cfg["mlflow"]["tracking_uri"].removeprefix("sqlite:///"))
    db.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        art = path(cfg["mlflow"]["artifact_dir"])
        art.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=art.as_uri())
    mlflow.set_experiment(name)
    return name


def train_all(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    seed = cfg["training"]["random_state"]
    etl = run_etl(cfg)
    spec, train_f, test_f = etl["spec"], etl["train_f"], etl["test_f"]
    cols, cap = spec["feature_columns"], spec["rul_cap"]

    X, y, groups = train_f[cols], train_f["rul"], train_f["unit"]
    test_last = last_cycle_rows(test_f)
    X_test, y_test = test_last[cols], test_last["rul"]
    cv = GroupKFold(n_splits=cfg["training"]["cv_folds"])

    setup_mlflow(cfg)
    results, test_preds, oof_all, fitted = [], {"unit": test_last["unit"], "true_rul": y_test}, {}, {}
    for name, params in cfg["training"]["models"].items():
        t0 = time.perf_counter()
        with mlflow.start_run(run_name=name) as run:
            oof = np.zeros(len(y))
            for tr, va in cv.split(X, y, groups):  # whole engines go to either side
                m = make_pipeline(name, params, seed).fit(X.iloc[tr], y.iloc[tr])
                oof[va] = m.predict(X.iloc[va])
            oof = np.clip(oof, 0, cap)
            cv_m = regression_report(y, oof)

            model = make_pipeline(name, params, seed).fit(X, y)
            pred = np.clip(model.predict(X_test), 0, cap)
            test_m = regression_report(y_test, pred)
            test_capped = regression_report(y_test.clip(upper=cap), pred)

            metrics = {**{f"cv_{k}": v for k, v in cv_m.items()},
                       **{f"test_{k}": v for k, v in test_m.items()},
                       "test_capped_rmse": test_capped["rmse"], "test_capped_mae": test_capped["mae"],
                       "train_seconds": time.perf_counter() - t0}
            mlflow.log_params({"model": name, "subset": spec["subset"], "rul_cap": cap,
                               "window": spec["window"], "n_features": len(cols),
                               "cv": f"GroupKFold({cv.n_splits}) by unit", **params})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"feature_version": "v1", "dataset": "NASA C-MAPSS"})
            pd.DataFrame({"unit": test_last["unit"], "true_rul": y_test, "pred_rul": pred}).to_csv(
                path("mlruns") / "_tmp_pred.csv", index=False)
            mlflow.log_artifact(str(path("mlruns") / "_tmp_pred.csv"), "evaluation")
            mlflow.sklearn.log_model(model, name="model", input_example=X_test.head(3),
                                     signature=infer_signature(X_test, pred),
                                     serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
            results.append({"model": name, "run_id": run.info.run_id, "params": params, **metrics})
            test_preds[name] = pred
            oof_all[name] = oof
            fitted[name] = model
            log.info("%-24s cv_rmse=%.2f test_rmse=%.2f test_mae=%.2f r2=%.3f (%.1fs)", name,
                     metrics["cv_rmse"], metrics["test_rmse"], metrics["test_mae"], metrics["test_r2"],
                     metrics["train_seconds"])
    (path("mlruns") / "_tmp_pred.csv").unlink(missing_ok=True)

    sel = cfg["training"]["selection_metric"]
    best = min(results, key=lambda r: r[sel])
    save_best(cfg, best, fitted[best["model"]], etl, results, test_preds, oof_all)
    return {"best": best, "results": results}


def feature_importance(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, seed: int) -> pd.DataFrame:
    m = pipeline.named_steps["model"]
    if hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    elif hasattr(m, "coef_"):
        imp = np.abs(m.coef_)  # on standardised inputs, so magnitudes are comparable
    else:  # e.g. HistGradientBoosting: permutation importance on a training sample
        s = X.sample(min(3000, len(X)), random_state=seed)
        imp = permutation_importance(pipeline, s, y.loc[s.index], n_repeats=3, random_state=seed,
                                     scoring="neg_root_mean_squared_error").importances_mean.clip(min=0)
    cols = list(X.columns)
    return pd.DataFrame({"feature": cols, "importance": imp / imp.sum()}).sort_values("importance", ascending=False)


def save_best(cfg, best, pipeline, etl, results, test_preds, oof_all):
    spec = etl["spec"]
    out = path(cfg["artifacts"]["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    meta = {
        "model_name": best["model"], "model_version": version, "mlflow_run_id": best["run_id"],
        "selection_metric": cfg["training"]["selection_metric"],
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {k: v for k, v in best.items() if k.startswith(("cv_", "test_"))},
        "data_summary": etl["summary"], **spec,
    }
    bundle = {"pipeline": pipeline, "sensors": spec["sensors"], "window": spec["window"], "rul_cap": spec["rul_cap"],
              "meta": meta, "drift_reference": build_reference(etl["train_raw"], spec["sensors"])}
    joblib.dump(bundle, out / cfg["artifacts"]["model_file"], compress=3)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "experiment_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(test_preds).to_csv(out / "test_predictions.csv", index=False)
    train_f = etl["train_f"]
    pd.DataFrame({"unit": train_f["unit"], "cycle": train_f["cycle"], "true_rul": train_f["rul"], **oof_all}).to_csv(
        out / "cv_predictions.csv.gz", index=False)
    feature_importance(pipeline, train_f[spec["feature_columns"]], train_f["rul"], cfg["training"]["random_state"]).to_csv(
        out / "feature_importance.csv", index=False)

    try:  # registry is a nice-to-have; never fail training because of it
        mlflow.register_model(f"runs:/{best['run_id']}/model", "aircraft-rul-model")
    except Exception as e:  # noqa: BLE001
        log.warning("Model registry skipped: %s", e)
    log.info("Best model: %s (%s=%.3f) saved to %s", best["model"], meta["selection_metric"],
             best[meta["selection_metric"]], out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    r = train_all()
    print(pd.DataFrame(r["results"])[["model", "cv_rmse", "cv_mae", "test_rmse", "test_mae", "test_r2", "test_nasa_score"]]
          .round(3).to_string(index=False))
    print("BEST:", r["best"]["model"])
