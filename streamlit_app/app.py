"""Aircraft Engine Predictive Maintenance - integrated MLOps dashboard.

Run from the project root:  streamlit run streamlit_app/app.py
"""
import json
import os
import shutil
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, path  # noqa: E402
from src.data import COLUMNS, SENSOR_COLS, SENSOR_NAMES, add_test_rul, add_train_rul, load_split, load_test_rul  # noqa: E402
from src.drift import drift_report  # noqa: E402
from src.features import FeatureStore, build_features  # noqa: E402
from src.predict import InputError, RULPredictor, default_model_path  # noqa: E402

CFG = load_config()
API_URL = os.getenv("RUL_API_URL", CFG["serving"]["api_url"]).rstrip("/")
PROM_URL = os.getenv("PROMETHEUS_URL", CFG["serving"]["prometheus_url"]).rstrip("/")
SUBSET = CFG["data"]["subset"]
CAP = CFG["labels"]["rul_cap"]
MODEL_DIR = path(CFG["artifacts"]["model_dir"])
BLUE, ORANGE, RED, GREEN = "#1f6feb", "#f0883e", "#d1242f", "#1a7f37"
SHORT_NAMES = {"linear_regression": "Linear Reg.", "random_forest": "Random Forest",
               "gradient_boosting": "Grad. Boost", "hist_gradient_boosting": "Hist GBM"}

st.set_page_config(page_title="Aircraft Engine RUL - MLOps", page_icon="✈️", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem;}
  div[data-testid="stMetric"] {background: rgba(31,111,235,0.06); border: 1px solid rgba(31,111,235,0.18);
      border-radius: 10px; padding: 10px 14px;}
  .pill {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.85rem; font-weight:600;}
  .ok {background:#dafbe1; color:#1a7f37;} .warn {background:#fff8c5; color:#9a6700;} .bad {background:#ffebe9; color:#d1242f;}
  .muted {color: #6e7781; font-size: 0.9rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def raw(split: str, subset: str = SUBSET) -> pd.DataFrame:
    df = load_split(split, subset)
    return add_train_rul(df, cap=None) if split == "train" else add_test_rul(df, load_test_rul(subset))


@st.cache_data(show_spinner=False)
def read_json(name: str):
    f = MODEL_DIR / name
    return json.loads(f.read_text()) if f.exists() else None


@st.cache_data(show_spinner=False)
def read_csv(name: str):
    f = MODEL_DIR / name
    return pd.read_csv(f) if f.exists() else None


@st.cache_resource(show_spinner=False)
def predictor():
    return RULPredictor() if default_model_path().exists() else None


def available_subsets():
    d = path(CFG["data"]["raw_dir"])
    return [s for s in CFG["data"]["available_subsets"] if (d / f"train_{s}.txt").exists()]


def api_get(route: str, timeout=2.0):
    try:
        r = requests.get(f"{API_URL}{route}", timeout=timeout)
        return r.status_code, (r.json() if "json" in r.headers.get("content-type", "") else r.text)
    except requests.RequestException as e:
        return None, str(e)


def prom(q: str):
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": q}, timeout=3)
    r.raise_for_status()
    return r.json()["data"]["result"]


def prom_range(q: str, minutes: int = 30):
    end = pd.Timestamp.now(tz="UTC").timestamp()
    r = requests.get(f"{PROM_URL}/api/v1/query_range",
                     params={"query": q, "start": end - minutes * 60, "end": end, "step": "15s"}, timeout=5)
    r.raise_for_status()
    rows = []
    for series in r.json()["data"]["result"]:
        label = ",".join(f"{k}={v}" for k, v in series["metric"].items() if k != "__name__") or "value"
        rows += [{"time": pd.to_datetime(t, unit="s"), "value": float(v), "series": label} for t, v in series["values"]]
    return pd.DataFrame(rows)


def run_cmd(args: list[str], timeout=8):
    if not shutil.which(args[0]):
        return None, f"`{args[0]}` is not installed in this environment."
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return p.returncode, (p.stdout or p.stderr).strip()
    except subprocess.TimeoutExpired:
        return None, f"`{' '.join(args)}` timed out"


def pill(text, kind):
    return f'<span class="pill {kind}">{text}</span>'


def band(rul):
    return ("CRITICAL", "bad") if rul <= 25 else ("WARNING", "warn") if rul <= 75 else ("HEALTHY", "ok")


def need_model():
    if predictor() is None:
        st.error("No trained model found. Run `python -m src.train` from the project root, then reload.")
        st.stop()


def need_data():
    try:
        raw("train")
    except FileNotFoundError as e:
        st.error(f"{e}")
        st.code("python -m src.data --download", language="bash")
        st.stop()


# --------------------------------------------------------------------------- pages
def page_overview():
    st.title("✈️ Aircraft Engine Predictive Maintenance")
    st.markdown(
        "Estimate the **Remaining Useful Life (RUL)** of turbofan engines - the number of operating cycles "
        "left before an engine reaches its failure threshold - from multivariate sensor histories of the "
        f"**NASA C-MAPSS** dataset (subset **{SUBSET}**). One project covers the full MLOps lifecycle: "
        "data, features, experiments, serving, CI/CD, containers, infrastructure and monitoring.")
    meta = read_json("metadata.json")
    tr, te = raw("train"), raw("test")
    c = st.columns(6)
    c[0].metric("Training engines", tr["unit"].nunique())
    c[1].metric("Test engines", te["unit"].nunique())
    c[2].metric("Sensor rows", f"{len(tr) + len(te):,}")
    if meta:
        m = meta["metrics"]
        c[3].metric("Best model", SHORT_NAMES.get(meta["model_name"], meta["model_name"]), help=meta["model_name"])
        c[4].metric("Test RMSE (cycles)", f"{m['test_rmse']:.2f}")
        c[5].metric("Test R²", f"{m['test_r2']:.3f}")
        st.caption(f"Metrics computed on the official {SUBSET} test set (RUL_{SUBSET}.txt) at each engine's last "
                   f"observed cycle. Model selected by engine-wise cross-validation RMSE, not by test score. "
                   f"Model version `{meta['model_version']}`, trained {meta['trained_utc']}.")
    else:
        c[3].warning("Model not trained yet")

    st.subheader("Architecture")
    st.graphviz_chart("""
    digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#eef4ff",
      color="#1f6feb", fontname="Helvetica", fontsize=11]; edge [color="#57606a"];
      raw [label="NASA C-MAPSS\\nraw text files"]; etl [label="ETL + validation\\nsrc/data.py"];
      fs [label="Feature store\\nParquet + metadata"]; train [label="Training (GroupKFold)\\nLR / RF / GB / HGB"];
      mlf [label="MLflow\\nruns + registry"]; model [label="Best model bundle\\nmodels/rul_model.joblib"];
      api [label="FastAPI\\n/predict /metrics", fillcolor="#fff1e5", color="#f0883e"];
      ui [label="Streamlit\\ndashboard", fillcolor="#fff1e5", color="#f0883e"];
      prom [label="Prometheus", fillcolor="#ffebe9", color="#d1242f"]; graf [label="Grafana", fillcolor="#ffebe9", color="#d1242f"];
      gha [label="GitHub Actions\\nCI tests", fillcolor="#dafbe1", color="#1a7f37"];
      dep [label="Docker Compose | Kubernetes | Terraform", fillcolor="#dafbe1", color="#1a7f37"];
      raw->etl->fs->train->mlf; train->model->api; ui->api; api->prom->graf; prom->ui; gha->dep->api; }""")

    st.subheader("Seven experiments, one project")
    st.dataframe(pd.DataFrame([
        ["1", "Git & GitHub", "Feature branches merged into main, tagged releases", "docs/GIT_WORKFLOW.md"],
        ["2", "Agile & Scrum", "Backlogs, roles and ceremonies for this project", "docs/SCRUM.md"],
        ["3", "CI/CD - GitHub Actions", "Install, train, pytest, Docker build on push/PR", ".github/workflows/ci.yml"],
        ["4", "ETL, Feature Store, MLflow", "Validation, causal features, 4 tracked models", "src/, feature_store/, mlruns/"],
        ["5", "Docker & Kubernetes", "API image, Compose stack, 3→5 replica scaling", "docker/, kubernetes/"],
        ["6", "Terraform", "Docker provider provisions the API container", "terraform/"],
        ["7", "Prometheus & Grafana", "Ops metrics, input drift, feedback MAE", "monitoring/"],
    ], columns=["#", "Experiment", "Implemented as", "Where"]), hide_index=True, width="stretch")

    code, body = api_get("/health")
    st.markdown("**Prediction API:** " + (pill("ONLINE", "ok") if code == 200 else pill("OFFLINE", "bad"))
                + f' <span class="muted">{API_URL}</span>', unsafe_allow_html=True)


def page_dataset():
    st.title("🗂️ Dataset Explorer")
    st.markdown("NASA C-MAPSS (Saxena & Goebel, 2008): simulated run-to-failure trajectories of a fleet of turbofan "
                "engines. Training engines run until failure; test engines stop earlier and their true RUL is "
                "given separately in `RUL_FDxxx.txt`.")
    subs = available_subsets()
    c1, c2 = st.columns(2)
    subset = c1.selectbox("Subset", subs, index=subs.index(SUBSET))
    split = c2.radio("Split", ["train", "test"], horizontal=True)
    df = raw(split, subset)
    life = df.groupby("unit")["cycle"].max()
    c = st.columns(4)
    c[0].metric("Engines", df["unit"].nunique())
    c[1].metric("Rows (cycles)", f"{len(df):,}")
    c[2].metric("Shortest history", int(life.min()))
    c[3].metric("Longest history", int(life.max()))
    if subset != SUBSET:
        st.info(f"The deployed model is trained on {SUBSET}. {subset} loads with the same code; "
                "multi-condition subsets (FD002/FD004) would need per-operating-condition normalisation.")

    left, right = st.columns([1, 1])
    fig = px.histogram(life, nbins=30, color_discrete_sequence=[BLUE],
                       labels={"value": "Cycles observed per engine"}, title="Engine history length")
    fig.update_layout(showlegend=False, height=330)
    left.plotly_chart(fig, width="stretch")
    if split == "test":
        rul = load_test_rul(subset)
        fig = px.histogram(rul, nbins=30, color_discrete_sequence=[ORANGE], title="True RUL at last test cycle",
                           labels={"value": "RUL (cycles)"})
    else:
        fig = px.scatter(df.groupby("unit")["cycle"].max().reset_index(), x="unit", y="cycle",
                         title="Failure cycle per training engine", color_discrete_sequence=[ORANGE])
    fig.update_layout(showlegend=False, height=330)
    right.plotly_chart(fig, width="stretch")

    st.subheader("Single engine")
    unit = st.selectbox("Engine unit", sorted(df["unit"].unique()))
    e = df[df.unit == unit]
    sensors = st.multiselect("Sensors", SENSOR_COLS, default=["s2", "s3", "s4", "s7", "s11", "s12"],
                             format_func=lambda s: f"{s} - {SENSOR_NAMES[s]}")
    if sensors:
        long = e.melt(id_vars="cycle", value_vars=sensors, var_name="sensor")
        fig = px.line(long, x="cycle", y="value", facet_col="sensor", facet_col_wrap=3, height=420)
        fig.update_yaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig, width="stretch")
    st.dataframe(e, hide_index=True, width="stretch", height=260)

    with st.expander("Column dictionary and sensor variability"):
        std = df[SENSOR_COLS].std()
        st.dataframe(pd.DataFrame({"column": SENSOR_COLS, "description": [SENSOR_NAMES[s] for s in SENSOR_COLS],
                                   "std": std.values.round(5),
                                   "usable": np.where(std.values > CFG["features"]["min_std"], "yes", "constant")}),
                     hide_index=True, width="stretch")


def page_eda():
    st.title("🔍 Exploratory Data Analysis")
    tr = raw("train")
    meta = read_json("metadata.json") or {}
    sensors = meta.get("sensors") or [s for s in SENSOR_COLS if tr[s].std() > CFG["features"]["min_std"]]
    st.markdown(f"Training set **{SUBSET}**, {tr['unit'].nunique()} engines run to failure. "
                f"{len(SENSOR_COLS) - len(sensors)} of 21 sensors are constant here and carry no information.")

    st.subheader("Degradation patterns vs remaining life")
    s = st.selectbox("Sensor", sensors, format_func=lambda x: f"{x} - {SENSOR_NAMES[x]}")
    units = sorted(tr["unit"].unique())[:: max(1, len(tr["unit"].unique()) // 12)]
    sub = tr[tr.unit.isin(units)]
    c1, c2 = st.columns(2)
    fig = px.line(sub, x="rul", y=s, color="unit", title=f"{s} for 12 engines (aligned at failure)", height=380)
    fig.update_xaxes(autorange="reversed", title="RUL (cycles to failure)")
    fig.update_layout(showlegend=False)
    c1.plotly_chart(fig, width="stretch")
    b = tr.assign(rul_bucket=pd.cut(tr["rul"], [-1, 25, 50, 100, 150, 400],
                                    labels=["0-25", "26-50", "51-100", "101-150", "150+"]))
    fig = px.box(b, x="rul_bucket", y=s, color_discrete_sequence=[BLUE], title=f"{s} distribution by remaining life",
                 height=380)
    c2.plotly_chart(fig, width="stretch")

    st.subheader("Fleet-average sensor trend (z-scored)")
    z = tr[["rul", *sensors]].copy()
    z[sensors] = (z[sensors] - z[sensors].mean()) / z[sensors].std()
    trend = z[z.rul <= 200].groupby("rul")[sensors].mean().reset_index().melt(id_vars="rul", var_name="sensor")
    fig = px.line(trend, x="rul", y="value", color="sensor", height=420,
                  labels={"value": "mean z-score", "rul": "RUL (cycles to failure)"})
    fig.update_xaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.caption("Sensors such as s2, s3, s4, s11 rise towards failure while s7, s12, s20, s21 fall - "
               "the signal the model learns. Sensors s9 and s14 behave differently per engine.")

    st.subheader("Correlations")
    corr = tr[[*sensors, "rul"]].assign(rul=tr["rul"].clip(upper=CAP)).corr()
    c1, c2 = st.columns([3, 2])
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=520, text_auto=".2f")
    fig.update_traces(textfont_size=8)
    c1.plotly_chart(fig, width="stretch")
    r = corr["rul"].drop("rul").sort_values()
    fig = px.bar(r, orientation="h", color=r.values, color_continuous_scale="RdBu_r", range_color=[-1, 1],
                 title=f"Correlation with RUL (capped at {CAP})", height=520, labels={"value": "Pearson r", "index": ""})
    fig.update_layout(coloraxis_showscale=False)
    c2.plotly_chart(fig, width="stretch")


def page_etl():
    st.title("⚙️ ETL & Feature Store")
    st.markdown("""
| Stage | What happens | Code |
|---|---|---|
| **Extract** | Read space-separated NASA text files (26 columns) | `src/data.py::load_split` |
| **Validate** | Schema, numeric types, NaN/inf, duplicate (unit, cycle), cycle ≥ 1 | `src/data.py::validate` |
| **Label** | Train: `RUL = last cycle − cycle`, capped at %d. Test: `RUL = RUL_file + last cycle − cycle` | `add_train_rul`, `add_test_rul` |
| **Clean** | Drop sensors that are constant in training data | `src/features.py::select_sensors` |
| **Transform** | Per-engine *causal* rolling mean/std, slope, delta-from-baseline | `src/features.py::build_features` |
| **Load** | Parquet tables + JSON metadata (version, schema, SHA-256) | `src/features.py::FeatureStore` |
""" % CAP)
    store = FeatureStore(path(CFG["features"]["feature_store_dir"]))
    tables = store.list()
    if not tables:
        st.warning("Feature store is empty. Run `python -m src.train`.")
        return
    st.subheader("Feature store tables")
    metas = [store.metadata(t) for t in tables]
    st.dataframe(pd.DataFrame([{"table": m["name"], "rows": m["rows"], "columns": len(m["columns"]),
                                "version": m["feature_version"], "created (UTC)": m["created_utc"],
                                "label": m.get("label"), "sha256": m["sha256"][:16] + "…"} for m in metas]),
                 hide_index=True, width="stretch")
    m = metas[0]
    c = st.columns(4)
    c[0].metric("Sensors kept", len(m["sensors"]))
    c[1].metric("Sensors dropped", len(m["dropped_sensors"]), help=", ".join(m["dropped_sensors"]))
    c[2].metric("Model features", len(m["feature_columns"]))
    c[3].metric("Rolling window", f"{m['window']} cycles")

    st.subheader("Engineered features for one engine")
    f = store.read(tables[-1] if "train" in tables[-1] else tables[0])
    unit = st.selectbox("Training engine", sorted(f["unit"].unique()))
    s = st.selectbox("Sensor ", m["sensors"], index=m["sensors"].index("s4") if "s4" in m["sensors"] else 0)
    e = f[f.unit == unit]
    c1, c2 = st.columns(2)
    fig = go.Figure([go.Scatter(x=e.cycle, y=e[s], name="raw", line=dict(color="#afb8c1")),
                     go.Scatter(x=e.cycle, y=e[f"{s}_mean"], name=f"rolling mean ({m['window']})", line=dict(color=BLUE))])
    fig.update_layout(title=f"{s}: raw vs rolling mean", height=330, xaxis_title="cycle")
    c1.plotly_chart(fig, width="stretch")
    fig = go.Figure([go.Scatter(x=e.cycle, y=e[f"{s}_slope"], name="slope", line=dict(color=ORANGE)),
                     go.Scatter(x=e.cycle, y=e[f"{s}_delta"], name="delta from baseline", yaxis="y2",
                                line=dict(color=GREEN))])
    fig.update_layout(title=f"{s}: trend features", height=330, xaxis_title="cycle",
                      yaxis2=dict(overlaying="y", side="right"))
    c2.plotly_chart(fig, width="stretch")
    fig = px.line(e, x="cycle", y="rul", title=f"Piece-wise linear training label (cap = {CAP})", height=280,
                  color_discrete_sequence=[RED])
    st.plotly_chart(fig, width="stretch")
    with st.expander("All feature columns"):
        st.write(", ".join(m["feature_columns"]))
        st.dataframe(e.tail(10), hide_index=True, width="stretch")


@st.cache_data(ttl=30, show_spinner=False)
def mlflow_runs():
    import mlflow
    db = path(CFG["mlflow"]["tracking_uri"].removeprefix("sqlite:///"))
    if not db.exists():
        return None
    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
    return mlflow.search_runs(experiment_names=[CFG["mlflow"]["experiment_name"]], order_by=["start_time DESC"])


def page_experiments():
    st.title("🧪 Model Experiments (MLflow)")
    res = read_json("experiment_results.json")
    meta = read_json("metadata.json")
    if not res:
        st.warning("No experiments yet. Run `python -m src.train`.")
        return
    st.markdown(f"Each model is a `StandardScaler → regressor` pipeline evaluated with **{CFG['training']['cv_folds']}-fold "
                "GroupKFold by engine** (an engine never appears in both training and validation folds), then refit on "
                f"all training engines and scored once on the official test set. Selection metric: "
                f"**{CFG['training']['selection_metric']}**.")
    df = pd.DataFrame(res)
    best = meta["model_name"] if meta else None
    show = df[["model", "cv_rmse", "cv_mae", "cv_r2", "test_rmse", "test_mae", "test_r2", "test_nasa_score",
               "train_seconds"]].round(3)
    st.dataframe(show.style.apply(lambda r: ["background-color: rgba(26,127,55,0.15)" if r["model"] == best else ""
                                             for _ in r], axis=1), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    long = df.melt(id_vars="model", value_vars=["cv_rmse", "test_rmse", "cv_mae", "test_mae"])
    fig = px.bar(long, x="model", y="value", color="variable", barmode="group", height=380,
                 title="Error in cycles (lower is better)", color_discrete_sequence=[BLUE, "#79c0ff", ORANGE, "#ffc680"])
    c1.plotly_chart(fig, width="stretch")
    fig = px.bar(df, x="model", y="test_nasa_score", color_discrete_sequence=[RED], height=380,
                 title="NASA asymmetric score on test set (lower is better)")
    c2.plotly_chart(fig, width="stretch")

    fi = read_csv("feature_importance.csv")
    if fi is not None and len(fi):
        fig = px.bar(fi.head(20).iloc[::-1], x="importance", y="feature", orientation="h", height=480,
                     title=f"Top 20 features - {best}", color_discrete_sequence=[BLUE])
        st.plotly_chart(fig, width="stretch")

    st.subheader("MLflow tracking store")
    try:
        runs = mlflow_runs()
    except Exception as e:  # noqa: BLE001
        runs, err = None, e
        st.warning(f"Could not read MLflow store: {err}")
    if runs is not None and len(runs):
        cols = ["run_id", "tags.mlflow.runName", "status", "start_time"] + \
               [c for c in runs.columns if c.startswith("metrics.") and c.split(".")[1] in
                ("cv_rmse", "test_rmse", "test_mae", "test_r2")]
        st.dataframe(runs[cols], hide_index=True, width="stretch")
        st.caption(f"{len(runs)} runs in experiment `{CFG['mlflow']['experiment_name']}`. Best model registered as "
                   "`aircraft-rul-model` in the MLflow Model Registry.")
    st.code("mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000", language="bash")


def page_evaluation():
    st.title("📈 Model Evaluation")
    need_model()
    tp = read_csv("test_predictions.csv")
    models = [c for c in tp.columns if c not in ("unit", "true_rul")]
    best = read_json("metadata.json")["model_name"]
    m = st.selectbox("Model", models, index=models.index(best))
    y, p = tp["true_rul"], tp[m]
    err = p - y
    from src.metrics import regression_report
    rep = regression_report(y, p)
    c = st.columns(5)
    c[0].metric("MAE", f"{rep['mae']:.2f}")
    c[1].metric("RMSE", f"{rep['rmse']:.2f}")
    c[2].metric("R²", f"{rep['r2']:.3f}")
    c[3].metric("NASA score", f"{rep['nasa_score']:.1f}")
    c[4].metric("Late predictions", f"{(err > 0).mean():.0%}", help="Predicted RUL > true RUL (riskier)")
    st.caption(f"{len(tp)} test engines, prediction at last observed cycle vs true RUL from RUL_{SUBSET}.txt (uncapped).")

    c1, c2 = st.columns(2)
    fig = px.scatter(tp, x="true_rul", y=m, hover_data=["unit"], color=err, color_continuous_scale="RdBu_r",
                     range_color=[-40, 40], labels={m: "Predicted RUL", "true_rul": "True RUL", "color": "error"},
                     title="Predicted vs actual RUL", height=420)
    lim = max(y.max(), p.max()) + 5
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="gray"))
    fig.add_hline(y=CAP, line_dash="dot", line_color=ORANGE, annotation_text=f"model cap {CAP}")
    c1.plotly_chart(fig, width="stretch")
    fig = px.histogram(err, nbins=30, title="Error distribution (pred − true)", height=420,
                       color_discrete_sequence=[BLUE], labels={"value": "error (cycles)"})
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(showlegend=False)
    c2.plotly_chart(fig, width="stretch")

    st.subheader("Error by remaining life")
    b = pd.DataFrame({"bucket": pd.cut(y, [-1, 25, 50, 75, 100, 125, 400],
                                       labels=["0-25", "26-50", "51-75", "76-100", "101-125", ">125"]),
                      "abs_error": err.abs(), "error": err})
    agg = b.groupby("bucket", observed=True).agg(engines=("error", "size"), mae=("abs_error", "mean"),
                                                 bias=("error", "mean")).reset_index()
    c1, c2 = st.columns([2, 1])
    fig = px.bar(agg, x="bucket", y="mae", text="engines", color_discrete_sequence=[ORANGE], height=330,
                 title="MAE per true-RUL bucket (bar label = #engines)")
    c1.plotly_chart(fig, width="stretch")
    c2.dataframe(agg.round(2), hide_index=True, width="stretch")
    st.caption("Errors are smallest close to failure, where accuracy matters most. Engines with true RUL above the "
               f"{CAP}-cycle cap are systematically under-predicted by design (conservative).")

    st.subheader("RUL trajectory of a test engine")
    pr = predictor()
    te = raw("test")
    unit = st.selectbox("Test engine", sorted(te.unit.unique()))
    e = te[te.unit == unit]
    f = build_features(e, pr.sensors, pr.window)
    traj = np.clip(pr.pipeline.predict(f[pr.features]), 0, CAP)
    fig = go.Figure([go.Scatter(x=e.cycle, y=e.rul, name="true RUL", line=dict(color="gray", dash="dash")),
                     go.Scatter(x=e.cycle, y=e.rul.clip(upper=CAP), name=f"true RUL capped", line=dict(color="#afb8c1")),
                     go.Scatter(x=e.cycle, y=traj, name="predicted RUL", line=dict(color=BLUE, width=3))])
    fig.update_layout(height=380, xaxis_title="cycle", yaxis_title="RUL (cycles)")
    st.plotly_chart(fig, width="stretch")

    cv = read_csv("cv_predictions.csv.gz")
    if cv is not None and m in cv:
        with st.expander("Cross-validation (out-of-fold) residuals on training engines"):
            s = cv.sample(min(5000, len(cv)), random_state=0)
            fig = px.scatter(s, x="true_rul", y=s[m] - s["true_rul"], opacity=0.3, height=360,
                             labels={"y": "OOF error", "true_rul": "true RUL (capped)"})
            st.plotly_chart(fig, width="stretch")


def call_predict(df: pd.DataFrame, engine_id: str, use_api: bool):
    if use_api:
        r = requests.post(f"{API_URL}/predict", json={"engine_id": engine_id,
                                                       "records": df.drop(columns=["unit"], errors="ignore").to_dict("records")},
                          timeout=15)
        if r.status_code != 200:
            raise InputError(f"API returned {r.status_code}: {r.json().get('detail', r.text)}")
        return r.json(), "FastAPI"
    out = predictor().predict_history(df)
    return {**out, "health_status": band(out["predicted_rul"])[0]}, "local model"


def show_prediction(out, source, true_rul=None):
    rul = out["predicted_rul"]
    label, kind = band(rul)
    c1, c2 = st.columns([1, 1])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=rul, number={"suffix": " cycles"},
        gauge={"axis": {"range": [0, CAP]}, "bar": {"color": "#24292f"},
               "steps": [{"range": [0, 25], "color": "#ffebe9"}, {"range": [25, 75], "color": "#fff8c5"},
                         {"range": [75, CAP], "color": "#dafbe1"}]},
        title={"text": "Predicted remaining useful life"}))
    fig.update_layout(height=300, margin=dict(t=60, b=10))
    c1.plotly_chart(fig, width="stretch")
    c2.markdown(f"### {pill(label, kind)}", unsafe_allow_html=True)
    c2.markdown(f"- Last observed cycle: **{out['last_cycle']}** ({out['cycles_observed']} cycles of history)\n"
                f"- Model: `{out['model_name']}` v`{out['model_version']}` via **{source}**"
                + (f"\n- API latency: {out['latency_ms']} ms" if "latency_ms" in out else ""))
    if true_rul is not None:
        c2.markdown(f"- True RUL (from NASA labels): **{true_rul:.0f}** → error **{rul - true_rul:+.1f}** cycles")
    action = {"CRITICAL": "Ground the aircraft and schedule engine maintenance immediately.",
              "WARNING": "Plan maintenance within the next maintenance window.",
              "HEALTHY": "Continue normal operation and monitoring."}[label]
    c2.info(action)


def page_predict():
    st.title("🔮 RUL Prediction")
    need_model()
    pr = predictor()
    code, _ = api_get("/health")
    api_ok = code == 200
    use_api = st.toggle("Use FastAPI backend", value=api_ok,
                        help=f"{API_URL}/predict. Off = load the same model bundle in-process.")
    if use_api and not api_ok:
        st.warning(f"API not reachable at {API_URL}. Start it with `uvicorn api.main:app --port 8000` or turn off the toggle.")
    tab1, tab2 = st.tabs(["Example test engine", "Upload sensor history CSV"])

    with tab1:
        te = raw("test")
        c1, c2 = st.columns(2)
        unit = c1.selectbox("FD001 test engine", sorted(te.unit.unique()), index=0)
        e = te[te.unit == unit]
        cut = c2.slider("Use history up to cycle", int(e.cycle.min()), int(e.cycle.max()), int(e.cycle.max()))
        h = e[e.cycle <= cut]
        true_rul = float(h["rul"].iloc[-1])
        fig = px.line(h.melt(id_vars="cycle", value_vars=pr.sensors[:6], var_name="sensor"), x="cycle", y="value",
                      facet_col="sensor", facet_col_wrap=6, height=220)
        fig.update_yaxes(matches=None, showticklabels=False, title="")
        st.plotly_chart(fig, width="stretch")
        if st.button("Predict RUL", type="primary", key="p1"):
            try:
                out, src = call_predict(h[COLUMNS], f"FD001-{unit}", use_api)
                show_prediction(out, src, true_rul)
            except (InputError, requests.RequestException) as ex:
                st.error(str(ex))

    with tab2:
        need = ["cycle", *pr.sensors]
        st.markdown(f"CSV with columns **{', '.join(need)}** (other columns ignored). Optional `unit` column for "
                    "several engines. Raw NASA `.txt` files (26 space-separated columns) are also accepted.")
        tmpl = raw("test")
        tmpl = tmpl[tmpl.unit == 1][["unit", *need]]
        st.download_button("Download example CSV (test engine 1)", tmpl.to_csv(index=False), "engine_history_example.csv",
                           "text/csv")
        up = st.file_uploader("Sensor history", type=["csv", "txt"])
        if up is not None:
            try:
                text = up.getvalue().decode("utf-8")
                df = (pd.read_csv(StringIO(text), sep=r"\s+", header=None, names=COLUMNS) if up.name.endswith(".txt")
                      else pd.read_csv(StringIO(text)))
                if "unit" not in df:
                    df["unit"] = 1
                st.write(f"{len(df)} rows, {df['unit'].nunique()} engine(s)")
                st.dataframe(df.head(), hide_index=True, width="stretch")
                results = []
                for u, g in df.groupby("unit"):
                    out, src = call_predict(g, f"upload-{u}", use_api)
                    results.append({"unit": u, **out})
                res = pd.DataFrame(results)
                if len(res) == 1:
                    show_prediction(results[0], src)
                else:
                    fig = px.bar(res.sort_values("predicted_rul"), x="unit", y="predicted_rul", color="health_status",
                                 color_discrete_map={"CRITICAL": RED, "WARNING": "#bf8700", "HEALTHY": GREEN}, height=360)
                    fig.update_xaxes(type="category")
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(res, hide_index=True, width="stretch")
                    st.download_button("Download predictions", res.to_csv(index=False), "rul_predictions.csv")
                rep = drift_report(pr.drift_reference, df)
                if (rep["status"] != "stable").any():
                    st.warning("Input drift vs training data detected for: "
                               + ", ".join(rep.loc[rep.status != "stable", "sensor"]))
            except (InputError, ValueError, KeyError, requests.RequestException) as ex:
                st.error(f"Could not score file: {ex}")


def parse_metrics(text: str) -> dict:
    from prometheus_client.parser import text_string_to_metric_families
    out = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            out.setdefault(s.name, []).append((s.labels, s.value))
    return out


def page_monitoring():
    st.title("📡 Monitoring")
    st.markdown("Three different questions, three kinds of signal:\n"
                "1. **Operational** - is the service up, fast and error-free? (requests, errors, latency)\n"
                "2. **Input data drift** - do live sensor values still look like the training data? (PSI)\n"
                "3. **Model performance** - are predictions accurate? Needs ground truth, which for RUL only "
                "arrives later (after maintenance/failure). Offline: test-set metrics; online: `/feedback` MAE.")
    code, health = api_get("/health")
    c = st.columns(3)
    c[0].markdown("**API** " + (pill("UP", "ok") if code == 200 else pill("DOWN", "bad")), unsafe_allow_html=True)
    c[0].caption(API_URL)
    try:
        up = prom("up")
        prom_ok = True
    except Exception:  # noqa: BLE001
        up, prom_ok = [], False
    c[1].markdown("**Prometheus** " + (pill("UP", "ok") if prom_ok else pill("UNREACHABLE", "bad")),
                  unsafe_allow_html=True)
    c[1].caption(PROM_URL)
    c[2].markdown("**Grafana** dashboard: [localhost:3000](http://localhost:3000) (admin / admin)")

    if prom_ok:
        st.subheader("Scrape targets")
        st.dataframe(pd.DataFrame([{"job": r["metric"].get("job"), "instance": r["metric"].get("instance"),
                                    "up": "UP" if r["value"][1] == "1" else "DOWN"} for r in up]),
                     hide_index=True, width="stretch")
        jobs = sorted({r["metric"]["job"] for r in up if r["metric"].get("job", "").startswith("rul-api")})
        job = st.selectbox("Deployment to inspect", jobs) if jobs else None
        if job:
            sel = f'job="{job}"'

            def scalar(q):
                r = prom(q)
                return float(r[0]["value"][1]) if r else 0.0

            c = st.columns(5)
            c[0].metric("HTTP requests", f"{scalar(f'sum(rul_http_requests_total{{{sel}}})'):.0f}")
            c[1].metric("Predictions", f"{scalar(f'sum(rul_predictions_total{{{sel}}})'):.0f}")
            c[2].metric("Prediction errors", f"{scalar(f'sum(rul_prediction_errors_total{{{sel}}})'):.0f}")
            p95 = scalar(f"histogram_quantile(0.95, sum by (le) (rate(rul_prediction_latency_seconds_bucket{{{sel}}}[5m])))")
            c[3].metric("p95 latency (5m)", f"{p95 * 1000:.0f} ms" if p95 == p95 else "n/a")
            c[4].metric("Feedback MAE", f"{scalar(f'max(rul_feedback_rolling_mae{{{sel}}})'):.2f}")
            minutes = st.select_slider("Window", [5, 15, 30, 60, 180], value=30, format_func=lambda m: f"{m} min")
            c1, c2 = st.columns(2)
            for col, title, q in [
                (c1, "Request rate (req/s) by status", f"sum by (status) (rate(rul_http_requests_total{{{sel},endpoint=\"/predict\"}}[1m]))"),
                (c2, "Prediction latency p50 / p95 (s)", f"histogram_quantile(0.95, sum by (le) (rate(rul_prediction_latency_seconds_bucket{{{sel}}}[1m])))"),
                (c1, "Errors by type (cumulative)", f"sum by (type) (rul_prediction_errors_total{{{sel}}})"),
                (c2, "Rolling MAE from ground-truth feedback", f"rul_feedback_rolling_mae{{{sel}}}"),
            ]:
                try:
                    d = prom_range(q, minutes)
                    if title.startswith("Prediction latency"):
                        d2 = prom_range(q.replace("0.95", "0.5"), minutes).assign(series="p50")
                        d = pd.concat([d.assign(series="p95"), d2])
                    if d.empty:
                        col.info(f"{title}: no data in window yet")
                    else:
                        col.plotly_chart(px.line(d, x="time", y="value", color="series", title=title, height=300),
                                         width="stretch")
                except Exception as e:  # noqa: BLE001
                    col.warning(f"{title}: {e}")

            st.subheader("Input data drift (PSI per sensor)")
            psi = prom(f"rul_input_psi{{{sel}}}")
            if psi:
                d = pd.DataFrame([{"sensor": r["metric"]["sensor"], "psi": float(r["value"][1])} for r in psi])
                d["status"] = pd.cut(d.psi, [-1, 0.1, 0.25, 1e9], labels=["stable", "moderate", "significant"])
                fig = px.bar(d.sort_values("psi"), x="sensor", y="psi", color="status", height=340,
                             color_discrete_map={"stable": GREEN, "moderate": "#bf8700", "significant": RED})
                fig.add_hline(y=0.1, line_dash="dot")
                fig.add_hline(y=0.25, line_dash="dot", line_color=RED)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No drift values yet - the API needs ≥ 50 buffered input rows.")
    else:
        st.info("Prometheus is not reachable, so no time series. Start the stack with "
                "`docker compose -f docker/docker-compose.yml up -d`. Showing the API's raw /metrics instead.")
        if code == 200:
            _, text = api_get("/metrics")
            mt = parse_metrics(text)

            def total(name):
                return sum(v for _, v in mt.get(name, []))
            c = st.columns(4)
            c[0].metric("HTTP requests", f"{total('rul_http_requests_total'):.0f}")
            c[1].metric("Predictions", f"{total('rul_predictions_total'):.0f}")
            c[2].metric("Prediction errors", f"{total('rul_prediction_errors_total'):.0f}")
            n, s = total("rul_prediction_latency_seconds_count"), total("rul_prediction_latency_seconds_sum")
            c[3].metric("Mean latency", f"{s / n * 1000:.1f} ms" if n else "n/a")
            _, d = api_get("/drift")
            if isinstance(d, dict) and d.get("sensors"):
                st.dataframe(pd.DataFrame(d["sensors"]).round(4), hide_index=True, width="stretch")

    st.subheader("Model performance (offline, ground truth available)")
    meta = read_json("metadata.json")
    if meta:
        m = meta["metrics"]
        c = st.columns(4)
        c[0].metric("Test MAE", f"{m['test_mae']:.2f}")
        c[1].metric("Test RMSE", f"{m['test_rmse']:.2f}")
        c[2].metric("CV RMSE", f"{m['cv_rmse']:.2f}")
        c[3].metric("NASA score", f"{m['test_nasa_score']:.1f}")
    st.code("python scripts/simulate_traffic.py --n 100              # normal traffic + ground-truth feedback\n"
            "python scripts/simulate_traffic.py --n 60 --error-rate 0.3 --drift 3   # errors + drifted sensors",
            language="bash")


IN_CONTAINER = Path("/.dockerenv").exists()
NO_CLI_NOTE = ("Not available here: the dashboard is running inside a Docker container, which deliberately has no "
               "access to the host's Docker daemon or kubectl (mounting the Docker socket would give the container "
               "root on the host). Deployment health above comes from Prometheus instead. For full CLI status run "
               "the dashboard on the host: `streamlit run streamlit_app/app.py`.")


def cli_status(args: list[str], empty: str):
    rc, out = run_cmd(args)
    if rc is None and IN_CONTAINER:
        st.info(NO_CLI_NOTE)
    else:
        st.code(out or empty, language="text")


def page_pipeline():
    st.title("🚀 MLOps Pipeline")
    st.caption("Status below is queried live from Prometheus, GitHub, git, Docker, kubectl and the Terraform state "
               "file. Nothing here is hard-coded.")
    repo = CFG["github"]["repo"]

    st.subheader("Deployments (live health via Prometheus scrapes)")
    targets = {"rul-api-compose": ("Docker Compose", ":8000"), "rul-api-terraform": ("Terraform", ":8001"),
               "rul-api-kubernetes": ("Kubernetes", ":30080")}
    try:
        up = {r["metric"]["job"]: r["value"][1] == "1" for r in prom('up{job=~"rul-api.*"}')}
        c = st.columns(3)
        for col, (job, (label, port)) in zip(c, targets.items()):
            state = pill("UP", "ok") if up.get(job) else pill("DOWN", "bad") if job in up else pill("NOT SCRAPED", "warn")
            col.markdown(f"**{label}** {state}<br><span class='muted'>job {job}, port {port}</span>",
                         unsafe_allow_html=True)
    except Exception:  # noqa: BLE001
        st.info(f"Prometheus not reachable at {PROM_URL}; start `docker compose -f docker/docker-compose.yml up -d`.")

    t = st.tabs(["Git & GitHub Actions", "Docker", "Kubernetes", "Terraform"])

    with t[0]:
        st.markdown(f"**Repository:** [github.com/{repo}](https://github.com/{repo})")
        rc, out = run_cmd(["git", "log", "--oneline", "--graph", "--decorate", "-n", "25"])
        if rc == 0:
            st.code(out, language="text")
        else:  # no local git (e.g. inside the container): read the history from GitHub
            try:
                r = requests.get(f"https://api.github.com/repos/{repo}/commits?per_page=20", timeout=5)
                r.raise_for_status()
                st.caption("Commit history from the GitHub API (no local git in this environment)")
                st.dataframe(pd.DataFrame([{"sha": x["sha"][:7], "message": x["commit"]["message"].splitlines()[0],
                                            "author": x["commit"]["author"]["name"],
                                            "date": x["commit"]["author"]["date"]} for x in r.json()]),
                             hide_index=True, width="stretch")
            except requests.RequestException as e:
                st.info(f"No local git and GitHub API not reachable: {e}")
        st.markdown("**GitHub Actions runs**")
        try:
            r = requests.get(f"https://api.github.com/repos/{repo}/actions/runs?per_page=15", timeout=5)
            runs = r.json().get("workflow_runs", []) if r.ok else []
            if runs:
                st.dataframe(pd.DataFrame([{"workflow": x["name"], "branch": x["head_branch"], "event": x["event"],
                                            "status": x["status"], "conclusion": x["conclusion"],
                                            "commit": x["head_commit"]["message"].splitlines()[0],
                                            "started": x["run_started_at"], "url": x["html_url"]} for x in runs]),
                             hide_index=True, width="stretch",
                             column_config={"url": st.column_config.LinkColumn("run")})
            else:
                st.info(f"No workflow runs visible (HTTP {r.status_code}). Push to GitHub to trigger CI; "
                        "private repos need authentication.")
        except requests.RequestException as e:
            st.info(f"GitHub API not reachable: {e}")
        st.code("git checkout -b feature/my-change\n# edit code\ngit commit -am \"Describe change\"\n"
                "git push -u origin feature/my-change   # CI runs on push + pull request", language="bash")

    with t[1]:
        cli_status(["docker", "ps", "-a", "--filter", "name=rul", "--format",
                    "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"], "no rul containers")
        cli_status(["docker", "images", "rul-api", "--format",
                    "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"], "no rul-api image")
        st.code("docker build -f docker/Dockerfile -t rul-api:1.0.0 .\n"
                "docker compose -f docker/docker-compose.yml up -d --build\n"
                "docker compose -f docker/docker-compose.yml ps", language="bash")

    with t[2]:
        cli_status(["kubectl", "get", "deploy,pods,svc", "-n", "rul-mlops", "-o", "wide"], "namespace rul-mlops is empty")
        st.code("kubectl apply -f kubernetes/\nkubectl get pods -n rul-mlops\n"
                "kubectl scale deployment rul-api -n rul-mlops --replicas=5\n"
                "kubectl get pods -n rul-mlops", language="bash")

    with t[3]:
        state = ROOT / "terraform" / "terraform.tfstate"
        if state.exists():
            s = json.loads(state.read_text() or "{}")
            res = s.get("resources", [])
            if res:
                st.success(f"Terraform state has {len(res)} managed resource(s) (serial {s.get('serial')}).")
                st.dataframe(pd.DataFrame([{"type": r["type"], "name": r["name"],
                                            "id": r["instances"][0]["attributes"].get("id", "")[:20] if r["instances"] else ""}
                                           for r in res]), hide_index=True, width="stretch")
            else:
                st.info("Terraform state exists but is empty (resources destroyed).")
        else:
            st.info("No Terraform state yet - `terraform apply` has not been run.")
        st.code("cd terraform\nterraform init\nterraform validate\nterraform plan\nterraform apply -auto-approve\n"
                "terraform destroy -auto-approve", language="bash")


PAGES = {
    "🏠 Project Overview": page_overview,
    "🗂️ Dataset Explorer": page_dataset,
    "🔍 Exploratory Data Analysis": page_eda,
    "⚙️ ETL & Feature Store": page_etl,
    "🧪 Model Experiments": page_experiments,
    "📈 Model Evaluation": page_evaluation,
    "🔮 RUL Prediction": page_predict,
    "📡 Monitoring": page_monitoring,
    "🚀 MLOps Pipeline": page_pipeline,
}

with st.sidebar:
    st.markdown("## ✈️ Engine RUL MLOps")
    st.caption(f"NASA C-MAPSS {SUBSET} · v{CFG['project']['version']}")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Data: NASA Prognostics Center of Excellence, Turbofan Engine Degradation Simulation "
               "(Saxena & Goebel, 2008).")

need_data()
PAGES[choice]()
