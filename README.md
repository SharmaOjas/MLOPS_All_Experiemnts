# ✈️ Aircraft Engine Predictive Maintenance: End-to-End MLOps

Predict the **Remaining Useful Life (RUL)** of turbofan engines, meaning the number of operating cycles left
before an engine reaches its failure threshold, from multivariate sensor histories in the
**NASA C-MAPSS** dataset. One project covers all seven MLOps experiments: Git, Scrum, CI/CD,
ETL + feature store + MLflow, Docker + Kubernetes, Terraform, and Prometheus + Grafana. A single
**Streamlit dashboard** ties them together.

```
NASA C-MAPSS ─► ETL + validation ─► Feature store ─► GroupKFold experiments ─► MLflow (runs + registry)
                                                                   │
                                                                   ▼
             Streamlit dashboard ──► FastAPI /predict ◄── best model bundle (models/rul_model.joblib)
                     ▲                    │ /metrics
                     └──── Prometheus ◄───┘ ──► Grafana
   GitHub Actions: tests on push/PR  ·  deployed via Docker Compose | Kubernetes (3→5 replicas) | Terraform
```

## Results (FD001, measured on the official test set)

Every model is a `StandardScaler → regressor` pipeline. The model is **selected by 5-fold GroupKFold CV
RMSE (by engine)** and only then scored once on the official test set (`RUL_FD001.txt`, prediction at
each engine's last observed cycle, uncapped true RUL).

| Model | CV RMSE | CV MAE | Test RMSE | Test MAE | Test R² | NASA score | Train time (s) |
|---|---|---|---|---|---|---|---|
| linear_regression | 15.99 | 12.49 | 17.45 | 14.06 | 0.824 | 477.5 | 0.9 |
| random_forest | 12.29 | 8.23 | 14.50 | 10.66 | 0.878 | 303.2 | 32.7 |
| gradient_boosting | 11.20 | 7.74 | 12.60 | 9.07 | 0.908 | 202.6 | 152.4 |
| hist_gradient_boosting **(selected)** | 10.86 | 7.31 | 13.00 | 9.17 | 0.902 | 230.9 | 13.5 |

*These numbers come from one local run (Windows, Python 3.13). Re-running `python -m src.train` regenerates
them in `models/experiment_results.json` and in MLflow. Gradient Boosting scores slightly better on the
test set, but selection uses CV only, so the test set never influences the choice.*

## Project structure

```
aircraft-rul-mlops/
├── config/config.yaml          # all paths, hyper-parameters, URLs
├── data/raw/CMAPSS/            # NASA files (downloaded, git-ignored)
├── src/
│   ├── data.py                 # extract, validate, RUL labels, download
│   ├── features.py             # causal per-engine features + FeatureStore
│   ├── train.py                # ETL → feature store → CV experiments → MLflow → best model
│   ├── predict.py              # RULPredictor (shared by API, dashboard, tests)
│   ├── metrics.py              # MAE, RMSE, R², NASA score
│   └── drift.py                # PSI input-drift reference + report
├── feature_store/              # Parquet tables + JSON metadata (generated)
├── models/                     # best model bundle + evaluation outputs (generated)
├── mlruns/                     # MLflow tracking DB + artifacts (generated)
├── api/main.py                 # FastAPI + Prometheus metrics
├── streamlit_app/app.py        # the main dashboard (9 sections)
├── scripts/simulate_traffic.py # real test-engine traffic + ground-truth feedback
├── tests/                      # pytest: data, features, metrics, model, API
├── docker/                     # Dockerfile (API), Dockerfile.streamlit, docker-compose.yml
├── kubernetes/                 # namespace, Deployment (3 replicas), Service
├── terraform/                  # Docker provider → API container on :8001
├── monitoring/                 # prometheus.yml, alert rules, Grafana provisioning + dashboard
├── .github/workflows/ci.yml    # CI: unit tests → train → full tests → Docker smoke test → infra validate
└── docs/                       # SCRUM.md, GIT_WORKFLOW.md, ASSUMPTIONS.md, RUNBOOK.md
```

## Quick start (Windows, PowerShell)

Requires Python 3.13 (3.12 also works), plus Docker Desktop for the container parts.

```powershell
cd aircraft-rul-mlops
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt

python -m src.data --download          # NASA C-MAPSS (~12 MB) -> data/raw/CMAPSS
python -m src.train                    # ETL, feature store, 4 MLflow runs, best model (~3-4 min)
pytest -v                              # 25 tests

uvicorn api.main:app --port 8000       # terminal 1: API  -> http://127.0.0.1:8000/docs
streamlit run streamlit_app/app.py     # terminal 2: dashboard -> http://localhost:8501
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000   # terminal 3: MLflow UI
```

Manual dataset download: get *Turbofan Engine Degradation Simulation Data Set* from the
[NASA PCoE Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
(direct: <https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip>).
Unzip it, unzip the inner `CMAPSSData.zip`, and copy the `train_FD00x.txt`, `test_FD00x.txt` and
`RUL_FD00x.txt` files to `data/raw/CMAPSS/`.

Docker, Kubernetes, Terraform, monitoring and GitHub Actions commands are in **[docs/RUNBOOK.md](docs/RUNBOOK.md)**,
along with the screenshot checklist for the practical submission.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness/readiness (503 if the model failed to load) |
| GET | `/model/info` | model name, version, metrics, required columns |
| POST | `/predict` | `{"engine_id": "...", "records": [{"cycle": 1, "s2": 642.1, ...}, ...]}` → RUL + health band |
| POST | `/feedback` | report true RUL later → live model-error metrics |
| GET | `/drift` | PSI per sensor for recent inputs |
| GET | `/metrics` | Prometheus exposition |

Status codes: 400 for semantic input errors (missing sensors, duplicate cycles, non-finite values),
422 for schema errors, and 503 when no model is loaded.

## How the ML is kept honest

- **Labels:** training RUL = `last_cycle − cycle`, capped at 125 (piece-wise linear target). Test RUL =
  `RUL_FD001[unit] + last_observed_cycle − cycle`, uncapped for evaluation.
- **No leakage:** GroupKFold by engine unit. Features are causal (cycle *t* only sees cycles ≤ *t* of the
  same engine), and `tests/test_features.py` checks this.
- **No train/serve skew:** training, API, dashboard and tests all call the same `build_features`. The model
  bundle stores the sensor list, window and cap.
- **No hard-coded results:** every number in the dashboard is read from training outputs, MLflow, the live
  API or Prometheus. Services that are down are shown as down.

## Seven experiments → where they live

| # | Experiment | In this project |
|---|---|---|
| 1 | Git & GitHub | feature branches merged with `--no-ff`, tag `v1.0.0`. See [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) |
| 2 | Agile & Scrum | [docs/SCRUM.md](docs/SCRUM.md) |
| 3 | CI/CD | [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| 4 | ETL, Feature Store, MLflow | `src/`, `feature_store/`, `mlruns/` |
| 5 | Docker & Kubernetes | `docker/`, `kubernetes/` |
| 6 | Terraform | `terraform/` |
| 7 | Prometheus & Grafana | `monitoring/`, `api/main.py` |

## Dataset attribution

A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", *Proc. 1st Int. Conf. on Prognostics and Health Management (PHM08)*, 2008.
Data: NASA Ames Prognostics Center of Excellence (PCoE) Data Set Repository, "Turbofan Engine Degradation
Simulation Data Set". The data is downloaded from NASA and is not redistributed in this repository.

Assumptions and limitations: [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md).
