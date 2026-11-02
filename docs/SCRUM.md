# Experiment 2 - Agile & Scrum: Aircraft Engine RUL Predictive Maintenance

## Real-time scenario

An airline maintenance organisation wants to move from fixed-interval engine overhauls to
**condition-based maintenance**. Engineers need a system that reads engine sensor histories
and estimates the **Remaining Useful Life (RUL)** in operating cycles, so engines are serviced
before failure but not wastefully early. The team builds it incrementally in Scrum sprints,
using the NASA C-MAPSS FD001 dataset as the fleet's historical data.

## Scrum team

If one student performs the experiment, they hold all five roles.

| Role | Responsibilities in this project |
|---|---|
| **Product Owner** | Talks to maintenance engineers; defines the goal "predict RUL within ±15 cycles RMSE on FD001 and serve it through an API and dashboard"; owns and prioritises the Product Backlog; accepts completed stories |
| **Scrum Master** | Runs Sprint Planning, Daily Scrum, Review and Retrospective; removes blockers (Docker Desktop, Kubernetes, package versions); protects the sprint scope |
| **Data Engineer** | Downloads C-MAPSS, validates schema, builds RUL labels, engineers features, maintains the feature store |
| **ML Engineer** | Chooses algorithms, runs MLflow experiments with engine-wise CV, selects and packages the best model, builds the FastAPI service |
| **Tester / MLOps Engineer** | Writes pytest suites, GitHub Actions CI, Docker/K8s/Terraform deployment, Prometheus/Grafana monitoring; verifies predictions |

**Definition of Done:** code merged to `main` through a feature branch, CI green, tests cover the
change, documentation updated, and the feature visible in the Streamlit dashboard.

## Product Backlog

| ID | User story | Priority | Story points | Sprint |
|---|---|---|---|---|
| US1 | As a data engineer, I want to download and load the NASA C-MAPSS data so that the team has a reliable training source | High | 2 | 1 |
| US2 | As a data engineer, I want schema and quality validation so that corrupt sensor files are rejected early | High | 3 | 1 |
| US3 | As an ML engineer, I want correct RUL labels for train (run-to-failure) and test (separate RUL file) so that evaluation is honest | High | 3 | 1 |
| US4 | As an ML engineer, I want rolling, trend and baseline features per engine so that degradation is visible to the model | High | 5 | 1 |
| US5 | As an ML engineer, I want a reusable feature store so that training and inference use identical features | High | 3 | 1 |
| US6 | As an ML engineer, I want to compare Linear Regression, Random Forest and Gradient Boosting in MLflow using engine-wise CV so that the best model is chosen without leakage | High | 5 | 2 |
| US7 | As a maintenance planner, I want MAE, RMSE, R² and the NASA score so that I understand prediction risk | High | 2 | 2 |
| US8 | As a developer, I want a FastAPI `/predict` endpoint so that other systems can request RUL | High | 5 | 2 |
| US9 | As a maintenance engineer, I want a Streamlit dashboard to explore data and predict RUL for an engine or uploaded CSV | High | 8 | 3 |
| US10 | As a team, we want GitHub Actions CI that runs tests on every push and pull request | Medium | 3 | 3 |
| US11 | As an operator, I want the API containerised with Docker Compose and deployed on Kubernetes with 3→5 replicas | Medium | 5 | 4 |
| US12 | As an operator, I want Terraform to provision the API container reproducibly | Medium | 3 | 4 |
| US13 | As an operator, I want Prometheus metrics and Grafana panels for requests, errors, latency and input drift | Medium | 5 | 4 |
| US14 | As the product owner, I want README, assumptions and limitations documented | Low | 2 | 4 |
| US15 | (Future) Support FD002/FD004 with per-operating-condition normalisation | Low | 5 | Backlog |
| US16 | (Future) Automated retraining when drift or feedback MAE exceeds thresholds | Low | 8 | Backlog |

*Why High first:* data, labels and features are dependencies of every later story; a model or
dashboard built on wrong RUL labels would have to be thrown away.

## Sprint 1 - Data foundation

- **Duration:** 1 week  - **Sprint Goal:** "A validated, labelled FD001 feature table in the feature store."

| Task | Owner | Estimate | Status |
|---|---|---|---|
| Download dataset script (`src/data.py --download`) | Data Engineer | 1 h | Done |
| Validation rules + unit tests | Data Engineer / Tester | 2 h | Done |
| Train & test RUL label construction + tests | ML Engineer | 2 h | Done |
| Causal per-engine rolling/slope/delta features | Data Engineer | 4 h | Done |
| Parquet feature store with metadata hash | Data Engineer | 2 h | Done |

## Sprint 2 - Models and API

- **Sprint Goal:** "Best model tracked in MLflow and served by FastAPI."

| Task | Owner | Estimate | Status |
|---|---|---|---|
| GroupKFold CV training loop for 4 models | ML Engineer | 4 h | Done |
| MLflow tracking + model registry | ML Engineer | 2 h | Done |
| NASA scoring function + metrics tests | Tester | 1 h | Done |
| FastAPI `/predict`, `/health`, `/model/info`, `/metrics` | ML Engineer | 4 h | Done |
| API tests with TestClient | Tester | 2 h | Done |

## Sprint 3 - Dashboard and CI

- **Sprint Goal:** "Engineers can use the dashboard; every push is tested automatically."

| Task | Owner | Estimate | Status |
|---|---|---|---|
| Streamlit dashboard, 9 sections | ML Engineer | 8 h | Done |
| GitHub Actions workflow (test, Docker smoke test, infra validation) | MLOps Engineer | 3 h | Done |
| Demonstrate failing and fixed CI run | Tester | 1 h | Done |

## Sprint 4 - Deployment and monitoring

- **Sprint Goal:** "The API runs in Docker, Kubernetes and Terraform and is monitored."

| Task | Owner | Estimate | Status |
|---|---|---|---|
| Dockerfile + Compose stack | MLOps Engineer | 3 h | Done |
| K8s Deployment/Service, scale 3→5 | MLOps Engineer | 2 h | Done |
| Terraform Docker-provider config | MLOps Engineer | 2 h | Done |
| Prometheus metrics, alert rules, Grafana dashboard, drift PSI | MLOps Engineer | 4 h | Done |

## Daily Scrum (sample, Sprint 2 day 2)

| Member | Yesterday | Today | Blockers |
|---|---|---|---|
| Product Owner | Agreed acceptance metric: test RMSE < 15 cycles | Review CV results with the team | None |
| Scrum Master | Ran Sprint Planning | Facilitate Daily Scrum; fix Docker Desktop install | Docker Desktop needed WSL2 update |
| Data Engineer | Finished feature store | Pair with ML engineer on feature importance | None |
| ML Engineer | Ran LR and RF baselines | Add Gradient Boosting; register best model | MLflow 3 refuses to log tree models with default skops format → switched to cloudpickle |
| Tester | Wrote label tests | Write API tests | Need trained model in CI → CI now trains before tests |

## Sprint Review (end of Sprint 2)

- Demonstrated: MLflow UI with 4 runs, best model selected by **CV RMSE** (not test score), `/predict` returning RUL for real FD001 test engines.
- Sprint Goal achieved: **Yes**. Exact metrics are in `models/metadata.json` and the dashboard's *Model Experiments* page.
- Product Owner feedback: wants a health band (HEALTHY/WARNING/CRITICAL) and CSV upload → added to Sprint 3.

## Sprint Retrospective

| What went well | What went wrong | Improvements |
|---|---|---|
| Engine-wise CV prevented optimistic scores | Classic GradientBoosting took ~6.6 min per training run, too slow for CI | Tuned GB to fewer/shallower trees; HistGradientBoosting kept as the fast strong model |
| Shared `build_features` removed train/serve skew | Random row splits were tempting but would leak engine histories | Added a causality test to CI |
| Tests caught schema errors early | Package version mismatch between local and Docker | Pinned versions in `requirements.txt` |

## Sprint Increment (deliverables)

Working RUL model and API · Streamlit dashboard · Feature store · MLflow experiments ·
CI pipeline · Docker/K8s/Terraform deployment · Prometheus/Grafana monitoring · Documentation.
