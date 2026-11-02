# Experiment 1: Source code management with Git & GitHub

Remote: <https://github.com/SharmaOjas/MLOPS_All_Experiemnts>

## Branching model

`main` always holds working, tested code. Each piece of work gets a short-lived branch that is merged back
with `--no-ff`, so the merge shows up in history (the Scrum stories in [SCRUM.md](SCRUM.md) map onto these
branches):

| Branch | Content | Story |
|---|---|---|
| `feature/etl-feature-store` | data loading, validation, labels, features, feature store, drift + tests | US1-US5 |
| `feature/mlflow-training` | GroupKFold experiments, MLflow tracking/registry, predictor + model tests | US6-US7 |
| `feature/fastapi-service` | FastAPI service, Prometheus instrumentation, traffic simulator + API tests | US8 |
| `feature/streamlit-dashboard` | 9-section Streamlit dashboard | US9 |
| `feature/ci-cd` | GitHub Actions workflow | US10 |
| `feature/containerization` | Dockerfiles, Compose, Kubernetes manifests | US11 |
| `feature/terraform` | Terraform Docker-provider configuration | US12 |
| `feature/monitoring` | Prometheus config, alert rules, Grafana provisioning | US13 |
| `docs/project-documentation` | README, Scrum, runbook, assumptions | US14 |
| `demo/ci-failure` | deliberately failing commit + fix to demonstrate CI | Exp 3 |

Release: tag `v1.0.0` on `main`.

## Commands used

```bash
git config --global user.name "SharmaOjas"
git config --global user.email "<your email>"

git init -b main
git add .gitignore requirements.txt config/ pytest.ini
git commit -m "Initial project skeleton"

git checkout -b feature/etl-feature-store
git add src/ tests/
git commit -m "Add ETL, validation, RUL labels and feature store"
git checkout main
git merge --no-ff feature/etl-feature-store -m "Merge branch 'feature/etl-feature-store'"
# ... same for every feature branch ...

git tag -a v1.0.0 -m "First complete release"
git remote add origin https://github.com/SharmaOjas/MLOPS_All_Experiemnts.git
git push -u origin main --tags
git push origin --all          # publish the feature branches too
```

Useful inspection commands: `git status`, `git log --oneline --graph --all`, `git branch -a`,
`git show --stat HEAD`, `git diff main~1`.

## Merge conflict exercise

```bash
git checkout -b feature/window-15
# edit config/config.yaml: rolling_window: 15
git commit -am "Try a 15-cycle window"
git checkout main
git checkout -b feature/window-20
# edit the same line: rolling_window: 20
git commit -am "Try a 20-cycle window"
git checkout main
git merge feature/window-15        # fast-forward
git merge feature/window-20        # CONFLICT in config/config.yaml
# edit the file and keep one value, then:
git add config/config.yaml
git commit -m "Resolve window conflict"
```
